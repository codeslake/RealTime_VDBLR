import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import collections
import torch.nn.utils as torch_utils
import copy

from utils import *
from data_loader.utils import *
from models.utils import *
from models.baseModel import baseModel
from models.archs.liteFlowNet import liteFlowNet

from data_loader.datasets_reds_lmdb import datasets
import models.trainers.lr_scheduler as lr_scheduler

from data_loader.data_sampler import DistIterSampler

from ptflops import get_model_complexity_info
import importlib
from shutil import copy2
import os

from torch import autograd
# torch.autograd.set_detect_anomaly(mode = False)

class Model(baseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.ws = torch.distributed.get_world_size() if config.dist else 1

        ### NETWORKS ###
        if self.rank <= 0 : print(toGreen('Loading Model...'))
        self.network = DeblurNet(config, is_distill = False if self.is_train is True else None).to(torch.device('cuda'))
        self.network1 = DeblurNet(config, is_distill = False).to(torch.device('cuda'))
        self.network2 = DeblurNet(config, is_distill = True).to(torch.device('cuda'))


        if self.is_train and self.config.resume is None or self.is_train and os.path.exists('./models/archs/{}.py'.format(config.network)):
            copy2('./models/archs/{}.py'.format(config.network), self.config.LOG_DIR.offset)
            copy2('./models/trainers/{}.py'.format(config.trainer), self.config.LOG_DIR.offset)

        self.liteFlowNet = liteFlowNet().to(torch.device('cuda'))
        print('liteFlowNet: ', self.liteFlowNet.load_state_dict(torch.load('./ckpt/liteFlowNet.pytorch')))

        ### PROFILE ###
        if self.rank <= 0:
            with torch.no_grad():
                Macs,params = get_model_complexity_info(self.network, (1, 3, 720, 1280), input_constructor = self.network.input_constructor, as_strings=False,print_per_layer_stat=config.is_verbose)


        ### INIT for training ###
        if self.is_train:
            self.itr_global = {'train': 0, 'valid': 0}
            self.itr_inc = {'train': self.config.frame_itr_num, 'valid': 1}
            self.network.init()

            self._set_optim()
            self._set_lr_scheduler()
            self._set_dataloader()

            if config.is_verbose:
                for name, param in self.network.named_parameters():
                    if self.rank <= 0: print(name, ', ', param.requires_grad)
        else:
            self._set_dataloader()

        ### DDP ###
        if config.dist:
            if self.rank <= 0: print(toGreen('Building Dist Parallel Model...'))
            self.network = DDP(self.network, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), broadcast_buffers=True, find_unused_parameters=False)
            for name, param in self.network.named_parameters():
                param.requires_grad = False
            # print('MTU_large: ', self.network.load_state_dict(torch.load('./ckpt/MTU_large.pytorch')))
            self.network1 = DDP(self.network1, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), broadcast_buffers=True, find_unused_parameters=True)
            self.network2 = DDP(self.network2, device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(), broadcast_buffers=True, find_unused_parameters=True)
        else:
            self.network = DP(self.network).to(torch.device('cuda'))
            for name, param in self.network.named_parameters():
                param.requires_grad = False
            self.network1 = DP(self.network1).to(torch.device('cuda'))
            self.network2 = DP(self.network2).to(torch.device('cuda'))

        # self.network1 = copy.deepcopy(self.network)
        # self.network2 = copy.deepcopy(self.network)
        net1_params = dict(self.network1.named_parameters())
        net2_params = dict(self.network2.named_parameters())
        for name, param in self.network.named_parameters():
            if name in net1_params.keys():
                net1_params[name].data = param.data
            if name in net2_params.keys():
                net2_params[name].data = param.data

        if self.rank <= 0:
            print(toGreen('Computing model complexity...'))
            print('{:<30}  {:<8} B'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
            print('{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2))
            if self.is_train:
                with open(config.LOG_DIR.offset + '/cost.txt', 'w') as f:
                    f.write('{:<30}  {:<8} B\n'.format('Computational complexity (Macs): ', Macs / 1000 ** 3 ))
                    f.write('{:<30}  {:<8} M'.format('Number of parameters: ',params / 1000 ** 2))
                    f.close()

        # for name, param in self.network.named_parameters():
        #     if self.rank <= 0: print(name, param.requires_grad)

    def get_itr_per_epoch(self, state):
        if state == 'train':
            return len(self.data_loader_train) * self.itr_inc[state]
        else:
            return len(self.data_loader_eval) * self.itr_inc[state]

    def _set_loss(self, lr = None):
        if self.rank <= 0: print(toGreen('Building Loss...'))
        self.MSE = torch.nn.MSELoss().to(torch.device('cuda'))
        self.MAE = torch.nn.L1Loss().to(torch.device('cuda'))
        self.CSE = torch.nn.CrossEntropyLoss(reduction='none').to(torch.device('cuda'))
        self.MSE_sum = torch.nn.MSELoss(reduction = 'sum').to(torch.device('cuda'))

    def _set_optim(self, lr = None):
        if self.rank <= 0: print(toGreen('Building Optim...'))
        self._set_loss()
        lr = self.config.lr_init if lr is None else lr

        for name, param in self.liteFlowNet.named_parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam([
            # {'params': self.network.parameters(), 'lr': self.config.lr_init, 'initial_lr': self.config.lr_init}
            {'params': [param for name, param in self.network1.named_parameters() if 'motion_layer' not in name], 'lr': self.config.lr_init, 'initial_lr': self.config.lr_init, 'lr_lambda': 1.0}
            ], eps= 1e-8, betas=(self.config.beta1, 0.999))

        # for g in optimizer.param_groups:
        #     for k, v in g.items():
        #         if 'param' not in k:
        #             if self.rank <= 0: print('!!! ', k,': ', v)

        self.optimizers.append(optimizer)

        # names = [name for name, param in self.network.named_parameters() if 'motion_layer' not in name]
        # for name in names:
        #     if self.rank<=0: print(name)

        optimizer = torch.optim.Adam([
            # {'params': self.network.parameters(), 'lr': self.config.lr_init * self.config.distill_lambda, 'initial_lr': self.config.lr_init * self.config.distill_lambda}
            {'params': [param for name, param in self.network2.named_parameters() if 'HGs.{}'.format(self.config.HG_num-1) not in name and 'out' not in name and 'deblur_layer' not in name], 'lr': self.config.lr_init * self.config.distill_lambda, 'initial_lr': self.config.lr_init * self.config.distill_lambda, 'lr_lambda': self.config.distill_lambda}
            ], eps= 1e-8, betas=(self.config.beta1, 0.999))

        # names = [name for name, param in self.network.named_parameters() if 'HGs.{}'.format(self.config.HG_num-1) not in name and 'out' not in name and 'deblur_layer' not in name]
        # for name in names:
        #     if self.rank<=0: print(name)

        self.optimizers.append(optimizer)

    def _set_lr_scheduler(self):
        if self.rank <= 0: print(toGreen('Loading Learning Rate Scheduler...'))
        if self.config.LRS == 'CA':
            if self.rank <= 0: print(toRed('\tCosine annealing scheduler...'))
            for i, optimizer in enumerate(self.optimizers):
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingLR_Restart(
                        optimizer, self.config.T_period, eta_min= self.config.eta_min if i == 0 else 1e-8,
                        restarts= self.config.restarts, weights= self.config.restart_weights))
        elif self.config.LRS == 'LD':
            if self.rank <= 0: print(toRed('\tLR dacay scheduler...'))
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LR_decay(
                        optimizer, decay_period = self.config.decay_period,
                        decay_rate = self.config.decay_rate))

    def _set_dataloader(self):
        if self.rank <= 0: print(toGreen('Loading Data Loader...'))

        if self.is_train:
            self.sampler_train = None
            self.dataset_train = datasets(self.config, is_train = True)

            if self.config.dist == True:
                self.sampler_train = DistIterSampler(self.dataset_train, self.ws, self.rank)
            else:
                self.sampler_train = None
                
            self.data_loader_train = self._create_dataloader(self.dataset_train, sampler = self.sampler_train, is_train = True)

        self.sampler_eval = None
        self.dataset_eval = datasets(self.config, is_train = False)

        if self.config.dist == True:
            self.sampler_eval = DistIterSampler(self.dataset_eval, self.ws, self.rank, is_train=False)
        else:
            self.sampler_eval = None

        self.data_loader_eval = self._create_dataloader(self.dataset_eval, sampler = self.sampler_eval, is_train = False)

    def _update(self, errs, warmup_itr = -1):

        self.network.zero_grad()
        self.network1.zero_grad()
        self.network2.zero_grad()
        errs.backward()
        torch_utils.clip_grad_norm_(self.network1.parameters(), self.config.gc)
        torch_utils.clip_grad_norm_(self.network2.parameters(), self.config.gc)

        # total_norm = 0
        # for name, p in self.network1.named_parameters():
        #     if p.grad is None:
        #         if self.config.is_verbose and self.rank<=0:print('\n\nnetwork1 grad none param: ', name, '\n\n')
        #     else:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm1 = total_norm ** (1. / 2)

        # total_norm = 0
        # for name, p in self.network2.named_parameters():
        #     if p.grad is None:
        #         if self.config.is_verbose and self.rank<=0:print('\n\nnetwork2 grad none param: ', name, '\n\n')
        #     else:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm2 = total_norm ** (1. / 2)

        for optimizer in self.optimizers:
            optimizer.step()

        net1_params = dict(self.network1.named_parameters())
        net2_params = dict(self.network2.named_parameters())
        for name, params in self.network.named_parameters():
            if name in net1_params.keys() and name in net2_params.keys():
                params.data = net1_params[name].data - (params.data - net2_params[name].data)
            elif name in net1_params.keys() and name not in net2_params.keys():
                params.data = net1_params[name].data
            elif name not in net1_params.keys() and name in net2_params.keys():
                params.data = net2_params[name].data
            else:
                if self.rank<=0: print('\n\n\n!!update error\n\n\n')
                exit()
            # net1_params[name].data = params.data
            # net2_params[name].data = params.data

        lr = self._update_learning_rate(self.itr_global['train'], warmup_itr)
        # lr['gn0'] = total_norm1
        # lr['gn1'] = total_norm2

        return lr

    ######################################################################################################
    ########################### Edit from here for training/testing scheme ###############################
    ######################################################################################################

    def _set_results(self, inputs, outs, errs, lr, norm_=1, is_train=False):
        if self.rank <=0 and self.config.save_sample:
            ## save visuals (inputs)
            self.results['vis'] = collections.OrderedDict()

            self.results['vis']['input'] = outs['I_curr']
            self.results['vis']['gt'] = outs['gt_curr']

            ## save visuals (outputs)
            self.results['vis']['result'] = outs['result']

            if outs['aux_l'] is not None:
                self.results['vis']['aux'] = outs['aux_l'][:, -3:, :, :]

            if outs['flow'] is not None:
                gt_prev_warped = warp(outs['gt_prev'], outs['flow'])
                self.results['vis']['gt_prev_warped'] = gt_prev_warped

            if is_train:
                self.results['vis']['gt_prev_warped_gt'] = outs['warped_gt']
            self.results['vis']['gt_prev'] = inputs['gt'][:, -4, :, :, :]

            if 'pre_f_prev' in outs.keys():
                self.results['vis']['pre_f_prev'] = norm_feat_vis(outs['pre_f_prev'][:1])
            if 'pre_f' in outs.keys():
                self.results['vis']['pre_f'] = norm_feat_vis(outs['pre_f'][:1])
            self.results['vis']['post_f'] = norm_feat_vis(outs['post_f'][:1])
            self.results['vis']['post_f_prev'] = norm_feat_vis(outs['post_f_prev'][:1])
            self.results['vis']['post_f_prev_warped'] = norm_feat_vis(outs['post_f_prev_warped'][:1])
            if outs['flow'] is not None:
                self.results['vis']['flow_pred'] = OF_vis(outs['flow'][0])

            if is_train:
                self.results['vis']['flow_gt'] = OF_vis(outs['flow_gt'][0])
                self.results['vis']['flow_mask'] = outs['flow_mask']

        ## Essentials ##
        # save scalars
        self.results['errs'] = errs
        self.results['norm'] = norm_
        # learning rate
        self.results['log_etc'] = lr

    def _get_results(self, I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame, gt_prev, gt_curr, is_train):

        if is_train:
            outs = self.network1(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
            outs_distill = self.network2(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
        else:
            outs = self.network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)

        ## loss
        if self.config.is_train:
            errs = collections.OrderedDict()
            errs['total'] = 0.
            errs['image'] = self.MSE(outs['result'], gt_curr) + self.MAE(outs['result'], gt_curr)
            errs['total'] = errs['total'] + errs['image']

            if is_train:
                if outs['aux_l'] is not None:
                    gt_curr_repeat = gt_curr.repeat(1, self.config.HG_num - 1, 1, 1)
                    errs['aux'] = self.config.aux_lambda * (self.MSE(outs['aux_l'], gt_curr_repeat) + self.MAE(outs['aux_l'], gt_curr_repeat))
                    errs['total'] = errs['total'] + errs['aux']

                if outs_distill['corr_l'] is not None:
                    mc_scale = self.config.mc_scale
                    with torch.no_grad():
                        b, c, h, w = gt_prev.size()

                        flow_gt = upsample(self.liteFlowNet(gt_curr, gt_prev), h, w, 'area')
                        flow_mask = upsample(warp(torch.ones(b, 1, h, w).to(torch.device('cuda')).type(torch.cuda.FloatTensor), flow_gt), h/4, h/4, 'nearest')

                        flow_gt_ = upsample(flow_gt, h/mc_scale, w/mc_scale, 'area') / mc_scale
                        flow_gt_clamp = torch.clamp(flow_gt_, -1*self.config.max_displacement, self.config.max_displacement)

                        corr_gt = (
                            ((flow_gt_clamp[:, 0, :, :] + self.config.max_displacement) / 1.).round() +
                            ((flow_gt_clamp[:, 1, :, :] + self.config.max_displacement) / 1.).round() * (self.config.max_displacement * 2 + 1)
                        )
                        # hrf = 14
                        # corr_gt = corr_gt[:, hrf:-hrf, hrf:-hrf]
                        # corr_gt = corr_gt.reshape(b, -1).repeat(1, self.config.HG_num - 1)
                        # corr_gt = corr_gt.view(b, -1).repeat(1, self.config.HG_num - 1)
                        _, _, rp, _, _ = outs_distill['corr_l'].size()
                        corr_gt = corr_gt.view(b, -1).repeat(1, rp)
                        corr_gt = corr_gt.type(torch.cuda.LongTensor)

                        flow_gt_clamp_up = upsample(flow_gt_clamp, h, w, 'nearest') * mc_scale

                        if self.config.save_sample:
                            outs['warped_gt'] = warp(gt_prev, flow_gt_clamp_up)
                            outs['flow_gt'] = flow_gt_clamp_up
                            outs['flow_mask'] = upsample(flow_mask, h, w, 'nearest')

                    # if self.rank <= 0: print(torch.mean(corr_gt.type(torch.cuda.FloatTensor)))
                    # corrs = outs_distill['corr_l'].view(b, (self.config.max_displacement * 2 + 1)**2, self.config.HG_num - 1, int(h/4), int(w/4))
                    # corrs = corrs[:, :, :, hrf:-hrf, hrf:-hrf]
                    # corrs = corrs.reshape(b, (self.config.max_displacement * 2 + 1)**2, -1)
                    corrs = outs_distill['corr_l'].view(b, (self.config.max_displacement * 2 + 1)**2, -1)
                    errs['distill'] = torch.mean(self.CSE(corrs, corr_gt))
                    errs['total'] = errs['total'] + errs['distill']

            errs['psnr'] = get_psnr2(outs['result'], gt_curr)
            return errs, outs
        else:
            return outs

    def iteration(self, inputs, epoch, max_epoch, is_log, is_train):
        # init for logging
        state = 'train' if is_train else 'valid'
        self.itr_global[state] += self.itr_inc[state]

        # prepare data
        Is = refine_image_pt(inputs['input'].to(torch.device('cuda')), self.config.refine_val)
        GTs = refine_image_pt(inputs['gt'].to(torch.device('cuda')), self.config.refine_val)

        norm_ = 0
        b, f, c, h, w = Is.size()
        errs_total = collections.OrderedDict()
        for i in range(Is.size()[1]-4):
            is_first_frame = i == 0 if is_train else inputs['is_first'][0].item()
            # if self.rank <= 0: print('\n\n\n\n', i, is_first_frame, '\n\n\n\n')

            # run network & get errs and outputs
            I_prev_prev = Is[:, i, :, :, :]
            I_prev = Is[:, i+1, :, :, :]
            I_curr = Is[:, i+2, :, :, :]
            I_next = Is[:, i+3, :, :, :]
            I_next_next = Is[:, i+4, :, :, :]

            gt_prev = GTs[:, i+1, :, :, :]
            gt_curr = GTs[:, i+2, :, :, :]

            self.R_prev = I_prev if is_first_frame else self.R_prev

            # with autograd.detect_anomaly():
            net1_params = dict(self.network1.named_parameters())
            net2_params = dict(self.network2.named_parameters())
            for name, params in self.network.named_parameters():
                if name in net1_params.keys():
                    net1_params[name].data = params.data
                if name in net2_params.keys():
                    net2_params[name].data = params.data

            errs, outs = self._get_results(I_prev_prev, I_prev, I_curr, I_next, I_next_next, self.R_prev, is_first_frame, gt_prev, gt_curr, is_train)
            lr = self._update(errs['total'], warmup_itr = self.config.warmup_itr) if is_train else None

            # self.R_prev = outs['result'].detach()
            self.R_prev = outs['result'].clone().detach()

            norm_ += b
            for k, v in errs.items():
                # if state =='valid' and self.rank<=0: print(k)
                v_t = 0 if i == 0 else errs_total[k]
                # errs_total[k] = v_t + v * b
                errs_total[k] = v_t + v.clone().detach() * b

        assert norm_ != 0

        # set results for the log
        outs['gt_curr'] = gt_curr
        outs['gt_prev'] = gt_prev
        outs['I_curr'] = I_curr
        # self._set_results(inputs, outs, errs_total, lr, norm_, is_train)
        outs_ = collections.OrderedDict()
        for k, v in outs.items():
            if v is not None:
                outs_[k] = v.clone().detach()
        self._set_results(inputs, outs_, errs_total, lr, norm_, is_train)

    def evaluation(self, inputs):
        # init for logging
        state = 'valid'

        # prepare data
        Is = refine_image_pt(inputs['input'].to(torch.device('cuda')), self.config.refine_val)
        GTs = refine_image_pt(inputs['gt'].to(torch.device('cuda')), self.config.refine_val)

        norm_ = 0
        b, f, c, h, w = Is.size()
        errs_total = collections.OrderedDict()
        for i in range(Is.size()[1]-4):
            is_first_frame = inputs['is_first'][0].item()
            # if self.rank <= 0: print('\n\n\n\n', i, is_first_frame, '\n\n\n\n')

            # run network & get errs and outputs
            I_prev_prev = Is[:, i, :, :, :]
            I_prev = Is[:, i+1, :, :, :]
            I_curr = Is[:, i+2, :, :, :]
            I_next = Is[:, i+3, :, :, :]
            I_next_next = Is[:, i+4, :, :, :]

            gt_prev = GTs[:, i+1, :, :, :]
            gt_curr = GTs[:, i+2, :, :, :]

            self.R_prev = I_prev if is_first_frame else self.R_prev

            # with autograd.detect_anomaly():
            net1_params = dict(self.network1.named_parameters())
            net2_params = dict(self.network2.named_parameters())
            for name, params in self.network.named_parameters():
                if name in net1_params.keys():
                    net1_params[name].data = params.data
                if name in net2_params.keys():
                    net2_params[name].data = params.data

            outs = self._get_results(I_prev_prev, I_prev, I_curr, I_next, I_next_next, self.R_prev, is_first_frame, gt_prev, gt_curr, is_train=False)
            lr = None

            # self.R_prev = outs['result'].detach()
            self.R_prev = outs['result'].clone().detach()

        # set results for the log
        outs['gt'] = gt_curr
        outs['input'] = I_curr

        return outs

class DeblurNet(nn.Module):
    def __init__(self, config, is_distill):
        super(DeblurNet, self).__init__()
        self.rank = torch.distributed.get_rank() if config.dist else -1
        self.config = config

        if self.rank <= 0: print(toRed('\tinitializing deblurring network'))

        lib = importlib.import_module('models.archs.{}'.format(config.network))
        self.Network = lib.Network(config, is_distill)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):

            # torch.nn.init.xavier_uniform_(m.weight, gain = self.config.wi)
            torch.nn.init.normal_(m.weight.data, 0.0, self.config.win)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
        elif type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)

    def init(self):
        self.Network.apply(self.weights_init)


    def input_constructor(self, res):
        b, c, h, w = res[:]
        img = torch.FloatTensor(np.random.randn(b, c, h, w)).cuda()

        return {'I_prev_prev': img, 'I_prev': img, 'I_curr': img, 'I_next': img, 'I_next_next': img, 'R_prev': img, 'is_first_frame': True}

    #####################################################
    def forward(self, I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame):

        outs = self.Network.forward(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
        return outs
