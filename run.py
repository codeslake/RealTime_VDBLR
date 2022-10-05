import torch
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import json

import time
import numpy
import os
import sys
import collections
import numpy as np
import gc
import math
import random

from models import create_model
from utils import *
from ckpt_manager import CKPT_Manager

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

class Trainer():
    def __init__(self, config, rank = -1):
        self.rank = rank
        if config.dist:
            self.pg = dist.new_group(range(dist.get_world_size()))

        self.config = config
        if self.rank <= 0: self.summary = SummaryWriter(config.LOG_DIR.log_scalar)

        ## model
        self.model = create_model(config)
        if self.rank <= 0 and config.is_verbose:
            self.model.print()

        ## checkpoint manager
        self.ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num, is_descending = True)

        ## training vars
        self.states = ['train', 'valid']
        # self.states = ['valid', 'train']
        self.max_epoch = int(math.ceil(config.total_itr / self.model.get_itr_per_epoch('train')))

        if self.rank <= 0: print(toGreen('Max Epoch: {}'.format(self.max_epoch)))
        self.epoch_range = np.arange(1, self.max_epoch + 1)
        self.err_epoch = {'train': {}, 'valid': {}}
        self.norm = torch.tensor(0).to(torch.device('cuda'))
        self.lr = 0

        if self.config.resume is not None:
            if self.rank <= 0: print(toGreen('Resume Trianing...'))
            if self.rank <= 0: print(toRed('\tResuming {}..'.format(self.config.resume)))
            resume_state = self.ckpt_manager.resume(self.model.get_network(), self.config.resume, self.rank)
            self.epoch_range = np.arange(resume_state['epoch'] + 1, self.max_epoch + 1)
            self.model.resume_training(resume_state)

    def train(self):
        torch.backends.cudnn.benchmark = True
        if self.rank <= 0 : print(toYellow('\n\n=========== TRAINING START ============'))
        for epoch in self.epoch_range:
            if self.rank <= 0 and epoch == 1:
                if self.config.resume is None:
                    self.ckpt_manager.save(self.model.get_network(), self.model.get_training_state(0), 0, score = [1e-8, 1e8])
            is_log = epoch == 1 or epoch % self.config.write_ckpt_every_epoch == 0 or epoch > self.max_epoch - 10
            if self.config.resume is not None and epoch == int(self.config.resume) + 1:
                is_log = True

            epoch_time = time.time()
            for state in self.states:
                if state == 'train':
                    self.model.train()
                    self.iteration(epoch, state, is_log)
                elif is_log:
                    self.model.eval()
                    with torch.no_grad():
                        self.iteration(epoch, state, is_log)

                if state == 'valid' or state == 'train' : # add "or state == 'train" if you want to save train logs
                    if is_log:
                        #if state == 'train':
                        if config.dist: dist.all_reduce(self.norm, op=dist.ReduceOp.SUM, group=self.pg, async_op=False)
                        assert self.norm != 0
                        for k, v in self.err_epoch[state].items():
                            # print('\n\n!!KEY: ', k, '\n\n\n\n')
                            if config.dist:  dist.all_reduce(self.err_epoch[state][k], op=dist.ReduceOp.SUM, group=self.pg, async_op=False)
                            self.err_epoch[state][k] = (self.err_epoch[state][k] / self.norm).item()

                            if self.rank <= 0:
                                self.summary.add_scalar('loss/epoch_{}_{}'.format(k, state), self.err_epoch[state][k], epoch)
                                self.summary.add_scalar('loss/{}_{}'.format(k, state), self.err_epoch[state][k], self.model.itr_global['train'])

                        if self.rank <= 0:
                            #torch.cuda.synchronize()
                            if state == 'train':
                                print_logs(state.upper() + ' TOTAL', self.config.mode, epoch, self.max_epoch, epoch_time, iter = self.model.itr_global[state], iter_total = self.config.total_itr, errs = self.err_epoch[state], log_etc = self.lr, is_overwrite = False)
                            else:
                                print_logs(state.upper() + ' TOTAL', self.config.mode, epoch, self.max_epoch, epoch_time, errs = self.err_epoch[state], log_etc = self.lr, is_overwrite = False)
                                print('\n')
                            epoch_time = time.time()

                            if state == 'valid':
                                is_saved = False
                                while is_saved == False:
                                    #print(self.rank)
                                    try:
                                        if math.isnan(self.err_epoch['valid']['psnr']) == False:
                                            self.ckpt_manager.save(self.model.get_network(), self.model.get_training_state(epoch), epoch, score = [self.err_epoch['valid']['psnr']])
                                        is_saved = True
                                    except Exception as ex:
                                        is_saved = False
                                #if math.isnan(self.err_epoch['valid']['psnr']) == False:
                                #    self.ckpt_manager.save(self.model.get_network(), self.model.get_training_state(epoch), epoch, score = [self.err_epoch['valid']['psnr'].item()])
                                #is_saved = False

                        self.err_epoch[state] = {}
                        if config.dist:
                            dist.barrier()

            if self.rank <= 0:
                if epoch % self.config.refresh_image_log_every_epoch['train'] == 0:
                    remove_file_end_with(self.config.LOG_DIR.sample, '*.jpg')
                    remove_file_end_with(self.config.LOG_DIR.sample, '*.png')
                if epoch % self.config.refresh_image_log_every_epoch['valid'] == 0:
                    remove_file_end_with(self.config.LOG_DIR.sample_val, '*.jpg')
                    remove_file_end_with(self.config.LOG_DIR.sample_val, '*.png')

            gc.collect()
            if self.model.itr_global['train'] >= self.config.total_itr:
                break

    def iteration(self, epoch, state, is_log):
        is_train = True if state == 'train' else False
        data_loader = self.model.data_loader_train if is_train else self.model.data_loader_eval
        if config.dist:
            if is_train: self.model.sampler_train.set_epoch(epoch)


        itr = 0
        self.norm = torch.tensor(0).to(torch.device('cuda'))
        itr_time = time.time()
        for inputs in data_loader:
            lr = None

            self.model.iteration(inputs, epoch, self.max_epoch, is_log, is_train)
            itr += 1

            if is_log:
                bs = inputs['gt'].size()[0]
                errs = self.model.results['errs']
                norm = self.model.results['norm']
                for k, v in errs.items():
                    if itr == 1:
                        self.err_epoch[state][k] = v
                    else:
                        if k in self.err_epoch[state].keys():
                            self.err_epoch[state][k] += v
                        else:
                            self.err_epoch[state][k] = v
                self.norm = self.norm + norm

                if self.rank <= 0:
                    if config.save_sample:
                        # saves image patches for logging
                        vis = self.model.results['vis']
                        sample_dir = self.config.LOG_DIR.sample if is_train else self.config.LOG_DIR.sample_val
                        if itr == 1 or self.model.itr_global[state] % config.write_log_every_itr[state] == 0:
                            try:
                                i = 1
                                for key, val in vis.items():
                                    if val.dim() == 5:
                                        for j in range(val.size()[1]):
                                            vutils.save_image(val[:, j, :, :, :], '{}/E{:02}_I{:06}_{:02}_{}_{:03}.jpg'.format(sample_dir, epoch, self.model.itr_global[state], i, key, j), nrow=math.ceil(math.sqrt(val.size()[0])), padding = 0, normalize = False)
                                    else:
                                        vutils.save_image(val, '{}/E{:02}_I{:06}_{:02}_{}.jpg'.format(sample_dir, epoch, self.model.itr_global[state], i, key), nrow=math.ceil(math.sqrt(val.size()[0])), padding = 0, normalize = False)
                                    i += 1
                            except Exception as ex:
                                print('\n\n\n\nsaving error: ', ex, '\n\n\n\n')

                    self.lr = self.model.results['log_etc']

                    #torch.cuda.synchronize()

                    errs_itr = collections.OrderedDict()
                    for k, v in errs.items():
                        errs_itr[k] = v / norm
                    print_logs(state.upper(), self.config.mode, epoch, self.max_epoch, itr_time, itr * self.model.itr_inc[state], self.model.get_itr_per_epoch(state), errs = errs_itr, log_etc = self.lr, is_overwrite = itr > 1)
                    itr_time = time.time()

##########################################################
def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)

if __name__ == '__main__':
    project = 'PG2022_RealTime_VDBLR'
    mode = 'PG2022_RealTime_VDBLR'

    from configs.config import set_train_path
    import importlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', action = 'store_true', default = False, help = 'whether to delete log')
    parser.add_argument('--config', type = str, default = None, help = 'config name') # do not change the default value
    parser.add_argument('--mode', type = str, default = mode, help = 'mode name')
    parser.add_argument('--project', type = str, default = project, help = 'project name')
    parser.add_argument('-data', '--data', type=str, default = 'DVD', help = 'dataset to train (DVD|nah) or test (DVD|nah|random)')
    parser.add_argument('-LRS', '--learning_rate_scheduler', type=str, default = 'LD', help = 'learning rate scheduler to use [LD or CA]')
    parser.add_argument('-b', '--batch_size', type = int, default = 8, help = 'number of batch')
    parser.add_argument('-trainer', '--trainer', type = str, default = 'trainer', help = 'model name')
    parser.add_argument('-net', '--network', type = str, default = 'MTU', help = 'network name')
    args, _ = parser.parse_known_args()

    if args.is_train:
        config_lib = importlib.import_module('configs.{}'.format(args.config))
        config = config_lib.get_config(args.project, args.mode, args.config, args.data, args.learning_rate_scheduler, args.batch_size)
        config.is_train = True

        ## DEFAULT
        parser.add_argument('-r', '--resume', type = str, default = config.resume, help = 'name of state or ckpt (names are the same)')
        parser.add_argument('-dl', '--delete_log', action = 'store_true', default = False, help = 'whether to delete log')
        parser.add_argument('-lr', '--lr_init', type = float, default = config.lr_init, help = 'leraning rate')
        parser.add_argument('-th', '--thread_num', type = int, default = config.thread_num, help = 'number of thread')
        parser.add_argument('-dist', '--dist', action = 'store_true', default = config.dist, help = 'whether to distributed pytorch')
        parser.add_argument('-vs', '--is_verbose', action = 'store_true', default = False, help = 'whether to delete log')
        parser.add_argument('-ss', '--save_sample', action = 'store_true', default = False, help = 'whether to save_sample')
        parser.add_argument("--local_rank", type=int)

        ## CUSTOM
        parser.add_argument('-wi', '--weights_init', type = float, default = config.wi, help = 'weights_init')
        parser.add_argument('-proc', '--proc', type = str, default = 'proc', help = 'dummy process name for killing')
        parser.add_argument('-gc', '--gc', type = float, default = config.gc, help = 'gradient clipping')
        parser.add_argument('-no_distill', '--no_distill', action = 'store_false', default = config.is_distill, help = 'gradient clipping')
        parser.add_argument('-no_aux', '--no_aux', action = 'store_false', default = config.is_aux, help = 'gradient clipping')
        parser.add_argument('-max_D', '--max_D', type=int , default = 10, help = 'max displacement')

        args, _ = parser.parse_known_args()

        ## default
        config.trainer = args.trainer
        config.network = args.network

        config.resume = args.resume
        config.delete_log = False if config.resume is not None else args.delete_log
        config.lr_init = args.lr_init
        config.batch_size = args.batch_size
        config.thread_num = args.thread_num
        config.dist = args.dist
        config.data = args.data
        config.LRS = args.learning_rate_scheduler
        config.is_verbose = args.is_verbose
        config.save_sample = args.save_sample
        config.is_distill = args.no_distill
        config.is_aux = args.no_aux
        config.max_displacement = args.max_D

        # CUSTOM
        config.wi = args.weights_init
        config.gc = args.gc

        # set datapath
        config = set_train_path(config, config.data)

        if config.dist:
            init_dist()
            rank = dist.get_rank()
        else:
            rank = -1

        if rank <= 0:
            handle_directory(config, config.delete_log)
            print(toGreen('Laoding Config...'))
            config_lib.print_config(config)
            config_lib.log_config(config.LOG_DIR.config, config)
            print(toRed('\tProject : {}'.format(config.project)))
            print(toRed('\tMode : {}'.format(config.mode)))
            print(toRed('\tConfig: {}'.format(config.config)))
            print(toRed('\tNetwork: {}'.format(config.network)))
            print(toRed('\tTrainer: {}'.format(config.trainer)))
            print(toRed('\tData: {}'.format(config.data)))

        if config.dist:
            dist.barrier()

        ## random seed
        seed = config.manual_seed
        if seed is None:
            seed = random.randint(1, 10000)
        if rank <= 0 and config.is_verbose: print('Random seed: {}'.format(seed))

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        trainer = Trainer(config, rank)
        if config.dist:
            dist.barrier()
        trainer.train()

    else:
        from configs.config import get_config, set_eval_path, set_log_path
        from easydict import EasyDict as edict
        print(toGreen('Laoding Config for evaluation'))
        if args.config is None:
            config_ori = get_config(args.project, args.mode, None)
            with open('{}/config.txt'.format(config_ori.LOG_DIR.config)) as json_file:
                json_data = json.load(json_file)
                # config_lib = importlib.import_module('configs.{}'.format(json_data['config']))
                config = edict(json_data)
                # print(config['config'])
        else:
            config_lib = importlib.import_module('configs.{}'.format(args.config))
            config = config_lib.get_config(args.project, args.mode, args.config)

        config.is_train = False
        ## EVAL
        parser.add_argument('-ckpt_name', '--ckpt_name', type=str, default = None, help='ckpt name')
        parser.add_argument('-ckpt_abs_name', '--ckpt_abs_name', type=str, default = None, help='ckpt abs name')
        parser.add_argument('-ckpt_epoch', '--ckpt_epoch', type=int, default = None, help='ckpt epoch')
        parser.add_argument('-ckpt_sc', '--ckpt_score', action = 'store_true', help='ckpt name')
        parser.add_argument('-dist', '--dist', action = 'store_true', default = False, help = 'whether to distributed pytorch')
        parser.add_argument('-eval_mode', '--eval_mode', type=str, default = 'quan', help = 'evaluation mode. qual(qualitative)/quan(quantitative)')
        parser.add_argument('-is_quan', '--is_quan', action = 'store_true', help='ckpt name')
        parser.add_argument('-save_input_gt', '--save_input_gt', action = 'store_true', help='ckpt name')

        parser.add_argument('-max_D', '--max_D', type=int , default = config.max_displacement, help = 'max displacement')
        args, _ = parser.parse_known_args()

        config.trainer = args.trainer
        config.network = args.network
        config.EVAL.ckpt_name = args.ckpt_name
        config.EVAL.ckpt_abs_name = args.ckpt_abs_name
        config.EVAL.ckpt_epoch = args.ckpt_epoch
        config.EVAL.load_ckpt_by_score = args.ckpt_score
        config.EVAL.is_quan = args.is_quan
        config.EVAL.save_input_gt = args.save_input_gt

        config.max_displacement = args.max_D

        config.dist = args.dist
        config.EVAL.eval_mode = args.eval_mode
        config.EVAL.data = args.data

        handle_directory(config, False)
        config = set_eval_path(config, config.EVAL.data)
        if args.config is None:
            config = set_log_path(config, config_ori.log_offset, config_ori.mode)

        print(toRed('\tProject : {}'.format(config.project)))
        print(toRed('\tMode : {}'.format(config.mode)))
        print(toRed('\tConfig: {}'.format(config.config)))
        print(toRed('\tNetwork: {}'.format(config.network)))
        print(toRed('\tTrainer: {}'.format(config.trainer)))
        print(toRed('\tdata: {}'.format(config.EVAL.data)))
        print(toRed('\tis_quan: {}'.format(config.EVAL.is_quan)))
        print(toRed('\tsave_input_gt: {}'.format(config.EVAL.save_input_gt)))

        if config.EVAL.data == 'REDS':
            from eval_lmdb import *
        else:
            from eval import *
        eval(config)
