import torch
import torchvision.utils as vutils
import torch.nn.functional as F

import os
import sys
import datetime
import time
import gc
from pathlib import Path

import numpy as np
import cv2
import math
from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity
import collections

from utils import *
from data_loader.utils import refine_image, read_frame, load_file_list, norm
from ckpt_manager import CKPT_Manager
from models.utils import warp, norm_res_vis, upsample

from models import create_model

def mae(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])

def ssim(img1, img2, PIXEL_MAX = 1.0):
    return structural_similarity(img1, img2, data_range=PIXEL_MAX, multichannel=True)

def ssim_masked(img1, img2, mask, PIXEL_MAX = 1.0):
    _, s = structural_similarity(img1, img2, data_range=PIXEL_MAX, multichannel=True, full=True)
    s = s * mask
    mssim = np.sum(s)/np.sum(mask)
    return mssim

def psnr(img1, img2, PIXEL_MAX = 1.0):
    mse_ = np.mean( (img1 - img2) ** 2 )
    return 10 * math.log10(PIXEL_MAX / mse_)

def psnr_masked(img1, img2, mask, PIXEL_MAX = 1.0):
    mse_ = np.sum( ( (img1 - img2) ** 2) * mask) / np.sum(mask)
    return 10 * math.log10(PIXEL_MAX / mse_)

def init(config, mode = 'deblur'):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cuda.matmul.allow_tf32 = False

    date = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')

    model = create_model(config)
    network = model.get_network().eval()

    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num)
    load_state, ckpt_name = ckpt_manager.load_ckpt(network, by_score = config.EVAL.load_ckpt_by_score, name = config.EVAL.ckpt_name, abs_name = config.EVAL.ckpt_abs_name, epoch = config.EVAL.ckpt_epoch)
    print('\nLoading checkpoint \'{}\' on model \'{}\': {}'.format(ckpt_name, config.mode, load_state))

    save_path_root = config.EVAL.LOG_DIR.save

    save_path_root_deblur = os.path.join(save_path_root, mode, ckpt_name.split('.')[0])
    save_path_root_deblur_score = save_path_root_deblur
    Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), os.path.join(save_path_root_deblur, ckpt_name))
    save_path_root_deblur = os.path.join(save_path_root_deblur, config.EVAL.data, date)

    blur_folder_path_list, blur_file_path_list, _ = load_file_list(config.EVAL.data_path, config.EVAL.input_path)
    gt_file_path_list = None
    if config.EVAL.gt_path is not None:
        _, gt_file_path_list, _ = load_file_list(config.EVAL.data_path, config.EVAL.gt_path)

    return network, save_path_root_deblur, save_path_root_deblur_score, ckpt_name, blur_folder_path_list, blur_file_path_list, gt_file_path_list

def eval_quan_qual(config):
    mode = config.EVAL.eval_mode
    network, save_path_root_deblur, save_path_root_deblur_score, ckpt_name,\
    blur_folder_path_list, blur_file_path_list, gt_file_path_list = init(config, mode)

    ##
    total_norm = 0
    total_itr_time = 0
    PSNR_mean_total = 0.
    SSIM_mean_total = 0.

    for i in range(len(blur_file_path_list)):
        ## read frame
        video_name = blur_folder_path_list[i].split(os.sep)[-2]
        frame_list = []
        print('[Reading Frames: {}][{}/{}]'.format(video_name, i + 1, len(blur_file_path_list)))
        for frame_name in blur_file_path_list[i]:
            frame = refine_image(read_frame(frame_name), config.refine_val)
            frame = np.expand_dims(frame.transpose(0, 3, 1, 2), 0)
            frame_list.append(frame)

        if gt_file_path_list is not None:
            frame_list_gt = []
            for frame_name in gt_file_path_list[i]:
                frame = refine_image(read_frame(frame_name), config.refine_val)
                frame = np.expand_dims(frame.transpose(0, 3, 1, 2), 0)
                frame_list_gt.append(frame)

        ## evaluation start
        PSNR_mean = 0.
        SSIM_mean = 0.
        total_itr_time_video = 0
        for j in range(len(frame_list)):
            total_norm = total_norm + 1

            ## prepare input tensors
            index = np.array(list(range(j - 2, j + 3))).clip(min=0, max=len(frame_list)-1)
            Is = torch.cat([torch.FloatTensor(frame_list[k]).to(torch.device('cuda'), non_blocking=True) for k in index], dim = 1)
            I_center = torch.FloatTensor(frame_list[j][0]).to(torch.device('cuda'), non_blocking=True)

            if j == 0:
                R_prev = Is[:, 0, :, :, :]
                b, c, h, w = R_prev.size()
                is_first_frame = True
            else:
                is_first_frame = False

            I_prev_prev = Is[:, 0, :, :, :]
            I_prev = Is[:, 1, :, :, :]
            I_curr = Is[:, 2, :, :, :]
            I_next = Is[:, 3, :, :, :]
            I_next_next = Is[:, 4, :, :, :]

            #######################################################################
            ## run network
            with torch.no_grad():
                torch.cuda.synchronize()
                if 'amp' not in config.mode:
                    init_time = time.time()
                    out = network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
                else:
                    with torch.cuda.amp.autocast():
                        init_time = time.time()
                        out = network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
                torch.cuda.synchronize()
                itr_time = time.time() - init_time
                out['result'] = torch.clamp(out['result'], 0, 1)
                R_prev = out['result']
            #######################################################################

            total_itr_time_video = total_itr_time_video + itr_time
            total_itr_time = total_itr_time + itr_time

            ## evaluation
            inp = I_center
            output = out['result']
            output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)

            PSNR = 0
            SSIM = 0
            if gt_file_path_list is not None:
                gt = torch.FloatTensor(frame_list_gt[j][0]).cuda()

                # quantitative
                gt_cpu = gt.cpu().numpy()[0].transpose(1, 2, 0)

                if config.EVAL.is_quan:
                    PSNR = psnr(output_cpu, gt_cpu)
                    SSIM = ssim(output_cpu, gt_cpu)

                PSNR_mean += PSNR
                SSIM_mean += SSIM

            frame_name = os.path.basename(blur_file_path_list[i][j])
            print('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec)'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), j + 1, len(frame_list), frame_name, PSNR, SSIM, itr_time))
            with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'w' if (i == 0 and j == 0) else 'a') as file:
                file.write('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), j + 1, len(frame_list), frame_name, PSNR, SSIM, itr_time))
                file.close()

            # qualitative

            ## create output dir for a video
            for iformat in ['png']:
                frame_name_no_ext = frame_name.split('.')[0]
                save_path_deblur = os.path.join(save_path_root_deblur, iformat)
                Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

                Path(os.path.join(save_path_deblur, 'output', video_name)).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_output = os.path.join(save_path_deblur, 'output', video_name, '{}.{}'.format(frame_name_no_ext, iformat))
                #vutils.save_image(output, '{}'.format(save_file_path_deblur_output), nrow=1, padding = 0, normalize = False)
                cv2.imwrite(save_file_path_deblur_output, cv2.cvtColor(output_cpu*255, cv2.COLOR_BGR2RGB))

                if config.EVAL.save_input_gt:
                    Path(os.path.join(save_path_deblur, 'input', video_name)).mkdir(parents=True, exist_ok=True)
                    save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', video_name, '{}.{}'.format(frame_name_no_ext, iformat))
                    #jvutils.save_image(inp, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False)
                    input_cpu = inp.cpu().numpy()[0].transpose(1, 2, 0)
                    cv2.imwrite(save_file_path_deblur_input, cv2.cvtColor(input_cpu*255, cv2.COLOR_BGR2RGB))

                    if gt_file_path_list is not None:
                        Path(os.path.join(save_path_deblur, 'gt', video_name)).mkdir(parents=True, exist_ok=True)
                        save_file_path_deblur_gt = os.path.join(save_path_deblur, 'gt', video_name, '{}.{}'.format(frame_name_no_ext, iformat))
                        #vutils.save_image(gt, '{}'.format(save_file_path_deblur_gt), nrow=1, padding = 0, normalize = False)
                        gt_cpu = gt.cpu().numpy()[0].transpose(1, 2, 0)
                        cv2.imwrite(save_file_path_deblur_gt, cv2.cvtColor(gt_cpu*255, cv2.COLOR_BGR2RGB))

        # video average
        PSNR_mean_total += PSNR_mean
        PSNR_mean = PSNR_mean / len(frame_list)

        SSIM_mean_total += SSIM_mean
        SSIM_mean = SSIM_mean / len(frame_list)

        print('[MEAN EVAL {}|{}|{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), frame_name, PSNR_mean, SSIM_mean, total_itr_time_video / len(frame_list)))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'a') as file:
            file.write('[MEAN EVAL {}|{}|{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), frame_name, PSNR_mean, SSIM_mean, total_itr_time_video / len(frame_list)))
            file.close()

        gc.collect()

    # total average
    total_itr_time = total_itr_time / total_norm
    PSNR_mean_total = PSNR_mean_total / total_norm
    SSIM_mean_total = SSIM_mean_total / total_norm

    sys.stdout.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time))
    with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'a') as file:
        file.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time))
        file.close()
    print('\nSaving root: ', save_path_root_deblur)

def eval_MC_cost(config):
    mode = config.EVAL.eval_mode
    network, save_path_root_deblur, save_path_root_deblur_score, ckpt_name,\
    blur_folder_path_list, blur_file_path_list, gt_file_path_list = init(config, mode)

    ##
    total_norm = 0
    total_itr_time = 0
    total_MC_cost = 0
    PSNR_mean_total = 0.
    SSIM_mean_total = 0.

    for i in range(len(blur_file_path_list)):
        ## read frame
        video_name = blur_folder_path_list[i].split(os.sep)[-2]
        frame_list = []
        print('[Reading Frames: {}][{}/{}]'.format(video_name, i + 1, len(blur_file_path_list)))
        for frame_name in blur_file_path_list[i]:
            frame = refine_image(read_frame(frame_name), config.refine_val)
            frame = np.expand_dims(frame.transpose(0, 3, 1, 2), 0)
            frame_list.append(frame)

        if gt_file_path_list is not None:
            frame_list_gt = []
            for frame_name in gt_file_path_list[i]:
                frame = refine_image(read_frame(frame_name), config.refine_val)
                frame = np.expand_dims(frame.transpose(0, 3, 1, 2), 0)
                frame_list_gt.append(frame)

        ## evaluation start
        PSNR_mean = 0.
        SSIM_mean = 0.
        total_itr_time_video = 0
        total_MC_time_video = 0
        for j in range(len(frame_list)):
            total_norm = total_norm + 1

            ## prepare input tensors
            index = np.array(list(range(j - 2, j + 3))).clip(min=0, max=len(frame_list)-1)
            Is = torch.cat([torch.FloatTensor(frame_list[k]).to(torch.device('cuda'), non_blocking=True) for k in index], dim = 1)
            I_center = torch.FloatTensor(frame_list[j][0]).to(torch.device('cuda'), non_blocking=True)

            if j == 0:
                R_prev = Is[:, 0, :, :, :]
                b, c, h, w = R_prev.size()
                is_first_frame = True
            else:
                is_first_frame = False

            I_prev_prev = Is[:, 0, :, :, :]
            I_prev = Is[:, 1, :, :, :]
            I_curr = Is[:, 2, :, :, :]
            I_next = Is[:, 3, :, :, :]
            I_next_next = Is[:, 4, :, :, :]

            #######################################################################
            ## run network
            with torch.no_grad():
                torch.cuda.synchronize()
                if 'amp' not in config.mode:
                    init_time = time.time()
                    out, MC_cost = network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
                else:
                    with torch.cuda.amp.autocast():
                        init_time = time.time()
                        out, MC_cost = network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
                torch.cuda.synchronize()
                out['result'] = torch.clamp(out['result'], 0, 1)
                itr_time = time.time() - init_time
            #######################################################################

            total_itr_time_video = total_itr_time_video + itr_time
            total_itr_time = total_itr_time + itr_time
            total_MC_time_video = total_MC_time_video + MC_cost 
            total_MC_cost = total_MC_cost + MC_cost

            PSNR = 0
            SSIM = 0
            PSNR_mean = 0
            SSIM_mean = 0

            frame_name = os.path.basename(blur_file_path_list[i][j])
            print('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec/MC_time:{:.5f}sec)'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), j + 1, len(frame_list), frame_name, PSNR, SSIM, itr_time, MC_cost))
            with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'w' if (i == 0 and j == 0) else 'a') as file:
                file.write('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec/MC_time:{:.5f}sec)\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), j + 1, len(frame_list), frame_name, PSNR, SSIM, itr_time, MC_cost))
                file.close()

        # video average
        PSNR_mean_total += PSNR_mean
        PSNR_mean = PSNR_mean / len(frame_list)

        SSIM_mean_total += SSIM_mean
        SSIM_mean = SSIM_mean / len(frame_list)

        print('[MEAN EVAL {}|{}|{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec/MC_time:{:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), frame_name, PSNR_mean, SSIM_mean, total_itr_time_video / len(frame_list), total_MC_time_video / len(frame_list)))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'a') as file:
            file.write('[MEAN EVAL {}|{}|{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec/MC_time:{:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), frame_name, PSNR_mean, SSIM_mean, total_itr_time_video / len(frame_list), total_MC_time_video / len(frame_list)))
            file.close()

        gc.collect()

    # total average
    total_itr_time = total_itr_time / total_norm
    total_MC_cost = total_MC_cost / total_norm
    PSNR_mean_total = PSNR_mean_total / total_norm
    SSIM_mean_total = SSIM_mean_total / total_norm

    sys.stdout.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec/MC_time:{:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time, total_MC_cost))
    with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'a') as file:
        file.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec/MC_time:{:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time, total_MC_cost))
        file.close()
    print('\nSaving root: ', save_path_root_deblur)

def eval_warp(config):
    mode = config.EVAL.eval_mode
    network, save_path_root_deblur, save_path_root_deblur_score, ckpt_name,\
    blur_folder_path_list, blur_file_path_list, gt_file_path_list = init(config, mode)

    ##
    total_norm = 0
    total_itr_time = 0
    PSNR_mean_total = 0.
    SSIM_mean_total = 0.

    for i in range(len(blur_file_path_list)):
        ## read frame
        video_name = blur_folder_path_list[i].split(os.sep)[-2]
        frame_list = []
        print('[Reading Frames: {}][{}/{}]'.format(video_name, i + 1, len(blur_file_path_list)))
        for frame_name in blur_file_path_list[i]:
            frame = refine_image(read_frame(frame_name), config.refine_val)
            frame = np.expand_dims(frame.transpose(0, 3, 1, 2), 0)
            frame_list.append(frame)

        if gt_file_path_list is not None:
            frame_list_gt = []
            for frame_name in gt_file_path_list[i]:
                frame = refine_image(read_frame(frame_name), config.refine_val)
                frame = np.expand_dims(frame.transpose(0, 3, 1, 2), 0)
                frame_list_gt.append(frame)

        ## evaluation start
        PSNR_mean = 0.
        SSIM_mean = 0.
        total_itr_time_video = 0
        for j in range(len(frame_list)):
            total_norm = total_norm + 1

            ## prepare input tensors
            index = np.array(list(range(j - 2, j + 3))).clip(min=0, max=len(frame_list)-1)
            Is = torch.cat([torch.FloatTensor(frame_list[k]).to(torch.device('cuda'), non_blocking=True) for k in index], dim = 1)
            Is_gt = torch.cat([torch.FloatTensor(frame_list_gt[k]).to(torch.device('cuda'), non_blocking=True) for k in index], dim = 1)
            I_center = torch.FloatTensor(frame_list[j][0]).to(torch.device('cuda'), non_blocking=True)

            if j == 0:
                R_prev = Is[:, 0, :, :, :]
                b, c, h, w = R_prev.size()
                is_first_frame = True
            else:
                is_first_frame = False

            I_prev_prev = Is[:, 0, :, :, :]
            I_prev = Is[:, 1, :, :, :]
            I_curr = Is[:, 2, :, :, :]
            I_next = Is[:, 3, :, :, :]
            I_next_next = Is[:, 4, :, :, :]

            I_prev_gt = Is_gt[:, 1, :, :, :]
            I_curr_gt = Is_gt[:, 2, :, :, :]

            #######################################################################
            ## run network
            with torch.no_grad():
                torch.cuda.synchronize()
                if 'amp' not in config.mode:
                    init_time = time.time()
                    out = network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
                else:
                    with torch.cuda.amp.autocast():
                        init_time = time.time()
                        out = network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
                torch.cuda.synchronize()
                itr_time = time.time() - init_time
                out['result'] = torch.clamp(out['result'], 0, 1)
                R_prev = out['result']
            #######################################################################

            total_itr_time_video = total_itr_time_video + itr_time
            total_itr_time = total_itr_time + itr_time

            ## evaluation
            inp = I_center
            output = out['result']
            flow = out['flow']

            I_prev_gt_warped = warp(I_prev_gt, flow)
            I_prev_gt_warped_mask = warp(torch.ones_like(I_prev_gt), flow)

            if 'downsample' in config.EVAL.eval_mode:
                b, c, h, w = I_prev_gt_warped.size()
                # flow_gt = upsample(self.liteFlowNet(gt_curr, gt_prev), h, w, 'area')
                I_prev_gt_warped = upsample(I_prev_gt_warped, h//4, w//4, 'area')
                I_prev_gt_warped_mask = upsample(I_prev_gt_warped_mask, h//4, w//4, 'area')

            PSNR = 0
            SSIM = 0
            if gt_file_path_list is not None:
                gt = torch.FloatTensor(frame_list_gt[j][0]).cuda()
                if 'downsample' in config.EVAL.eval_mode:
                    gt = upsample(gt, h//4, w//4, 'area')

                # quantitative
                output_cpu = I_prev_gt_warped.cpu().numpy()[0].transpose(1, 2, 0)
                output_cpu_mask = I_prev_gt_warped_mask.cpu().numpy()[0].transpose(1, 2, 0)
                gt_cpu = gt.cpu().numpy()[0].transpose(1, 2, 0)

                PSNR = psnr_masked(output_cpu, gt_cpu, output_cpu_mask)
                SSIM = ssim_masked(output_cpu, gt_cpu, output_cpu_mask)

                PSNR_mean += PSNR
                SSIM_mean += SSIM

            frame_name = os.path.basename(blur_file_path_list[i][j])
            print('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec)'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), j + 1, len(frame_list), frame_name, PSNR, SSIM, itr_time))
            with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'w' if (i == 0 and j == 0) else 'a') as file:
                file.write('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), j + 1, len(frame_list), frame_name, PSNR, SSIM, itr_time))
                file.close()

            # qualitative

            ## create output dir for a video
            for iformat in ['png']:
                frame_name_no_ext = frame_name.split('.')[0]
                save_path_deblur = os.path.join(save_path_root_deblur, iformat)
                Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

                Path(os.path.join(save_path_deblur, 'output_warp', video_name)).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_output = os.path.join(save_path_deblur, 'output_warp', video_name, '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(I_prev_gt_warped, '{}'.format(save_file_path_deblur_output), nrow=1, padding = 0, normalize = False)

        # video average
        PSNR_mean_total += PSNR_mean
        PSNR_mean = PSNR_mean / len(frame_list)

        SSIM_mean_total += SSIM_mean
        SSIM_mean = SSIM_mean / len(frame_list)

        print('[MEAN EVAL {}|{}|{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), frame_name, PSNR_mean, SSIM_mean, total_itr_time_video / len(frame_list)))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'a') as file:
            file.write('[MEAN EVAL {}|{}|{}][{}/{}] {} PSNR: {:.5f}, SSIM: {:.5f} ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, video_name, i + 1, len(blur_file_path_list), frame_name, PSNR_mean, SSIM_mean, total_itr_time_video / len(frame_list)))
            file.close()

        gc.collect()

    # total average
    total_itr_time = total_itr_time / total_norm
    PSNR_mean_total = PSNR_mean_total / total_norm
    SSIM_mean_total = SSIM_mean_total / total_norm

    sys.stdout.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time))
    with open(os.path.join(save_path_root_deblur_score, 'score_{}.txt'.format(config.EVAL.data)), 'a') as file:
        file.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time))
        file.close()
    print('\nSaving root: ', save_path_root_deblur)


def eval_feat(config):
    # date = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')
    import matplotlib.pyplot as plt
    #colormap = plt.get_cmap('plasma')
    colormap = plt.get_cmap('jet')
    mode = 'eval_feat'
    network, save_path_root_deblur, save_path_root_deblur_score, ckpt_name,\
    blur_folder_path_list, blur_file_path_list, gt_file_path_list = init(config, mode)
    Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)

    video_idx = 0
    frame_end = 29
    #video_idx = 7
    #frame_end = 9

    video_name = blur_folder_path_list[video_idx].split(os.sep)[-2]
    frame_list = []

    print('[{}/{}]reading frames of {}'.format(0 + 1, len(blur_file_path_list), video_name))
    ## read frame
    video_name = blur_folder_path_list[video_idx].split(os.sep)[-2]
    frame_list = []
    print('[Reading Frames: {}][{}/{}]'.format(video_name, video_idx + 1, len(blur_file_path_list)))
    for i, frame_name in enumerate(blur_file_path_list[video_idx]):
        frame = refine_image(read_frame(frame_name), config.refine_val)
        frame = np.expand_dims(frame.transpose(0, 3, 1, 2), 0)
        frame_list.append(frame)
        if i == frame_end:
            break;

    if gt_file_path_list is not None:
        frame_list_gt = []
        for i, frame_name in enumerate(gt_file_path_list[video_idx]):
            frame = refine_image(read_frame(frame_name), config.refine_val)
            frame = np.expand_dims(frame.transpose(0, 3, 1, 2), 0)
            frame_list_gt.append(frame)
            if i == frame_end:
                break;

    ## evaluation start
    total_itr_time_video = 0
    for j in range(len(frame_list)):

        ## prepare input tensors
        index = np.array(list(range(j - 2, j + 3))).clip(min=0, max=len(frame_list)-1)
        Is = torch.cat([torch.FloatTensor(frame_list[k]).to(torch.device('cuda'), non_blocking=True) for k in index], dim = 1)
        I_center = torch.FloatTensor(frame_list[j][0]).to(torch.device('cuda'), non_blocking=True)

        Is_gt = torch.cat([torch.FloatTensor(frame_list_gt[k]).to(torch.device('cuda'), non_blocking=True) for k in index], dim = 1)
        I__gt_center = torch.FloatTensor(frame_list_gt[j][0]).to(torch.device('cuda'), non_blocking=True)

        if j == 0:
            R_prev = Is[:, 0, :, :, :]
            b, c, h, w = R_prev.size()
            is_first_frame = True
        else:
            is_first_frame = False

        I_prev_prev = Is[:, 0, :, :, :]
        I_prev = Is[:, 1, :, :, :]
        I_curr = Is[:, 2, :, :, :]
        I_next = Is[:, 3, :, :, :]
        I_next_next = Is[:, 4, :, :, :]

        I_prev_gt = Is_gt[:, 1, :, :, :]
        I_curr_gt = Is_gt[:, 2, :, :, :]

        #######################################################################
        ## run network
        with torch.no_grad():
            torch.cuda.synchronize()
            if 'amp' not in config.mode:
                init_time = time.time()
                out = network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
            else:
                with torch.cuda.amp.autocast():
                    init_time = time.time()
                    out = network(I_prev_prev, I_prev, I_curr, I_next, I_next_next, R_prev, is_first_frame)
            torch.cuda.synchronize()
            out['result'] = torch.clamp(out['result'], 0, 1)
            itr_time = time.time() - init_time
            ##

        flow = out['flow']
        flow_b = out['flow_b']
        flow_d = out['flow_d']
        feat_b = out['feat_b']
        feat_d = out['feat_d']
        feat_bd = out['feat_bd']
        feat_bd_sum = out['feat_bd_sum']
        feat_d_N_prev = out['feat_d_N_prev']
        feat_d_N_prev_warped = out['feat_d_N_prev_warped']

        if j == frame_end:
            print(j)
            break

    ### 
    l2norm = torch.sqrt(torch.sum(torch.mul(feat_b,feat_b),1,keepdim=True))
    feat_b = feat_b / (l2norm + 1e-8)
    feat_b = feat_b.permute(1, 0, 2, 3)

    l2norm = torch.sqrt(torch.sum(torch.mul(feat_d,feat_d),1,keepdim=True))
    feat_d = feat_d / (l2norm + 1e-8)
    feat_d = feat_d.permute(1, 0, 2, 3)

    feat_bd = feat_bd - torch.unsqueeze(torch.unsqueeze(torch.min(feat_bd.view(52, -1), dim=1, keepdim=True)[0], 1), 0)
    feat_bd = feat_bd / torch.unsqueeze(torch.unsqueeze(torch.max(feat_bd.view(52, -1), dim=1, keepdim=True)[0], 1), 0)
    feat_bd = feat_bd.permute(1, 0, 2, 3)

    # l2norm = torch.sqrt(torch.sum(torch.mul(feat_bd,feat_bd),1,keepdim=True))
    # feat_bd = feat_bd / (l2norm + 1e-8)
    # feat_bd = feat_bd.permute(1, 0, 2, 3)

    l2norm = torch.sqrt(torch.sum(torch.mul(feat_bd_sum,feat_bd_sum),1,keepdim=True))
    feat_bd_sum = feat_bd_sum / (l2norm + 1e-8)
    feat_bd_sum = feat_bd_sum.permute(1, 0, 2, 3)

    l2norm = torch.sqrt(torch.sum(torch.mul(feat_d_N_prev,feat_d_N_prev),1,keepdim=True))
    feat_d_N_prev = feat_d_N_prev / (l2norm + 1e-8)
    feat_d_N_prev = feat_d_N_prev.permute(1, 0, 2, 3)

    l2norm = torch.sqrt(torch.sum(torch.mul(feat_d_N_prev_warped,feat_d_N_prev_warped),1,keepdim=True))
    feat_d_N_prev_warped = feat_d_N_prev_warped / (l2norm + 1e-8)
    feat_d_N_prev_warped = feat_d_N_prev_warped.permute(1, 0, 2, 3)

    print('feat_b')
    print(feat_b.size())
    print(torch.sum(feat_b))
    print(torch.mean(feat_b))
    print(torch.min(feat_b))
    print(torch.max(feat_b))

    print('feat_d')
    print(feat_d.size())
    print(torch.sum(feat_d))
    print(torch.mean(feat_d))
    print(torch.min(feat_d))
    print(torch.max(feat_d))

    print('feat_bd')
    print(feat_bd.size())
    print(torch.sum(feat_bd))
    print(torch.mean(feat_bd))
    print(torch.min(feat_bd))
    print(torch.max(feat_bd))


    vutils.save_image(I_curr, os.path.join(save_path_root_deblur, '00_I.jpg'), nrow=1, padding = 0, normalize = False)
    vutils.save_image((out['result']), os.path.join(save_path_root_deblur, '01_I_r.jpg'), nrow=1, padding = 0, normalize = False)
    res = abs(out['I_res'])
    res = res - torch.min(res)
    res = res / torch.max(res)
    vutils.save_image(res, os.path.join(save_path_root_deblur, '02_I_d.jpg'), nrow=1, padding = 0, normalize = False)

    vutils.save_image(feat_b, os.path.join(save_path_root_deblur, '03_feat_b.jpg'), nrow=8, padding=0, normalize=False)
    vutils.save_image(feat_d, os.path.join(save_path_root_deblur, '04_feat_d.jpg'), nrow=8, padding=0, normalize=False)
    vutils.save_image(feat_bd_sum, os.path.join(save_path_root_deblur, '05_feat_bd_sum.jpg'), nrow=8, padding=0, normalize=False)
    vutils.save_image(feat_bd, os.path.join(save_path_root_deblur, '06_feat_bd.jpg'), nrow=8, padding=0, normalize=False)
    vutils.save_image(feat_d_N_prev, os.path.join(save_path_root_deblur, '07_feat_d_N_prev.jpg'), nrow=8, padding=0, normalize=False)
    vutils.save_image(feat_d_N_prev_warped, os.path.join(save_path_root_deblur, '08_feat_d_N_prev_warped.jpg'), nrow=8, padding=0, normalize=False)

    I_prev_warped_bd = warp(I_prev_gt, flow)
    I_prev_warped_b = warp(I_prev_gt, flow_b)
    I_prev_warped_d = warp(I_prev_gt, flow_d)

    vutils.save_image(I_prev_warped_bd, os.path.join(save_path_root_deblur, '08_I_prev_warped_bd.jpg'), nrow=1, padding = 0, normalize = False)
    vutils.save_image(I_prev_warped_b, os.path.join(save_path_root_deblur, '09_I_prev_warped_b.jpg'), nrow=1, padding = 0, normalize = False)
    vutils.save_image(I_prev_warped_d, os.path.join(save_path_root_deblur, '10_I_prev_warped_d.jpg'), nrow=1, padding = 0, normalize = False)

    I_diff_prev_warped_bd = abs(I_curr_gt - I_prev_warped_bd)
    I_diff_prev_warped_bd = colormap(cv2.cvtColor(I_diff_prev_warped_bd.cpu().numpy()[0].transpose(1, 2, 0), cv2.COLOR_BGR2GRAY))[:, :, :3]
    I_diff_prev_warped_bd = torch.Tensor(I_diff_prev_warped_bd)[None, :].permute(0, 3, 1, 2)

    I_diff_prev_warped_b = abs(I_curr_gt - I_prev_warped_b)
    I_diff_prev_warped_b = colormap(cv2.cvtColor(I_diff_prev_warped_b.cpu().numpy()[0].transpose(1, 2, 0), cv2.COLOR_BGR2GRAY))[:, :, :3]
    I_diff_prev_warped_b = torch.Tensor(I_diff_prev_warped_b)[None, :].permute(0, 3, 1, 2)

    I_diff_prev_warped_d = abs(I_curr_gt - I_prev_warped_d)
    I_diff_prev_warped_d = colormap(cv2.cvtColor(I_diff_prev_warped_d.cpu().numpy()[0].transpose(1, 2, 0), cv2.COLOR_BGR2GRAY))[:, :, :3]
    I_diff_prev_warped_d = torch.Tensor(I_diff_prev_warped_d)[None, :].permute(0, 3, 1, 2)

    vutils.save_image(I_diff_prev_warped_bd, os.path.join(save_path_root_deblur, '11_I_diff_prev_warped_bd.jpg'), nrow=1, padding = 0, normalize = False)
    vutils.save_image(I_diff_prev_warped_b, os.path.join(save_path_root_deblur, '12_I_diff_prev_warped_b.jpg'), nrow=1, padding = 0, normalize = False)
    vutils.save_image(I_diff_prev_warped_d, os.path.join(save_path_root_deblur, '13_I_diff_prev_warped_d.jpg'), nrow=1, padding = 0, normalize = False)


    path_d = os.path.join(save_path_root_deblur, 'detail')
    path_b = os.path.join(save_path_root_deblur, 'structure')
    path_bd = os.path.join(save_path_root_deblur, 'bd')
    path_bd_sum = os.path.join(save_path_root_deblur, 'bd_sum')

    Path(path_d).mkdir(parents=True, exist_ok=True)
    Path(path_b).mkdir(parents=True, exist_ok=True)
    Path(path_bd).mkdir(parents=True, exist_ok=True)
    Path(path_bd_sum).mkdir(parents=True, exist_ok=True)

    for i in range(feat_d.size(0)):
        f_d = torch.unsqueeze(feat_d[i], 0)
        f_b = torch.unsqueeze(feat_b[i], 0)
        f_bd_sum = torch.unsqueeze(feat_bd_sum[i], 0)
        vutils.save_image(f_d, os.path.join(path_d, '{0:02d}.jpg'.format(i)), nrow=1, padding=0, normalize=True)
        vutils.save_image(f_b, os.path.join(path_b, '{0:02d}.jpg'.format(i)), nrow=1, padding=0, normalize=True)
        vutils.save_image(f_bd_sum, os.path.join(path_bd_sum, '{0:02d}.jpg'.format(i)), nrow=1, padding=0, normalize=True)

    for i in range(feat_bd.size(0)):
        f_bd = torch.unsqueeze(feat_bd[i], 0)
        vutils.save_image(f_bd, os.path.join(path_bd, '{0:02d}.jpg'.format(i)), nrow=1, padding=0, normalize=True)


def eval(config):
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.backends.cudnn.benchmark = False
    print(config.EVAL.eval_mode)

    if config.EVAL.eval_mode == 'eval_feat':
        config.save_sample=True
        eval_feat(config)
    elif 'warp' in config.EVAL.eval_mode:
        config.save_sample=True
        eval_warp(config)
    elif 'MC_cost' in config.EVAL.eval_mode:
        config.is_verbose = True
        eval_MC_cost(config)
    else:
        eval_quan_qual(config)

