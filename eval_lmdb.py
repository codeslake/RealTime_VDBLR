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
from data_loader.utils import refine_image_pt, read_frame, load_file_list, norm
from models.utils import warp
from ckpt_manager import CKPT_Manager
# from eval_input import eval_input
# from eval_specific_video import eval_specific_video

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
    date = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')

    model = create_model(config)
    model.eval()
    network = model.get_network().eval()

    ckpt_manager = CKPT_Manager(config.LOG_DIR.ckpt, config.mode, config.max_ckpt_num, is_descending = False)
    load_state, ckpt_name = ckpt_manager.load_ckpt(network, by_score = config.EVAL.load_ckpt_by_score, name = config.EVAL.ckpt_name, abs_name = config.EVAL.ckpt_abs_name, epoch = config.EVAL.ckpt_epoch)
    print('\nLoading checkpoint \'{}\' on model \'{}\': {}'.format(ckpt_name, config.mode, load_state))

    save_path_root = config.EVAL.LOG_DIR.save

    save_path_root_deblur = os.path.join(save_path_root, mode, ckpt_name.split('.')[0])
    save_path_root_deblur_score = save_path_root_deblur
    Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), os.path.join(save_path_root_deblur, ckpt_name))
    save_path_root_deblur = os.path.join(save_path_root_deblur, config.EVAL.data, date)
    # Path(save_path_root_deblur).mkdir(parents=True, exist_ok=True)

    return network, model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name

def eval_quan_qual(config):
    mode = config.EVAL.eval_mode
    network, model, save_path_root_deblur, save_path_root_deblur_score, ckpt_name = init(config, mode)

    ##
    total_norm = 0
    total_itr_time = PSNR_mean_total = SSIM_mean_total = 0
    total_itr_time_video = PSNR_mean = SSIM_mean = 0
    frame_len_prev = 0

    for i, inputs in enumerate(model.data_loader_eval):
        is_first_frame = inputs['is_first'][0].item()

        if is_first_frame:
            if i > 0:
                PSNR_mean_total = PSNR_mean_total + PSNR_mean
                SSIM_mean_total = SSIM_mean_total + SSIM_mean
                total_itr_time = total_itr_time + total_itr_time_video

                PSNR_mean = PSNR_mean / frame_len_prev
                SSIM_mean = SSIM_mean / frame_len_prev
                total_itr_time_video = total_itr_time_video / frame_len_prev

                print('[MEAN EVAL {}|{}|{}][{}/{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0], inputs['video_len'][0], PSNR_mean, SSIM_mean, total_itr_time_video))
                with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'a') as file:
                    file.write('[MEAN EVAL {}|{}|{}][{}/{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0], inputs['video_len'][0], PSNR_mean, SSIM_mean, total_itr_time_video))
                    file.close()

            total_itr_time_video = PSNR_mean = SSIM_mean = 0

        #########################
        init_time = time.time()
        with torch.no_grad():
            results = model.evaluation(inputs)
        # gc.collect()
        # torch.cuda.empty_cache()
        itr_time = time.time() - init_time
        #########################

        ## evaluation
        outs = results

        inp = outs['input']
        outs['result'] = torch.clamp(outs['result'], 0, 1)
        output = outs['result']

        PSNR = SSIM = 0
        gt = outs['gt']

        # quantitative
        output_cpu = output.cpu().numpy()[0].transpose(1, 2, 0)
        gt_cpu = gt.cpu().numpy()[0].transpose(1, 2, 0)


        if config.EVAL.is_quan:
            PSNR = psnr(output_cpu, gt_cpu)
            SSIM = ssim(output_cpu, gt_cpu)

        PSNR_mean = PSNR_mean + PSNR
        SSIM_mean = SSIM_mean + SSIM

        frame_name = inputs['frame_name'][0]
        print('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, PSNR, SSIM, itr_time))
        with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'w' if (i == 0) else 'a') as file:
            file.write('[EVAL {}|{}|{}][{}/{}][{}/{}] {} PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(config.mode, config.EVAL.data, inputs['video_name'][0], inputs['video_idx'][0]+1, inputs['video_len'][0], inputs['frame_idx'][0]+1, inputs['frame_len'][0], frame_name, PSNR, SSIM, itr_time))
            file.close()

        # qualitative
        ## create output dir for a video
        for iformat in ['png']:
            frame_name_no_ext = frame_name.split('.')[0]
            save_path_deblur = os.path.join(save_path_root_deblur, iformat)
            Path(save_path_deblur).mkdir(parents=True, exist_ok=True)

            Path(os.path.join(save_path_deblur, 'output', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
            save_file_path_deblur_output = os.path.join(save_path_deblur, 'output', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
            vutils.save_image(output, '{}'.format(save_file_path_deblur_output), nrow=1, padding = 0, normalize = False)

            if config.EVAL.save_input_gt:
                Path(os.path.join(save_path_deblur, 'input', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                save_file_path_deblur_input = os.path.join(save_path_deblur, 'input', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                vutils.save_image(inp, '{}'.format(save_file_path_deblur_input), nrow=1, padding = 0, normalize = False)

                if 'gt' in inputs.keys():
                    Path(os.path.join(save_path_deblur, 'gt', inputs['video_name'][0])).mkdir(parents=True, exist_ok=True)
                    save_file_path_deblur_gt = os.path.join(save_path_deblur, 'gt', inputs['video_name'][0], '{}.{}'.format(frame_name_no_ext, iformat))
                    vutils.save_image(gt, '{}'.format(save_file_path_deblur_gt), nrow=1, padding = 0, normalize = False)

        total_itr_time_video = total_itr_time_video + itr_time
        total_norm = total_norm + 1
        frame_len_prev = inputs['frame_len'][0]

    # total average
    total_itr_time = (total_itr_time + total_itr_time_video) / total_norm
    PSNR_mean_total = (PSNR_mean_total + PSNR_mean) / total_norm
    SSIM_mean_total = (SSIM_mean_total + SSIM_mean) / total_norm

    sys.stdout.write('\n[TOTAL {}|{}] PSNR: {:.5f}  SSIM: {:.5f} ({:.5f}sec)'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time))
    with open(os.path.join(save_path_root_deblur_score, 'score_{}_{}.txt'.format(config.EVAL.data, config.EVAL.eval_mode)), 'a') as file:
        file.write('\n[TOTAL {}|{}] PSNR: {:.5f} SSIM: {:.5f} ({:.5f}sec)\n'.format(ckpt_name, config.EVAL.data, PSNR_mean_total, SSIM_mean_total, total_itr_time))
        file.close()

def eval(config):
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        eval_quan_qual(config)
