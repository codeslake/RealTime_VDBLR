from configs.config import get_config as main_config
from configs.config import log_config, print_config
import math
import torch
import numpy as np

def get_config(project = '', mode = '', config = '', data = '', LRS = '', batch_size = 8):

    ## GLOBAL
    config = main_config(project, mode, config, data, LRS, batch_size)

    ## LOCAL
    # tarining
    actual_batch_size = config.batch_size * torch.cuda.device_count()
    config.lr_init = 1e-4
    config.trainer = 'trainer_multi_opt_v2'

    # networks
    config.network = 'MTUv2_stack_multi_opt_v2-skip_C'

    config.HG_num = 4
    config.RB_num = 8#4
    config.ch = 26#32
    config.max_displacement = 10

    config.is_aux = True
    config.is_distill = True
    config.aux_lambda = 1e-1
    config.distill_lambda = 1e-2
    config.decay_distill = 0

    config.skip_corr_index = [1,2,3] # 0(m)1(skip)2(m)3

    config.wi = 1.1 # weight init (xavier)
    config.win = 0.03 # weight init (normal)

    ## data
    config.frame_itr_num = 13
    config.frame_num = 5
    config.refine_val = 4

    ## training schedule
    # config.write_ckpt_every_epoch = 1

    if config.data == 'nah':
        total_frame_num = int(6309/3)
        video_num = 22
    elif config.data == 'DVD':
        total_frame_num = int(11416/2)
        video_num = 61

    config.total_itr = 300000
    IpE = math.ceil((len(list(range(0, total_frame_num - (config.frame_itr_num-1), config.frame_itr_num)))) / actual_batch_size) * config.frame_itr_num
    #our_epoch = math.ceil(config.total_itr / IpE)

    if config.LRS == 'LD':
        # lr_decay
        config.decay_period = []
        #config.decay_period = [400000, 450000]
        config.decay_rate = 0.25
        config.warmup_itr = -1
    elif config.LRS == 'CA':
        # Cosine Anealing
        config.warmup_itr = -1
        config.T_period = [0, config.total_itr]
        config.restarts = np.cumsum(config.T_period)[:-1].tolist()
        config.restart_weights = np.ones_like(config.restarts).tolist()
        config.eta_min = config.lr_min

    return config
