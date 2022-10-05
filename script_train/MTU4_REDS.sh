#!/bin/bash

py3clean ./
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=9000 run.py \
CUDA_VISIBLE_DEVICES=0,1,2,3 nice -n 20 python -B -m torch.distributed.launch --nproc_per_node=4 --master_port=9002 run.py \
                        --is_train \
                        --mode MTU4_REDS \
                        --config config_MTU4 \
                        --network MTU \
                        --trainer trainer_multi_opt_lmdb\
                        --data REDS \
                        -LRS LD \
                        -b 2 \
                        -th 4 \
                        -dl \
                        -dist
