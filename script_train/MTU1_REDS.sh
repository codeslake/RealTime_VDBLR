#!/bin/bash

py3clean ./
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=9001 run.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -B -m torch.distributed.launch --nproc_per_node=8 --master_port=9001 run.py \
                        --is_train \
                        --mode MTU1_REDS \
                        --config config_MTU1 \
                        --network MTU \
                        --trainer trainer_multi_opt_lmdb \
                        --data REDS \
                        -LRS LD \
                        -b 1 \
                        -th 4 \
                        -dl \
                        -dist
