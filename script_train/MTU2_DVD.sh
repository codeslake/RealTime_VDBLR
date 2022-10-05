#!/bin/bash

py3clean ./
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=9001 run.py \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -B -m torch.distributed.launch --nproc_per_node=8 --master_port=9001 run.py \
                        --is_train \
                        --mode MTU2_DVD \
                        --config config_MTU2 \
                        --network MTU \
                        --trainer trainer_multi_opt \
                        --data DVD \
                        -LRS LD \
                        -b 1 \
                        -th 4 \
                        -dl \
                        -dist
