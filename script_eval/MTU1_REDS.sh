#!/bin/bash

py3clean ./

CUDA_VISIBLE_DEVICES=0 python run.py \
    --config config_MTU1 \
    --mode MTU1_REDS \
    --network MTU \
    --trainer trainer_multi_opt_lmdb \
    --data REDS \
    --ckpt_abs_name ckpt/MTU1_REDS.pytorch \
    --eval_mode eval \
    --is_quan \
    --is_qual
