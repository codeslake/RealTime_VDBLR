#!/bin/bash

py3clean ./

CUDA_VISIBLE_DEVICES=0 python run.py \
    --config config_MTU2 \
    --mode MTU2_amp_REDS \
    --network MTU \
    --trainer trainer_multi_opt_lmdb_amp \
    --data REDS \
    --ckpt_abs_name ckpt/MTU2_amp_REDS.pytorch \
    --eval_mode eval \
    --is_quan \
    --is_qual

