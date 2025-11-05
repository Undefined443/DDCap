#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4

torchrun --nproc-per-node auto train.py --out_dir results --tag caption_diff_vitb16
