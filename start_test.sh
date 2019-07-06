#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 test.py --ca --gpu --output_dir $2
