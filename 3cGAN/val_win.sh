#!/bin/bash

CUDA_VISIBLE_DEVICES="$1" C:/Users/win/anaconda3/envs/torch/python.exe "3cGAN_validation.py" --save_name $2 --ismerged $3 --mergew $4>$2.log &