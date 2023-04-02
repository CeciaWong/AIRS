#!/bin/bash

CUDA_VISIBLE_DEVICES="$1" python "/media/wxx/AIRS/3cGAN/3cGAN_test.py" --save_name $2 >AIRS/3cGAN/$2.log & --test_model $2 --epoch $3
