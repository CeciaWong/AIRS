#!/bin/bash

CUDA_VISIBLE_DEVICES="$1" python "/media/wxx/AIRS/3cGAN/3cGAN_testing.py" --test_model $2 --epoch $3 >$2.log & 
