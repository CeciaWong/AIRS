#!/bin/bash

CUDA_VISIBLE_DEVICES="$1" C:/Users/win/anaconda3/envs/torch/python.exe "E:\AIRS\Visually_Navigated_Bronchoscopy\3cGAN\3cGAN\3cGAN_testing.py" --test_model $2 --epoch $3
