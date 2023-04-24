#!/bin/bash
CUDA_VISIBLE_DEVICES="$1" python "3cGAN_testing.py" --test_model $2 --epoch $3
