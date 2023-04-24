#!/bin/bash
CUDA_VISIBLE_DEVICES="$1" python "3cGAN_validation.py" --save_name $2 --ismerged $3 --mergew $4>$2.log &