import numpy as np
from PIL import Image
import os
import cv2

inpath = r"E:\AIRS\AIRS\3cGAN_dataset\Testing\osteo-testing\C2"
for img in os.listdir(inpath):
    x = Image.open(os.path.join(inpath, img))
    x = np.asarray(x)
    x = x/np.max(x)*255
    cv2.imwrite(os.path.join(inpath, img.replace(".png","_1.png")), x)