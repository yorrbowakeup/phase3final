from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from skimage import data_dir, io, transform, color
import cv2
import matplotlib.pyplot as plt

def img_preprocessing(funs, **args):
    rgb = io.imread(funs)
    rgb = cv2.resize(rgb, (256,256))
    # gray = color.rgb2gray(rgb)
    # dst = transform.resize(gray, (256*256))
    # return dst
    return rgb