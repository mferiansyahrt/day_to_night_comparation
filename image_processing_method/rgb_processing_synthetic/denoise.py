import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import rawpy
import pickle
from skimage.color import rgb2lab, lab2rgb

# Denoising

def denoising_image(synth_rgb_im,synth_processed_raw2rgb_im,raw_processed):

    if raw_processed:
        srgb_image_uint = (synth_processed_raw2rgb_im * 255).round().astype(np.uint8)
        im = cv2.fastNlMeansDenoisingColored(srgb_image_uint, None, 10, 10, 7, 15) 
    else:
        im = cv2.fastNlMeansDenoisingColored(synth_rgb_im, None, 10, 10, 7, 15)
        
    return im