import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import rawpy
import pickle
from skimage.color import rgb2lab, lab2rgb

# Unsharp Masking

def unsharp_masking(image,sigma): 
    im_zeros = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    
    for ch in range(image.shape[2]):    
        gaussian_filtered = cv2.GaussianBlur(image[:,:,ch], (0, 0), sigma)
        sharpened_image = image[:,:,ch] + (image[:,:,ch] - gaussian_filtered)
        sharpened_image = np.clip(sharpened_image, 0, 255)
        im_zeros[:,:,ch] = sharpened_image
        
    return im_zeros

