import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import rawpy
import pickle
from skimage.color import rgb2lab, lab2rgb

# Grayness Index

def rgb2lab(rgb, gamma):
    # Ensure input is in the range [0, 255] and convert to float
    rgb = np.clip(rgb, 0, 255).astype(float)

    # Normalize RGB values to the range [0, 1]
    rgb /= 255.0

    # Apply gamma correction
    rgb = np.where(rgb <= 0.04045, rgb / (gamma * 12.92), ((rgb / gamma + 0.055) / (1.055)) ** (2.4 * gamma))

    # Linear transformation to XYZ color space
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalize to D65 white point
    x /= 0.950456
    z /= 1.088754

    # Nonlinear transformation to LAB color space using vectorized operations
    epsilon = 6 / 29
    f = np.where(y > epsilon, y ** (1/3), (y / (3 * epsilon ** 2) + 4/29))
    l = 116 * f - 16
    a = (x - y) * 500
    b = (y - z) * 200

    return np.stack((l, a, b), axis=-1)

def lab2rgb(lab, gamma):
    # Nonlinear transformation from LAB to XYZ color space
    epsilon = 6 / 29
    f_inv = lambda t: np.where(t > epsilon, t ** (1/3), (t / (3 * epsilon ** 2) + 4/29))
    y = (lab[:, :, 0] + 16) / 116
    x = lab[:, :, 1] / 500 + y
    z = y - lab[:, :, 2] / 200

    x = f_inv(x)
    y = f_inv(y)
    z = f_inv(z)

    x *= 0.950456
    z *= 1.088754

    # Linear transformation from XYZ to RGB color space
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    # Apply gamma correction
    r = np.where(r <= 0.0031308, 12.92 * r, (1.055 * r) ** (1/gamma) - 0.055)
    g = np.where(g <= 0.0031308, 12.92 * g, (1.055 * g) ** (1/gamma) - 0.055)
    b = np.where(b <= 0.0031308, 12.92 * b, (1.055 * b) ** (1/gamma) - 0.055)

    # Scale RGB values to the range [0, 255]
    rgb = np.stack((r, g, b), axis=-1) * 255

    return np.clip(rgb, 0, 255).astype(np.uint8)

def grayness_index(image,threshold):
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    #lab_image = rgb2lab(image)
    #lab_image = rgb2lab(im_unsharped_masking,gamma)

    # Calculate the Grayness Index using the L channel
    l_channel = lab_image[:, :, 0]
    grayness_index = l_channel  
    achromatic_pixels = grayness_index <= threshold
    correction_factor = np.mean(l_channel[achromatic_pixels])
    balanced_l_channel = l_channel / correction_factor

    # Merge the LAB channels back
    balanced_lab_image = lab_image.copy()
    balanced_lab_image[:, :, 0] = balanced_l_channel

    # Convert the balanced LAB image back to BGR color space
    balanced_image = cv2.cvtColor(balanced_lab_image, cv2.COLOR_Lab2RGB)
    #balanced_image = lab2rgb(balanced_lab_image)
    #balanced_image = lab2rgb(balanced_lab_image,gamma)
    return balanced_image