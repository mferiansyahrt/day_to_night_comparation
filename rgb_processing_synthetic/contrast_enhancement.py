import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import rawpy
import pickle
from skimage.color import rgb2lab, lab2rgb

# Local Contrast Enhancement

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.float64(ycbcr)

def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype('float64')
    return cv2.LUT(image, table)

def gamma_filter(image, gamma):
    # Gamma correction 
    # Increase gamma for darkening, decrease for brightening
    filtr = np.power(image, gamma).clip(0, 255).astype(np.uint8) 
    return filtr

def luminance_corr(image,gamma):
    im_ycbcr = rgb2ycbcr(image)
    Y = im_ycbcr[:,:,0]
    M = gamma_filter(Y,gamma)
    eks = gamma ** ((0.5 - (1-M))/0.5)
    Y_ = []
    for i,j in zip(Y,eks):
        temp =[]
        for val,eksponen in zip(i,j):
            temp.append(val**eksponen.astype("float32"))
        Y_.append(temp)
    Y_ = np.array(Y_)
    return Y,Y_

def new_ycbcr(image,lum):
    origin = rgb2ycbcr(image)
    im_ycbcr = rgb2ycbcr(image)
    im_ycbcr[:,:,0] = lum
    return origin,im_ycbcr

def contrast_fixing(im_ycbcr,Y_stretched):
    # Extract the Y (luminance) channel from YCbCr
    Y = im_ycbcr[:, :, 0]

    # Compute the chroma radius (Cb and Cr channels) and chrominance
    Cb = im_ycbcr[:, :, 1]
    Cr = im_ycbcr[:, :, 2]
    chroma_radius = np.sqrt(Cb ** 2 + Cr ** 2)

    # Define thresholds for dark pixels
    luminance_threshold = 0.14
    chroma_threshold = 0.07

    # Find dark pixels
    dark_pixel_mask = (Y < luminance_threshold) & (chroma_radius < chroma_threshold)

    # Calculate cumulative histograms for Y
    hist_y = cv2.calcHist([Y], [0], None, [256], [0, 256])
    cumulative_hist_y = np.cumsum(hist_y)

    # Calculate cumulative histograms for Y hat -> After applying LCC
    hist_y_hat = cv2.calcHist([Y_stretched], [0], None, [256], [0, 256])
    cumulative_hist_y_hat = np.cumsum(hist_y_hat)

    # If there are dark pixels, calculate lower range
    if np.any(dark_pixel_mask):
        # Determine the threshold for the darkest 30% pixels
        total_pixels = Y.size
        percentile_threshold = 0.3
        threshold_value = int(total_pixels * percentile_threshold)

        # Find the bin value corresponding to the threshold in cumulative_hist_y_hat
        lower_range_y_hat = np.where(cumulative_hist_y_hat >= threshold_value)[0][0]

        # Find the bin value corresponding to the threshold in cumulative_hist_y
        lower_range_y = np.where(cumulative_hist_y >= threshold_value)[0][0]

        # Calculate lower range difference
        lower_range_difference = lower_range_y_hat - lower_range_y

        # Clip lower and upper ranges (maximum 50 bins)
        lower_range_difference = max(0, min(lower_range_difference, 50))

        # Apply histogram stretching to Y channel
        Y_stretched = cv2.normalize(Y, None, lower_range_difference, 255, cv2.NORM_MINMAX)
    else:
        # If no dark pixels, define lower range as the second percentile
        lower_range_y_hat = np.percentile(Y_stretched, 2)
        Y_stretched = Y

    # Calculate the upper range (always corresponds to the 98th percentile)
    upper_range_y_hat = np.percentile(Y_stretched, 98)

    # Stretch the image histogram using the determined ranges
    Y_stretched = cv2.normalize(Y_stretched, None, 0, 255, cv2.NORM_MINMAX)

    # Update the Y channel in the YCbCr image
    im_ycbcr[:, :, 0] = Y_stretched
    return im_ycbcr

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float64)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def saturation_fixing(rgb_after_lccC,ycbcr_before_lcc,ycbcr_after_lcc):
    Y = ycbcr_before_lcc[:,:,0]
    Y_ = ycbcr_after_lcc[:,:,0]
    
    R = rgb_after_lccC[:,:,0]
    G = rgb_after_lccC[:,:,1]
    B = rgb_after_lccC[:,:,2]
    
    init = np.zeros((ycbcr_before_lcc.shape[0],ycbcr_before_lcc.shape[1],ycbcr_before_lcc.shape[2]))
    init[:,:,0], init[:,:,1], init[:,:,2]  = R + Y, G + Y, B + Y    # C + Y
    
    init2 = np.zeros((ycbcr_before_lcc.shape[0],ycbcr_before_lcc.shape[1],ycbcr_before_lcc.shape[2]))
    init2[:,:,0], init2[:,:,1], init2[:,:,2]  = R - Y, G - Y, B - Y # C - Y
    
    idx_zero = np.where(Y == 0.0)
    for i,j in zip(idx_zero[0],idx_zero[1]):
        Y[i][j] = 10e-06
        
    div_factor = Y_/Y
    
    init[:,:,0], init[:,:,1], init[:,:,2] = 0.5 * div_factor * init[:,:,0], 0.5 * div_factor * init[:,:,1], 0.5 * div_factor * init[:,:,2]
    
    rgb_saturation_fixed = init + init2
    return rgb_saturation_fixed

# Global Mean Contrast

def glob_mean_contrast(im,beta):
    r,g,b = im[:,:,0],im[:,:,1],im[:,:,2]
    r_mean,g_mean,b_mean = np.mean(r),np.mean(g),np.mean(b)
    
    def calc(channel,mean):
        glob_mean = mean + beta * (channel-mean)
        return glob_mean
    
    r_glob,g_glob,b_glob = calc(r,r_mean),calc(g,g_mean),calc(b,b_mean)
    rgb_glob = np.dstack((r_glob,g_glob,b_glob))
    return rgb_glob

# S-Curve Correction

def s_curve_corr1(image,alpha,lambda_val):
    im_zeros = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    for ch in range(image.shape[2]):
        c = image[:,:,ch]
        for idx_w,w in enumerate(c):
            for idx_h,h in enumerate(w):
                if h >= alpha:
                    pix = alpha + ((1 - alpha) * (((h - alpha)/(1 - alpha))**lambda_val))
                else:
                    pix = alpha - (alpha * ((1 - (h/alpha))**lambda_val))
                im_zeros[idx_w,idx_h,ch] = pix
    return im_zeros

def s_curve_corr2(image,alpha,lambda_value):
    im_zeros = np.zeros((image.shape[0],image.shape[1],image.shape[2]))
    for channel in range(3):  
        C = image[:, :, channel]
        S_CurveC = np.where(C >= alpha, alpha + (1 - alpha) * ((C - alpha) / (1 - alpha)) ** lambda_value, alpha - alpha * (1 - (C / alpha)) ** lambda_value)
        im_zeros[:, :, channel] = S_CurveC
    return im_zeros

# Histogram Stretching

def histogram_stretching(image):
    min_value = np.min(image)
    max_value = np.max(image)

    stretched_image = 255 * (image - min_value) / (max_value - min_value)
    return stretched_image

# Conditional Contrast Correction

def cond_contrast_corr(image,lower_t,upper_t,alpha,lambda_val,gamma):

    histed_ycbcr = rgb2ycbcr(image)
    luminance = histed_ycbcr[:,:,0]
    mean_luminance = np.mean(luminance)
    
    if mean_luminance <= lower_t:
        cond_im = s_curve_corr1(image,alpha,lambda_val)
        txt = 'S-Curve Correction Applied'
    
    elif mean_luminance >= upper_t:
        cond_im = gamma_filter(image,gamma) 
        txt = 'Gamma Filter Applied'
    
    else:
        cond_im = image
        txt = 'No Conditional Contrast Correction method Applied'
        
    return cond_im,mean_luminance,txt
