import math
import numpy as np
import cv2
from scipy import special

def entropy(pk,
            base,
            axis
            ):
    
    if base is not None and base <= 0:
        raise ValueError("`base` must be a positive number or `None`.")

    pk = np.asarray(pk)
    pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
    vec = special.entr(pk)
   
    S = np.sum(vec, axis=axis)
    if base is not None:
        S /= np.log(base)
    return S


def calculate_vpmi(image, vpmi_show_metrics):
    
    ##AIE
    
    image_r = image[:,:,2]
    image_g = image[:,:,1]
    image_b = image[:,:,0]

    
    #gray_image_r = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)
    #gray_image_g = cv2.cvtColor(image_g, cv2.COLOR_BGR2GRAY)
    #gray_image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    
    _bins = 128
   # histr, _ = np.histogram(gray_image_r.ravel(), bins=_bins, range=(0, _bins))
   # histg, _ = np.histogram(gray_image_g.ravel(), bins=_bins, range=(0, _bins))
   # histb, _ = np.histogram(gray_image_b.ravel(), bins=_bins, range=(0, _bins))

    histr, _ = np.histogram(image_r.ravel(), bins=_bins, range=(0, _bins))
    histg, _ = np.histogram(image_g.ravel(), bins=_bins, range=(0, _bins))
    histb, _ = np.histogram(image_b.ravel(), bins=_bins, range=(0, _bins))
    
    prob_distr = histr / histr.sum()
    prob_distg = histg / histg.sum()
    prob_distb = histb / histb.sum()
    
    image_r_entropy = entropy(prob_distr, base=2,axis=0)
    image_g_entropy = entropy(prob_distg, base=2,axis=0)
    image_b_entropy = entropy(prob_distb, base=2,axis=0)
    
   # image_r_entropy = entropy2(prob_distr)
    #image_g_entropy = entropy2(prob_distg)
    #image_b_entropy = entropy2(prob_distb )
    
    average_ie = np.sqrt(( pow(((image_r_entropy +image_g_entropy + image_b_entropy) / 3),2) ))
    
    
    ##ABWF
    
    #bwfr = ( np.max(image_r) - np.min(image_r) + 1) / 256
    #bwfg = (np.max(image_g) - np.min(image_g) + 1) / 256
    #bwfb = (np.max(image_b) - np.min(image_b) + 1) / 256
    
    #bwfr = (bandwidth(image_r)[1] - (bandwidth(image_r)[0]) + 1 ) / 256
    #bwfg = (bandwidth(image_g)[1] - (bandwidth(image_g)[0]) + 1 ) / 256
    #bwfb = (bandwidth(image_b)[1] - (bandwidth(image_b)[0]) + 1 ) / 256
    
    bwfr = np.average((image_r[:,-1] - image_r[:,0] + 1 ) / 256)
    bwfg = np.average((image_g[:,-1] - image_g[:,0] + 1 ) / 256)
    bwfb = np.average((image_b[:,-1] - image_b[:,0] + 1 ) / 256)
  
    average_bwf =  np.sqrt(( pow(((bwfr +bwfg + bwfb )/ 3),2) ) )
    
    ## ACLF
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for channel_id in range(3):
        histogram, bin_edges = np.histogram(
        image_rgb[:, :, channel_id], bins=256, range=(0, 256)
        )
        if(channel_id==0):
            clfr = (1/256) * np.count_nonzero(histogram)
        elif(channel_id==1):
            clfg = (1/256) * np.count_nonzero(histogram)
        else :
            clfb = (1/256) * np.count_nonzero(histogram)
            
    
    average_clf = np.sqrt(( pow(((clfr +clfg + clfb)/ 3),2) ) )
    
    ## AL
    
    triple_image_r = image.copy()
    triple_image_r[:, :, 1] = 0
    triple_image_r[:, :, 0] = 0


    triple_image_g = image.copy()
    triple_image_g[:, :, 0] = 0
    triple_image_g[:, :, 2] = 0

    triple_image_b = image.copy()
    triple_image_b[:, :, 1] = 0
    triple_image_b[:, :, 2] = 0
    
    lumr =  cv2.cvtColor(triple_image_r, cv2.COLOR_BGR2HSV)[...,2] 
    lumg = cv2.cvtColor(triple_image_g, cv2.COLOR_BGR2HSV)[...,2] 
    lumb = cv2.cvtColor(triple_image_b, cv2.COLOR_BGR2HSV)[...,2] 
    
    al_lumr = np.average( cv2.cvtColor(triple_image_r, cv2.COLOR_BGR2HSV)[...,2] )
    al_lumg = np.average(cv2.cvtColor(triple_image_g, cv2.COLOR_BGR2HSV)[...,2] )
    al_lumb = np.average(cv2.cvtColor(triple_image_b, cv2.COLOR_BGR2HSV)[...,2] )
    #print(lumr.shape)
   # print("\n")
    #print(al_lumr)
    average_al = (al_lumr + al_lumg + al_lumb) / 3
    average_al_op = 127.5
    ## LNPF 
    #print(average_al)
    lnpf = 1 - (abs((average_al-average_al_op))/average_al_op)
    
    ##AG
    average_gr = ag(image)

    ##VPMI
    
    vpmi_value = average_ie * math.log2(average_gr) * lnpf * average_clf * average_bwf
    if vpmi_show_metrics:
        return average_ie , average_bwf, average_clf, average_al , lnpf, average_gr, vpmi_value
    else:
        return vpmi_value
    
def ag(image):
    image_r = image[:,:,2]
    image_g = image[:,:,1]
    image_b = image[:,:,0]
    
    # Calculate luminance for each channel
    lumr = 0.299 * image_r
    lumg = 0.587 * image_g
    lumb = 0.114 * image_b
    
    def calculate_gradient(luminance, direction='x'):
        # Calculate the gradient in the specified direction (x or y)
        if direction == 'x':
            gradient = np.gradient(luminance, axis=1)
            #gradient = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
        elif direction == 'y':
            gradient = np.gradient(luminance, axis=0)
            #gradient = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
        else:
            raise ValueError("Invalid direction. Use 'x' or 'y'.")

        return gradient

    def calculate_magnitude(gradient_x, gradient_y, p, mu):
        # Calculate the magnitude of the gradient
        magnitude = 1 / np.sqrt(2) * np.sqrt((gradient_x)**2 + (gradient_y)**2)
        return magnitude

    def calculate_average_gradient(gradient_x, gradient_y, M, N):
        # Calculate the average gradient AG
        ag = np.sum(abs(gradient_x - np.roll(gradient_x, shift=(1, 1), axis=(0, 1)))) + \
             np.sum(abs(gradient_y - np.roll(gradient_y, shift=(1, 1), axis=(0, 1))))

        return ag / (M * N)
   
    def calculate_each_channel(lum_c):
        gradient_x = calculate_gradient(lum_c,'x')
        gradient_y = calculate_gradient(lum_c,'y')
        
        M,N = lum_c.shape
        ag = calculate_average_gradient(gradient_x,gradient_y,M,N)
        return ag
    
    agr = calculate_each_channel(lumr)
    agg = calculate_each_channel(lumg)
    agb = calculate_each_channel(lumb)
    avg_g = np.sqrt(1/3*(agr**2 + agb**2 + agg**2))
    
    #gr_x = calculate_gradient(lumb,'x')
    #gr_y = calculate_gradient(lumb,'y')
    return avg_g