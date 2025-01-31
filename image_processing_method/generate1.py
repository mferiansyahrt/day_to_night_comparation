import prepare_data1
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import rawpy
import pickle
from skimage.color import rgb2lab, lab2rgb

from utils.relight1 import relight_locally, apply_local_lights_rgb
from pipeline.pipeline import run_pipeline
from pipeline.pipeline_utils import normalize, denormalize, get_visible_raw_image, ratios2floats, white_balance, \
    get_metadata
from utils.gen_utils import check_dir
from noise_profiler.image_synthesizer import load_noise_model, synthesize_noisy_image_v2

def generate_synthetic_image(sample,main_path,noise_model_path,night_illum_path,
                                   stages,params,noisy_im = True,**params_synth):
    
    # Read Single Image
    main = main_path
    day = main + '/day/'
    night = main + '/night_real/clean_raw/'

    noise_model_path = noise_model_path
    noise_model, iso2b1_interp_splines, iso2b2_interp_splines = load_noise_model(path=noise_model_path)

    sampel = sample
    im = os.path.join(day,sorted(os.listdir(day))[sampel])
    with rawpy.imread(im) as raw:
        rgb = raw.postprocess()

    def get_illum_normalized_by_g(illum_in_arr):
        return illum_in_arr[:, 0] / illum_in_arr[:, 1], illum_in_arr[:, 1] / illum_in_arr[:, 1], illum_in_arr[:, 2] / illum_in_arr[:, 1]

    # load nighttime illuminants
    gt_illum = scipy.io.loadmat(night_illum_path)
    gt_illum = gt_illum['night_dict']

    gt_illum[:, 0], gt_illum[:, 1], gt_illum[:, 2] = get_illum_normalized_by_g(gt_illum)

    gt_illum_mean = np.mean(gt_illum, 0) * 0.5 
    gt_illum_cov = np.cov(np.transpose(gt_illum)) * 0.5 
    
    relight_local = params_synth['relight_local']

    if relight_local:
        light_mask_dirname = 'masks'
    else:
        light_mask_dirname = None

    img_bayer = get_visible_raw_image(im)
    meta_data = get_metadata(im)
        
    # Generate Clean Synthetic Night Image
    '''Namespace(base_address='./dataset/day/', 
    savefolderpath='synthetic_datasets', 
    savefoldername='night', how_many_train=60, dim=True, 
    relight=True, discard_black_level=False, clip=True, relight_local=True, 
    min_num_lights=5, max_num_lights=5, min_light_size=0.5, 
    max_light_size=1.0, save_light_masks=False, num_sat_lights=5, iso_list='1600,3200')'''
        
    results_ = prepare_data1.synth_night_imgs(in_img_or_path = img_bayer, in_day_meta_data = meta_data,
                                                gt_illum_mean = gt_illum_mean, 
                                                gt_illum_cov = gt_illum_cov,
                                                **params_synth)
    
    # num_sat_lights = berapa banyak "bola lampu dalam gambar"
    # min_num_lights & max num_lights = seberapa terang
    # min_light_size & max_light_size = seberapa terang

    '''    
    :param min_num_lights: Minimum number of local illuminants, in case of local relighting.
    :param max_num_lights: Maximum number of local illuminants, in case of local relighting.
    :param min_light_size: Minimum local light size as percent of image dimension, in case of local relighting.
    :param max_light_size: Maximum local light size as percent of image dimension, in case of local relighting.
    :param num_sat_lights: Number of small saturated local lights, in case of local relighting.'''
    print('a')
    if relight_local:
        example_night_synth, meta_data_night, local_lights = results_
    else:
        example_night_synth, meta_data_night = results_
        local_lights = None

    meta_data_night_raw = meta_data_night

    as_shot_neutral = meta_data_night['as_shot_neutral']  # keep as_shot_neutral
    if relight_local:
        meta_data_night['as_shot_neutral'] = meta_data_night['avg_night_illuminant']  # modify as_shot_neutral
    night_synth_srgb_avg = run_pipeline(example_night_synth, params=params, metadata=meta_data_night, stages=stages)
    night_synth_srgb_avg = (night_synth_srgb_avg * 255).astype(np.uint8)
    meta_data_night['as_shot_neutral'] = as_shot_neutral  # restore as_shot_neutral
    print('b')
    # Generate Noisy Image
    if noisy_im:
        noisy_night_image = synthesize_noisy_image_v2(example_night_synth, model=noise_model,
                                        dst_iso=meta_data_night['iso'], min_val=0,
                                        max_val=1023,
                                        iso2b1_interp_splines=iso2b1_interp_splines,
                                        iso2b2_interp_splines=iso2b2_interp_splines)

        noisy_night_image = noisy_night_image.astype(np.uint16)

        as_shot_neutral = meta_data_night['as_shot_neutral']  # keep as_shot_neutral
        if relight_local:
            meta_data_night['as_shot_neutral'] = meta_data_night['avg_night_illuminant']  # modify as_shot_neutral
        noisy_night_image_srgb = run_pipeline(noisy_night_image, params=params, metadata=meta_data_night, stages=stages)
        noisy_night_image_srgb = (noisy_night_image_srgb * 255).astype(np.uint8)
        meta_data_night['as_shot_neutral'] = as_shot_neutral  # restore as_shot_neutral
        print('c')
        save_light_masks = True

        if relight_local and save_light_masks and local_lights is not None:

            neutral_image = np.ones(night_synth_srgb_avg.shape, dtype=np.float32)

            # save individual local light masks
            for k, light in enumerate(local_lights[1:]):
                neutral_image_relight = apply_local_lights_rgb(neutral_image, [local_lights[0], light], clip=True,
                                                               invert_wb=True)
                # save combined light mask
                neutral_image_relight = apply_local_lights_rgb(neutral_image, local_lights, clip=True, invert_wb=True)
    else:
        noisy_night_image_srgb = None
        noisy_night_image = None
    print('d')
    return rgb,night_synth_srgb_avg, example_night_synth.astype(np.uint16), \
            meta_data_night_raw, noisy_night_image_srgb, noisy_night_image

def save_clean_raw_image(no,save_path,clean,clean_raw,metadata_raw,noisy,noisy_raw):
    # save image raw clean and raw noisy

    path = save_path

    # CLEAN
    cv2.imwrite(path + f'clean{no}' + '.png',
                clean[:,:,[2,1,0]])
    
    # CLEAN RAW
    cv2.imwrite(path + f'clean_raw{no}' + '.png',
                clean_raw)
    
    # META DATA
    pickle.dump(metadata_raw,
                open(path + f'metadata_raw{no}' + '.p', "wb"))

    # NOISY
    cv2.imwrite(path + f'noisy{no}' + '.png',
                noisy[:,:,[2,1,0]])
    
    # NOISY RAW
    cv2.imwrite(path + f'noisy_raw{no}' + '.png',
                noisy_raw)
    
    #pickle.dump(meta_data_night_noisy,
    #            open(path + f'noisy_raw{no}' + '.p', "wb"))


