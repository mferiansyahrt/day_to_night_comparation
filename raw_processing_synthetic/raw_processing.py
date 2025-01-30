from raw_prc_pipeline.pipeline_utils import get_visible_raw_image, get_metadata
from raw_prc_pipeline.pipeline import PipelineExecutor, RawProcessingPipelineDemo
from raw_prc_pipeline import expected_landscape_img_height, expected_landscape_img_width, expected_img_ext
from utilsbaseline import fraction_from_json, json_read
from pathlib import Path
import json
import numpy as np
import cv2
import os
import pickle

def pickle_to_json_saved(path,filename):
    # rubah format metadata pickle jd json
    
    # open pickle file
    with open(path + filename + '.p', 'rb') as infile:
        obj = pickle.load(infile)
        pickle_obj = obj

    # convert pickle object to json object
    json_obj = json.loads(json.dumps(obj, default=str))
    
    # write the json file
    with open(
            os.path.splitext(path + filename + '.p')[0] + '.json',
            'w',
            encoding='utf-8'
        ) as outfile:
        json.dump(json_obj, outfile, ensure_ascii=False, indent=4)
    return pickle_obj,json_obj

def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
    # using opencv edge-aware demosaicing
    if alg_type != '':
        alg_type = '_' + alg_type
    if output_channel_order == 'BGR':
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2BGR' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2BGR' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2BGR' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
            print("CFA pattern not identified.")
    else:  # RGB
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2RGB' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2RGB' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2RGB' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
            print("CFA pattern not identified.")
    return opencv_demosaic_flag

def demosaic(bayer_image, cfa_pattern, output_channel_order='RGB', alg_type='VNG'):
    """
    Demosaic a Bayer image.
    :param bayer_image: Image in Bayer format, single channel.
    :param cfa_pattern: Bayer/CFA pattern.
    :param output_channel_order: Either RGB or BGR.
    :param alg_type: algorithm type. options: '', 'EA' for edge-aware, 'VNG' for variable number of gradients
    :return: Demosaiced image.
    """
    if alg_type == 'VNG':
        max_val = 255
        wb_image = (bayer_image * max_val).astype(dtype=np.uint8)
    else:
        max_val = 16383
        wb_image = (bayer_image * max_val).astype(dtype=np.uint16)

    if alg_type in ['', 'EA', 'VNG']:
        opencv_demosaic_flag = get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type=alg_type)
        demosaiced_image = cv2.cvtColor(wb_image, opencv_demosaic_flag)
    elif alg_type == 'menon2007':
        cfa_pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
        demosaiced_image = demosaicing_CFA_Bayer_Menon2007(wb_image, pattern=cfa_pattern_str)
    else:
        raise ValueError('Unsupported demosaicing algorithm, alg_type = {}'.format(alg_type))

    demosaiced_image = demosaiced_image.astype(dtype=np.float32) / max_val

    return demosaiced_image

def raw_processing_from_saved_im(main_path,metadata_filename,raw_im_filename,g_awb = True):
    #Baseline

    pipeline_params = {
        'tone_mapping': 'Flash', # options: Flash, Storm, Base, Linear, Drago, Mantiuk, Reinhard
        'illumination_estimation': 'gw', # ie algorithm, options: "gw", "wp", "sog", "iwp"
        'denoise_flg': True,
        'out_landscape_width': None,
        'out_landscape_height': None
    }

    pipeline_demo = RawProcessingPipelineDemo(**pipeline_params)
    
    # Meta Data Pickle
    pickle_path = main_path
    name = metadata_filename

    meta_data_night_pickle, meta_data_night_json = pickle_to_json_saved(path = pickle_path,filename = name)

    #Load Raw Image

    path = main_path
    im_filename = raw_im_filename
    metadata_filename = metadata_filename
    #name = 'RAW_000_2021_08_26_03_51_41_105_000050_000001668607'

    png_path = Path(path + im_filename + '.png')
    raw_image = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    meta_data = json_read(path + metadata_filename + '.json', object_hook=fraction_from_json)
    
    ## linearize
    linearized_image = pipeline_demo.linearize_raw(raw_image, meta_data_night_json)
    
    ## normalize
    normalized_image = pipeline_demo.normalize(linearized_image, meta_data_night_json)                  

    ## demosaic
    demosaic_image = demosaic(normalized_image, meta_data_night_pickle['cfa_pattern'])

    # Gray World AWB
    white_balanced_image = pipeline_demo.white_balance(demosaic_image, meta_data_night_pickle)

    if g_awb:
        demosaic_image = demosaic(linearized_image, meta_data_night_pickle['cfa_pattern'])
        g_awb_direct = pipeline_demo.white_balance(demosaic_image, meta_data_night_pickle)
        xyz_image_g_awb = pipeline_demo.xyz_transform(g_awb_direct, meta_data_night_pickle)
        srgb_image_g_awb = pipeline_demo.srgb_transform(xyz_image_g_awb, meta_data_night_pickle)
    else:
        srgb_image_g_awb = None
       
    #convert sRGB
    xyz_image = pipeline_demo.xyz_transform(white_balanced_image, meta_data_night_pickle)
    srgb_image = pipeline_demo.srgb_transform(xyz_image, meta_data_night_pickle)
                    
    return srgb_image,srgb_image_g_awb
                    
def raw_processing(image,meta_data,g_awb = True):
    #Baseline

    pipeline_params = {
        'tone_mapping': 'Flash', # options: Flash, Storm, Base, Linear, Drago, Mantiuk, Reinhard
        'illumination_estimation': 'gw', # ie algorithm, options: "gw", "wp", "sog", "iwp"
        'denoise_flg': True,
        'out_landscape_width': None,
        'out_landscape_height': None
    }

    pipeline_demo = RawProcessingPipelineDemo(**pipeline_params)
    
    # Meta Data
    meta_data = meta_data
    
    # Raw Image
    raw_image = image
    
    ## linearize
    linearized_image = pipeline_demo.linearize_raw(raw_image, meta_data)
    
    ## normalize
    normalized_image = pipeline_demo.normalize(linearized_image, meta_data)                  

    ## demosaic
    demosaic_image = demosaic(normalized_image, meta_data['cfa_pattern'])

    # Gray World AWB
    white_balanced_image = pipeline_demo.white_balance(demosaic_image, meta_data)

    if g_awb:
        demosaic_image = demosaic(linearized_image, meta_data['cfa_pattern'])
        g_awb_direct = pipeline_demo.white_balance(demosaic_image, meta_data)
        xyz_image_g_awb = pipeline_demo.xyz_transform(g_awb_direct, meta_data)
        srgb_image_g_awb = pipeline_demo.srgb_transform(xyz_image_g_awb, meta_data)
    else:
        srgb_image_g_awb = None
        
    #convert sRGB
    xyz_image = pipeline_demo.xyz_transform(white_balanced_image, meta_data)
    srgb_image = pipeline_demo.srgb_transform(xyz_image, meta_data)
                    
    return srgb_image,srgb_image_g_awb