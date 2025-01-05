

from __future__ import annotations

from collections import OrderedDict
import torch
import yaml
import os
import os.path as osp
import cv2
import numpy as np
import tyro
import subprocess

from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

from src.utils.io import load_image_rgb
from src.live_portrait_pipeline import LivePortraitPipeline
from time import time
import pickle

import os
import time
import json
from natsort import natsorted

def remove_ddp_dumplicate_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath("./")), fn)

device = "cpu"

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")

tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
if osp.exists(ffmpeg_dir):
    os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

if not fast_check_ffmpeg():
    raise ImportError(
        "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
    )

fast_check_args(args)

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)
crop_cfg = partial_fields(CropConfig, args.__dict__)

live_portrait_pipeline = LivePortraitPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg
)


def get_descriptor_information(driving):
    #Part 2
    # flag_is_driving_video = False
    driving_img_rgb = load_image_rgb(driving)
    output_fps = 25
    driving_rgb_lst = [driving_img_rgb]

    # Part 4.1 excluding 4.1.1
    ret_d = live_portrait_pipeline.cropper.crop_driving_video(driving_rgb_lst)
    driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
    driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]

    # Consistent block minus some dumping pkl logic
    c_d_eyes_lst, c_d_lip_lst = live_portrait_pipeline.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
    # save the motion template
    I_d_lst = live_portrait_pipeline.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
    driving_template_dct = live_portrait_pipeline.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

    return {"c_d_eyes_lst": c_d_eyes_lst, "c_d_lip_lst": c_d_lip_lst, "driving_template_dct": driving_template_dct}

def generate_descriptors_for_folder(folder_path, filename):
    descriptors = {}
    
    # Record start time
    start_time = time.time()
    i = 0
    # Walk through all files in the given folder path
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            descriptor = get_descriptor_information(file_path)  # Get descriptor information
            descriptors["/".join(file_path.split("/")[-6:])] = descriptor  # Add to dictionary            
            if i % 1000 == 0:
                print(i, file_path)
            i+=1

    # Record end time
    end_time = time.time()
    
    # Calculate and print processing time
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time:.2f} seconds")

    return descriptors

## 
def generate_descriptors_meads(actor_folder_path, actor_id):
    emotions_list = natsorted(os.listdir(actor_folder_path))
    for emo in emotions_list:
        # if emo == "angry" and actor_id == "M007":
        #     continue
        emo_path = os.path.join(actor_folder_path, emo)
        emotion_levels_list = natsorted(os.listdir(emo_path))
        for level in emotion_levels_list:
            emo_level_path = os.path.join(emo_path, level)
            file_name = f'Data/{actor_id}/MEAD_descriptor_{actor_id}_{emo}_{level}.pkl'
            descriptors = generate_descriptors_for_folder(emo_level_path, "")
            with open(file_name, 'wb') as file:
                pickle.dump(descriptors, file)
            del descriptors

# def get_descriptors_for_a_folder_in_sequence(folder_path):
#     descriptor_list = []
#     files = natsorted(os.listdir(folder_path))
#     for file in files:
#        descriptor_list.append(get_descriptor_information(os.path.join(folder_path, file)))
    
#     return descriptor_list

# descriptor_list = get_descriptors_for_a_folder_in_sequence("/home/sawaiz/Documents/Lab/In_Progress/Current/Dr. Immadullah/Phase 1/code_base/pipeline/LivePortrait_descriptor_extractor/test_videos/01-01-01-01-01-01-01")


actor_id = "M007"
actor_folder_path = f"/mnt/18EEB7A0EEB7749A/PortraitVideoSynthesis/Data/DatasetMEAD_Frames/{actor_id}/front/"
os.makedirs(f"Data/{actor_id}/", exist_ok = True)
generate_descriptors_meads(actor_folder_path, actor_id)

# # Save dictionary as a pickle file
# with open("descriptor_list.pkl", 'wb') as file:
#     pickle.dump(descriptor_list, file)



