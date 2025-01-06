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

import sys

# Assume `src` is located in the parent directory
sys.path.append(os.path.abspath('../'))


from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

from src.utils.io import load_image_rgb, resize_to_limit, load_video
from src.utils.helper import basename, dct2device, calc_motion_multiplier
from src.utils.crop import prepare_paste_back, paste_back
from src.utils.video import concat_frames
from src.utils.crop import prepare_paste_back, paste_back
from src.utils.camera import get_rotation_matrix
from src.utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream


from src.live_portrait_pipeline import LivePortraitPipeline


def remove_ddp_dumplicate_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new

def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath("./")), fn)

device = "cuda"

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

def map_image_using_descriptor(source, c_d_eyes_lst, c_d_lip_lst, driving_template_dct, modify_kp = True):

    n_frames = 1

    # Part 1
    flag_is_source_video = False
    img_rgb = load_image_rgb(source)

    source_division: int = 2 # make sure the height and width of source image or video can be divided by this number
    source_max_dim: int = 1280 # the max dim of height and width of source image or video, you can change it to a larger number, e.g., 1920

    img_rgb = resize_to_limit(img_rgb, source_max_dim, source_division)
    source_rgb_lst = [img_rgb]

    #Part 5.1
    c_d_eyes_lst = c_d_eyes_lst*n_frames
    c_d_lip_lst = c_d_lip_lst*n_frames

    #Part 5.1
    c_d_eyes_lst = c_d_eyes_lst*n_frames
    c_d_lip_lst = c_d_lip_lst*n_frames

    #Consistent block
    I_p_pstbk_lst = None

    #part 6.1
    I_p_pstbk_lst = []

    #Consistent block
    I_p_lst = []
    R_d_0, x_d_0_info = None, None
    flag_normalize_lip = inference_cfg.flag_normalize_lip  # not overwrite
    flag_source_video_eye_retargeting = inference_cfg.flag_source_video_eye_retargeting  # not overwrite
    lip_delta_before_animation, eye_delta_before_animation = None, None

    #Part 7.2 --> 7.2.1
    crop_info = live_portrait_pipeline.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
    if crop_info is None:
        raise Exception("No face detected in the source image!")

    img_crop_256x256 = crop_info['img_crop_256x256']

    I_s = live_portrait_pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)
    x_s_info = live_portrait_pipeline.live_portrait_wrapper.get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
    f_s = live_portrait_pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
    x_s = live_portrait_pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)

    if modify_kp:
        kp = x_s_info['kp'].cpu()    # (bs, k, 3)
        t, exp = driving_template_dct["motion"][0]['t'], driving_template_dct["motion"][0]['exp']
        scale = driving_template_dct["motion"][0]['scale']

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[1] // 3  # Bx(num_kpx3)
        else:
            num_kp = kp.shape[1]  # Bxnum_kpx3

        rot_mat = driving_template_dct["motion"][0]['R']

        kp = kp if isinstance(kp, torch.Tensor) else torch.tensor(kp, dtype=torch.float32)

        # Convert `rot_mat` and `exp` from numpy to PyTorch tensors if needed
        rot_mat = torch.from_numpy(rot_mat).float().cpu() if not isinstance(rot_mat, torch.Tensor) else rot_mat
        exp = torch.from_numpy(exp).float().cpu() if not isinstance(exp, torch.Tensor) else exp

        # Ensure `scale` is a tensor and on the same device as other tensors
        scale = scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, dtype=torch.float32)

        # Convert translation vector `t` to a tensor if it's not already
        t = t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)

        # Eqn.2: s * (R * x_c,s + exp) + t
        kp_transformed = kp.cpu().view(bs, num_kp, 3) @ rot_mat.cpu() + exp.cpu().view(bs, num_kp, 3)
        kp_transformed *= scale.cpu()[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)
        kp_transformed[:, :, 0:2] += t.cpu()[:, None, 0:2]  # remove z, only apply tx ty

        driving_template_dct["motion"][0]["kp"] = x_s_info['kp']
        driving_template_dct["motion"][0]["scale"] = x_s_info['scale']
        driving_template_dct["motion"][0]["t"] = x_s_info['t']
        driving_template_dct["motion"][0]["R"] = R_s
        driving_template_dct["motion"][0]["x_s"] = kp_transformed

    #Part 7.2.4
    mask_ori_float = prepare_paste_back(inference_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

    #Consistent block
    from rich.progress import track
    for i in range(n_frames):

    #Part 10.2
        x_d_i_info = driving_template_dct['motion'][i]

    #Consistent blcok
        x_d_i_info = dct2device(x_d_i_info, "cuda")
        R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

    # Part 11.1
        R_d_0 = R_d_i
        x_d_0_info = x_d_i_info.copy()

    # Part 12.1.1 (truncated if)
        R_new =(R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s

    # Part 12.2 --> 12.2.2 --> 12.2.2.2
    delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inference_cfg.lip_array).to(dtype=torch.float32, device=device))

    #Part 13.1
    if flag_is_source_video:
        scale_new = x_s_info['scale']
    else:
        scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])

    #Part 14.1
    if flag_is_source_video:
        t_new = x_s_info['t']
    else:
        t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])

    #Consistent block
    t_new[..., 2].fill_(0)  # zero tz
    x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

    #Part 18 --> 18.2
    x_d_i_new = live_portrait_pipeline.live_portrait_wrapper.stitching(x_s, x_d_i_new)

    #consistent block
    x_d_i_new = x_s + (x_d_i_new - x_s) * inference_cfg.driving_multiplier
    out = live_portrait_pipeline.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
    I_p_i = live_portrait_pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
    I_p_lst.append(I_p_i)

    #Part 20 --> 20.2
    I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)

    #Consistent block in 20
    I_p_pstbk_lst.append(I_p_pstbk)

    return I_p_lst[-1][..., ::-1]


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
def get_cluster_data(id, cluster_mapping):
    data = cluster_mapping[id]
    return data["c_d_eyes_lst"], data["c_d_lip_lst"], data["driving_template_dct"]

# cluster_id =0
# c_d_eyes_lst, c_d_lip_lst, driving_template_dct = get_cluster_data(cluster_id ,cluster_mapping)
# processed_image = map_image_using_descriptor("1.jpg", c_d_eyes_lst, c_d_lip_lst, driving_template_dct)


import os
from PIL import Image

# Define function to retrieve data and process images based on a list of cluster IDs
def process_images(image_path, cluster_ids, cluster_mapping, output_dir="actor_6_22/", modify_kp=True):
    """
    Processes a sequence of images using cluster IDs and saves each processed image.

    Parameters:
    - image_path (str): The path of the image to process.
    - cluster_ids (list): A list of cluster IDs to process.
    - output_dir (str): Directory where processed images will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i, cluster_id in enumerate(cluster_ids):
        # Retrieve cluster data for current ID
        c_d_eyes_lst, c_d_lip_lst, driving_template_dct = get_cluster_data(cluster_id, cluster_mapping)

        # Process image with current cluster data
        processed_image = map_image_using_descriptor(image_path, c_d_eyes_lst, c_d_lip_lst, driving_template_dct, modify_kp = modify_kp)

        # Define filename for processed image
        output_filename = os.path.join(output_dir, f"processed_{i}.jpg")

        # Convert the processed image (NumPy array) to a PIL Image
        if isinstance(processed_image, np.ndarray):
            pil_image = Image.fromarray(processed_image[:, :, ::-1])

            # Define filename for processed image
            output_filename = os.path.join(output_dir, f"processed_{i}.jpg")

            # Save the processed image
            pil_image.save(output_filename)
            # print(f"Processed image for cluster ID {cluster_id} saved as {output_filename}")
        else:
            print(f"Error: Processed image for cluster ID {cluster_id} is not a NumPy array.")

# Define function to retrieve data and process images based on a list of cluster IDs
def process_images_with_descriptors(image_path, descriptors, output_dir="actor_6_22/", modify_kp=True):
    """
    Processes a sequence of images using cluster IDs and saves each processed image.

    Parameters:
    - image_path (str): The path of the image to process.
    - cluster_ids (list): A list of cluster IDs to process.
    - output_dir (str): Directory where processed images will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for i, descriptor in enumerate(descriptors):
        # Retrieve cluster data for current ID
        c_d_eyes_lst, c_d_lip_lst, driving_template_dct = descriptor["c_d_eyes_lst"], descriptor["c_d_lip_lst"],  descriptor["driving_template_dct"]
        # Process image with current cluster data
        processed_image = map_image_using_descriptor(image_path, c_d_eyes_lst, c_d_lip_lst, driving_template_dct, modify_kp = modify_kp)

        # Define filename for processed image
        output_filename = os.path.join(output_dir, f"processed_{i}.jpg")

        # Convert the processed image (NumPy array) to a PIL Image
        if isinstance(processed_image, np.ndarray):
            pil_image = Image.fromarray(processed_image[:, :, ::-1])

            # Define filename for processed image
            output_filename = os.path.join(output_dir, f"processed_{i}.jpg")

            # Save the processed image
            pil_image.save(output_filename)
            # print(f"Processed image for cluster ID {cluster_id} saved as {output_filename}")
        else:
            print(f"Error: Processed image for cluster ID {i} is not a NumPy array.")

############################################################# Demo video generation #############################################################

# import pickle
# import numpy as np
# file_path = f'autoencoder_pipeline_descriptors_live_portrait_descriptor_all_with_mead.pkl'
# print(file_path)

# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     descriptors_list = pickle.load(file)

# file_path = f'live_portrait_descriptor_all_with_mead.pkl'
# print(file_path)

# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     descriptors_list_2 = pickle.load(file)

# for i, key in enumerate(descriptors_list):
#     print(key)
#     process_images_with_descriptors(f"source_folder/2.jpg", [descriptors_list[key]], output_dir=f"video_generation_test/temp/{i}")
#     process_images_with_descriptors(f"source_folder/2.jpg", [descriptors_list_2[key]], output_dir=f"video_generation_test/temp/{i+1}")
#     break

# actor = 1
# process_images_with_descriptors(f"source_folder/{actor}.jpg", descriptors_list, output_dir=f"video_generation_test/stiching_clip_test/1_original/")

# # file_path = f'extended_descriptors/final_descriptors.pkl'
# # print(file_path)
# # # Open the file in binary read mode
# # with open(file_path, 'rb') as file:
# #     video_clusters = pickle.load(file)

# file_path = f'raw_lstm_descriptors.pkl'
# print(file_path)
# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     video_descriptors = pickle.load(file)

# # video_clusters = {
# # "01-01-02-01-01-01-07 ": [1465, 7025, 3139, 256, 3139, 5407, 3139, 5407, 3139, 5407, 3139, 256, 3139, 256, 3139, 7025, 6045, 2061, 4455, 2061, 257, 9683, 257, 9683, 257, 9683, 257, 1499, 257, 1499, 7355, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 7355, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 7355, 5000, 7355, 5000, 7355, 5000, 7355, 5000, 3614, 8447, 1403, 8447, 1403, 8447, 1403, 8447, 1403, 8447, 1403, 8447, 2597, 257, 2061, 257, 2061, 257, 2061, 257, 8447, 3614],
# # "01-01-06-01-02-02-18 ": [6833, 4348, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 257, 2061, 257, 2061, 257, 2996, 257, 1403, 1499, 5000, 8350, 5000, 8350, 7744, 8350, 7744, 8350, 7355, 502, 8350, 502, 8350, 502, 8350, 502, 8350, 502, 8080, 5680, 8080, 7744, 8080, 7744, 8080, 7744, 8080, 7744, 8080, 7744, 8080, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 7744, 7355, 1403, 2474, 1403, 2474, 1403, 2474, 4353, 2474, 1403, 2474, 1403, 2474, 1403, 2474, 4353, 2474, 4353, 2474, 1403],
# # "01-01-07-01-02-01-08 ": [6833, 4348, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 4455, 2061, 257, 1403, 257, 1403, 8447, 1403, 8447, 1403, 7355, 1499, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 7355, 5000, 7355, 5000, 7355, 5000, 3614, 7744, 3614, 7744, 5000, 7355, 5000, 7355, 5000, 7355, 5000, 7355, 5000, 7355, 3614, 7744, 3614, 8447, 3614, 8447, 3614, 8447, 2597, 8447, 2597, 8447, 2597, 8447, 2597, 8447, 2597, 8447, 2597, 8447, 2597, 8447, 2597, 257, 2061, 257, 2061, 257, 2061, 257, 2061, 257, 2061, 257, 2061, 257, 2996, 4285],
# # "01-01-07-02-01-02-21 ": [6833, 4348, 2061, 4455, 2061, 4455, 2061, 4455, 6045, 9637, 6045, 9637, 6045, 9637, 6045, 1465, 7964, 1465, 7964, 2061, 257, 1403, 7355, 1499, 7355, 5000, 7355, 5000, 8350, 8186, 8350, 8186, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 9390, 1146, 5371, 1146, 5371, 1146, 9070, 1146, 9070, 1146, 5680, 9070, 7419, 9070, 7419, 9070, 5680, 9070, 5680, 7355, 502, 7355, 5000, 8350, 694, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 8350, 5000, 3614, 7744, 694, 7744, 5000, 3614, 7744, 3614, 7744, 3614, 7744, 3614, 8447, 2597, 8447, 4285, 2996, 4285, 2996, 4285, 2996, 4285, 257, 8447, 2597, 8447]
# # }

# for actor_video in video_descriptors:
#     actor = str(int(actor_video.split("-")[-1]))
#     process_images_with_descriptors(f"source_folder/{actor}.jpg", video_descriptors[actor_video], output_dir=f"video_generation_test/temp/second_batch/{actor_video}/")

############################################################### Interpolated video Gen #####################################################################
import pickle
folder = "video_generation_test/video_gen_descriptor_8"
all_pkls = os.listdir(folder)
for pkl in all_pkls:
    path = os.path.join(folder, pkl)
    # Open the file in binary read mode
    with open(path, 'rb') as file:
        descriptor_list = pickle.load(file)

    actor = str(int(pkl.split(".")[0].split("-")[-1]))
    process_images_with_descriptors(f"source_folder/{actor}.jpg", descriptor_list, output_dir=f"video_generation_test/temp/fourth_batch/{pkl.split('.')[0]}/")

############################################################# Cluster analysis #############################################################

# import pickle
# import numpy as np
# from natsort import natsorted

# # sample he karta ha and drops it

# base = "cluster_analysis/nov/18"
# base_1 = "nov/18"
# pca = False
# encoded = False

# if pca:
#     norm = "unpca"
# elif encoded:
#     norm = "unencoded"
# else:
#     norm = "unnormalized"

# for folder in natsorted(os.listdir(base))[-1:]:

#     print(folder)
#     # Path to your pickle file
#     file_path = f'{base}/{folder}/averaged_descriptors_raw.pkl'
#     print(file_path)

#     # Open the file in binary read mode
#     with open(file_path, 'rb') as file:
#         cluster_mapping = pickle.load(file)

#     # Path to your pickle file
#     file_path = f'{base}/{folder}/cluster_to_{norm}_original_dict.pkl'
#     print(file_path)

#     # Open the file in binary read mode
#     with open(file_path, 'rb') as file:
#         cluster_org_groups = pickle.load(file)

#     import random
#     # Step 1: Filter clusters with more than 30 points
#     filtered_clusters = {cluster_id: points for cluster_id, points in cluster_org_groups.items() if len(points) >= 10 and  len(points) <= 50}

#     for i, cluster in enumerate(filtered_clusters):
#         process_images_with_descriptors("source_folder/2.jpg", filtered_clusters[cluster], output_dir=f"video_generation_test/{base_1}/{folder}/clusters/cluster_{i}")
#         process_images("source_folder/2.jpg", [cluster], cluster_mapping, output_dir=f"video_generation_test/{base_1}/{folder}/cluster_averages/cluster_{i}")
#         if i == 10:
#             break

# import pickle
# import numpy as np

# for folder in natsorted(os.listdir(base))[-1:]:

#     file_path = f'{base}/{folder}/averaged_descriptors_raw.pkl'
#     print(file_path)

#     # Open the file in binary read mode
#     with open(file_path, 'rb') as file:
#         cluster_mapping = pickle.load(file)

#     # Path to your pickle file
#     file_path = f'{base}/{folder}/frame_to_cluster_mapping.pkl'
#     print(file_path)


#     # Open the file in binary read mode
#     with open(file_path, 'rb') as file:
#         frame_to_clus = pickle.load(file)

#     video = {}
#     for i, vid in enumerate(frame_to_clus):

#         if i in [0, 60, 120, 180]:
#             # 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900

#             video[vid] = [val[1] for val in frame_to_clus[vid]]


#             process_images(f"source_folder/{int(vid.split('/')[-1].split('-')[-1])}.jpg", video[vid], cluster_mapping, output_dir=f"video_generation_test/{base_1}/{folder}/video_results/{vid}/")

#             with open(f"video_generation_test/{base_1}/{folder}/video_results/{vid}/cluster_ids.txt", "w") as file:
#                 for number in video[vid]:
#                     file.write(f"{number}\n")

#         if i == 360:
#             break

########################################################################## MEADs script ###########################################################
# import pickle
# import numpy as np

# for folder in natsorted(os.listdir(base))[-1:]:

#     file_path = f'{base}/{folder}/averaged_descriptors_raw.pkl'
#     print(file_path)

#     # Open the file in binary read mode
#     with open(file_path, 'rb') as file:
#         cluster_mapping = pickle.load(file)

#     # Path to your pickle file
#     file_path = f'{base}/{folder}/frame_to_cluster_mapping.pkl'
#     print(file_path)


#     # Open the file in binary read mode
#     with open(file_path, 'rb') as file:
#         frame_to_clus = pickle.load(file)

#     video = {}
#     vids = ["02-01-03-01-02-01-13", "01-01-06-01-02-01-13", "01-01-03-01-02-01-13", "01-01-03-01-02-01-13", "02-01-03-01-02-01-13", "01-01-03-01-02-02-11"]
#     for i, vid in enumerate(vids):


#         video[vid] = [val[1] for val in frame_to_clus[vid]]


#         process_images(f"source_folder/{int(vid.split('/')[-1].split('-')[-1])}.jpg", video[vid], cluster_mapping, output_dir=f"video_generation_test/{base_1}/{folder}/video_results/{vid}/")

#         with open(f"video_generation_test/{base_1}/{folder}/video_results/{vid}/cluster_ids.txt", "w") as file:
#             for number in video[vid]:
#                 file.write(f"{number}\n")



############################################################ Cluster analysis #############################################################
# frame_to_cluster_mapping_path = "cluster_analysis/nov/20/21_52_51_10000_0.8_0.05_0.05_0.05_0.05"
# file_path = f'{frame_to_cluster_mapping_path}/averaged_descriptors_raw.pkl'
# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     cluster_mapping = pickle.load(file)

# # Open the file in binary read mode
# with open(f"{frame_to_cluster_mapping_path}/all_nonzero_transitions_dict.pkl", 'rb') as file:
#     all_non_zero_transitions = pickle.load(file)

# with open(f"{frame_to_cluster_mapping_path}/interpolated_descriptors.pkl", 'rb') as file:
#     interpolated_clusters_dict = pickle.load(file)

# for cluster_id in interpolated_clusters_dict:
#     print("Transition from", cluster_id)
#     cluster_trans = interpolated_clusters_dict[cluster_id]
#     for trans in cluster_trans:
#         print("to", trans)
#         descriptors = []
#         for descriptor in cluster_trans[trans]:
#             descriptors.append(descriptor)
#         process_images_with_descriptors("source_folder/2.jpg", descriptors, output_dir=f"video_generation_test/temp/intermediate_clustering_out/{cluster_id}_{trans}/")



############################################################################ Last minute intepolated video generation #################################################
# import torch
# import numpy as np
# import pickle


# # Extract Euler angles from rotation matrix
# def get_euler_angles_from_rotation_matrix(rot_matrix):
#     rot_matrix = rot_matrix.permute(0, 2, 1)
#     yaw = torch.arcsin(-rot_matrix[:, 2, 0])
#     near_pi_over_2 = torch.isclose(torch.cos(yaw), torch.tensor(0.0), atol=1e-6)

#     pitch = torch.where(
#         ~near_pi_over_2,
#         torch.atan2(rot_matrix[:, 2, 1], rot_matrix[:, 2, 2]),
#         torch.atan2(rot_matrix[:, 1, 2], rot_matrix[:, 1, 1])
#     )

#     roll = torch.where(
#         ~near_pi_over_2,
#         torch.atan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0]),
#         torch.zeros_like(yaw)
#     )

#     pitch = pitch * 180 / torch.pi
#     yaw = yaw * 180 / torch.pi
#     roll = roll * 180 / torch.pi

#     return pitch, yaw, roll


# def get_rotation_matrix(pitch_, yaw_, roll_):
#     """ the input is in degree
#     """
#     # transform to radian
#     pitch = pitch_ / 180 * torch.pi
#     yaw = yaw_ / 180 * torch.pi
#     roll = roll_ / 180 * torch.pi

#     device = pitch.device

#     if pitch.ndim == 1:
#         pitch = pitch.unsqueeze(1)
#     if yaw.ndim == 1:
#         yaw = yaw.unsqueeze(1)
#     if roll.ndim == 1:
#         roll = roll.unsqueeze(1)

#     # calculate the euler matrix
#     bs = pitch.shape[0]
#     ones = torch.ones([bs, 1]).to(device)
#     zeros = torch.zeros([bs, 1]).to(device)
#     x, y, z = pitch, yaw, roll

#     rot_x = torch.cat([
#         ones, zeros, zeros,
#         zeros, torch.cos(x), -torch.sin(x),
#         zeros, torch.sin(x), torch.cos(x)
#     ], dim=1).reshape([bs, 3, 3])

#     rot_y = torch.cat([
#         torch.cos(y), zeros, torch.sin(y),
#         zeros, ones, zeros,
#         -torch.sin(y), zeros, torch.cos(y)
#     ], dim=1).reshape([bs, 3, 3])

#     rot_z = torch.cat([
#         torch.cos(z), -torch.sin(z), zeros,
#         torch.sin(z), torch.cos(z), zeros,
#         zeros, zeros, ones
#     ], dim=1).reshape([bs, 3, 3])

#     rot = rot_z @ rot_y @ rot_x
#     return rot.permute(0, 2, 1)  # transpose

# # Convert yaw, pitch, and roll to a rotation matrix
# def euler_angles_to_rotation_matrix(pitch, yaw, roll):
#     PI = np.pi
#     # Convert to torch tensors and add batch dimension
#     pitch_ = torch.tensor([pitch], dtype=torch.float32)
#     yaw_ = torch.tensor([yaw], dtype=torch.float32)
#     roll_ = torch.tensor([roll], dtype=torch.float32)

#     # Get rotation matrix using provided function
#     R = get_rotation_matrix(pitch_, yaw_, roll_)

#     # Convert to numpy and reshape to (1,3,3)
#     R = R.cpu().numpy().astype(np.float32)

#     return R


# # Function to extract the full 208-dimensional vector from frame data
# def extract_full_vector(frame_data):
#     c_d_eyes = frame_data['c_d_eyes_lst'][0].reshape(-1)  # 2 values
#     c_d_lip = frame_data['c_d_lip_lst'][0].reshape(-1)    # 1 value

#     driving_template = frame_data['driving_template_dct']
#     c_eyes = driving_template['motion'][0]['c_eyes_lst'][0].reshape(-1)  # 2 values
#     c_lip = driving_template['motion'][0]['c_lip_lst'][0].reshape(-1)    # 1 value

#     motion = driving_template['motion'][0]
#     scale = np.array(motion['scale']).reshape(-1)         # 1 value
#     t = motion['t'].reshape(-1)                           # 3 values
#     R = motion['R'].reshape(1, 3, 3)                      # 9 values in matrix form
#     exp = motion['exp'].reshape(-1)                       # 63 values
#     x_s = motion['x_s'].reshape(-1)                       # 63 values
#     kp = motion['kp'].reshape(-1)                         # 63 values actual value now becomes 202

#     # Convert R to pitch, yaw, and roll using the function
#     pitch, yaw, roll = get_euler_angles_from_rotation_matrix(torch.tensor(R))
#     euler_angles = np.array([pitch.item(), yaw.item(), roll.item()])

#     if not np.array_equal(c_d_eyes, c_eyes):
#         print("Eyes arrays not equal")
#     if not np.array_equal(c_d_lip, c_lip):
#         print("Lip arrays not equal")

#     # print(c_d_eyes.shape, c_d_lip.shape, c_eyes.shape, c_lip.shape, scale.shape, t.shape, euler_angles.shape, exp.shape, x_s.shape, kp.shape)
#     # print("(2,) (1,) (2,) (1,) (1,) (3,) (3,) (63,) (63,) (63,)")
#     # 202 values

#     # Combine the components into a full vector excluding R
#     vector = np.concatenate([c_d_eyes, c_d_lip, c_eyes, c_lip, scale, t, euler_angles, exp, x_s, kp])

#     return vector

# def unflatten_vector(avg_vector):
#     # Convert flattened vector back to original format
#     c_d_eyes = np.array(avg_vector[0:2], dtype=np.float32).reshape(1, 2)
#     c_d_lip = np.array(avg_vector[2:3], dtype=np.float32).reshape(1, 1)
#     c_eyes = np.array(avg_vector[3:5], dtype=np.float32).reshape(1, 2)
#     c_lip = np.array(avg_vector[5:6], dtype=np.float32).reshape(1, 1)
#     scale = np.array(avg_vector[6:7], dtype=np.float32).reshape(1, 1)
#     t = np.array(avg_vector[7:10], dtype=np.float32).reshape(1, 3)

#     # Convert to rotation matrix and update
#     R = euler_angles_to_rotation_matrix(avg_vector[10], avg_vector[11], avg_vector[12])

#     # Expression, shape and keypoint parameters
#     exp = np.array(avg_vector[13:76], dtype=np.float32).reshape(1, 21, 3)
#     x_s = np.array(avg_vector[76:139], dtype=np.float32).reshape(1, 21, 3)
#     kp = np.array(avg_vector[139:202], dtype=np.float32).reshape(1, 21, 3)

#     # Return dictionary in original format
#     return {
#         'c_d_eyes_lst': c_d_eyes,
#         'c_d_lip_lst': c_d_lip,
#         'driving_template_dct': {
#             'motion': [{
#                 'scale': scale,
#                 'R': R,
#                 't': t,
#                 'c_eyes_lst': c_eyes,
#                 'c_lip_lst': c_lip,
#                 'exp': exp,
#                 'x_s': x_s,
#                 'kp': kp
#             }],

#         }
#     }


# def interpolate_clusters(cluster_id_1, cluster_id_2, average_descriptor_dict, num_intermediate_clusters=9):
#     # Extract descriptors for the given cluster IDs
#     descriptor_1 = average_descriptor_dict[cluster_id_1]
#     descriptor_2 = average_descriptor_dict[cluster_id_2]

#     # Flatten the descriptors using the extract_vectors function
#     flattened_1 = extract_full_vector(descriptor_1)
#     flattened_2 = extract_full_vector(descriptor_2)

#     # Interpolate between the two flattened descriptors
#     intermediate_vectors = []
#     for t in range(1, num_intermediate_clusters + 1):  # Generate specified number of intermediate vectors
#         alpha = t / (num_intermediate_clusters + 1)  # Adjust alpha to control the number of intermediate clusters
#         intermediate_vector = (1 - alpha) * flattened_1 + alpha * flattened_2
#         intermediate_vectors.append(unflatten_vector(intermediate_vector))

#     return descriptor_1, descriptor_2, intermediate_vectors

# def run_length_set(numbers):
#     """
#     Creates a run-length set from a list of numbers by selecting one representative
#     from each contiguous group of numbers.

#     Parameters:
#     - numbers (list): The list of numbers to process.

#     Returns:
#     - list: A list containing one representative per contiguous group.
#     """
#     if not numbers:
#         return []

#     result = [numbers[0]]  # Start with the first number
#     for i in range(1, len(numbers)):
#         if numbers[i] != numbers[i - 1]:
#             result.append(numbers[i])

#     return result

# def generate(ids, average_descriptor_dict, num_intermediate_clusters=3):
#     """
#     Generates a list of descriptors by applying the interpolate_clusters function
#     to consecutive pairs of IDs in the input list.

#     Parameters:
#     - ids (list): A list of cluster IDs to process.
#     - average_descriptor_dict (dict): A dictionary containing average descriptors for each cluster ID.
#     - num_intermediate_clusters (int): Number of intermediate clusters to generate.

#     Returns:
#     - all_des (list): A list of descriptor sequences for each consecutive pair.
#     """
#     # Initialize the list to store all descriptors
#     all_des = []

#     # Iterate through consecutive pairs of IDs
#     for i in range(len(ids) - 1):
#         cluster_id_1 = ids[i]
#         cluster_id_2 = ids[i + 1]

#         # Interpolate intermediate descriptors
#         descriptor_1, descriptor_2, intermediate_vectors = interpolate_clusters(
#             cluster_id_1, cluster_id_2, average_descriptor_dict, num_intermediate_clusters
#         )

#         # Combine descriptors into a list and append to the result
#         output = [descriptor_1] + intermediate_vectors + [descriptor_2]
#         all_des = all_des + output

#     return all_des

# frame_to_cluster_mapping_path = "cluster_analysis/nov/18/23_18_56_10000_0.6_0.1_0.1_0.1_0.1"
# file_path = f'{frame_to_cluster_mapping_path}/averaged_descriptors_raw.pkl'
# # Open the file in binary read mode
# with open(file_path, 'rb') as file:
#     cluster_mapping = pickle.load(file)

# # Open the file in binary read mode
# with open(f"{frame_to_cluster_mapping_path}/all_nonzero_transitions_dict.pkl", 'rb') as file:
#     all_non_zero_transitions = pickle.load(file)

# with open(f"{frame_to_cluster_mapping_path}/interpolated_descriptors.pkl", 'rb') as file:
#     interpolated_clusters_dict = pickle.load(file)


# for folder in natsorted(os.listdir(base))[-1:]:
#     # Path to your pickle file
#     file_path = f'{base}/{folder}/frame_to_cluster_mapping.pkl'
#     print(file_path)

#     # Open the file in binary read mode
#     with open(file_path, 'rb') as file:
#         frame_to_clus = pickle.load(file)

#     video = {}
#     for i, vid in enumerate(frame_to_clus):
#         if i in [120, 180, 240, 300, 360, 420]: #480, 540, 600, 660, 720, 780, 840, 900
#             video[vid] = [val[1] for val in frame_to_clus[vid]]
#             descriptors = generate(run_length_set(video[vid]), cluster_mapping, num_intermediate_clusters=5)

#             process_images_with_descriptors(f"source_folder/{int(vid.split('/')[-1].split('-')[-1])}.jpg", descriptors, output_dir=f"video_generation_test/{base_1}/{folder}/video_results/{vid}/")

#             with open(f"video_generation_test/{base_1}/{folder}/video_results/{vid}/cluster_ids.txt", "w") as file:
#                 for number in video[vid]:
#                     file.write(f"{number}\n")
