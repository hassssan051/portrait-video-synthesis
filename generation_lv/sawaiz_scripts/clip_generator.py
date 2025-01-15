
from __future__ import annotations

from collections import OrderedDict
import yaml
import os
import os.path as osp
import cv2
import numpy as np
import tyro
import subprocess
import torch

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


from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from src.utils.crop import prepare_paste_back, paste_back
from src.utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from src.utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from src.utils.filter import smooth




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

def get_driving_dict(descriptors_list, fps=25):
    template_dct = {
        'n_frames': 0,
        'output_fps': fps,
        'motion': [],
        'c_eyes_lst': [],
        'c_lip_lst': [],
    }

    for i, des in enumerate(descriptors_list):
        motion = des["driving_template_dct"]["motion"][0]
        c_eyes_lst = des["c_d_eyes_lst"][0]
        c_lip_lst = des["c_d_eyes_lst"][0]
        template_dct["motion"].append(motion)
        template_dct["c_eyes_lst"].append(c_eyes_lst)
        template_dct["c_lip_lst"].append(c_lip_lst)
        template_dct["n_frames"] += 1

    final_dict = {
        "c_d_eyes_lst": template_dct["c_eyes_lst"],
        "c_d_lip_lst": template_dct["c_lip_lst"],
        "driving_template_dct": template_dct,
    }
    return final_dict

def map_image_using_descriptor_lst(source, descriptor_lst, output_dir, output_vid_name, inference_cfg, crop_cfg, fps = 30):

    audio_priority = 'driving'

    flag_source_has_audio = False
    flag_driving_has_audio = False
    output_fps = fps

     # for convenience
    inf_cfg = inference_cfg
    device = "cuda"
    crop_cfg = crop_cfg
    # Part 1
    flag_is_source_video = False
    source_fps = None
    flag_is_source_video = False
    flag_is_driving_video = True

    img_rgb = load_image_rgb(source)
    img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
    source_rgb_lst = [img_rgb]

    ######## process driving info ########
    # flag_load_from_template = is_template(driving)
    flag_load_from_template = False
    driving_rgb_crop_256x256_lst = None
    wfp_template = None

    final_dict = get_driving_dict(descriptor_lst, fps=30)
    driving_n_frames = len(final_dict["driving_template_dct"]["motion"])

# Part 3.3
    print("Part 3.3")
    n_frames = driving_n_frames

    driving_template_dct, c_d_eyes_lst, c_d_lip_lst = final_dict["driving_template_dct"], final_dict["c_d_eyes_lst"], final_dict["c_d_lip_lst"]

    I_p_pstbk_lst = None
    if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
        print("Part 6.1")
        I_p_pstbk_lst = []

    I_p_lst = []
    R_d_0, x_d_0_info = None, None
    flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
    flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
    lip_delta_before_animation, eye_delta_before_animation = None, None

    if inf_cfg.flag_do_crop:
        print("7.2.1")
        crop_info = live_portrait_pipeline.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
        if crop_info is None:
            raise Exception("No face detected in the source image!")
        source_lmk = crop_info['lmk_crop']
        img_crop_256x256 = crop_info['img_crop_256x256']
    else:
        print("7.2.2")
        source_lmk = live_portrait_pipeline.cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])
        img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256

    I_s = live_portrait_pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)
    x_s_info = live_portrait_pipeline.live_portrait_wrapper.get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
    f_s = live_portrait_pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
    x_s = live_portrait_pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)

    # let lip-open scalar to be 0 at first
    if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
        print("7.2.3")
        c_d_lip_before_animation = [0.]
        combined_lip_ratio_tensor_before_animation = live_portrait_pipeline.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)
        if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:
            lip_delta_before_animation = live_portrait_pipeline.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)

    if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
        print("7.2.4")
        mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

    from rich.progress import track

    for i in track(range(n_frames), description='🚀Animating...', total=n_frames):
                ######## process source info ########
        if flag_is_source_video:
            print("7.1")

            source_rgb_lst = source_rgb_lst[:n_frames]
            if inf_cfg.flag_do_crop:
                print("7.1.1")
                ret_s = live_portrait_pipeline.cropper.crop_source_video(source_rgb_lst, crop_cfg)
                if len(ret_s["frame_crop_lst"]) is not n_frames:
                    n_frames = min(n_frames, len(ret_s["frame_crop_lst"]))
                img_crop_256x256_lst, source_lmk_crop_lst, source_M_c2o_lst = ret_s['frame_crop_lst'], ret_s['lmk_crop_lst'], ret_s['M_c2o_lst']
            else:
                print("7.1.2")
                source_lmk_crop_lst = live_portrait_pipeline.cropper.calc_lmks_from_cropped_video(source_rgb_lst)
                img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in source_rgb_lst]  # force to resize to 256x256

            c_s_eyes_lst, c_s_lip_lst = live_portrait_pipeline.live_portrait_wrapper.calc_ratio(source_lmk_crop_lst)
            # save the motion template
            I_s_lst = live_portrait_pipeline.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
            source_template_dct = live_portrait_pipeline.make_motion_template(I_s_lst, c_s_eyes_lst, c_s_lip_lst, output_fps=source_fps)

            key_r = 'R' if 'R' in driving_template_dct['motion'][0].keys() else 'R_d'  # compatible with previous keys
            if inf_cfg.flag_relative_motion:
                print("7.1.3")
                if flag_is_driving_video:
                    print("7.1.3.1")
                    x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + driving_template_dct['motion'][i]['exp'] - driving_template_dct['motion'][0]['exp'] for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
                else:
                    print("7.1.3.2")
                    x_d_exp_lst = [source_template_dct['motion'][i]['exp'] + (driving_template_dct['motion'][0]['exp'] - inf_cfg.lip_array) for i in range(n_frames)]
                    x_d_exp_lst_smooth = [torch.tensor(x_d_exp[0], dtype=torch.float32, device=device) for x_d_exp in x_d_exp_lst]
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    print("7.1.4")
                    if flag_is_driving_video:
                        print("7.1.4.1")
                        x_d_r_lst = [(np.dot(driving_template_dct['motion'][i][key_r], driving_template_dct['motion'][0][key_r].transpose(0, 2, 1))) @ source_template_dct['motion'][i]['R'] for i in range(n_frames)]
                        x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, device, inf_cfg.driving_smooth_observation_variance)
                    else:
                        print("7.1.4.2")
                        x_d_r_lst = [source_template_dct['motion'][i]['R'] for i in range(n_frames)]
                        x_d_r_lst_smooth = [torch.tensor(x_d_r[0], dtype=torch.float32, device=device) for x_d_r in x_d_r_lst]
            else:
                print("7.1.5")
                if flag_is_driving_video:
                    print("7.1.5.1")
                    x_d_exp_lst = [driving_template_dct['motion'][i]['exp'] for i in range(n_frames)]
                    x_d_exp_lst_smooth = smooth(x_d_exp_lst, source_template_dct['motion'][0]['exp'].shape, device, inf_cfg.driving_smooth_observation_variance)
                else:
                    print("7.1.5.2")
                    x_d_exp_lst = [driving_template_dct['motion'][0]['exp']]
                    x_d_exp_lst_smooth = [torch.tensor(x_d_exp[0], dtype=torch.float32, device=device) for x_d_exp in x_d_exp_lst]*n_frames
                if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                    print("7.1.6")
                    if flag_is_driving_video:
                        print("7.1.6.1")
                        x_d_r_lst = [driving_template_dct['motion'][i][key_r] for i in range(n_frames)]
                        x_d_r_lst_smooth = smooth(x_d_r_lst, source_template_dct['motion'][0]['R'].shape, device, inf_cfg.driving_smooth_observation_variance)
                    else:
                        print("7.1.6")
                        x_d_r_lst = [driving_template_dct['motion'][0][key_r]]
                        x_d_r_lst_smooth = [torch.tensor(x_d_r[0], dtype=torch.float32, device=device) for x_d_r in x_d_r_lst]*n_frames
            if flag_is_source_video and not flag_is_driving_video:
                print("10.1")
                x_d_i_info = driving_template_dct['motion'][0]
            else:
                print("10.2")
                x_d_i_info = driving_template_dct['motion'][i]

        x_d_i_info = driving_template_dct['motion'][i]
        #######################################################################################

        x_d_i_info = dct2device(x_d_i_info, device)
        if 'R' in x_d_i_info.keys():
            print("10.3")
            R_d_i = x_d_i_info['R']
        else:
            print("10.4")
            R_d_i = x_d_i_info['R_d']  # compatible with previous keys

        if i == 0:  # cache the first frame
            print("11.1")
            R_d_0 = R_d_i
            x_d_0_info = x_d_i_info.copy()

        delta_new = x_s_info['exp'].clone()

        #######################################################################################
        if inf_cfg.flag_relative_motion:
            print("12.1")

            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                print("12.1.1")
                R_new = x_d_r_lst_smooth[i] if flag_is_source_video else (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
            else:
                #######################################################################################
                print("12.1.2")
                R_new = R_s

            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                print("12.2")
                if flag_is_source_video:
                    print("12.2.1")
                    for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                        delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :]
                    delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1]
                    delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2]
                    delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2]
                    delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:]
                else:
                    print("12.2.2")
                    if flag_is_driving_video:
                        print("12.2.2.1")
                        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
                    else:
                        print("12.2.2.2")
                        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device))
            elif inf_cfg.animation_region == "lip":
                #######################################################################################
                print("12.3")
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    if flag_is_source_video:
                        print("12.3.1")
                        delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :]
                    elif flag_is_driving_video:
                        #######################################################################################
                        print("12.3.2")
                        delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, lip_idx, :]
                    else:
                        print("12.3.3")
                        delta_new[:, lip_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)))[:, lip_idx, :]
            elif inf_cfg.animation_region == "eyes":
                print("12.4")
                for eyes_idx in [11, 13, 15, 16, 18]:
                    if flag_is_source_video:
                        print("12.4.1")
                        delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :]
                    elif flag_is_driving_video:
                        print("12.4.2")
                        delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp']))[:, eyes_idx, :]
                    else:
                        print("12.4.3")
                        delta_new[:, eyes_idx, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, eyes_idx, :]

            if inf_cfg.animation_region == "all":
                print("13.1")
                scale_new = x_s_info['scale'] if flag_is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
            else:
                #######################################################################################
                print("13.2")
                scale_new = x_s_info['scale']
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                print("14.1")
                t_new = x_s_info['t'] if flag_is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])
            else:
                #######################################################################################
                print("14.2")
                t_new = x_s_info['t']
        else:
            print("15.1")
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                print("15.1.1")
                R_new = x_d_r_lst_smooth[i] if flag_is_source_video else R_d_i
            else:
                print("15.1.2")
                R_new = R_s
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "exp":
                for idx in [1,2,6,11,12,13,14,15,16,17,18,19,20]:
                    delta_new[:, idx, :] = x_d_exp_lst_smooth[i][idx, :] if flag_is_source_video else x_d_i_info['exp'][:, idx, :]
                delta_new[:, 3:5, 1] = x_d_exp_lst_smooth[i][3:5, 1] if flag_is_source_video else x_d_i_info['exp'][:, 3:5, 1]
                delta_new[:, 5, 2] = x_d_exp_lst_smooth[i][5, 2] if flag_is_source_video else x_d_i_info['exp'][:, 5, 2]
                delta_new[:, 8, 2] = x_d_exp_lst_smooth[i][8, 2] if flag_is_source_video else x_d_i_info['exp'][:, 8, 2]
                delta_new[:, 9, 1:] = x_d_exp_lst_smooth[i][9, 1:] if flag_is_source_video else x_d_i_info['exp'][:, 9, 1:]
            elif inf_cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = x_d_exp_lst_smooth[i][lip_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, lip_idx, :]
            elif inf_cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = x_d_exp_lst_smooth[i][eyes_idx, :] if flag_is_source_video else x_d_i_info['exp'][:, eyes_idx, :]
            scale_new = x_s_info['scale']
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":
                t_new = x_d_i_info['t']
            else:
                t_new = x_s_info['t']

        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

        if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly" and not flag_is_source_video and flag_is_driving_video:
            #######################################################################################
            print("16")
            if i == 0:
                print("16.1")
                x_d_0_new = x_d_i_new
                motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
                # motion_multiplier *= inf_cfg.driving_multiplier
            x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
            x_d_i_new = x_d_diff + x_s

        # Algorithm 1:
        if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
            print("17")
            # without stitching or retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                print("17.1")
                x_d_i_new += lip_delta_before_animation
            if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                print("17.2")
                x_d_i_new += eye_delta_before_animation
            else:
                pass
        elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:
            #######################################################################################
            print("18")
            # with stitching and without retargeting
            if flag_normalize_lip and lip_delta_before_animation is not None:
                print("18.1")
                x_d_i_new = live_portrait_pipeline.live_portrait_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
            else:
                #######################################################################################
                print("18.2")
                x_d_i_new = live_portrait_pipeline.live_portrait_wrapper.stitching(x_s, x_d_i_new)
            if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                print("18.3")
                x_d_i_new += eye_delta_before_animation
        else:
            print("19")
            eyes_delta, lip_delta = None, None
            if inf_cfg.flag_eye_retargeting and source_lmk is not None:
                print("19.1")
                c_d_eyes_i = c_d_eyes_lst[i]
                combined_eye_ratio_tensor = live_portrait_pipeline.live_portrait_wrapper.calc_combined_eye_ratio(c_d_eyes_i, source_lmk)
                # ∆_eyes,i = R_eyes(x_s; c_s,eyes, c_d,eyes,i)
                eyes_delta = live_portrait_pipeline.live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)
            if inf_cfg.flag_lip_retargeting and source_lmk is not None:
                print("19.2")
                c_d_lip_i = c_d_lip_lst[i]
                combined_lip_ratio_tensor = live_portrait_pipeline.live_portrait_wrapper.calc_combined_lip_ratio(c_d_lip_i, source_lmk)
                # ∆_lip,i = R_lip(x_s; c_s,lip, c_d,lip,i)
                lip_delta = live_portrait_pipeline.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

            if inf_cfg.flag_relative_motion:  # use x_s
                print("19.3")
                x_d_i_new = x_s + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)
            else:  # use x_d,i
                print("19.4")
                x_d_i_new = x_d_i_new + \
                    (eyes_delta if eyes_delta is not None else 0) + \
                    (lip_delta if lip_delta is not None else 0)

            if inf_cfg.flag_stitching:
                print("19.5")
                x_d_i_new = live_portrait_pipeline.live_portrait_wrapper.stitching(x_s, x_d_i_new)

        x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
        out = live_portrait_pipeline.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p_i = live_portrait_pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
        I_p_lst.append(I_p_i)

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:
                #######################################################################################
                print("20")
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU
                if flag_is_source_video:
                    print("20.1")
                    I_p_pstbk = paste_back(I_p_i, source_M_c2o_lst[i], source_rgb_lst[i], mask_ori_float)
                else:
                    print("20.2")
                    #######################################################################################
                    I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)


    mkdir(output_dir)
    wfp_concat = None
    ######### build the final concatenation result #########
    # driving frame | source frame | generation
    if flag_is_source_video and flag_is_driving_video:
        print("21.1")
        frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, img_crop_256x256_lst, I_p_lst)
    elif flag_is_source_video and not flag_is_driving_video:
        print("21.2")
        if flag_load_from_template:
            print("21.2.1")
            frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, img_crop_256x256_lst, I_p_lst)
        else:
            print("21.2.2")
            frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst*n_frames, img_crop_256x256_lst, I_p_lst)
    else:
        print("21.3")
        frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, [img_crop_256x256], I_p_lst)

    if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
        print("22")
        if flag_driving_has_audio:
            flag_source_has_audio = flag_is_source_video and has_audio_stream(source)
            flag_driving_has_audio = (not flag_load_from_template) and has_audio_stream(output_vid_name)

        # wfp_concat = osp.join(output_dir, f'{output_vid_name}_concat.mp4')

        # NOTE: update output fps
        # output_fps = source_fps if flag_is_source_video else output_fps
        # images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)

        if flag_source_has_audio or flag_driving_has_audio:
            print("22.1")
            # final result with concatenation
            wfp_concat_with_audio = osp.join(output_dir, f'{output_vid_name}_concat_with_audio.mp4')
            audio_from_which_video = output_vid_name if ((flag_driving_has_audio and audio_priority == 'driving') or (not flag_source_has_audio)) else source
            add_audio_to_video(wfp_concat, audio_from_which_video, wfp_concat_with_audio)
            os.replace(wfp_concat_with_audio, wfp_concat)

        # save the animated result
        wfp = osp.join(output_dir, f'{output_vid_name}.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            print("22.2.1")
            images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)
        else:
            print("22.2.2")
            images2video(I_p_lst, wfp=wfp, fps=output_fps)

        ######### build the final result #########
        if flag_source_has_audio or flag_driving_has_audio:
            print("22.2.")
            wfp_with_audio = osp.join(output_dir, f'{output_vid_name}_with_audio.mp4')
            audio_from_which_video = output_vid_name if ((flag_driving_has_audio and audio_priority == 'driving') or (not flag_source_has_audio)) else source
            add_audio_to_video(wfp, audio_from_which_video, wfp_with_audio)
            os.replace(wfp_with_audio, wfp)

    else:
        print("23")
        # wfp_concat = osp.join(output_dir, f'{output_vid_name}_concat.jpg')
        # cv2.imwrite(wfp_concat, frames_concatenated[0][..., ::-1])
        wfp = osp.join(output_dir, f'{output_vid_name}.jpg')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            print("23.1")
            cv2.imwrite(wfp, I_p_pstbk_lst[0][..., ::-1])
        else:
            print("23.2")
            cv2.imwrite(wfp, frames_concatenated[0][..., ::-1])

    return True


# map_image_using_descriptor_lst("source", [], "output_dir", "output_vid_name", inference_cfg, crop_cfg, fps = 30)

import pickle
output_dir = "animations/"
output_vid_name = "test_2"
path = "video_dict.pkl"
# Open the file in binary read mode
with open(path, 'rb') as file:
    descriptor_list = pickle.load(file)

for i, video_name in enumerate(descriptor_list):
    if i == 1:
        continue
    descriptor_list_for_video = []
    for frame_number, descriptor in descriptor_list[video_name]:
        descriptor_list_for_video.append(descriptor)
    map_image_using_descriptor_lst("source_folder/1.jpg", descriptor_list_for_video, output_dir=output_dir, output_vid_name= output_vid_name, inference_cfg = inference_cfg, crop_cfg=crop_cfg, fps = 30)
