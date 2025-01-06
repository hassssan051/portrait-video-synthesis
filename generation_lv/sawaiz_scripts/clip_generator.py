
from __future__ import annotations

from collections import OrderedDict
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
     # for convenience
    inf_cfg = inference_cfg
    device = "cuda"
    crop_cfg = crop_cfg
    # Part 1
    flag_is_source_video = False
    source_fps = None
    flag_is_source_video = False
    img_rgb = load_image_rgb(source)
    img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
    source_rgb_lst = [img_rgb]

 ### cons block
    ######## process driving info ########
    flag_load_from_template = False
    driving_rgb_crop_256x256_lst = None
    wfp_template = None

    # Part 2.1
    flag_is_driving_video = True
    # load from video file, AND make motion template
    output_fps = fps ## not needed

    # driving_rgb_lst = load_video(driving) ## not needed
    # print("Part 2.1")
    final_dict = get_driving_dict(descriptor_lst, fps=fps)


    driving_n_frames = len(final_dict["driving_template_dct"]["motion"])

# Part 3.3
    print("Part 3.3")
    n_frames = driving_n_frames
# # Part 4.2
#     print("Part 4.2")
#     driving_lmk_crop_lst = live_portrait_pipeline.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
#     driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256

# # cons block

#     c_d_eyes_lst, c_d_lip_lst = live_portrait_pipeline.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst) # Eye/lip close and open ratio
#     # save the motion template
#     I_d_lst = live_portrait_pipeline.live_portrait_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)
#     driving_template_dct = live_portrait_pipeline.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)
#     # print(driving_template_dct)
    driving_template_dct, c_d_eyes_lst, c_d_lip_lst = final_dict["driving_template_dct"], final_dict["c_d_eyes_lst"], final_dict["c_d_lip_lst"]
    
# Cons block 2
    I_p_pstbk_lst = None

# Part 6.1
    print("Part 6.1")
    I_p_pstbk_lst = []

### cons block
    I_p_lst = []
    R_d_0, x_d_0_info = None, None
    flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite
    flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite
    lip_delta_before_animation, eye_delta_before_animation = None, None

# 7.2
# 7.2.1

    print("7.2.1")
    crop_info = live_portrait_pipeline.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)
    if crop_info is None:
        raise Exception("No face detected in the source image!")
    source_lmk = crop_info['lmk_crop']
    img_crop_256x256 = crop_info['img_crop_256x256']


    I_s = live_portrait_pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)
    x_s_info = live_portrait_pipeline.live_portrait_wrapper.get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
    f_s = live_portrait_pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
    x_s = live_portrait_pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)


    #modify kp code from before

# 7.2.4
    print("7.2.4")
    mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

##3 CoNS BLCOCK
    from rich.progress import track
    for i in track(range(n_frames), description='🚀Animating...', total=n_frames):

# 10.2
        x_d_i_info = driving_template_dct['motion'][i]

    # constant block
        x_d_i_info = dct2device(x_d_i_info, device)
        R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys

# 11.1
        if i == 0:  # cache the first frame
            print("11.1")
            R_d_0 = R_d_i
            x_d_0_info = x_d_i_info.copy()

        delta_new = x_s_info['exp'].clone()
# 12.1
# 12.1.1
            
        R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s

# 12.2
# 12.2.2
    # 12.2.2.1 # difference here in lip array
        delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])
# 13.1

        scale_new = x_s_info['scale'] if flag_is_source_video else x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])
# 14.1
        t_new = x_s_info['t'] if flag_is_source_video else x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])

# consistent block

        t_new[..., 2].fill_(0)  # zero tz
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

# 16
# 16.1 ## difference
        print("16")
        if i == 0:
            print("16.1")
            x_d_0_new = x_d_i_new
            motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)
            # motion_multiplier *= inf_cfg.driving_multiplier
        x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier
        x_d_i_new = x_d_diff + x_s

# 18
# 18.2
        print("18.2")
        x_d_i_new = live_portrait_pipeline.live_portrait_wrapper.stitching(x_s, x_d_i_new)

# Consistent block
        x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
        out = live_portrait_pipeline.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
        I_p_i = live_portrait_pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
        I_p_lst.append(I_p_i)

# 20
# 20.2
        I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)
        I_p_pstbk_lst.append(I_p_pstbk)

# 21.3
    print("21.3")
    frames_concatenated = concat_frames(driving_rgb_crop_256x256_lst, [img_crop_256x256], I_p_lst)

    if flag_is_driving_video or (flag_is_source_video and not flag_is_driving_video):
        print("22")
        flag_source_has_audio = flag_is_source_video and has_audio_stream(source)
        flag_driving_has_audio = False

        wfp_concat = osp.join(args.output_dir, f'{basename(args.source)}--{basename(output_vid_name)}_concat.mp4')

        # NOTE: update output fps
        output_fps = source_fps if flag_is_source_video else output_fps
        images2video(frames_concatenated, wfp=wfp_concat, fps=output_fps)

        if flag_source_has_audio or flag_driving_has_audio:
            print("22.1")
            # final result with concatenation
            wfp_concat_with_audio = osp.join(output_dir, f'{basename(source)}--{basename(output_vid_name)}_concat_with_audio.mp4')
            audio_from_which_video = args.driving if ((flag_driving_has_audio and args.audio_priority == 'driving') or (not flag_source_has_audio)) else source
            add_audio_to_video(wfp_concat, audio_from_which_video, wfp_concat_with_audio)
            os.replace(wfp_concat_with_audio, wfp_concat)

        # save the animated result
        wfp = osp.join(args.output_dir, f'{basename(source)}--{basename(output_vid_name)}.mp4')
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            print("22.2.1")
            images2video(I_p_pstbk_lst, wfp=wfp, fps=output_fps)

    ######### build the final result #########
    if flag_source_has_audio or flag_driving_has_audio:
        wfp_with_audio = osp.join(args.output_dir, f'{basename(source)}--{basename(output_vid_name)}_with_audio.mp4')
        audio_from_which_video = args.driving if ((flag_driving_has_audio and args.audio_priority == 'driving') or (not flag_source_has_audio)) else args.source
        add_audio_to_video(wfp, audio_from_which_video, wfp_with_audio)
        os.replace(wfp_with_audio, wfp)


output_dir = "animations/"
output_vid_name = "test_2"
import pickle
with open(f"descriptor_list.pkl", 'rb') as file:
    template_dct = pickle.load(file)

map_image_using_descriptor_lst("source_folder/2.jpg", template_dct, output_dir=output_dir, output_vid_name= output_vid_name, inference_cfg = inference_cfg, crop_cfg=crop_cfg, fps = 30)
