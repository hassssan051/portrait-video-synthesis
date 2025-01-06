import os
import cv2
import numpy as np
from natsort import natsorted
import sys

# Assume `src` is located in the parent directory
sys.path.append(os.path.abspath('../'))
from src.utils.video import images2video
from src.utils.video import add_audio_to_video


def load_images(folder_path):
    """
    Load images from a folder in natural order and return a list of numpy arrays in uint8 format.

    Parameters:
        folder_path (str): Path to the folder containing images.

    Returns:
        list: List of numpy arrays representing the images in uint8 format.
    """
    # Get the list of files in the folder
    file_list = os.listdir(folder_path)

    # Sort the files using natsort
    sorted_files = natsorted(file_list)

    # Initialize a list to store images
    image_list = []

    # Iterate through sorted files and load images
    for file_name in sorted_files:
        file_path = os.path.join(folder_path, file_name)

        # Ensure the file is an image
        if os.path.isfile(file_path) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')):
            # Load the image using OpenCV
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Ensure the image was loaded successfully
            if image is not None:
                # Convert to uint8 (though OpenCV loads as uint8 by default)
                image_uint8 = np.asarray(image, dtype=np.uint8)
                image_list.append(image_uint8)

    return image_list

# Example usage
# Replace 'your_folder_path_here' with the path to your folder
folder_path = 'video_generation_test/temp/nine_batch'
for fold in os.listdir(folder_path):
    video_path = os.path.join(folder_path, fold)
    images = load_images(video_path)
    video_path = f"video_generation_test/stitching_code_2/{fold}.mp4" 
    # video_path = f"video_generation_test/stiched_videos/01-01-02-02-02-02-02.mp4" 
    # audio_video_path = "/home/sawaiz/Documents/Lab/In_Progress/Current/Dr. Immadullah/Phase 1/code_base/pipeline/Generation_LivePortrait/sawaiz_scripts/01-01-01-01-01-01-01.mp4"
    images2video(images, video_path, fps=60)
    # add_audio_to_video(video_path, audio_video_path, video_path)
