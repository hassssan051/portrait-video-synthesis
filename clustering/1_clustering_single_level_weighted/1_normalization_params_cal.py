# Load necessary libraries
import pickle
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import torch
from tqdm import tqdm
from collections import defaultdict

# Extract Euler angles from rotation matrix
def get_euler_angles_from_rotation_matrix(rot_matrix):
    rot_matrix = rot_matrix.permute(0, 2, 1)
    yaw = torch.arcsin(-rot_matrix[:, 2, 0])
    near_pi_over_2 = torch.isclose(torch.cos(yaw), torch.tensor(0.0), atol=1e-6)

    pitch = torch.where(
        ~near_pi_over_2,
        torch.atan2(rot_matrix[:, 2, 1], rot_matrix[:, 2, 2]),
        torch.atan2(rot_matrix[:, 1, 2], rot_matrix[:, 1, 1])
    )

    roll = torch.where(
        ~near_pi_over_2,
        torch.atan2(rot_matrix[:, 1, 0], rot_matrix[:, 0, 0]),
        torch.zeros_like(yaw)
    )

    pitch = pitch * 180 / torch.pi
    yaw = yaw * 180 / torch.pi
    roll = roll * 180 / torch.pi

    return pitch, yaw, roll


# Function to extract the full 208-dimensional vector from frame data
def extract_full_vector(frame_data):
    c_d_eyes = frame_data['c_d_eyes_lst'][0].reshape(-1)  # 2 values
    c_d_lip = frame_data['c_d_lip_lst'][0].reshape(-1)    # 1 value

    driving_template = frame_data['driving_template_dct']
    c_eyes = driving_template['c_eyes_lst'][0].reshape(-1)  # 2 values
    c_lip = driving_template['c_lip_lst'][0].reshape(-1)    # 1 value

    motion = driving_template['motion'][0]
    scale = np.array(motion['scale']).reshape(-1)         # 1 value
    t = motion['t'].reshape(-1)                           # 3 values
    R = motion['R'].reshape(1, 3, 3)                      # 9 values in matrix form
    exp = motion['exp'].reshape(-1)                       # 63 values
    x_s = motion['x_s'].reshape(-1)                       # 63 values
    kp = motion['kp'].reshape(-1)                         # 63 values actual value now becomes 202

    # Convert R to pitch, yaw, and roll using the function
    pitch, yaw, roll = get_euler_angles_from_rotation_matrix(torch.tensor(R))
    euler_angles = np.array([pitch.item(), yaw.item(), roll.item()])

    if not np.array_equal(c_d_eyes, c_eyes):
        print("Eyes arrays not equal")
    if not np.array_equal(c_d_lip, c_lip):
        print("Lip arrays not equal")

    # print(c_d_eyes.shape, c_d_lip.shape, c_eyes.shape, c_lip.shape, scale.shape, t.shape, euler_angles.shape, exp.shape, x_s.shape, kp.shape)
    # print("(2,) (1,) (2,) (1,) (1,) (3,) (3,) (63,) (63,) (63,)")
    # 202 values

    # Combine the components into a full vector excluding R
    vector = np.concatenate([c_d_eyes, c_d_lip, c_eyes, c_lip, scale, t, euler_angles, exp, x_s, kp])
    overall_vector.append(vector.copy())

    return vector

def extract_subset_vector(vector):

    output_vector = []
    output_vector.extend(vector[3:5]) # c_eyes
    output_vector.extend(vector[5:6]) # lip
    output_vector.extend(vector[6:7]) # scale
    output_vector.extend(vector[7:10]) # t
    output_vector.extend(vector[10:13]) # yaw, pitch, roll
    output_vector.extend(vector[13:76]) # exp

    return np.array(output_vector)

# Apply weights to the components of the vector
def apply_weights(vector, means, stds):
    # First perform z-score normalization
    normalized_vector = (vector - means) / stds

    eyes = normalized_vector[3:5]  # Eyes
    lip = normalized_vector[5:6]  # Lip
    scale = normalized_vector[6:7]  # Scale
    t = normalized_vector[7:10]  # Translation
    yaw_pitch_roll = normalized_vector[10:13]  # Yaw, pitch, roll
    exp = normalized_vector[13:76]  # Expression (exp)

    # Apply the weights to the respective parts
    weighted_vector = np.copy(normalized_vector)
    weighted_vector[3:5] = eyes * weights['eyes']
    weighted_vector[5:6] = lip * weights['lip']
    weighted_vector[6:7] = scale * weights['scale']
    weighted_vector[7:10] = t * weights['t']
    weighted_vector[10:13] = yaw_pitch_roll * weights['yaw_pitch_roll']
    weighted_vector[13:76] = exp * weights['exp']

    output_vector = []
    output_vector.extend(weighted_vector[3:5])  # c_eyes
    output_vector.extend(weighted_vector[5:6])  # lip
    output_vector.extend(weighted_vector[6:7])  # scale
    output_vector.extend(weighted_vector[7:10])  # t
    output_vector.extend(weighted_vector[10:13])  # yaw, pitch, roll
    output_vector.extend(weighted_vector[13:76])  # exp

    return np.array(output_vector)

overall_vector = []

# Load the pickled data
with open('pkls/full_dataset_descriptors/live_portrait_descriptor_all.pkl', 'rb') as file:
    all_descriptors = pickle.load(file)

# Weightage for each component
weights = {
    'yaw_pitch_roll': 1,
    'exp': 1,
    'eyes': 1,
    'lip': 1,
    'scale': 1,
    't': 1
} 

# Dictionary to store video frames
video_dict = defaultdict(list)

# Populate the video_dict with frame arrays in order
for key, value in all_descriptors.items():
    parts = key.split('/')
    video_name = parts[1]
    frame_number = int(parts[2].split('.')[0])
    video_dict[video_name].append((frame_number, value))

# Clustering using selected components
all_vectors = []
frame_to_cluster_mapping = {}

for video_name, frames in tqdm(video_dict.items(), desc="Extracting vectors for clustering"):
    video_frame_vectors = []
    for frame_number, frame_data in frames:
        vector_full = extract_full_vector(frame_data)
        # Apply weights to each part of the vector
        all_vectors.append(vector_full)

# Convert to numpy array for easier computation
all_vectors = np.array(all_vectors)

# Compute mean and std for each column (feature)
means = np.mean(all_vectors, axis=0)
stds = np.std(all_vectors, axis=0)

# Avoid division by zero
stds[stds == 0] = 1

normalization_params = {
    'means': means,
    'stds': stds
}

with open('pkls/norm_params/normalization_params.pkl', 'wb') as f:
    pickle.dump(normalization_params, f)

