# Load necessary libraries
import pickle
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from collections import defaultdict
import math
import os
import yaml
from datetime import datetime
from utils import extract_full_vector, apply_encoder, unflatten_vector, apply_encoder_exp_lip
from utils import Encoder
import torch.nn as nn
import torch

# Record start time for performance tracking
start_time = datetime.now()
######################################################################## Model config ########################################################################################
config_path = "config.yaml"
######################################################################### Model config processing #######################################################################################3
# Weightage for each component
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

n_clusters = config['n_clusters']

experimental_setup = {
    "n_clusters": n_clusters
}

day = str(datetime.now().day)
month = datetime.now().strftime("%b").lower()
time = datetime.now().strftime("%H_%M_%S") + f"_{n_clusters}"

# Load the pickled data
with open('pkls/full_dataset_descriptors/live_portrait_descriptor_all_with_mead.pkl', 'rb') as file:
    all_descriptors = pickle.load(file)

final_files_save_path = f'pkls/clustering_output/{month}/{day}/{time}/'
os.makedirs(final_files_save_path, exist_ok=True)

input_dim = 72
encoder_layers = [input_dim, 512, 256, 16]  # Number of neurons in each encoder layer
encoder_activations = [nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU]  # Activation functions for encoder
encoder = Encoder(input_dim=input_dim, layer_dims=encoder_layers, activations=encoder_activations).cuda()
# Load the saved weights into the encoder
encoder.load_state_dict(torch.load(config["encoder_save_path"]))
# Ensure the loaded encoder is in evaluation mode (for inference)
encoder.eval()
print("Encoder loaded successfully and ready for inference.")

print(final_files_save_path)
###################################### Data preprocessing ###################cd##############################################################################################

# Dictionary to store video frames
video_dict = defaultdict(list)

# Populate the video_dict with frame arrays in order
for key, value in all_descriptors.items():
    parts = key.split('/')
    if key[0] == "M":
        video_name = "/".join(parts[:-1])
        frame_number = parts[-1].split('.')[0].split("_")[-1]
    else:
        video_name = parts[1]
        frame_number = parts[-1].split('.')[0]
    video_dict[video_name].append((frame_number, value))

# Sort video_dict by keys
video_dict = dict(sorted(video_dict.items()))

# Sort each video's frames by frame number
for video_name in video_dict:
    video_dict[video_name].sort(key=lambda x: int(x[0]))

# Clustering using selected components
subset_vectors_weighted_normalized = []
frame_to_cluster_mapping = {}
all_vectors_full = []
for video_name, frames in tqdm(video_dict.items(), desc="Extracting vectors for clustering"):
    video_frame_vectors = []
    for frame_number, frame_data in frames:
        vector_full = extract_full_vector(frame_data)
        all_vectors_full.append(vector_full)

        # Apply weights to each part of the vector
        weighted_vector = apply_encoder(vector_full, encoder)
        subset_vectors_weighted_normalized.append(weighted_vector)
        
        video_frame_vectors.append((frame_number, weighted_vector))

    frame_to_cluster_mapping[video_name] = video_frame_vectors

subset_vectors_weighted_normalized = np.vstack(subset_vectors_weighted_normalized)
all_vectors_full = np.vstack(all_vectors_full)

# Remove video_dict from RAM
del video_dict

####################################################### RUN K MEANS ####################################################################################
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(subset_vectors_weighted_normalized)

# Save KMeans model
kmeans_save_path = os.path.join(final_files_save_path, 'kmeans_model.pkl')
with open(kmeans_save_path, 'wb') as f:
    pickle.dump(kmeans, f)
    
# Load KMeans model back (commented out since we just created it)
# with open(kmeans_save_path, 'rb') as f:
#     kmeans = pickle.load(f)
print("KMeans model saved successfully")
####################################################### RUN K MEANS ####################################################################################


###################################################### Extract key information and store ###############################################################
# Create mapping of cluster ID to all vectors in that cluster while unnormalizing
cluster_to_unnormalized_original_vectors = {i: [] for i in range(n_clusters)}
cluster_to_unnormalized_original_dict = {i: [] for i in range(n_clusters)}
cluster_to_normalized_original_vectors = {i: [] for i in range(n_clusters)}

idx = 0
for video_name, frames in frame_to_cluster_mapping.items():
    for i, (frame_number, vector) in enumerate(frames):
        cluster_id = kmeans.labels_[idx]
        frame_to_cluster_mapping[video_name][i] = (f"{frame_number}.jpg", cluster_id)

        # Get original unnormalized vector and normalized vector
        original_vector = all_vectors_full[idx]
        normalized_vector = subset_vectors_weighted_normalized[idx]

        # Append normalized vector
        cluster_to_normalized_original_vectors[cluster_id].append(normalized_vector)

        # Append unnormalized vector
        cluster_to_unnormalized_original_vectors[cluster_id].append(original_vector)
        cluster_to_unnormalized_original_dict[cluster_id].append(unflatten_vector(original_vector))

        idx += 1

with open(f'{final_files_save_path}frame_to_cluster_mapping.pkl', 'wb') as f:
    pickle.dump(frame_to_cluster_mapping, f)
del frame_to_cluster_mapping

with open(f'{final_files_save_path}cluster_to_unencoded_original_dict.pkl', 'wb') as f:
    pickle.dump(cluster_to_unnormalized_original_dict, f)
del cluster_to_unnormalized_original_dict

## Iterate through UNnormalized vectors
averaged_descriptors_raw = {}
for cluster_id, vectors in tqdm(cluster_to_unnormalized_original_vectors.items(), desc="Averaging cluster unnormalized vectors"):
    if vectors:
        # Average all vectors in cluster
        avg_vector = np.mean(vectors, axis=0)
        # Convert averaged vector back to original format
        unflattened_avg = unflatten_vector(avg_vector)
        averaged_descriptors_raw[cluster_id] = unflattened_avg

del all_vectors_full

# Save results
with open(f'{final_files_save_path}averaged_descriptors_raw.pkl', 'wb') as f:
    pickle.dump(averaged_descriptors_raw, f)

del averaged_descriptors_raw

## Iterate through normalized weighted vector original
averaged_descriptors_normalized = {}
for cluster_id, vectors in tqdm(cluster_to_normalized_original_vectors.items(), desc="Averaging cluster normalized vectors"):
    if vectors:
        # Average all vectors in cluster
        avg_vector = np.mean(vectors, axis=0)
        # Convert averaged vector back to original format
        averaged_descriptors_normalized[cluster_id] = avg_vector

with open(f'{final_files_save_path}averaged_descriptors_encoded.pkl', 'wb') as f:
    pickle.dump(averaged_descriptors_normalized, f)

with open(f'{final_files_save_path}experimental_setup.pkl', 'wb') as f:
    pickle.dump(experimental_setup, f)

del averaged_descriptors_normalized
del experimental_setup

######################################################################## Get Updated live portrait descriptors (weighted and normalized) ################################################################
for key, value in all_descriptors.items():
    all_descriptors[key] = apply_encoder(extract_full_vector(value), encoder)

for key in all_descriptors:
    print(all_descriptors[key].shape)
    break

print("Updated live portrait descriptors given the encoder")

with open(f'{final_files_save_path}live_portrait_descriptors_all_encoder.pkl', 'wb') as file:
    pickle.dump(all_descriptors, file)

del all_descriptors
################################################################################## Post run analysis statistics ###################################################################

final_files_save_path = f'pkls/clustering_output/{month}/{day}/{time}/'

# Path to your pickle file
file_path = f'{final_files_save_path}/cluster_to_unencoded_original_dict.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    cluster_org_groups = pickle.load(file)

# Assuming your dictionary is named `clusters_dict`
cluster_sizes = [len(points) for points in cluster_org_groups.values()]
# Compute descriptive statistics
mean_size = np.mean(cluster_sizes)
median_size = np.median(cluster_sizes)
std_dev_size = np.std(cluster_sizes)
min_size = np.min(cluster_sizes)
max_size = np.max(cluster_sizes)
q1_size = np.percentile(cluster_sizes, 25)
q3_size = np.percentile(cluster_sizes, 75)

# Count the number of clusters with exactly 1 point
single_point_clusters = sum(1 for size in cluster_sizes if size == 1)

# Save to text file
with open(f"{final_files_save_path}/cluster_statistics.txt", "w") as file:
    file.write("Cluster Analysis Results:\n")
    file.write(f"Total Clusters (n_clusters): {n_clusters}\n\n")
    
    file.write("Descriptive Statistics for Cluster Sizes:\n")
    file.write(f"Mean: {mean_size}\n")
    file.write(f"Median: {median_size}\n")
    file.write(f"Standard Deviation: {std_dev_size}\n")
    file.write(f"Min: {min_size}\n")
    file.write(f"Max: {max_size}\n")
    file.write(f"25th Percentile (Q1): {q1_size}\n")
    file.write(f"75th Percentile (Q3): {q3_size}\n")
    file.write(f"Number of clusters with exactly 1 point: {single_point_clusters}\n\n")

########################################################################### Compute run time #################################################################################
end_time = datetime.now()
run_time = end_time - start_time
print("Total run time is", run_time)