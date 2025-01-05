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
from utils import extract_full_vector, apply_weights, unflatten_vector

# Record start time for performance tracking
start_time = datetime.now()
######################################################################## Model config ########################################################################################
config_path = "config.yaml"
divide_exp_by = 21

######################################################################### Model config processing #######################################################################################3
# Weightage for each component
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Check if weights sum to 1
weight_sum = sum(config['weights'].values())
if abs(weight_sum - 1.0) > 1e-10:  # Using small epsilon for floating point comparison
    raise ValueError(f"Weights must sum to 1.0, but they sum to {weight_sum}. Please adjust weights in config.yaml")

# Weightage for each component
weights = {
    'exp': math.sqrt(config['weights']['exp']) / divide_exp_by,
    'eyes': math.sqrt(config['weights']['eyes']), 
    'lip': math.sqrt(config['weights']['lip']),
    't': math.sqrt(config['weights']['t']),
    'yaw_pitch_roll': math.sqrt(config['weights']['yaw_pitch_roll']),
}

n_clusters = config['n_clusters']

experimental_setup = {
    "weights": weights,
    "n_clusters": n_clusters
}

day = str(datetime.now().day)
month = datetime.now().strftime("%b").lower()
weights_str = f"_{config['weights']['exp']}_{config['weights']['eyes']}_{config['weights']['lip']}_{config['weights']['t']}_{config['weights']['yaw_pitch_roll']}"
time = datetime.now().strftime("%H_%M_%S") + f"_{n_clusters}" + weights_str

# Load the pickled data
with open('pkls/full_dataset_descriptors/live_portrait_descriptor_all.pkl', 'rb') as file:
    all_descriptors = pickle.load(file)

# Load the pickled data
with open(str(config["norm_params_path"]), 'rb') as file:
    norm_params = pickle.load(file)

final_files_save_path = f'pkls/clustering_output/{month}/{day}/{time}/'
os.makedirs(final_files_save_path, exist_ok=True)

means = norm_params['means']
stds = norm_params['stds']

print(final_files_save_path)
###################################### Data preprocessing #################################################################################################################

# Dictionary to store video frames
video_dict = defaultdict(list)

# Populate the video_dict with frame arrays in order
for key, value in all_descriptors.items():
    parts = key.split('/')
    video_name = parts[1]
    frame_number = int(parts[2].split('.')[0])
    video_dict[video_name].append((frame_number, value))

# Sort video_dict by keys
video_dict = dict(sorted(video_dict.items()))

# Sort each video's frames by frame number
for video_name in video_dict:
    video_dict[video_name].sort(key=lambda x: x[0])

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
        weighted_vector = apply_weights(vector_full, means, stds, weights)
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

## Iterate through UNnormalized vectors
averaged_descriptors_raw = {}
for cluster_id, vectors in tqdm(cluster_to_unnormalized_original_vectors.items(), desc="Averaging cluster unnormalized vectors"):
    if vectors:
        # Average all vectors in cluster
        avg_vector = np.mean(vectors, axis=0)
        # Convert averaged vector back to original format
        unflattened_avg = unflatten_vector(avg_vector)
        averaged_descriptors_raw[cluster_id] = unflattened_avg

## Iterate through normalized weighted vector original
averaged_descriptors_normalized = {}
for cluster_id, vectors in tqdm(cluster_to_normalized_original_vectors.items(), desc="Averaging cluster normalized vectors"):
    if vectors:
        # Average all vectors in cluster
        avg_vector = np.mean(vectors, axis=0)
        # Convert averaged vector back to original format
        averaged_descriptors_normalized[cluster_id] = avg_vector

# Save results
with open(f'{final_files_save_path}averaged_descriptors_raw.pkl', 'wb') as f:
    pickle.dump(averaged_descriptors_raw, f)

with open(f'{final_files_save_path}averaged_descriptors_normalized.pkl', 'wb') as f:
    pickle.dump(averaged_descriptors_normalized, f)

with open(f'{final_files_save_path}frame_to_cluster_mapping.pkl', 'wb') as f:
    pickle.dump(frame_to_cluster_mapping, f)

with open(f'{final_files_save_path}cluster_to_unnormalized_original_dict.pkl', 'wb') as f:
    pickle.dump(cluster_to_unnormalized_original_dict, f)

with open(f'{final_files_save_path}experimental_setup.pkl', 'wb') as f:
    pickle.dump(experimental_setup, f)

with open(f'{final_files_save_path}norm_params.pkl', 'wb') as f:
    pickle.dump(norm_params, f)


######################################################################## Get Updated live portrait descriptors (weighted and normalized) ################################################################
for key, value in all_descriptors.items():
    all_descriptors[key] = apply_weights(extract_full_vector(value), means, stds, weights)

for key in all_descriptors:
    print(all_descriptors[key].shape)
    break

print("Updated live portrait descriptors given the weights and normalization")

with open(f'{final_files_save_path}live_portrait_descriptors_all_weighted_normalized.pkl', 'wb') as file:
    pickle.dump(all_descriptors, file)
################################################################################## Post run analysis statistics ###################################################################
# Unsquare the weights by squaring them again
weights["exp"] = weights["exp"] * divide_exp_by

unsquared_weights = {}
for component, weight in weights.items():
    unsquared_weights[component] = round(weight * weight, 10)

weights = unsquared_weights

final_files_save_path = f'pkls/clustering_output/{month}/{day}/{time}/'

# Path to your pickle file
file_path = f'{final_files_save_path}/cluster_to_unnormalized_original_dict.pkl'

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
    
    file.write("Weightage for Each Component:\n")
    for component, weight in weights.items():
        file.write(f"{component}: {weight}\n")

########################################################################### Compute run time #################################################################################
end_time = datetime.now()
run_time = end_time - start_time
print("Total run time is", run_time)