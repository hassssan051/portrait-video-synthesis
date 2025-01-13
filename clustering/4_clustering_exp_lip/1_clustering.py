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
from utils import extract_lip_data

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

print(final_files_save_path)
###################################### Data preprocessing #################################################################################################################

# Dictionary to store video frames
video_dict = defaultdict(list)

removed = 0
# Populate the video_dict with frame arrays in order
for key, value in all_descriptors.items():
    parts = key.split('/')
    if key[0] == "M":
        video_name = "/".join(parts[:-1])
        frame_number = parts[-1].split('.')[0].split("_")[-1]
    else:
        if key.split("/")[1][:2] == "02":
            removed+=1
            continue
        video_name = parts[1]
        frame_number = parts[-1].split(')[0]
    video_dict[video_name].append((frame_number, value))

print("Removed data:", removed)

# Sort video_dict by keys
video_dict = dict(sorted(video_dict.items()))

# Sort each video's frames by frame number
for video_name in video_dict:
    video_dict[video_name].sort(key=lambda x: int(x[0]))

# Clustering using selected components
frame_to_cluster_mapping = {}
all_lip_vectors_full = []
for video_name, frames in tqdm(video_dict.items(), desc="Extracting vectors for clustering"):
    video_frame_vectors = []
    for frame_number, frame_data in frames:
        lip_vector = extract_lip_data(frame_data)
        all_lip_vectors_full.append(lip_vector)
        
        video_frame_vectors.append((frame_number, lip_vector))

    frame_to_cluster_mapping[video_name] = video_frame_vectors

all_lip_vectors_full = np.vstack(all_lip_vectors_full)

# Remove video_dict from RAM
del video_dict

####################################################### RUN K MEANS ####################################################################################
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(all_lip_vectors_full)
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
cluster_to_exp_lip_vectors = {i: [] for i in range(n_clusters)}

idx = 0
for video_name, frames in frame_to_cluster_mapping.items():
    for i, (frame_number, vector) in enumerate(frames):
        cluster_id = kmeans.labels_[idx]
        frame_to_cluster_mapping[video_name][i] = (f"{frame_number}.jpg", cluster_id)
        # Get original unnormalized vector and normalized vector
        lip_vector = all_lip_vectors_full[idx]
        # Append normalized vector
        cluster_to_exp_lip_vectors[cluster_id].append(lip_vector)
        idx += 1

with open(f'{final_files_save_path}frame_to_cluster_mapping.pkl', 'wb') as f:
    pickle.dump(frame_to_cluster_mapping, f)
del frame_to_cluster_mapping

## Iterate through UNnormalized vectors
averaged_descriptors_lips = {}
for cluster_id, vectors in tqdm(cluster_to_exp_lip_vectors.items(), desc="Averaging cluster lip vectors from exp"):
    if vectors:
        # Average all vectors in cluster
        avg_vector = np.mean(vectors, axis=0)
        # Convert averaged vector back to original format
        unflattened_avg = avg_vector
        averaged_descriptors_lips[cluster_id] = unflattened_avg

del all_lip_vectors_full

with open(f'{final_files_save_path}cluster_to_exp_lip_vectors_dict.pkl', 'wb') as f:
    pickle.dump(cluster_to_exp_lip_vectors, f)
del cluster_to_exp_lip_vectors

# Save results
with open(f'{final_files_save_path}averaged_descriptors_exp_lips.pkl', 'wb') as f:
    pickle.dump(averaged_descriptors_lips, f)

del averaged_descriptors_lips


with open(f'{final_files_save_path}experimental_setup.pkl', 'wb') as f:
    pickle.dump(experimental_setup, f)

del experimental_setup
######################################################################## Get Updated live portrait descriptors (weighted and normalized) ################################################################
for key, value in all_descriptors.items():
    all_descriptors[key] = extract_lip_data(value)

for key in all_descriptors:
    print(all_descriptors[key].shape)
    break

print("Updated live portrait descriptors given the weights and normalization")

with open(f'{final_files_save_path}live_portrait_descriptors_exp_lip_all.pkl', 'wb') as file:
    pickle.dump(all_descriptors, file)

del all_descriptors
################################################################################## Post run analysis statistics ###################################################################
final_files_save_path = f'pkls/clustering_output/{month}/{day}/{time}/'

# Path to your pickle file
file_path = f'{final_files_save_path}cluster_to_exp_lip_vectors_dict.pkl'

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