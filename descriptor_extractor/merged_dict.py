import pickle
import numpy as np
# Path to your pickle file
file_path_1 = 'descriptor_pkls/mead_descriptors/mead_live_portrait_descriptor_all.pkl'
file_path_2 = 'descriptor_pkls/rawdes_full_dataset_descriptors/rawdes_live_portrait_descriptor_all.pkl'

# Open the file in binary read mode
with open(file_path_1, 'rb') as file:
    six = pickle.load(file)


# Open the file in binary read mode
with open(file_path_2, 'rb') as file:
    sixteen = pickle.load(file)

merged_dict = six | sixteen

# Specify the filename
filename = 'descriptor_pkls/combined_descriptors/live_portrait_descriptor_all_with_mead.pkl'

# Save dictionary as a pickle file
with open(filename, 'wb') as file:
    pickle.dump(merged_dict, file)