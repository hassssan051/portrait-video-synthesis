import pickle
from utils import extract_lip_data, reverse_extract_lip_data
import numpy as np

# Load the pickled data
with open('pkls/full_dataset_descriptors/live_portrait_descriptor_all_with_mead.pkl', 'rb') as file:
    all_descriptors = pickle.load(file)

i = 0
for descriptor in all_descriptors:
    org_frame_Data = all_descriptors[descriptor]
    lip_flatten_vector = extract_lip_data(org_frame_Data)
    new_frame_data = reverse_extract_lip_data(org_frame_Data, lip_flatten_vector)
    
    driving_template = new_frame_data['driving_template_dct']
    motion = driving_template['motion'][0]
    exp_new = motion['exp'] 

    driving_template = org_frame_Data['driving_template_dct']
    motion = driving_template['motion'][0]
    exp_org = motion['exp'] 

    # Check if shapes match
    if exp_org.shape != exp_new.shape:
        print(f"Shape mismatch for {descriptor}: exp_org {exp_org.shape} vs exp_new {exp_new.shape}")
        continue

    # Check if values are equal within small numerical tolerance
    if not np.allclose(exp_org, exp_new, rtol=1e-5, atol=1e-8):
        print(f"Value mismatch for {descriptor}")
        print("Max absolute difference:", np.max(np.abs(exp_org - exp_new)))
        continue

    i+=1
    if i ==20:
        break

print("Not mismatch")
