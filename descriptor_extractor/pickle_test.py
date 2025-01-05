import pickle
import numpy as np

# Path to your pickle file
file_path = 'clustering_output/averaged_descriptors_custom_distance.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    data = pickle.load(file)


for key in data:
    for k in data[key]["motion"]:
        print(k, data[key]["motion"][k].shape)
    break

# # Function to print structure
# # Example function to explore structure
# def print_detailed_structure(value, indent=0):
#     indent_space = "  " * indent
#     if isinstance(value, dict):
#         for k, v in value.items():
#             print(f"{indent_space}{k}: {type(v).__name__}")
#             print_detailed_structure(v, indent + 1)
#     elif isinstance(value, list):
#         if value:
#             print(f"{indent_space}List of {type(value[0]).__name__} (length {len(value)})")
#             # Recursively print structure of first element if it's a dictionary or array
#             if isinstance(value[0], dict) or isinstance(value[0], np.ndarray):
#                 print_detailed_structure(value[0], indent + 1)
#             else:
#                 print(f"{indent_space}  {type(value[0]).__name__} elements")
#         else:
#             print(f"{indent_space}Empty list")
#     elif isinstance(value, np.ndarray):
#         print(f"{indent_space}Numpy array of {value.dtype} with shape {value.shape}")
#     else:
#         print(f"{indent_space}{type(value).__name__}")

# print(len(data))