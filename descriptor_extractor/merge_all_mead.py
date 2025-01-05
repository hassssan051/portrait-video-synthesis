import os
import pickle

def merge_dicts_from_pkls(folder_path):
    """
    Reads all .pkl files from the given folder and its subfolders, 
    and merges the dictionaries contained in them.

    Args:
        folder_path (str): Path to the folder containing .pkl files.

    Returns:
        dict: Merged dictionary from all .pkl files.
    """
    merged_dict = {}

    # Walk through all subdirectories and files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)
                try:
                    # Load the .pkl file
                    with open(file_path, 'rb') as pkl_file:
                        data = pickle.load(pkl_file)
                        if isinstance(data, dict):
                            merged_dict.update(data)
                        else:
                            print(f"File {file_path} does not contain a dictionary, skipping.")
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    return merged_dict

# Example usage
if __name__ == "__main__":
    folder_path = "/home/sawaiz/Documents/Lab/In_Progress/Current/Dr. Immadullah/Phase 1/dataset/mead_descriptors"
    result = merge_dicts_from_pkls(folder_path)

    # Specify the filename
    filename = 'descriptor_pkls/mead_descriptors/mead_live_portrait_descriptor_all.pkl'

    # Save dictionary as a pickle file
    with open(filename, 'wb') as file:
        pickle.dump(result, file)