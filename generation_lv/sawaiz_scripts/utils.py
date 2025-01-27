import numpy as np

def reverse_extract_lip_data(frame_data, lip_flatten_vector):
    """
    Replaces the lip data in frame_data with the values from lip_flatten_vector.

    Args:
        frame_data (dict): The frame data dictionary.
        lip_flatten_vector (np.ndarray): The flattened lip vector to replace.

    Returns:
        dict: The updated frame_data with replaced lip data.
    """

    # Define the lip indices to be updated
    lip_indices = [6, 12, 14, 17, 19, 20]

    # Ensure the lip_flatten_vector has the correct size
    expected_size = len(lip_indices) * 3  # Assuming each lip point has 3 features (e.g., x, y, z)
    if lip_flatten_vector.size != expected_size:
        raise ValueError(f"lip_flatten_vector must have {expected_size} elements, got {lip_flatten_vector.size}.")

    # Reshape the flattened vector to (6, 3) corresponding to the 6 lip points
    lip_reshaped = lip_flatten_vector.reshape(len(lip_indices), 3)

    # Access the 'exp' array within the frame_data
    exp = frame_data['driving_template_dct']['motion'][0]['exp']

    # Check if 'exp' has the expected shape
    if exp.shape != (1, 21, 3):
        raise ValueError(f"'exp' array must have shape (1, 21, 3), got {exp.shape}.")

    # Replace the lip vectors at the specified indices
    for i, lip_idx in enumerate(lip_indices):
        exp[0, lip_idx, :] = lip_reshaped[i]

    return frame_data