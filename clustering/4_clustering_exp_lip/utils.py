import torch
import numpy as np

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

    return vector

# Function to extract the full 208-dimensional vector from frame data
def extract_lip_data(frame_data):
    driving_template = frame_data['driving_template_dct']
    
    motion = driving_template['motion'][0]
                     # 9 values in matrix form
    exp = motion['exp']   
    lip_vectors = []
    for lip_idx in [6, 12, 14, 17, 19, 20]:  
        lip_vectors.append(exp[:, lip_idx, :])

    lip_flatten_vector = np.array(lip_vectors).reshape(-1)

    # Combine the components into a full vector excluding R
    vector = lip_flatten_vector

    return vector


def reverse_extract_lip_data(frame_data, lip_flatten_vector):
    """
    Replaces the lip data in frame_data with the values from lip_flatten_vector.

    Args:
        frame_data (dict): The frame data dictionary.
        lip_flatten_vector (np.ndarray): The flattened lip vector to replace.

    Returns:
        dict: The updated frame_data with replaced lip data.
    """
    import numpy as np

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


def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * torch.pi
    yaw = yaw_ / 180 * torch.pi
    roll = roll_ / 180 * torch.pi

    device = pitch.device

    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)
    x, y, z = pitch, yaw, roll

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose

# Convert yaw, pitch, and roll to a rotation matrix
def euler_angles_to_rotation_matrix(pitch, yaw, roll):
    PI = np.pi
    # Convert to torch tensors and add batch dimension
    pitch_ = torch.tensor([pitch], dtype=torch.float32)
    yaw_ = torch.tensor([yaw], dtype=torch.float32) 
    roll_ = torch.tensor([roll], dtype=torch.float32)
    
    # Get rotation matrix using provided function
    R = get_rotation_matrix(pitch_, yaw_, roll_)
    
    # Convert to numpy and reshape to (1,3,3)
    R = R.cpu().numpy().astype(np.float32)
    
    return R


def unflatten_vector(avg_vector):
    # Convert flattened vector back to original format
    c_d_eyes = np.array(avg_vector[0:2], dtype=np.float32).reshape(1, 2)
    c_d_lip = np.array(avg_vector[2:3], dtype=np.float32).reshape(1, 1)
    c_eyes = np.array(avg_vector[3:5], dtype=np.float32).reshape(1, 2)
    c_lip = np.array(avg_vector[5:6], dtype=np.float32).reshape(1, 1)
    scale = np.array(avg_vector[6:7], dtype=np.float32).reshape(1, 1)
    t = np.array(avg_vector[7:10], dtype=np.float32).reshape(1, 3)
    
    # Convert to rotation matrix and update
    R = euler_angles_to_rotation_matrix(avg_vector[10], avg_vector[11], avg_vector[12])
    
    # Expression, shape and keypoint parameters
    exp = np.array(avg_vector[13:76], dtype=np.float32).reshape(1, 21, 3)
    x_s = np.array(avg_vector[76:139], dtype=np.float32).reshape(1, 21, 3)
    kp = np.array(avg_vector[139:202], dtype=np.float32).reshape(1, 21, 3)

    # Return dictionary in original format
    return {
        'c_d_eyes_lst': c_d_eyes,
        'c_d_lip_lst': c_d_lip,
        'driving_template_dct': {
            'motion': [{
                'scale': scale,
                'R': R,
                't': t,
                'c_eyes_lst': c_eyes,
                'c_lip_lst': c_lip,
                'exp': exp,
                'x_s': x_s,
                'kp': kp
            }]
        }
    }