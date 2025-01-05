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

def extract_subset_vector(vector):

    output_vector = []
    output_vector.extend(vector[3:5]) # c_eyes
    output_vector.extend(vector[5:6]) # lip
    output_vector.extend(vector[7:10]) # t
    output_vector.extend(vector[10:13]) # yaw, pitch, roll
    output_vector.extend(vector[13:76]) # exp

    return np.array(output_vector)

# Apply weights to the components of the vector
def apply_weights(vector, means, stds, weights):
    # First perform z-score normalization
    normalized_vector = (vector - means) / stds

    eyes = normalized_vector[3:5]  # Eyes
    lip = normalized_vector[5:6]  # Lip
    t = normalized_vector[7:10]  # Translation
    yaw_pitch_roll = normalized_vector[10:13]  # Yaw, pitch, roll
    exp = normalized_vector[13:76]  # Expression (exp)

    # Apply the weights to the respective parts
    weighted_vector = np.copy(normalized_vector)
    weighted_vector[3:5] = eyes * weights['eyes']
    weighted_vector[5:6] = lip * weights['lip']
    weighted_vector[7:10] = t * weights['t']
    weighted_vector[10:13] = yaw_pitch_roll * weights['yaw_pitch_roll']
    weighted_vector[13:76] = exp * weights['exp']

    output_vector = []
    output_vector.extend(weighted_vector[3:5])  # c_eyes (2)
    output_vector.extend(weighted_vector[5:6])  # c_lip (1)
    output_vector.extend(weighted_vector[7:10])  # t (3)
    output_vector.extend(weighted_vector[10:13])  # yaw, pitch, roll (3)
    output_vector.extend(weighted_vector[13:76])  # exp (63)

    return np.array(output_vector)

# Apply weights to the components of the vector
def apply_weights_xs(vector, means, stds, weights):
    # First perform z-score normalization
    normalized_vector = (vector - means) / stds

    eyes = normalized_vector[3:5]  # Eyes
    lip = normalized_vector[5:6]  # Lip
    x_s = normalized_vector[76:139]  # Shape (x_s)
    # Apply the weights to the respective parts
    weighted_vector = np.copy(normalized_vector)
    weighted_vector[3:5] = eyes * weights['eyes']
    weighted_vector[5:6] = lip * weights['lip']
    output_vector = []
    output_vector.extend(weighted_vector[3:5])  # c_eyes (2)
    output_vector.extend(weighted_vector[5:6])  # c_lip (1)
    output_vector.extend(weighted_vector[76:139])  # x_s (63)

    return np.array(output_vector)



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


def unflatten_vector_lstm(vector, means, stds, weights):
    # Convert flattened vector back to original format
    c_d_eyes = np.array(vector[0:2], dtype=np.float32) / weights['eyes']
    c_d_lip = np.array(vector[2:3], dtype=np.float32) / weights['lip']
    c_eyes = np.array(vector[0:2], dtype=np.float32) / weights['eyes']
    c_lip = np.array(vector[2:3], dtype=np.float32) / weights['lip']
    scale = np.array(1.3, dtype=np.float32)
    t = np.array(vector[3:6], dtype=np.float32) / weights['t']
    
    # Convert to rotation matrix and update
    euler_angles = np.array([[vector[6], vector[7], vector[8]]]) / weights['yaw_pitch_roll']
    
    # Expression, shape and keypoint parameters
    exp = (np.array(vector[9:], dtype=np.float32)) / (weights['exp'] / 21)
    x_s = np.random.rand(1, 21, 3).astype(np.float32).reshape(-1)
    kp = np.random.rand(1, 21, 3).astype(np.float32).reshape(-1)

    # Flatten and concatenate the values in order
    flattened_vector = np.concatenate([
        c_d_eyes,
        c_d_lip,
        c_eyes,
        c_lip,
        scale,
        t,
        euler_angles,  # Euler angles
        exp,
        x_s,
        kp
    ])

    flattened_vector = (flattened_vector * stds) + means

    return unflatten_vector(flattened_vector)