import numpy as np
# Points generator
def get_grid(x, y, homogenous=False):
    coords = np.indices((x, y)).reshape(2, -1)
    return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords
# Define Transformations
def get_rotation_matrix(angle):
    angle = np.radians(angle)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
def get_translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
def get_scale_matrix(s):
    return np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, 1]
    ])

def transformation(x_points, y_points, angle_factor, translate_x_factor, translate_y_factor, scale_factor, center_x=0, center_y=0):
    z_points = np.ones_like(x_points)
    coords = np.array([x_points, y_points, z_points])
    TC_1 = get_translation_matrix(-center_x, -center_y)
    R1 = get_rotation_matrix(angle_factor)
    TC_2 = get_translation_matrix(center_x, center_y)
    T1 = get_translation_matrix(translate_x_factor, translate_y_factor)
    S1 = get_scale_matrix(scale_factor)

    coords_composite2 = TC_2 @ T1 @ S1 @  R1 @ TC_1 @ coords
    return coords_composite2[0], coords_composite2[1]



def test():
    x_points = np.array([0, 0, 20, 20])
    y_points = np.array([0, 20, 20, 0])
    coords_composite2 = transformation(x_points, y_points, 90, -2, 2, 2)
    # Apply transformation x' = Ax

    print(coords_composite2)

# test()