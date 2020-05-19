import numpy as np


def get_new_bounds(original_shape, rot_matrix):
    """
    Calculate the boundaries of the new image
    Args:
        original_shape (tuple): Shape of original image in (H, W)
        rot_matrix: A 2*3 rotation matrix

    Returns:
        x_min, x_max, y_min, y_max :
    """
    assert len(original_shape) == 2
    points = [[0, 0], [original_shape[1], 0], [0, original_shape[0]],
              [original_shape[1], original_shape[0]]]
    new_points = [rotate_pt((pt[0], pt[1]), rot_matrix) for pt in points]

    h_min = np.min([x[0] for x in new_points])
    h_max = np.max([x[0] for x in new_points])

    w_min = np.min([y[1] for y in new_points])
    w_max = np.max([y[1] for y in new_points])
    return h_min, h_max, w_min, w_max


def rotate_pt(pt, rot_matrix):
    """
    Rotates a point
    :param pt: (H, W co-ordinate of point to rotate)
    :param rot_matrix: A 2*3 rotation matrix
    :return: (h, w) co-ordinates of new point
    """
    h, w = pt
    a, b, c, d, e, f = rot_matrix
    i, j = a * h + b * w + c, d * h + e * w + f
    new_point = int(i), int(j)
    return new_point


def check_bounds(point, shape):
    """
    Checks if a point is in bounds of a shape
    :param point: (H co-ordinate , W, co-ordinate)
    :param shape: (H, W)
    :return:
    """
    x_min = 0
    x_max = shape[1]
    y_min = 0
    y_max = shape[0]
    if (point[1] <= x_min) or (point[1] >= x_max):
        return True
    elif (point[0] <= y_min) or (point[0] >= y_max):
        return True
    else:
        return False


def get_rotation_matrix(image, angle, adjust_boundaries=True):
    """
    Calculates the 2D rotation matrix
    :param image: An image in (H, W) shape
    :param angle: Angle of rotation in degrees
    :return: The 2D rotation matrix
    """
    angle = np.deg2rad(angle)

    h, w = image.shape
    cy, cx = h / 2, w / 2

    a = np.math.cos(angle)
    b = np.math.sin(angle)
    xc1 = -cx
    xc2 = cx
    yc1 = -cy
    yc2 = cy

    rot_matrix = [
        a, -b, xc1 * a - yc1 * b + xc2, b, a, xc1 * b + yc1 * a + yc2
    ]

    if adjust_boundaries:
        x_min, x_max, y_min, y_max = get_new_bounds((h, w), rot_matrix)
        new_img = np.zeros((y_max - y_min, x_max - x_min))

        new_h, new_w = new_img.shape
        yc_new, xc_new = new_h / 2, new_w / 2

        y_shift = cy - yc_new
        x_shift = cx - xc_new

        rot_matrix[2] -= x_shift
        rot_matrix[5] -= y_shift

    return rot_matrix


def rotate_image(image, rot_matrix):
    """
    Rotates a given image
    :param image: A grayscale image in (H, W) format
    :param rot_matrix: a 2 * 3 rotation matrix
    :return: Rotated image
    """
    h_min, h_max, w_min, w_max = get_new_bounds(
        (image.shape[0], image.shape[1]), rot_matrix)
    new_image = np.zeros((h_max, w_max))
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            new_point = rotate_pt((j, i), rot_matrix)
            if check_bounds((new_point[1], new_point[0]), new_image.shape):
                continue
            else:
                new_image[new_point[1]][new_point[0]] = image[i][j]
    return new_image
