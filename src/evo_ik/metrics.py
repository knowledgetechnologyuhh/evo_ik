import torch


def range_conversion(value, from_range, to_range):
    """
    Converts value from one range to another.

    :param value: value to convert
    :type value: float
    :param value: min and max of the input value's possibility space
    :type value: tuple(float, float)
    :param value: min and max of the range to convert to
    :type value: tuple(float, float)
    """
    from_min, from_max = from_range
    to_min, to_max = to_range
    return (value - from_min) / (from_max - from_min) * (to_max - to_min) + to_min


def euclidean_distance(pos_a, pos_b):
    return torch.linalg.norm(pos_b - pos_a, dim=1)


def quaternion_angle_distance(quat_a, quat_b):
    # solves scalar part of q_a * inv(q_b) which equals cos(theta/2) for theta (in radians)
    # absolute dot product is equivalent to min((theta/2), pi-(theta/2))
    # clamp prevents nan results due to floating point imprecision
    return torch.sum(quat_a * quat_b, dim=1).abs().clamp(max=1.0).arccos() * 2.0
