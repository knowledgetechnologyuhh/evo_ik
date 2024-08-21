import torch


def range_conversion(
    values: torch.Tensor, from_range: torch.Tensor, to_range: torch.Tensor
):
    """
    Convert a tensor of values from one range to another.

    Args:
        values (torch.Tensor): The tensor containing the values to convert.
        from_range (torch.Tensor): A tensor of size 2 indicating the minimum and maximum of the input range.
        to_range (torch.Tensor): A tensor of size 2 indicating the minimum and maximum of the target range.

    Returns:
        torch.Tensor: A tensor with values converted to the target range.
    """
    from_min, from_max = from_range
    to_min, to_max = to_range
    return (values - from_min) / (from_max - from_min) * (to_max - to_min) + to_min


def euclidean_distance(pos_a: torch.Tensor, pos_b: torch.Tensor):
    """
    Calculate the Euclidean distance between two sets of positions.

    Args:
        pos_a (torch.Tensor): A tensor of shape (N, D) representing N positions in D-dimensional space.
        pos_b (torch.Tensor): A tensor of shape (N, D) representing N positions in D-dimensional space.

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the Euclidean distances between each corresponding pair of positions in `pos_a` and `pos_b`.
    """
    return torch.linalg.norm(pos_b - pos_a, dim=1)


def quaternion_angle_distance(quat_a: torch.Tensor, quat_b: torch.Tensor):
    """
    Calculate the angular distance between pairs of quaternions.

    This function computes the angle (in radians) between each pair of quaternions in two input tensors.
    The angle is derived from the scalar part of the quaternion product q_a * inv(q_b), which can represent
    the cosine of half the angle between the two quaternions. The result is then scaled to full angles.

    Args:
        quat_a (torch.Tensor): A tensor of shape (N, 4), where N is the number of quaternions,
                               representing each quaternion's scalar and vector parts.
        quat_b (torch.Tensor): A tensor of shape (N, 4), where N is the number of quaternions,
                               representing each quaternion's scalar and vector parts.

    Returns:
        torch.Tensor: A tensor of shape (N,) containing the angular distances (in radians) between each
                      corresponding pair of quaternions in `quat_a` and `quat_b`.

    Note:
        - The inputs `quat_a` and `quat_b` should be normalized quaternions to represent valid rotations.
        - The absolute value of the dot product is used to ensure the shortest rotational path is taken,
          as quaternions q and -q represent the same rotation.
    """
    # clamp prevents nan results due to floating point imprecision
    return torch.sum(quat_a * quat_b, dim=1).abs().clamp(max=1.0).arccos() * 2.0
