import os
import pytest
import pytorch_kinematics as pk
import torch
from evo_ik import EvoIK
from os.path import dirname, abspath, join

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def ik_solver():
    yield EvoIK(
        join(dirname(abspath(__file__)), "nico_left_arm.urdf"),
        end_link_name="left_tcp",
        root_link_name="torso:11",
        device=device,
        logging=False,
    )


@pytest.fixture()
def sampler(ik_solver):
    yield torch.distributions.Uniform(*ik_solver.limits)


@pytest.mark.repeat(100)
def test_random_pose(ik_solver, sampler):
    random_angles = sampler.sample()
    fk = ik_solver.forward_kinematics(random_angles)
    target_mat = fk.get_matrix()
    target_pos = target_mat[:, :3, 3]
    target_quat = pk.matrix_to_quaternion(target_mat[:, :3, :3])
    joint_angles = ik_solver.inverse_kinematics(
        target_pos,
        target_quat,
        dist_acc=0.01,
        orient_acc=0.349,
        max_steps=200,
        initial_joints=sampler.sample(),
    )
    pos_error, ori_error = ik_solver.get_euclidean_errors(
        joint_angles, target_pos, target_quat
    )
    assert pos_error < 0.01
    assert torch.rad2deg(ori_error) < 20.0
    assert ik_solver.check_success(
        joint_angles,
        target_pos,
        target_quat,
        dist_acc=0.01,
        orient_acc=0.349,
    )
