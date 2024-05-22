import torch

import pytorch_kinematics as pk

from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.decorators import vectorized
from evotorch.logging import StdOutLogger

from evo_ik.metrics import (
    euclidean_distance,
    quaternion_angle_distance,
    range_conversion,
)


class EvoIK:
    def __init__(
        self, urdf, end_link_name, root_link_name="", device="cpu", logging=False
    ):
        self.chain = pk.build_serial_chain_from_urdf(
            open(urdf).read(), end_link_name, root_link_name
        )
        self.chain.to(device=device)
        self.joint_names = self.chain.get_joint_parameter_names()
        self.limits = torch.tensor(self.chain.get_joint_limits()).to(device)
        self.num_joints = len(self.joint_names)
        self.device = device
        self.logging = logging

    def create_problem(
        self, target_pos, target_quat, orientation_weight=torch.tensor(0.0573)
    ):
        orientation_weight = orientation_weight.to(self.device)

        @vectorized
        def inverse_kinematics(joints: torch.Tensor) -> torch.Tensor:
            rad_angles = range_conversion(joints, (-1.0, 1.0), self.limits)
            trans = self.forward_kinematics(rad_angles)
            mat = trans.get_matrix()
            pos = mat[:, :3, 3]
            pos_loss = euclidean_distance(pos, target_pos)
            rot = pk.matrix_to_quaternion(mat[:, :3, :3])
            rot_loss = quaternion_angle_distance(rot, target_quat) * orientation_weight
            oob_penalty = (joints - joints.clamp(-1.0, 1.0)).square().sum(-1)
            return pos_loss + rot_loss + oob_penalty

        return Problem(
            "min",
            inverse_kinematics,
            solution_length=self.num_joints,
            initial_bounds=(-1.0, 1.0),
            bounds=(-1.0, 1.0),
            device=self.device,
        )

    def forward_kinematics(self, joint_angles):
        return self.chain.forward_kinematics(joint_angles.to(self.device))

    def check_success(
        self,
        joint_angles,
        target_pos,
        target_quat,
        dist_acc=0.01,
        orient_acc=0.349,
    ):
        distance, orientation = self.get_euclidean_errors(
            joint_angles, target_pos, target_quat
        )
        return torch.logical_and(
            torch.less(distance, dist_acc), torch.less(orientation, orient_acc)
        )

    def get_euclidean_errors(self, joint_angles, target_pos, target_quat):
        # transfer to euclidean via forward kinematics
        trans = self.forward_kinematics(joint_angles)
        mat = trans.get_matrix()
        # extract position coordinates and orientation quaternion
        pos = mat[:, :3, 3]
        quat = pk.matrix_to_quaternion(mat[:, :3, :3])
        # calculate errors
        distance = euclidean_distance(pos, target_pos)
        orientation = quaternion_angle_distance(quat, target_quat)
        return distance, orientation

    def inverse_kinematics_from_euler(
        self,
        target_pos,
        target_euler,
        convention="ZYX",
        initial_joints=None,
        dist_acc=0.01,
        orient_acc=0.349,
        min_steps=20,
        max_steps=200,
        step_incr=20,
        orientation_weight=torch.tensor(0.0573),
    ):
        target_mat = pk.euler_angles_to_matrix(target_euler, convention)
        target_quat = pk.matrix_to_quaternion(target_mat)
        return self.inverse_kinematics(
            target_pos,
            target_quat,
            initial_joints,
            dist_acc,
            orient_acc,
            min_steps,
            max_steps,
            step_incr,
            orientation_weight,
        )

    def inverse_kinematics_from_matrix(
        self,
        mat,
        initial_joints=None,
        dist_acc=0.01,
        orient_acc=0.349,
        min_steps=20,
        max_steps=200,
        step_incr=20,
        orientation_weight=torch.tensor(0.0573),
    ):
        pos = mat[:3, 3]
        rot = pk.matrix_to_quaternion(mat[:3, :3])
        return self.inverse_kinematics(
            pos,
            rot,
            initial_joints,
            dist_acc,
            orient_acc,
            min_steps,
            max_steps,
            step_incr,
            orientation_weight,
        )

    def inverse_kinematics(
        self,
        target_pos,
        target_quat,
        initial_joints=None,
        dist_acc=0.01,
        orient_acc=0.349,
        min_steps=20,
        max_steps=200,
        step_incr=20,
        orientation_weight=torch.tensor(0.0573),
    ):
        target_pos = target_pos.to(self.device)
        target_quat = target_quat.to(self.device)
        if initial_joints is not None:
            initial_joints = range_conversion(
                initial_joints.to(self.device), self.limits, (-1.0, 1.0)
            )
        # create searcher and run for min_steps
        problem = self.create_problem(target_pos, target_quat, orientation_weight)
        searcher = CMAES(
            problem, stdev_init=0.05, popsize=32, center_init=initial_joints
        )
        if self.logging:
            _ = StdOutLogger(searcher)
        # run for min_steps
        searcher.run(min_steps)
        # get best joint angles
        joints = searcher.status["pop_best"].values.clamp(-1.0, 1.0)
        rad_angles = range_conversion(joints, (-1.0, 1.0), self.limits)
        # check success
        success = self.check_success(
            rad_angles, target_pos, target_quat, dist_acc, orient_acc
        )
        steps = min_steps
        # run additional steps if necessary
        while not success and steps < max_steps:
            searcher.run(min(step_incr, max_steps - steps))
            # get best joint angles
            joints = searcher.status["pop_best"].values.clamp(-1.0, 1.0)
            rad_angles = range_conversion(joints, (-1.0, 1.0), self.limits)
            # check success
            success = self.check_success(
                rad_angles, target_pos, target_quat, dist_acc, orient_acc
            )
            steps += step_incr
        if self.logging:
            print(f"Success: {success.detach().item()}")
            distance, angle = self.get_euclidean_errors(
                rad_angles, target_pos, target_quat
            )
            print(f"Distance: {distance.detach().item()}")
            print(f"Angle: {torch.rad2deg(angle).detach().item()}°")
        return rad_angles
