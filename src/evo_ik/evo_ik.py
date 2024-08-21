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
    """
    The EvoIK class provides methods for performing inverse kinematics (IK)
    using evolutionary algorithms on a robotic kinematic chain described by a URDF.

    Attributes:
        chain: A kinematic chain object from pytorch_kinematics representing the robot.
        joint_names: The names of the actuated joints in the kinematic chain.
        limits: The joint limits of the robot, stored in a PyTorch tensor.
        num_joints: The number of actuated joints in the kinematic chain.
        device: The device on which computations will be performed.
        logging: A flag indicating whether to enable logging of search progress.
    """

    def __init__(
        self,
        urdf: str,
        end_link_name: str,
        root_link_name: str = "",
        device: str = "cpu",
        logging: bool = False,
    ) -> None:
        """
        Initialize the EvoIK class with a kinematic model from a URDF file.

        Args:
            urdf (str): The path to the URDF file describing the kinematic chain.
            end_link_name (str): The name of the end-effector link.
            root_link_name (str, optional): The name of the root link. Default is "".
            device (str, optional): The device used for computations ('cpu' or 'cuda'). Default is "cpu".
            logging (bool, optional): Whether to enable logging. Default is False.
        """
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
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        orientation_weight: torch.Tensor = torch.tensor(0.0573),
    ) -> Problem:
        """
        Create an optimization problem for inverse kinematics.

        Args:
            target_pos (torch.Tensor): The target position of the end-effector.
            target_quat (torch.Tensor): The target orientation represented as a quaternion.
            orientation_weight (torch.Tensor, optional): The weight for orientation in the optimization loss. Default is 0.0573.

        Returns:
            Problem: An evotorch Problem instance representing the inverse kinematics optimization.
        """
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

    def forward_kinematics(self, joint_angles: torch.Tensor) -> pk.Transform3d:
        """
        Perform forward kinematics to get the transformation of the end-effector.

        Args:
            joint_angles (torch.Tensor): The joint angles for the kinematic chain.

        Returns:
            Transform3d: The resulting transformation of the end-effector.
        """
        return self.chain.forward_kinematics(joint_angles.to(self.device))

    def check_success(
        self,
        joint_angles: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        dist_acc: float = 0.01,
        orient_acc: float = 0.349,
    ) -> torch.Tensor:
        """
        Check whether the inverse kinematics solution meets the success criteria.

        Args:
            joint_angles (torch.Tensor): The joint angles to evaluate.
            target_pos (torch.Tensor): The target position of the end-effector.
            target_quat (torch.Tensor): The target orientation as a quaternion.
            dist_acc (float, optional): The maximum allowable position error. Default is 0.01.
            orient_acc (float, optional): The maximum allowable orientation error. Default is 0.349.

        Returns:
            torch.Tensor: A tensor indicating success (True) or failure (False) of the IK solution.
        """
        distance, orientation = self.get_euclidean_errors(
            joint_angles, target_pos, target_quat
        )
        return torch.logical_and(
            torch.less(distance, dist_acc), torch.less(orientation, orient_acc)
        )

    def get_euclidean_errors(
        self,
        joint_angles: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> tuple:
        """
        Compute the positional and orientation errors relative to target end-effector pose.

        Args:
            joint_angles (torch.Tensor): The joint angles of the kinematic chain.
            target_pos (torch.Tensor): The target position.
            target_quat (torch.Tensor): The target orientation as a quaternion.

        Returns:
            tuple: A tuple containing the positional error and orientation error.
        """
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
        target_pos: torch.Tensor,
        target_euler: torch.Tensor,
        convention: str = "ZYX",
        initial_joints: torch.Tensor = None,
        dist_acc: float = 0.01,
        orient_acc: float = 0.349,
        min_steps: int = 20,
        max_steps: int = 200,
        step_incr: int = 20,
        orientation_weight: torch.Tensor = torch.tensor(0.0573),
    ) -> torch.Tensor:
        """
        Perform inverse kinematics using euler angles for the target orientation.

        Args:
            target_pos (torch.Tensor): The target position.
            target_euler (torch.Tensor): The target orientation as Euler angles.
            convention (str, optional): The convention for Euler angles. Default is "ZYX".
            initial_joints (torch.Tensor, optional): The initial guess for joint angles. Default is None.
            dist_acc (float, optional): The distance accuracy for success. Default is 0.01.
            orient_acc (float, optional): The orientation accuracy for success. Default is 0.349.
            min_steps (int, optional): Minimum number of optimization steps. Default is 20.
            max_steps (int, optional): Maximum number of optimization steps. Default is 200.
            step_incr (int, optional): Increment for steps beyond min_steps if needed. Default is 20.
            orientation_weight (torch.Tensor, optional): Weight for orientation in the loss. Default is 0.0573.

        Returns:
            torch.Tensor: The computed joint angles that achieve the target end-effector pose.
        """
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
        mat: torch.Tensor,
        initial_joints: torch.Tensor = None,
        dist_acc: float = 0.01,
        orient_acc: float = 0.349,
        min_steps: int = 20,
        max_steps: int = 200,
        step_incr: int = 20,
        orientation_weight: torch.Tensor = torch.tensor(0.0573),
    ) -> torch.Tensor:
        """
        Perform inverse kinematics using a transformation matrix for the target pose.

        Args:
            mat (torch.Tensor): A transformation matrix representing the target end-effector pose.
            initial_joints (torch.Tensor, optional): The initial guess for joint angles. Default is None.
            dist_acc (float, optional): The distance accuracy for success. Default is 0.01.
            orient_acc (float, optional): The orientation accuracy for success. Default is 0.349.
            min_steps (int, optional): Minimum number of optimization steps. Default is 20.
            max_steps (int, optional): Maximum number of optimization steps. Default is 200.
            step_incr (int, optional): Increment for steps beyond min_steps if needed. Default is 20.
            orientation_weight (torch.Tensor, optional): Weight for orientation in the loss. Default is 0.0573.

        Returns:
            torch.Tensor: The computed joint angles that achieve the target end-effector pose.
        """
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
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        initial_joints: torch.Tensor = None,
        dist_acc: float = 0.01,
        orient_acc: float = 0.349,
        min_steps: int = 20,
        max_steps: int = 200,
        step_incr: int = 20,
        orientation_weight: torch.Tensor = torch.tensor(0.0573),
    ) -> torch.Tensor:
        """
        Perform inverse kinematics to find joint angles that match the target pose.

        Args:
            target_pos (torch.Tensor): The target position.
            target_quat (torch.Tensor): The target orientation as a quaternion.
            initial_joints (torch.Tensor, optional): The initial guess for joint angles. Default is None.
            dist_acc (float, optional): The distance accuracy for success. Default is 0.01.
            orient_acc (float, optional): The orientation accuracy for success. Default is 0.349.
            min_steps (int, optional): Minimum number of optimization steps. Default is 20.
            max_steps (int, optional): Maximum number of optimization steps. Default is 200.
            step_incr (int, optional): Increment for steps beyond min_steps if needed. Default is 20.
            orientation_weight (torch.Tensor, optional): Weight for orientation in the loss. Default is 0.0573.

        Returns:
            torch.Tensor: The computed joint angles that achieve the target end-effector pose.
        """
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
            print(f"Solution: {rad_angles}")
            print(f"Success: {success.detach().item()}")
            distance, angle = self.get_euclidean_errors(
                rad_angles, target_pos, target_quat
            )
            print(f"Distance: {distance.detach().item()}")
            print(f"Angle: {torch.rad2deg(angle).detach().item()}°")
        return rad_angles
