# PyTorch Evolutionary Inverse Kinematics Solver

Evolutionary inverse kinematics solver combining EvoTorch and PyTorch Robot Kinematics.

## Usage

```py
from evo_ik import EvoIK

ik_solver = EvoIK("nico_left_arm.urdf", "left_tcp", device=device, logging=False)

target_pos = torch.tensor([0.35, 0.08, 0.67])
target_rot = torch.tensor([0.0, 0.0, torch.pi / 2.0]) # Z, Y, X

rad_angles = ik_solver.inverse_kinematics_from_euler(target_pos, target_rot, convention="ZYX", initial_joints=torch.zeros(6))
```
