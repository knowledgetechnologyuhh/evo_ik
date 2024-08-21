# PyTorch Evolutionary Inverse Kinematics Solver

Evolutionary inverse kinematics solver combining EvoTorch and PyTorch Robot Kinematics.

## Installation

```bash
git clone https://github.com/knowledgetechnologyuhh/EvoIK.git
cd evo_ik
pip install .
```

### Test Installation

> **NOTE:**
While all test poses are generated within the robot's reach, some may still fail due to chance or numerical instabilities (see [Known Issues](#known-issues)). As long as the majority passes, everything should work as intended.

```bash
pip install .[tests]
pytest .
```

## Usage

To calculate the inverse kinematics for your chain, all you need is a `.urdf` file, the name of the end-effector, and a target pose:

```py
import torch
from evo_ik import EvoIK

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create ik solver from urdf and given end-effector
ik_solver = EvoIK(
    urdf="nico_left_arm.urdf", end_link_name="left_tcp", device=device, logging=False
)

# define end-effector target
target_pos = torch.tensor([0.35, 0.08, 0.67])  # x, y, z in meters
target_rot = torch.tensor([0.707, 0.707, 0.0, 0.0])  # w, x, y, z quaternion

# solve inverse kinematics
rad_angles = ik_solver.inverse_kinematics(
    target_pos, target_rot, initial_joints=torch.zeros(6)
)
```

if you prefer Euler angles over quaternions:

```py
# define end-effector target
target_pos = torch.tensor([0.35, 0.08, 0.67])  # x, y, z in meters
target_rot = torch.tensor([0.0, 0.0, torch.pi / 2.0])  # z, y, x Euler angles in radians

# solve inverse kinematics (convention is ZYX by default)
rad_angles = ik_solver.inverse_kinematics_from_euler(
    target_pos, target_rot, convention="ZYX", initial_joints=torch.zeros(6)
)
```

or the full transformation matrix:

```py
# define end-effector target
target_mat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.35],
        [0.0, 0.0, -1.0, 0.08],
        [0.0, 1.0, 0.0, 0.67],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# solve inverse kinematics
rad_angles = ik_solver.inverse_kinematics_from_matrix(
    target_mat, initial_joints=torch.zeros(6)
)
```

The `initial_joints` parameter can be used to define start angles. A good practice is to set these to the current motor angles of the robot or the previous pose in a planned trajectory to generate trajectories with minimal movement.

## Known Issues

- The CMAES solver from EvoTorch seems to have numerical stability issues for longer runs with very small errors, occasionaly causing `nan` solutions.

- In some cases, the solver may get stuck at a local optimum and not find a solution that meets the success criteria, even if the target is reachable.

## Citation

```bibtex
@inproceedings{gade2024domain,
  title={Domain Adaption as Auxiliary Task for Sim-to-Real Transfer in Vision-based Neuro-Robotic Control},
  author={G{\"a}de, Connor and Habekost, Jan-Gerrit and Wermter, Stefan},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)},
  year={2024}
}
```

## License

Copyright (C) 2024  Connor Gäde

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.