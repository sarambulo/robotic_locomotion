# Robotic Locomotion

Educational project for learning reinforcement learning, robot modeling, and physics simulation through hands-on experimentation with MuJoCo.

## Week 1 Goals
- Install and configure MuJoCo
- Create the simplest possible robot (one body, one actuated revolute joint)
- Load the robot into a simulation
- Step the simulation manually
- Apply a fixed torque to the joint
- Read back the joint position and velocity
- Visualize the simulation

## Week 2 Progress
Week 2 introduced a minimal Gymnasium-compatible environment around the existing MuJoCo robot.

Implemented so far:
- A simple `RobotEnv` wrapper in `environment/env.py`
- `reset()`, `step()`, `render()`, and `close()` methods
- A basic observation vector containing joint position and velocity
- A continuous torque action space
- A placeholder reward signal and fixed-length episode termination
- Tests covering reset, stepping, rewards, observations, and termination behavior

This milestone is intentionally simple and transparent. The goal is to verify that the environment interface behaves correctly before introducing any RL logic.

## Setup
```bash
conda activate robotics
pip install -r requirements.txt
python -m pytest -q
python scripts/week1_manual_control.py
```

## Observation and action vectors
The environment uses a very small state and action representation so the interface remains easy to inspect.

- Observation vector: a 2D NumPy array of the form `[joint_position, joint_velocity]`
  - `joint_position` is the current joint angle in radians
  - `joint_velocity` is the current joint angular velocity in radians per second
- Action vector: a 1D NumPy array with a single value representing the torque applied to the joint
  - The action is continuous and is clipped to the range `[-1, 1]`

The same contract is documented directly in the docstrings of the environment methods in [environment/env.py](environment/env.py).

## Visualizing the simulation
To view the robot while interacting with the environment, create the environment with `render_mode="human"` and call `render()` during stepping.

Example:
```python
from environment.env import RobotEnv

env = RobotEnv(render_mode="human")
obs, info = env.reset()

for _ in range(10):
    action = [0.2]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

env.close()
```

If you are running in a headless environment, the viewer may not appear. In that case, you can still use the environment programmatically and inspect the observations and actions without rendering.

## Project Philosophy
- Build incrementally, one concept at a time
- Each component independently testable
- Minimal abstraction, maximal learning
- Clean, readable code
- Prefer understanding the implementation over adding complexity too early
