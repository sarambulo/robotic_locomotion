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

## Setup
```bash
conda activate robotics
pip install -r requirements.txt
python week1_manual_control.py
```

## Project Philosophy
- Build incrementally, one concept at a time
- Each component independently testable
- Minimal abstraction, maximal learning
- Clean, readable code
