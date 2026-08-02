"""
Week 1: Manual simulation control
- Load robot into simulation
- Step simulation manually
- Apply fixed torque to joint
- Read back joint position and velocity
- Visualize simulation (optional, may not work in WSL without X11)
"""

from __future__ import annotations

import sys

import mujoco

# Load the model
model = mujoco.MjModel.from_xml_path("robot/robot.xml")
data = mujoco.MjData(model)

print("Model loaded successfully")
print(f"Number of joints: {model.njnt}")
print(f"Joint names: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]}")
print(f"Number of actuators: {model.nu}")
print(f"Actuator names: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]}")

# Apply a fixed torque to the joint
fixed_torque = 0.5
data.ctrl[0] = fixed_torque
print(f"\nApplying fixed torque: {fixed_torque} Nm")

# Step simulation manually for 100 steps
print("\nStepping simulation...")
print("Step | Position (rad) | Velocity (rad/s)")
print("-" * 40)

for step in range(100):
    mujoco.mj_step(model, data)

    # Read back joint position and velocity
    joint_pos = data.qpos[0]
    joint_vel = data.qvel[0]

    if step % 10 == 0:
        print(f"{step:4d} | {joint_pos:14.4f} | {joint_vel:14.4f}")

print("\nSimulation complete")
print(f"Final position: {data.qpos[0]:.4f} rad")
print(f"Final velocity: {data.qvel[0]:.4f} rad/s")

# Open viewer
skip_viewer = "--no-viewer" in sys.argv

if not skip_viewer:
    try:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Viewer launched successfully. Press Ctrl+C to exit.")
            while viewer.is_running():
                data.ctrl[0] = fixed_torque
                mujoco.mj_step(model, data)
                viewer.sync()
    except (AttributeError, ImportError) as e:
        print(f"Viewer not available: {e}")
        print("This is expected in WSL without proper X11 setup.")
        print("Run with --no-viewer flag to skip this step entirely.")
    except Exception as e:
        print(f"Viewer error: {e}")
        print("This is expected in WSL without proper X11 setup.")
        print("Run with --no-viewer flag to skip this step entirely.")
else:
    print("Skipping viewer (--no-viewer flag specified).")
