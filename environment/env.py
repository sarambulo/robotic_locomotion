"""Minimal Gymnasium-compatible environment for the MuJoCo pendulum robot."""

from pathlib import Path

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RobotEnv(gym.Env):
    """A simple one-joint environment for educational reinforcement learning."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None, episode_length: int = 10):
        self.render_mode = render_mode
        self._episode_length = episode_length
        self._episode_step = 0
        self._terminated = False
        self._truncated = False

        model_path = Path(__file__).resolve().parents[1] / "robot" / "robot.xml"
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -10.0], dtype=np.float32),
            high=np.array([np.pi, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the simulation to the initial state.

        Returns
        -------
        observation : np.ndarray
            A 2D vector of the form ``[joint_position, joint_velocity]``.
            ``joint_position`` is in radians and ``joint_velocity`` is in radians
            per second.
        info : dict
            Empty dictionary for this minimal environment.
        """
        super().reset(seed=seed)
        self._episode_step = 0
        self._terminated = False
        self._truncated = False

        self.data.qpos[0] = 0.0
        self.data.qvel[0] = 0.0
        self.data.ctrl[0] = 0.0
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        """Apply a control action and advance the simulation by one time step.

        Parameters
        ----------
        action : array-like
            A 1D action vector with a single value representing the torque applied
            to the robot's joint. The value is clipped to the range ``[-1, 1]``
            and interpreted as a continuous motor command.

        Returns
        -------
        observation : np.ndarray
            A 2D vector of the form ``[joint_position, joint_velocity]`` after
            the step has been applied.
        reward : float
            A placeholder reward value for the current milestone.
        terminated : bool
            Whether the episode has reached the fixed maximum length.
        truncated : bool
            Always ``False`` in this minimal environment.
        info : dict
            Empty dictionary for this minimal environment.
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (1,):
            raise ValueError("Action must be a single scalar torque value")

        torque = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        self.data.ctrl[0] = torque
        mujoco.mj_step(self.model, self.data)

        observation = self._get_observation()
        reward = -0.1 * (observation[0] ** 2) + 0.01 * np.cos(observation[0])
        self._episode_step += 1

        self._terminated = self._episode_step >= self._episode_length
        self._truncated = False

        return observation, reward, self._terminated, self._truncated, {}

    def render(self):
        """Render the current simulation state when a render mode is enabled."""
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            try:
                import mujoco.viewer
            except Exception:
                return None
            if not hasattr(self, "_viewer"):
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            return None
        return None

    def close(self):
        if hasattr(self, "_viewer"):
            self._viewer.close()
            del self._viewer

    def _get_observation(self):
        position = np.clip(self.data.qpos[0], self.observation_space.low[0], self.observation_space.high[0])
        velocity = np.clip(self.data.qvel[0], self.observation_space.low[1], self.observation_space.high[1])
        return np.array([position, velocity], dtype=np.float32)
