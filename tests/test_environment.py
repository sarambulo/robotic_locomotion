import numpy as np
import pytest
from gymnasium import spaces

from environment.env import RobotEnv


def test_environment_exposes_expected_spaces_and_defaults():
    env = RobotEnv()

    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (2,)
    assert env.action_space.shape == (1,)
    assert env.action_space.low.shape == (1,)
    assert env.action_space.high.shape == (1,)

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert np.isfinite(obs).all()
    assert info == {}


def test_reset_initializes_state_and_episode_metadata():
    env = RobotEnv()
    obs, info = env.reset(seed=7)

    assert obs.shape == (2,)
    assert env._episode_step == 0
    assert env._episode_length == 10
    assert env._terminated is False
    assert env._truncated is False


def test_step_returns_observation_reward_terminated_and_truncated_flags():
    env = RobotEnv()
    env.reset(seed=1)

    action = np.array([0.25], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (2,)
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info == {}


def test_episode_terminates_after_fixed_length():
    env = RobotEnv()
    env.reset(seed=1)

    for _ in range(env._episode_length - 1):
        _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
        assert not terminated
        assert not truncated

    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
    assert terminated is True
    assert truncated is False


def test_render_and_close_do_not_crash():
    env = RobotEnv()
    env.reset(seed=1)

    frame = env.render()
    assert frame is None or isinstance(frame, np.ndarray)

    env.close()
