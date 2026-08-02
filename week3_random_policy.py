"""Run a simple random-policy rollout for the MuJoCo robot environment.

This script is intentionally small and focused on validating that the
environment behaves correctly before any reinforcement learning logic is added.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from environment.env import RobotEnv


def run_episode_with_random_policy(env: RobotEnv, seed: int, episode_length: int) -> dict:
    """Run one episode with random actions and collect basic diagnostics."""
    observation, info = env.reset(seed=seed)

    rewards: list[float] = []
    actions: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    terminated = False
    truncated = False

    for step_index in range(episode_length):
        action = env.action_space.sample()  # Sample a random action
        next_observation, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        actions.append(action)
        observations.append(next_observation)

        print(
            f"episode={seed} step={step_index} action={action[0]:.4f} "
            f"obs={next_observation} reward={reward:.4f} terminated={terminated}"
        )

        if terminated or truncated:
            break

    cumulative_reward = float(np.sum(rewards))
    final_observation = observations[-1] if observations else observation

    return {
        "length": len(rewards),
        "cumulative_reward": cumulative_reward,
        "final_observation": final_observation,
        "actions": actions,
        "observations": observations,
        "terminated": terminated,
        "truncated": truncated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random-policy rollouts for the robot environment")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--episode-length", type=int, default=10, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = parser.parse_args()

    env = RobotEnv(episode_length=args.episode_length)

    episode_results = []
    for episode_index in range(args.episodes):
        episode = run_episode_with_random_policy(env, seed=episode_index, episode_length=args.episode_length)
        episode['episode_index'] = episode_index
        episode_results.append(episode)

    print("\nSummary")
    print("-------")
    for result in episode_results:
        print(
            f"episode={result['episode_index']} length={result['length']} "
            f"cumulative_reward={result['cumulative_reward']:.4f} "
            f"final_observation={result['final_observation']} terminated={result['terminated']}"
        )

    env.close()


if __name__ == "__main__":
    main()
