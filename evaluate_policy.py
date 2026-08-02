"""Evaluate a simple policy on a Gymnasium environment."""

from __future__ import annotations

import numpy as np
import torch


def evaluate_policy(env, policy, episodes: int = 3, max_steps: int = 1000):
    """Run a few episodes with a simple policy function and report reward statistics."""
    episode_rewards = []
    episode_lengths = []

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=episode_idx)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < max_steps:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist, value = policy(obs_tensor)
            action_sample = dist.sample()
            action = int(action_sample.item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            obs = next_obs

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    }


def main() -> None:
    import gymnasium as gym

    env = gym.make("CartPole-v1")

    def policy(obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((obs.shape[0], 1), dtype=torch.float32)

    stats = evaluate_policy(env, policy, episodes=2, max_steps=50)
    print(stats)
    env.close()


if __name__ == "__main__":
    main()
