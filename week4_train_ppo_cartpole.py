"""Train a tiny PPO agent on CartPole for a few iterations.

This script is intentionally simple and is meant to verify that the PPO module
works on a standard Gymnasium environment before integrating it with the
locomotion environment.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from reinforcement_learning import PPOTrainer
from evaluate_policy import evaluate_policy


def main() -> None:
    env = gym.make("CartPole-v1", max_episode_steps=50)
    trainer = PPOTrainer(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=32,
        lr=1e-3,
        action_space=env.action_space,
    )

    for epoch in range(50):
        obs, _ = env.reset(seed=epoch)
        rewards = []
        done = False
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            dist, value = trainer.policy(obs_tensor)
            action_sample = dist.sample()
            action = int(action_sample.item())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            obs = next_obs
            done = terminated or truncated
        print(f"epoch={epoch} reward_sum={sum(rewards):.2f}")

    stats = evaluate_policy(env, trainer.policy, episodes=5, max_steps=50)
    print("evaluation", stats)
    env.close()


if __name__ == "__main__":
    main()
