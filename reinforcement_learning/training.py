"""Training utilities for PPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from gymnasium import spaces

from .networks import ContinuousActorCritic, DiscreteActorCritic


class PPOTrainer:
    """Minimal PPO trainer interface for educational use."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        steps_per_epoch: int = 4,
        batch_size: int = 2,
        action_space=None,
    ):
        if action_space is not None and isinstance(action_space, spaces.Discrete):
            self.policy = DiscreteActorCritic(input_dim, hidden_dim, action_dim)
        else:
            self.policy = ContinuousActorCritic(input_dim, hidden_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

    def update_policy(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        old_log_prob: torch.Tensor,
        advantage: torch.Tensor,
        return_value: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        self.optimizer.zero_grad()
        dist, predicted_value = self.policy(obs)
        if isinstance(dist, torch.distributions.Categorical):
            action_tensor = action.long().reshape(-1)
            new_log_prob = dist.log_prob(action_tensor)
        else:
            new_log_prob = dist.log_prob(action).sum(dim=-1)
        ratio = torch.exp(new_log_prob - old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        surrogate = torch.min(ratio * advantage, clipped_ratio * advantage)
        value_loss = F.mse_loss(predicted_value, return_value)
        loss = -surrogate.mean() + 0.5 * value_loss
        loss.backward()
        self.optimizer.step()
        return loss.detach()


def GAE(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimates and returns."""
    rewards = rewards.float()
    values = values.float()
    next_value = next_value.float()

    returns = torch.zeros_like(values)
    gae = torch.zeros_like(values)
    next_return = next_value

    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t].view(1, -1) + gamma * next_return - values[t]
        gae[t] = delta + gamma * lam * gae[t + 1] if t + 1 < len(rewards) else delta
        next_return = values[t]
        returns[t] = gae[t] + values[t]

    advantages = returns - values
    return advantages, returns
