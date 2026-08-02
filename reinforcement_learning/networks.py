"""Policy and value networks for PPO."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class ContinuousActorCritic(nn.Module):
    """Simple MLP actor-critic shared backbone for continuous actions."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.shared(obs)
        mu = self.mu_head(features)
        log_std = torch.clamp(self.log_std_head(features), min=-2.0, max=2.0)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        value = self.value_head(features)
        return dist, value


class DiscreteActorCritic(nn.Module):
    """Simple MLP actor-critic shared backbone for discrete actions."""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.logits_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        features = self.shared(obs)
        logits = self.logits_head(features)
        dist = Categorical(logits=logits)
        value = self.value_head(features)
        return dist, value