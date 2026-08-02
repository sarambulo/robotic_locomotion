import numpy as np
import pytest
import torch
from gymnasium import spaces

from reinforcement_learning import (
    ContinuousActorCritic,
    DiscreteActorCritic,
    GAE,
    PPOBuffer,
    PPOTrainer,
    RolloutBuffer,
)


def test_actor_critic_outputs_expected_shapes():
    model = ContinuousActorCritic(input_dim=4, hidden_dim=8, action_dim=2)
    obs = torch.randn(3, 4)
    action_dist, value = model(obs)

    assert action_dist.mean.shape == (3, 2)
    assert action_dist.stddev.shape == (3, 2)
    assert value.shape == (3, 1)


def test_discrete_actor_critic_outputs_expected_shapes():
    model = DiscreteActorCritic(input_dim=4, hidden_dim=8, action_dim=3)
    obs = torch.randn(3, 4)
    action_dist, value = model(obs)

    assert isinstance(action_dist, torch.distributions.Categorical)
    assert value.shape == (3, 1)


def test_gae_computes_expected_advantages():
    rewards = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    values = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    next_value = torch.tensor([[2.5]], dtype=torch.float32)

    advantages, returns = GAE(rewards, values, next_value, gamma=0.9, lam=0.95)

    assert advantages.shape == (3, 1)
    assert returns.shape == (3, 1)
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(returns).all()


def test_rollout_buffer_collects_and_returns_batches():
    buffer = RolloutBuffer(capacity=4)
    for i in range(4):
        buffer.add(
            obs=torch.tensor([[float(i)]], dtype=torch.float32),
            action=torch.tensor([[float(i)]], dtype=torch.float32),
            reward=torch.tensor([float(i)]),
            value=torch.tensor([[float(i)]], dtype=torch.float32),
            log_prob=torch.tensor([0.1]),
            done=torch.tensor([False]),
        )

    batch = buffer.sample()
    assert batch["obs"].shape[0] == 4
    assert batch["action"].shape[0] == 4
    assert batch["reward"].shape[0] == 4


def test_ppo_buffer_normalizes_advantages():
    buffer = PPOBuffer()
    buffer.add(
        obs=torch.randn(2, 3),
        action=torch.randn(2, 2),
        reward=torch.randn(2),
        value=torch.randn(2, 1),
        log_prob=torch.randn(2),
        done=torch.tensor([False, True]),
        advantage=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        return_value=torch.tensor([[3.0], [4.0]], dtype=torch.float32),
    )

    advantages = buffer.advantages

    assert advantages.shape == (2, 1)
    assert torch.isfinite(advantages).all()
    assert torch.allclose(advantages.mean(), torch.tensor(0.0, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(advantages.std(unbiased=False), torch.tensor(1.0, dtype=torch.float32), atol=1e-6)


def test_trainer_can_update_policy_once():
    env = None
    trainer = PPOTrainer(input_dim=2, action_dim=1, hidden_dim=8, lr=1e-3, steps_per_epoch=4, batch_size=2)
    loss = trainer.update_policy(torch.randn(4, 2), torch.randn(4, 1), torch.randn(4, 1), torch.randn(4, 1), torch.randn(4, 1), torch.randn(4, 1))
    assert torch.isfinite(loss)


