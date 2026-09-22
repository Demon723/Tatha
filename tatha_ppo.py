"""
tatha_ppo.py
============
Policy Gradient (PPO) integration for EFE agents.

Replaces argmin EFE with learned policy for action selection.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO policy configuration."""
    hidden_size: int = 64
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 10
    mini_batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class PolicyNetwork:
    """Small MLP policy network for action selection."""
    def __init__(self, obs_dim: int, n_actions: int, config: PPOConfig):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config

        # Policy network: obs -> action probabilities
        self.W1 = np.random.randn(obs_dim, config.hidden_size) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(config.hidden_size)
        self.W2 = np.random.randn(config.hidden_size, config.hidden_size) * np.sqrt(2.0 / config.hidden_size)
        self.b2 = np.zeros(config.hidden_size)
        self.W3 = np.random.randn(config.hidden_size, n_actions) * 0.01
        self.b3 = np.zeros(n_actions)

        # Value network: obs -> value estimate
        self.VW1 = np.random.randn(obs_dim, config.hidden_size) * np.sqrt(2.0 / obs_dim)
        self.Vb1 = np.zeros(config.hidden_size)
        self.VW2 = np.random.randn(config.hidden_size, config.hidden_size) * np.sqrt(2.0 / config.hidden_size)
        self.Vb2 = np.zeros(config.hidden_size)
        self.VW3 = np.random.randn(config.hidden_size, 1) * 0.01
        self.Vb3 = np.zeros(1)

    def forward(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Forward pass.
        Returns: (action_probs, value_estimate)
        """
        # Policy
        h = np.maximum(0, obs @ self.W1 + self.b1)  # ReLU
        h = np.maximum(0, h @ self.W2 + self.b2)
        logits = h @ self.W3 + self.b3

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        action_probs = exp_logits / np.sum(exp_logits)

        # Value
        v_h = np.maximum(0, obs @ self.VW1 + self.Vb1)
        v_h = np.maximum(0, v_h @ self.VW2 + self.Vb2)
        value = v_h @ self.VW3 + self.Vb3

        return action_probs, float(value)

    def act(self, obs: np.ndarray) -> tuple[int, float]:
        """Sample action from policy."""
        probs, value = self.forward(obs)
        action = np.random.choice(self.n_actions, p=probs)
        return action, probs[action]

    def update(self, states, actions, rewards, next_states, dones):
        """PPO update step."""
        n = len(states)

        # Compute advantages using GAE
        values = []
        for s in states:
            _, v = self.forward(s)
            values.append(v)
        values = np.array(values)

        # TD targets
        td_targets = []
        for i in range(n):
            r = rewards[i]
            if i < n - 1 and not dones[i]:
                _, next_v = self.forward(next_states[i])
                td_target = r + self.config.gamma * next_v
            else:
                td_target = r
            td_targets.append(td_target)
        td_targets = np.array(td_targets)

        advantages = td_targets - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss = 0.0
        for _ in range(self.config.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.config.mini_batch_size):
                batch_idx = indices[start:start + self.config.mini_batch_size]
                batch_states = np.array([states[i] for i in batch_idx])
                batch_actions = np.array([actions[i] for i in batch_idx])
                batch_advantages = advantages[batch_idx]

                # Compute new probabilities
                new_probs, _ = self.forward(batch_states)
                old_probs = new_probs[batch_actions]  # approximate

                # Clipped surrogate objective
                ratio = new_probs[batch_actions] / (old_probs + 1e-8)
                surr1 = ratio * batch_advantages
                surr2 = np.clip(ratio, 1 - self.config.clip_epsilon,
                                1 + self.config.clip_epsilon) * batch_advantages
                policy_loss = -np.minimum(surr1, surr2).mean()

                # Value loss
                _, values_pred = self.forward(batch_states)
                value_loss = np.mean((values_pred - td_targets[batch_idx]) ** 2)

                # Entropy bonus
                entropy = -np.sum(new_probs * np.log(new_probs + 1e-8), axis=1).mean()

                # Total loss
                loss = (policy_loss + self.config.value_coef * value_loss
                        - self.config.entropy_coef * entropy)
                total_loss += loss

        return total_loss / (self.config.ppo_epochs * n)


class PPOAgent:
    """PPO wrapper for TathaEFEAgent."""
    def __init__(self, obs_dim: int, n_actions: int, config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.policy = PolicyNetwork(obs_dim, n_actions, self.config)
        self.memory = {'states': [], 'actions': [], 'rewards': [],
                       'next_states': [], 'dones': []}

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using learned policy."""
        action, _ = self.policy.act(obs)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['next_states'].append(next_state)
        self.memory['dones'].append(done)

    def learn(self):
        """PPO update from stored experiences."""
        if len(self.memory['states']) < self.config.mini_batch_size:
            return 0.0
        loss = self.policy.update(
            self.memory['states'],
            self.memory['actions'],
            self.memory['rewards'],
            self.memory['next_states'],
            self.memory['dones'],
        )
        # Clear memory
        self.memory = {'states': [], 'actions': [], 'rewards': [],
                       'next_states': [], 'dones': []}
        return loss
