"""
tatha_replay.py
===============
Prioritized experience replay with importance sampling.
"""
from __future__ import annotations
import numpy as np
import collections


class PrioritizedReplayBuffer:
    """Sum-tree based prioritized experience replay."""
    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 10000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._buffer: list = []
        self._priorities = np.zeros(capacity)
        self._pos = 0
        self._size = 0

    def push(self, obs, action, reward, next_obs, done, td_error: float = 1.0):
        """Push experience with priority based on TD error."""
        priority = (abs(td_error) + 1e-6) ** self.alpha

        if self._size < self.capacity:
            self._buffer.append((obs, action, reward, next_obs, done))
            self._priorities[self._pos] = priority
            self._pos = (self._pos + 1) % self.capacity
            self._size += 1
        else:
            self._buffer[self._pos] = (obs, action, reward, next_obs, done)
            self._priorities[self._pos] = priority
            self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[list, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        probs = self._priorities[:self._size] / self._priorities[:self._size].sum()
        indices = np.random.choice(self._size, size=batch_size, replace=False, p=probs)
        batch = [self._buffer[i] for i in indices]

        # Importance sampling weights
        beta = min(self.beta_start + (1 - self.beta_start) * self._current_frame() / self.beta_frames, 1.0)
        max_prob = probs.max()
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities after learning."""
        for idx, td in zip(indices, td_errors):
            self._priorities[idx % self.capacity] = (abs(td) + 1e-6) ** self.alpha

    def _current_frame(self) -> int:
        return len(self._buffer) if hasattr(self, '_buffer') else 0

    def __len__(self):
        return self._size


class CurriculumScheduler:
    """Auto-difficulty curriculum for training."""
    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels
        self.current_level = 0
        self._metrics_history: list[float] = []
        self._thresholds = [0.3, 0.5, 0.7, 0.85, 0.95]

    def get_difficulty(self) -> int:
        """Current curriculum level (0 = easiest, n-1 = hardest)."""
        return self.current_level

    def update(self, success_rate: float):
        """Update curriculum based on success rate."""
        self._metrics_history.append(success_rate)

        # Advance if success rate exceeds threshold at current level
        target = self._thresholds[self.current_level]
        if success_rate > target and self.current_level < self.n_levels - 1:
            self.current_level += 1
            return True  # Level advanced
        return False

    def get_config(self) -> dict:
        """Get current curriculum parameters."""
        level = self.current_level
        return {
            'level': level,
            'exploration_coef': max(0.05, 0.5 - level * 0.1),
            'noise_scale': max(0.01, 0.3 - level * 0.05),
            'reward_threshold': self._thresholds[level],
        }

    def reset(self):
        self.current_level = 0
        self._metrics_history = []


class RewardShaper:
    """Transform sparse rewards into dense reward signals."""
    def __init__(self, shaping_type: str = 'distance', gamma: float = 0.99):
        self.shaping_type = shaping_type
        self.gamma = gamma
        self._prev_state = None

    def shape(self, state, reward, next_state, done: bool) -> float:
        """Add reward shaping to sparse reward."""
        if self.shaping_type == 'distance':
            return self._distance_shaping(state, reward, next_state, done)
        elif self.shaping_type == 'potential':
            return self._potential_shaping(state, reward, next_state, done)
        elif self.shaping_type == 'progress':
            return self._progress_shaping(state, reward, next_state, done)
        return reward

    def _distance_shaping(self, state, reward, next_state, done: bool) -> float:
        """Shape reward based on distance to goal."""
        if self._prev_state is None:
            self._prev_state = state
            return reward

        # Distance-based bonus: closer to goal = more reward
        dist_prev = np.linalg.norm(state - self._goal(state))
        dist_next = np.linalg.norm(next_state - self._goal(next_state))
        bonus = (dist_prev - dist_next) * 0.1

        self._prev_state = next_state
        return reward + bonus * (1 if not done else 0)

    def _potential_shaping(self, state, reward, next_state, done: bool) -> float:
        """Potential-based shaping for guaranteed convergence."""
        phi = self._potential_function(state)
        phi_next = self._potential_function(next_state)
        bonus = (phi_next - phi) * self.gamma

        self._prev_state = next_state
        return reward + bonus

    def _progress_shaping(self, state, reward, next_state, done: bool) -> float:
        """Reward based on progress toward waypoints."""
        if self._prev_state is None:
            self._prev_state = state
            return reward

        progress = np.linalg.norm(next_state - state)
        bonus = progress * 0.01

        self._prev_state = next_state
        return reward + bonus

    def _goal(self, state):
        """Default goal (override per environment)."""
        return np.zeros_like(state)

    def _potential_function(self, state):
        """Default potential function."""
        return -np.linalg.norm(state)


class ReplayBuffer:
    """Simple uniform replay buffer."""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer = collections.deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self._buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int) -> list:
        if len(self._buffer) < batch_size:
            return list(self._buffer)
        return [self._buffer[i] for i in np.random.choice(
            len(self._buffer), batch_size, replace=False)]

    def __len__(self):
        return len(self._buffer)

    def update_priorities(self, indices, td_errors):
        pass  # No-op for uniform replay
