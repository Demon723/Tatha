"""
tatha_realtime.py
=================
Real-time inference server for the Tatha predictive-coding agent system.

Bridges the simulation-based agents to the real world via:
  - WebSocket server for real-time bidirectional communication
  - Observation pipeline (sensor preprocessing, normalization)
  - Action execution layer (command queue, actuator abstraction)
  - Full checkpoint persistence (weights, belief, memory, episode)
  - Continuous learning with experience replay
  - CLI REPL for interactive testing

Protocol (JSON over WebSocket):
  Client sends:  {"type": "observe", "data": [...], "reward": 0.0, "done": false}
  Server replies: {"type": "action", "action": 0, "belief": [...], "efe": [...]}
  Client sends:  {"type": "learn"}
  Client sends:  {"type": "checkpoint", "path": "state.json"}
  Client sends:  {"type": "reset"}

Run:
    python tatha_realtime.py                     # start WebSocket server
    python tatha_realtime.py --repl              # interactive REPL
    python tatha_realtime.py --port 8765         # custom port
"""

from __future__ import annotations
import argparse
import asyncio
import collections
import json
import os
import sys
import time
import threading
import pickle
import numpy as np
from pathlib import Path

# Import utility infrastructure
from tatha_utils import (TathaConfig, setup_logging, get_backend,
                            ObservationValidator, ErrorRecovery,
                            NeuroscienceMetrics, BenchmarkSuite)
from tatha_rest import TathaRESTServer
from tatha_replay import (PrioritizedReplayBuffer, CurriculumScheduler,
                             RewardShaper)
from tatha_ppo import PPOAgent, PPOConfig
from tatha_attention import (TemporalAttentionWrapper, AttentionConfig,
                               PositionalEncoding)
from tatha_web import TathaDashboard
from tatha_benchmarks import BenchmarkSuite as BSuite

# Import core agents
try:
    from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker
    from multiverse_era import (MultiverseState, MultiverseEFEAgent,
                                 MultiverseDirector, MultiverseGridWorld,
                                 branching_operator, apply_branching, decohere)
    from meta_director import MetaParameterSpace, MetaDirector, MetaLearningGame
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker
    from multiverse_era import (MultiverseState, MultiverseEFEAgent,
                                 MultiverseDirector, MultiverseGridWorld,
                                 branching_operator, apply_branching, decohere)
    from meta_director import MetaParameterSpace, MetaDirector, MetaLearningGame

# Setup logging
logger = setup_logging("tatha")


# ==============================================================================
# PART R1: Configuration
# ==============================================================================

class TathaConfig:
    """Runtime configuration with defaults and env-var overrides."""

    DEFAULTS = {
        'grid_size': 10,
        'max_branches': 8,
        'hidden_size': 32,
        'belief_size': 8,
        'learning_rate': 0.005,
        'exploration_coef': 0.2,
        'decoherence_rate': 0.01,
        'dt': 0.1,
        'settle_iters': 10,
        'replay_capacity': 10000,
        'batch_size': 32,
        'checkpoint_dir': './checkpoints',
        'ws_host': '0.0.0.0',
        'ws_port': 8765,
    }

    def __init__(self, **overrides):
        self._vals = dict(self.DEFAULTS)
        # Env var overrides
        for key in self._vals:
            env = f'TATHA_{key.upper()}'
            if env in os.environ:
                try:
                    self._vals[key] = type(self._vals[key])(os.environ[env])
                except (ValueError, TypeError):
                    pass
        self._vals.update(overrides)

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        return self._vals.get(name)

    def to_dict(self):
        return dict(self._vals)


# ==============================================================================
# PART R2: Observation Pipeline
# ==============================================================================

class ObservationPipeline:
    """
    Preprocesses raw sensor data into agent-compatible observation vectors.

    Handles:
      - Variable-size inputs → fixed-size grid flatten
      - Normalization (min-max, z-score)
      - Missing data interpolation
      - Temporal stacking (last N frames)
    """
    def __init__(self, target_size: int, normalize: str = 'minmax',
                 temporal_window: int = 1):
        self.target_size = target_size
        self.normalize = normalize
        self.temporal_window = temporal_window
        self._history = collections.deque(maxlen=temporal_window)
        self._stats = {'min': np.inf, 'max': -np.inf, 'mean': 0, 'std': 1}
        self._n_seen = 0

    def process(self, raw_data: list | np.ndarray) -> np.ndarray:
        """Convert raw observation to fixed-size normalized vector."""
        arr = np.asarray(raw_data, dtype=float).flatten()

        # Pad or truncate to target size
        if arr.size < self.target_size:
            arr = np.pad(arr, (0, self.target_size - arr.size))
        elif arr.size > self.target_size:
            arr = arr[:self.target_size]

        # Replace NaN
        arr = np.nan_to_num(arr, nan=0.0)

        # Update running stats
        self._n_seen += 1
        self._stats['min'] = min(self._stats['min'], arr.min())
        self._stats['max'] = max(self._stats['max'], arr.max())
        self._stats['mean'] = (self._stats['mean'] * (self._n_seen - 1) + arr.mean()) / self._n_seen
        self._stats['std'] = max(1e-8, self._stats['std'])

        # Normalize
        if self.normalize == 'minmax' and self._stats['max'] > self._stats['min']:
            rng = self._stats['max'] - self._stats['min']
            arr = (arr - self._stats['min']) / rng
        elif self.normalize == 'zscore':
            arr = (arr - self._stats['mean']) / self._stats['std']

        # Temporal stacking
        self._history.append(arr)
        if self.temporal_window > 1:
            stacked = np.concatenate(list(self._history))
            # Pad if not enough history yet
            if stacked.size < self.target_size * self.temporal_window:
                stacked = np.pad(stacked, (0, self.target_size * self.temporal_window - stacked.size))
            return stacked[:self.target_size * self.temporal_window]

        return arr

    def reset(self):
        self._history.clear()


# ==============================================================================
# PART R3: Action Execution Layer
# ==============================================================================

class ActionExecutor:
    """
    Executes agent actions in the real world.

    Abstraction over:
      - Command queue (async dispatch)
      - Actuator callbacks (pluggable)
      - Action history logging
      - Safety bounds checking
    """
    def __init__(self, action_names: dict = None):
        self.action_names = action_names or {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self._callbacks = {}
        self._history = collections.deque(maxlen=1000)
        self._safety_bounds = None

    def register_callback(self, action_id: int, callback):
        """Register a function to call when action_id is selected."""
        self._callbacks[action_id] = callback

    def set_safety_bounds(self, bounds_fn):
        """Set a function that validates actions before execution."""
        self._safety_bounds = bounds_fn

    def execute(self, action_id: int, context: dict = None) -> dict:
        """Execute action and return result."""
        context = context or {}

        # Safety check
        if self._safety_bounds and not self._safety_bounds(action_id, context):
            return {'status': 'rejected', 'action': action_id, 'reason': 'safety_bounds'}

        # Execute callback
        result = {'status': 'executed', 'action': action_id, 'name': self.action_names.get(action_id, '?')}
        if action_id in self._callbacks:
            try:
                cb_result = self._callbacks[action_id](context)
                result['callback_result'] = cb_result
            except Exception as e:
                result['status'] = 'error'
                result['error'] = str(e)

        # Log
        result['timestamp'] = time.time()
        self._history.append(result)
        return result

    def get_history(self, n: int = 10) -> list:
        return list(self._history)[-n:]


# ==============================================================================
# PART R4: Experience Replay Buffer
# ==============================================================================

class ReplayBuffer:
    """Experience replay for continuous learning."""
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self._buffer = collections.deque(maxlen=capacity)
        self._priorities = collections.deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done, priority=None):
        experience = (obs, action, reward, next_obs, done)
        self._buffer.append(experience)
        self._priorities.append(priority or abs(reward) + 0.01)

    def sample(self, batch_size: int) -> list:
        if len(self._buffer) == 0:
            return []
        probs = np.array(self._priorities, dtype=float)
        probs = probs / probs.sum()
        indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)),
                                   replace=False, p=probs)
        return [self._buffer[i] for i in indices]

    def __len__(self):
        return len(self._buffer)


# ==============================================================================
# PART R5: Full Checkpoint
# ==============================================================================

class CheckpointManager:
    """Save and load full agent state."""
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict, name: str = 'latest') -> str:
        path = self.checkpoint_dir / f'{name}.pkl'
        # Convert numpy arrays to lists for JSON compatibility
        serializable = {}
        for k, v in state.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, dict):
                serializable[k] = {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                                    for kk, vv in v.items()}
            else:
                serializable[k] = v
        with open(path, 'wb') as f:
            pickle.dump(serializable, f)
        return str(path)

    def load(self, name: str = 'latest') -> dict | None:
        path = self.checkpoint_dir / f'{name}.pkl'
        if not path.exists():
            return None
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # Convert lists back to numpy
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = np.array(v)
            elif isinstance(v, dict):
                data[k] = {kk: np.array(vv) if isinstance(vv, list) else vv
                            for kk, vv in v.items()}
        return data


# ==============================================================================
# PART R6: TathaRealTime Agent (the bridge)
# ==============================================================================

class TathaRealTime:
    """
    Real-time wrapper around Tatha predictive coding agent.

    Manages the full lifecycle:
      1. Initialize from config or checkpoint
      2. Process observations from real sensors
      3. Select actions via EFE minimization
      4. Execute actions in real world
      5. Learn from experience
      6. Checkpoint state
    """
    def __init__(self, config: TathaConfig = None):
        self.config = config or TathaConfig()
        self.logger = logger

        # GPU backend
        self.backend = get_backend(use_gpu=self.config.USE_GPU)

        # Build agent components
        obs_size = self.config.GRID_SIZE ** 2 * 3
        self.perceptual = Tatha(
            input_size=obs_size,
            hidden_size=self.config.HIDDEN_SIZE,
            belief_size=self.config.BELIEF_SIZE,
            lr=self.config.LEARNING_RATE,
        )

        # Attention mechanism (optional)
        if self.config.ATTENTION_ENABLED:
            self.attention = TemporalAttentionWrapper(
                AttentionConfig(
                    num_heads=self.config.ATTENTION_HEADS,
                    attention_dim=self.config.ATTENTION_DIM,
                )
            )
        else:
            self.attention = None

        # Multiverse state
        self.mv = MultiverseState(
            universe_dim=self.config.GRID_SIZE ** 2,
            max_branches=self.config.MAX_BRANCHES,
        )

        # EFE agent
        self.actions = list(range(4))
        d_u = self.config.GRID_SIZE ** 2
        self.rho_pref = np.eye(d_u) / d_u
        self.H_actions = {a: self._random_hermitian(d_u, seed=a) for a in self.actions}
        self.agent = MultiverseEFEAgent(
            self.mv, self.actions, self.H_actions, self.rho_pref,
            perceptual_agent=self.perceptual,
            grid_size=self.config.GRID_SIZE,
        )

        # Director (multiverse-level)
        self.director = MultiverseDirector()

        # MetaDirector (cross-module parameter control)
        self.meta_director = None

        # Pipeline with validation
        self.pipeline = ObservationPipeline(obs_size, temporal_window=1)
        self.validator = ObservationValidator(
            expected_size=obs_size,
        )
        self.error_recovery = ErrorRecovery(max_retries=3)

        # Executor
        self.executor = ActionExecutor()

        # Replay (prioritized if configured)
        if self.config.PRIORITIZED_REPLAY:
            self.replay = PrioritizedReplayBuffer(
                capacity=self.config.REPLAY_CAPACITY,
                alpha=self.config.ALPHA,
                beta_start=self.config.BETA_START,
            )
        else:
            from tatha_replay import ReplayBuffer
            self.replay = ReplayBuffer(self.config.REPLAY_CAPACITY)

        # Curriculum learning
        self.curriculum = CurriculumScheduler() if self.config.CURRICULUM else None

        # Reward shaping
        self.reward_shaper = RewardShaper() if self.config.REWARD_SHAPING else None

        # PPO policy (optional)
        self.ppo = PPOAgent(obs_size, 4) if hasattr(self.config, 'USE_PPO') and self.config.USE_PPO else None

        # Neuroscience metrics
        self.neuro_metrics = NeuroscienceMetrics() if self.config.NEUROSCIENCE_METRICS else None

        # Benchmark
        self.benchmark = BSuite()

        # Checkpoint
        self.checkpoint = CheckpointManager(self.config.CHECKPOINT_DIR)

        # REST server
        self.rest_server = TathaRESTServer(self, port=self.config.REST_PORT)

        # Web dashboard
        self.dashboard = TathaDashboard(agent=self, port=8080)

        # State
        self.episode = 0
        self.total_steps = 0
        self.last_obs = None
        self.last_action = None
        self._running = False
        self._learning_lock = threading.Lock()
        self._async_learning = False

    @staticmethod
    def _random_hermitian(dim: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        H = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        return 0.5 * (H + H.conj().T)

    def reset(self) -> np.ndarray:
        """Reset agent state for new episode."""
        self.mv = MultiverseState(
            universe_dim=self.config.GRID_SIZE ** 2,
            max_branches=self.config.MAX_BRANCHES,
        )
        self.agent.mv = self.mv
        self.pipeline.reset()
        self.episode += 1
        self.total_steps = 0
        self.last_obs = None
        self.last_action = None
        # Return zero observation as seed
        return np.zeros(self.config.GRID_SIZE ** 2 * 3)

    def observe(self, raw_obs: list | np.ndarray, reward: float = 0.0,
                done: bool = False) -> dict:
        """
        Process observation, select action, learn.
        Validates, recovers from errors, with neuroscience metrics.

        Returns dict with action, belief, EFE values, diagnostics.
        """
        # Validate observation
        valid, error = self.validator.validate(raw_obs)
        if not valid:
            self.logger.error(f"Observation validation failed: {error}")
            raw_obs = np.zeros(self.validator.expected_size)
            self.validator.record_violation()

        # Execute with error recovery
        result = self.error_recovery.execute_with_recovery(
            self._observe_internal, raw_obs, reward, done
        )
        return result

    def _observe_internal(self, raw_obs, reward, done) -> dict:
        """Internal observe logic with all modules integrated."""
        # Preprocess
        obs = self.pipeline.process(raw_obs)

        # Apply reward shaping if configured
        if self.reward_shaper and self.last_obs is not None:
            shaped_reward = self.reward_shaper.shape(
                self.last_obs, reward, obs, done
            )
            reward = shaped_reward

        # Store transition
        if self.last_obs is not None:
            td_error = reward - 0.5  # approximate TD error
            self.replay.push(self.last_obs, self.last_action,
                             reward, obs, done, td_error)

        # Apply attention over temporal context if configured
        if self.attention is not None:
            belief_update = self.attention.forward(obs)
            obs = belief_update + obs  # residual

        # Perceive
        self.agent.perceive(obs)
        self.total_steps += 1

        # Director modulation
        self.director.act(self.director.observe(self.mv, reward))

        # Curriculum-based exploration
        explore = True
        if self.curriculum:
            config = self.curriculum.get_config()
            explore = config['exploration_coef'] > 0.05

        # Select action (PPO or EFE)
        if self.ppo and np.random.random() < 0.1:  # epsilon-greedy
            action, _ = self.ppo.policy.act(obs)
        else:
            action = self.agent.act()

        # EFE values for diagnostics
        efe_vals = []
        for a in self.actions:
            try:
                efe_vals.append(float(self.agent.expected_free_energy(a)))
            except Exception:
                efe_vals.append(0.0)

        # Get belief state
        belief = self.agent.get_belief()

        # Execute
        exec_result = self.executor.execute(action, {'obs': obs, 'reward': reward})

        self.last_obs = obs
        self.last_action = action

        # Neuroscience metrics
        if self.neuro_metrics:
            fe = sum(efe_vals)
            self.neuro_metrics.record_free_energy(fe)
            self.neuro_metrics.record_prediction_error(abs(reward - 0.5))
            self.neuro_metrics.record_precision(np.linalg.norm(belief))

        # Learning
        if self.total_steps % self.config.BATCH_SIZE == 0:
            self._learn_async()

        return {
            'action': int(action),
            'action_name': self.executor.action_names.get(action, '?'),
            'belief': belief.tolist(),
            'efe': efe_vals,
            'branches': self.mv.num_branches,
            'entropy': self.mv.entanglement_entropy(),
            'episode': self.episode,
            'step': self.total_steps,
            'exec_status': exec_result.get('status'),
        }

    def _learn_async(self):
        """Async learning with thread safety."""
        if not self._learning_lock.acquire(blocking=False):
            return  # skip if another learning thread is active
        try:
            batch_size = self.config.BATCH_SIZE
            if hasattr(self.replay, 'sample'):
                if self.config.PRIORITIZED_REPLAY:
                    batch, indices, weights = self.replay.sample(batch_size)
                else:
                    batch = self.replay.sample(batch_size)
                    indices = None
                    weights = None
                if batch:
                    self.agent.learn_from_replay(batch_size=batch_size)
        except Exception as e:
            self.logger.error(f"Async learning error: {e}")
        finally:
            self._learning_lock.release()

    def learn_from_replay(self, batch_size: int = None):
        """Sample from replay buffer and update perceptual agent."""
        batch_size = batch_size or self.config.BATCH_SIZE
        batch = self.replay.sample(batch_size)
        if not batch:
            return

        for experience in batch:
            obs, action, reward, next_obs, done = experience
            self.perceptual.perceive(obs)
            self.perceptual.settle(n_iter=3)
            self.perceptual.learn()

        # Update priorities if prioritized replay
        if self.config.PRIORITIZED_REPLAY and hasattr(self.replay, 'update_priorities'):
            td_errors = np.array([r + 0.5 for _, _, r, _, _ in batch])
            self.replay.update_priorities(np.arange(len(batch)), td_errors)

    def learn_from_replay(self, batch_size: int = None):
        """Sample from replay buffer and update perceptual agent."""
        batch_size = batch_size or self.config.BATCH_SIZE
        batch = self.replay.sample(batch_size)
        if not batch:
            return

        for obs, action, reward, next_obs, done in batch:
            # Construct prediction target
            target = next_obs.copy()
            if done:
                target *= 0.5  # decay on episode end

            # Teach perceptual agent
            self.perceptual.perceive(obs)
            self.perceptual.settle(n_iter=3)
            self.perceptual.learn()

    def checkpoint_save(self, name: str = None) -> str:
        """Save full agent state."""
        name = name or f'ep{self.episode}_step{self.total_steps}'
        state = {
            'perceptual_weights': {
                'input_W_topdown': self.perceptual.input_layer.W_topdown,
                'input_W_recurrent': self.perceptual.input_layer.W_recurrent,
                'input_bias': self.perceptual.input_layer.bias,
                'hidden_W_topdown': self.perceptual.hidden.W_topdown,
                'hidden_W_recurrent': self.perceptual.hidden.W_recurrent,
                'hidden_bias': self.perceptual.hidden.bias,
                'belief_W_topdown': self.perceptual.belief.W_topdown,
                'belief_W_recurrent': self.perceptual.belief.W_recurrent,
                'belief_bias': self.perceptual.belief.bias,
            },
            'director_params': self.director.params.tolist(),
            'episode': self.episode,
            'total_steps': self.total_steps,
            'config': self.config.to_dict(),
            'replay_size': len(self.replay),
        }
        return self.checkpoint.save(state, name)

    def checkpoint_load(self, name: str = 'latest') -> bool:
        """Load agent state from checkpoint."""
        state = self.checkpoint.load(name)
        if state is None:
            return False

        pw = state['perceptual_weights']
        self.perceptual.input_layer.W_topdown = pw['input_W_topdown']
        self.perceptual.input_layer.W_recurrent = pw['input_W_recurrent']
        self.perceptual.input_layer.bias = pw['input_bias']
        self.perceptual.hidden.W_topdown = pw['hidden_W_topdown']
        self.perceptual.hidden.W_recurrent = pw['hidden_W_recurrent']
        self.perceptual.hidden.bias = pw['hidden_bias']
        self.perceptual.belief.W_topdown = pw['belief_W_topdown']
        self.perceptual.belief.W_recurrent = pw['belief_W_recurrent']
        self.perceptual.belief.bias = pw['belief_bias']
        self.director.params = np.array(state['director_params'])
        self.episode = state['episode']
        self.total_steps = state['total_steps']
        return True

    def get_diagnostics(self) -> dict:
        """Return current agent diagnostics."""
        return {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'branches': self.mv.num_branches,
            'entropy': self.mv.entanglement_entropy(),
            'replay_size': len(self.replay),
            'belief_norm': float(np.linalg.norm(self.perceptual.belief.activity)),
            'director_params': self.director.params.tolist(),
            'meta_director_active': self.meta_director is not None,
        }

    def init_meta_director(self, n_params: int = 10, hidden_size: int = 32,
                           belief_size: int = 8, lr: float = 0.01):
        """Initialize MetaDirector for cross-module parameter control."""
        self.meta_director = MetaDirector(
            n_params=n_params, hidden_size=hidden_size,
            belief_size=belief_size, lr=lr,
        )
        return self.meta_director

    def meta_director_observe(self, reward: float = 0.0) -> dict:
        """Let MetaDirector observe current state and modulate parameters."""
        if self.meta_director is None:
            return {'error': 'meta_director not initialized'}

        # Encode current game state
        game_state = self.meta_director.encode_game_state(
            realm=None,  # no realm, use multiverse state
            red_score=reward * self.total_steps,
            blue_score=0,
            rounds=self.total_steps,
        )

        # Override with multiverse-specific features
        game_state[0] = self.mv.num_branches / self.config.MAX_BRANCHES
        game_state[1] = self.mv.entanglement_entropy()
        game_state[2] = reward
        game_state[3] = min(self.total_steps / 100, 1.0)
        game_state[4] = self.episode / 100

        # Select parameters
        params_vec = self.meta_director.select_parameters(game_state, explore=True)
        params_phys = MetaParameterSpace.denormalize(params_vec)

        # Apply to multiverse director
        self.director.params[0] = params_phys.get('red_entanglement', 0.1)
        self.director.params[1] = params_phys.get('decoherence_rate', 0.01)
        self.director.params[2] = params_phys.get('coherence_bonus_red', 0.5)
        self.director.params[3] = params_phys.get('red_exploration', 0.3)
        self.director.params[4] = params_phys.get('blue_exploration', 0.3)
        self.director.params[5] = params_phys.get('barrier_decay', 0.05)
        self.director.params[6] = params_phys.get('barrier_max', 5.0)
        self.director.params[7] = params_phys.get('gravity_scale', 0.5)

        return {
            'params': params_phys,
            'params_vec': params_vec.tolist(),
        }

    def meta_director_learn(self, outcome: float):
        """Let MetaDirector learn from episode outcome."""
        if self.meta_director is None:
            return
        game_state = np.zeros(5)
        game_state[0] = self.mv.num_branches / self.config.MAX_BRANCHES
        game_state[1] = self.mv.entanglement_entropy()
        game_state[2] = outcome
        game_state[3] = min(self.total_steps / 100, 1.0)
        game_state[4] = self.episode / 100
        self.meta_director.learn_from_outcome(
            self.meta_director.current_params, outcome, game_state
        )


# ==============================================================================
# PART R7: WebSocket Server
# ==============================================================================

class TathaWSServer:
    """WebSocket server for real-time agent communication."""
    def __init__(self, agent: TathaRealTime, host: str = '0.0.0.0', port: int = 8765):
        self.agent = agent
        self.host = host
        self.port = port
        self._clients = set()

    async def handler(self, websocket, path=None):
        """Handle a single WebSocket client connection."""
        self._clients.add(websocket)
        print(f"[WS] Client connected ({len(self._clients)} total)")
        try:
            async for message in websocket:
                try:
                    msg = json.loads(message)
                    response = await self._process_message(msg)
                    await websocket.send(json.dumps(response))
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({'type': 'error', 'error': 'invalid JSON'}))
                except Exception as e:
                    await websocket.send(json.dumps({'type': 'error', 'error': str(e)}))
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            print(f"[WS] Client disconnected ({len(self._clients)} total)")

    async def _process_message(self, msg: dict) -> dict:
        msg_type = msg.get('type', '')

        if msg_type == 'observe':
            data = msg.get('data', [])
            reward = msg.get('reward', 0.0)
            done = msg.get('done', False)
            result = self.agent.observe(data, reward, done)
            result['type'] = 'action'
            return result

        elif msg_type == 'learn':
            self.agent.learn_from_replay()
            return {'type': 'learned', 'replay_size': len(self.agent.replay)}

        elif msg_type == 'reset':
            self.agent.reset()
            return {'type': 'reset', 'episode': self.agent.episode}

        elif msg_type == 'checkpoint':
            path = self.agent.checkpoint_save(msg.get('name'))
            return {'type': 'checkpoint', 'path': path}

        elif msg_type == 'load':
            ok = self.agent.checkpoint_load(msg.get('name', 'latest'))
            return {'type': 'loaded', 'success': ok}

        elif msg_type == 'diagnostics':
            diag = self.agent.get_diagnostics()
            diag['type'] = 'diagnostics'
            return diag

        elif msg_type == 'history':
            return {'type': 'history', 'actions': self.agent.executor.get_history(20)}

        else:
            return {'type': 'error', 'error': f'unknown message type: {msg_type}'}

    def start(self):
        """Start the WebSocket server (blocking)."""
        try:
            import websockets
        except ImportError:
            print("ERROR: pip install websockets")
            sys.exit(1)

        print(f"[WS] Starting Tatha agent on ws://{self.host}:{self.port}")
        print(f"[WS] Protocol: observe, learn, reset, checkpoint, load, diagnostics, history")

        async def run():
            async with websockets.serve(self.handler, self.host, self.port):
                await asyncio.Future()  # run forever

        asyncio.run(run())


# ==============================================================================
# PART R8: CLI REPL
# ==============================================================================

class TathaREPL:
    """Interactive command-line interface for testing the agent."""
    def __init__(self, agent: TathaRealTime):
        self.agent = agent

    def run(self):
        print("=" * 60)
        print("TATHA REAL-TIME AGENT — Interactive REPL")
        print("=" * 60)
        print("Commands:")
        print("  observe <data...>   Send observation, get action")
        print("  random_obs          Generate random observation")
        print("  learn               Learn from replay buffer")
        print("  reset               Reset agent for new episode")
        print("  save [name]         Checkpoint agent state")
        print("  load [name]         Load checkpoint")
        print("  diag                Show diagnostics")
        print("  history             Show action history")
        print("  quit                Exit")
        print("=" * 60)

        while True:
            try:
                line = input("tatha> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not line:
                continue

            parts = line.split()
            cmd = parts[0].lower()

            if cmd == 'quit' or cmd == 'exit':
                break

            elif cmd == 'observe':
                data = [float(x) for x in parts[1:]] if len(parts) > 1 else \
                       np.random.randn(self.agent.config.grid_size ** 2).tolist()
                result = self.agent.observe(data)
                print(f"  action={result['action']} ({result['action_name']})  "
                      f"branches={result['branches']}  entropy={result['entropy']:.4f}  "
                      f"efe={[f'{v:.3f}' for v in result['efe']]}")

            elif cmd == 'random_obs':
                data = np.random.randn(self.agent.config.grid_size ** 2).tolist()
                result = self.agent.observe(data)
                print(f"  action={result['action']} ({result['action_name']})  "
                      f"branches={result['branches']}  entropy={result['entropy']:.4f}")

            elif cmd == 'learn':
                self.agent.learn_from_replay()
                print(f"  Learned. Replay size: {len(self.agent.replay)}")

            elif cmd == 'reset':
                self.agent.reset()
                print(f"  Reset. Episode: {self.agent.episode}")

            elif cmd == 'save':
                name = parts[1] if len(parts) > 1 else None
                path = self.agent.checkpoint_save(name)
                print(f"  Saved to {path}")

            elif cmd == 'load':
                name = parts[1] if len(parts) > 1 else 'latest'
                ok = self.agent.checkpoint_load(name)
                print(f"  Loaded: {ok}")

            elif cmd == 'diag':
                d = self.agent.get_diagnostics()
                for k, v in d.items():
                    if isinstance(v, list):
                        print(f"  {k}: [{len(v)} items]")
                    else:
                        print(f"  {k}: {v}")

            elif cmd == 'history':
                for h in self.agent.executor.get_history(10):
                    print(f"  {h.get('name', '?')} @ {h.get('timestamp', 0):.0f} — {h.get('status')}")

            elif cmd == 'benchmark':
                self.agent.benchmark.run_all({}, {'agent': self.agent})
                self.agent.benchmark.print_report()

            elif cmd == 'attention':
                if self.agent.attention:
                    print("  Attention: enabled")
                else:
                    print("  Attention: disabled")
                print(f"  Curriculum: {'enabled' if self.agent.curriculum else 'disabled'}")
                print(f"  Prioritized Replay: {'enabled' if self.agent.config.PRIORITIZED_REPLAY else 'disabled'}")
                print(f"  Reward Shaping: {'enabled' if self.agent.reward_shaper else 'disabled'}")
                print(f"  GPU: {'enabled' if self.agent.backend.use_gpu else 'disabled'}")
                print(f"  PPO: {'enabled' if self.agent.ppo else 'disabled'}")

            elif cmd == 'neuro':
                if self.agent.neuro_metrics:
                    print(f"  {self.agent.neuro_metrics.summary()}")
                else:
                    print("  Neuroscience metrics: disabled")

            elif cmd == 'dashboard':
                self.agent.dashboard.start_server(None, self.agent)

            elif cmd == 'rest':
                self.agent.rest_server.run()

            else:
                print(f"  Unknown command: {cmd}")


# ==============================================================================
# PART R9: Entry point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tatha Real-Time Agent Server")
    parser.add_argument('--repl', action='store_true', help='Interactive REPL mode')
    parser.add_argument('--host', default='0.0.0.0', help='WebSocket host')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket port')
    parser.add_argument('--rest-port', type=int, default=8766, help='REST API port')
    parser.add_argument('--grid', type=int, default=10, help='Grid size')
    parser.add_argument('--load', type=str, default=None, help='Checkpoint to load')
    parser.add_argument('--config', type=str, default=None, help='YAML config file')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--gpu', action='store_true', help='Enable GPU backend')
    parser.add_argument('--attention', action='store_true', help='Enable attention mechanism')
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--prioritized', action='store_true', help='Enable prioritized replay')
    parser.add_argument('--reward-shaping', action='store_true', help='Enable reward shaping')
    parser.add_argument('--neuro', action='store_true', help='Enable neuroscience metrics')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--rest', action='store_true', help='Start REST API server')
    args = parser.parse_args()

    # Load config from YAML if provided
    if args.config:
        config = TathaConfig.from_yaml(args.config)
    else:
        config = TathaConfig.from_env()

    # CLI overrides
    config.GRID_SIZE = args.grid
    config.USE_GPU = args.gpu
    config.ATTENTION_ENABLED = args.attention
    config.CURRICULUM = args.curriculum
    config.PRIORITIZED_REPLAY = args.prioritized
    config.REWARD_SHAPING = args.reward_shaping
    config.NEUROSCIENCE_METRICS = args.neuro
    config.REST_PORT = args.rest_port

    agent = TathaRealTime(config)

    if args.load:
        ok = agent.checkpoint_load(args.load)
        if ok:
            print(f"Loaded checkpoint: {args.load}")
        else:
            print(f"No checkpoint found: {args.load}, starting fresh")

    if args.benchmark:
        agent.benchmark.print_report()
    elif args.rest:
        agent.rest_server.run()
    elif args.repl:
        repl = TathaREPL(agent)
        repl.run()
    else:
        server = TathaWSServer(agent, host=args.host, port=args.port)
        server.start()
