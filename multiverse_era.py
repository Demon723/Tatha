"""
multiverse_era.py
=================
Multiverse Era extension for the Tatha unified predictive-coding system.
Implements a true joint multiverse Hilbert space:

    |Psi> = sum_u c_u |u> ⊗ |psi_u>

Includes:
  - MultiverseState        (joint Hilbert space, evolution, entropy)
  - branching operators    (real 2x2 unitary branching + decoherence)
  - density-matrix EFE     (quantum KL, von Neumann entropy)
  - MultiverseEFEAgent     (action selection via quantum EFE + Tatha perceptual)
  - MultiverseGridWorld    (grid environment bridging quantum_realm patterns)
  - MultiverseDirector     (learned meta-controller, integrates with MetaParameterSpace)
  - MultiverseVisualizer   (visualization matching project conventions)
  - run()                  (end-to-end driver)
  - run_tests()            (built-in unit tests)

Integrates with: unified_pc_space.py, quantum_realm.py, quantum_swarm.py

Run:
    python multiverse_era.py
    python multiverse_era.py --test
"""

from __future__ import annotations
import argparse
import warnings
import numpy as np
from scipy.linalg import expm, logm
from scipy.sparse import kron, identity, csr_matrix
from scipy.sparse.linalg import expm_multiply

# Import utility infrastructure
try:
    from tatha_utils import (TathaConfig, setup_logging, get_backend,
                                 NeuroscienceMetrics)
    from tatha_replay import (CurriculumScheduler, RewardShaper,
                                  PrioritizedReplayBuffer)
    from tatha_ppo import PPOAgent, PPOConfig
    from tatha_attention import (TemporalAttentionWrapper, PositionalEncoding,
                                  AttentionConfig)
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tatha_utils import (TathaConfig, setup_logging, get_backend,
                                 NeuroscienceMetrics)
    from tatha_replay import (CurriculumScheduler, RewardShaper,
                                  PrioritizedReplayBuffer)
    from tatha_ppo import PPOAgent, PPOConfig
    from tatha_attention import (TemporalAttentionWrapper, PositionalEncoding,
                                  AttentionConfig)

# Import core predictive coding agents from unified system
try:
    from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker

warnings.filterwarnings("ignore", category=RuntimeWarning)
logger = setup_logging("multiverse_era")


# ---------------------------------------------------------------------------
# 1. Multiverse state: joint Hilbert space  branch_space ⊗ universe_space
# ---------------------------------------------------------------------------
class MultiverseState:
    def __init__(self, universe_dim: int, max_branches: int = 16):
        self.d_u = int(universe_dim)
        self.max_branches = int(max_branches)
        self.num_branches = 1

        self.c = np.zeros(self.max_branches, dtype=complex)
        self.c[0] = 1.0

        self.psi = np.zeros((self.max_branches, self.d_u), dtype=complex)
        self.psi[0, 0] = 1.0

        self.H_branch = np.zeros((self.max_branches, self.max_branches),
                                 dtype=complex)
        self.H_universe = np.zeros((self.d_u, self.d_u), dtype=complex)

    # -- representation -----------------------------------------------------
    def joint_vector(self) -> np.ndarray:
        n = self.num_branches
        joint = np.zeros(n * self.d_u, dtype=complex)
        for u in range(n):
            joint[u * self.d_u:(u + 1) * self.d_u] = self.c[u] * self.psi[u]
        return joint

    def set_joint_vector(self, vec: np.ndarray) -> None:
        vec = vec.reshape(self.num_branches, self.d_u)
        for u in range(self.num_branches):
            norm_u = np.linalg.norm(vec[u])
            if norm_u > 1e-12:
                self.psi[u] = vec[u] / norm_u
                self.c[u] = norm_u
            else:
                self.psi[u] = vec[u]
                self.c[u] = 0.0
        self.normalize()

    def normalize(self) -> None:
        for u in range(self.num_branches):
            nu = np.linalg.norm(self.psi[u])
            if nu > 1e-12:
                self.psi[u] /= nu
        nc = np.linalg.norm(self.c[:self.num_branches])
        if nc > 1e-12:
            self.c[:self.num_branches] /= nc

    # -- observables --------------------------------------------------------
    def branch_probs(self) -> np.ndarray:
        p = np.abs(self.c[:self.num_branches]) ** 2
        s = p.sum()
        return p / s if s > 1e-12 else p

    def reduced_universe(self, u: int) -> np.ndarray:
        psi_u = self.psi[u]
        return np.outer(psi_u, psi_u.conj())

    def entanglement_entropy(self) -> float:
        p = np.clip(self.branch_probs(), 1e-12, 1.0)
        return float(-np.sum(p * np.log(p)))

    # -- evolution ----------------------------------------------------------
    def evolve(self, dt: float,
               H_branch: np.ndarray | None = None,
               H_universe: np.ndarray | None = None) -> None:
        if H_branch is not None:
            self.H_branch = H_branch
        if H_universe is not None:
            self.H_universe = H_universe

        H_joint = (np.kron(self.H_branch[:self.num_branches, :self.num_branches],
                           np.eye(self.d_u))
                   + np.kron(np.eye(self.num_branches), self.H_universe))
        hnorm = np.linalg.norm(H_joint) * dt
        vec = self.joint_vector()
        if not np.isfinite(hnorm) or hnorm > 50:
            pass
        else:
            U = expm(-1j * H_joint * dt)
            vec = U @ vec
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        self.set_joint_vector(vec)

    def sparse_evolve(self, dt: float) -> None:
        H_joint = (kron(csr_matrix(self.H_branch[:self.num_branches,
                                                 :self.num_branches]),
                        identity(self.d_u))
                   + kron(identity(self.num_branches),
                          csr_matrix(self.H_universe)))
        vec = self.joint_vector()
        vec = expm_multiply(-1j * dt * H_joint, vec)
        self.set_joint_vector(vec)


# ---------------------------------------------------------------------------
# 2. Branching and decoherence
# ---------------------------------------------------------------------------
def branching_operator(theta: float = np.pi / 4,
                       phi: float = 0.0) -> np.ndarray:
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.exp(-1j * phi)],
        [np.sin(theta) * np.exp(1j * phi),  np.cos(theta)],
    ], dtype=complex)


def apply_branching(mv: MultiverseState, u_index: int,
                    theta: float = np.pi / 4, phi: float = 0.0) -> bool:
    if mv.num_branches >= mv.max_branches:
        return False

    B = branching_operator(theta, phi)
    old_n = mv.num_branches
    new_n = old_n + 1

    new_c = np.zeros_like(mv.c)
    new_psi = np.zeros_like(mv.psi)

    for i in range(old_n):
        if i == u_index:
            new_c[i]       += B[0, 0] * mv.c[i]
            new_c[new_n - 1] += B[1, 0] * mv.c[i]
            new_psi[i]        = mv.psi[i]
            new_psi[new_n - 1] = mv.psi[i].copy()
        else:
            new_c[i] += mv.c[i]
            new_psi[i] = mv.psi[i]

    mv.c = new_c
    mv.psi = new_psi
    mv.num_branches = new_n
    mv.normalize()
    return True


def decohere(mv: MultiverseState, gamma: float = 0.01) -> None:
    mv.c[:mv.num_branches] *= np.exp(-gamma)
    mv.normalize()


# ---------------------------------------------------------------------------
# 3. Density-matrix EFE
# ---------------------------------------------------------------------------
def von_neumann(rho: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(rho)
    vals = np.clip(vals.real, 1e-12, 1.0)
    return float(-np.sum(vals * np.log(vals)))


def quantum_kl(rho: np.ndarray, sigma: np.ndarray) -> float:
    d = rho.shape[0]
    eps = 1e-12 * np.eye(d)
    val = np.trace(rho @ (logm(rho + eps) - logm(sigma + eps)))
    return float(np.real(val))


def multiverse_efe(mv: MultiverseState, H_action: np.ndarray,
                   rho_pref: np.ndarray, info_gain: float = 0.0) -> float:
    vec = mv.joint_vector()
    if not np.all(np.isfinite(vec)):
        return 0.0
    rho = np.outer(vec, vec.conj())
    H_joint = np.kron(np.eye(mv.num_branches), H_action)

    hnorm = np.linalg.norm(H_joint)
    if not np.isfinite(hnorm) or hnorm > 20:
        d_u = mv.d_u
        rho_u = np.zeros((d_u, d_u), dtype=complex)
        for u in range(mv.num_branches):
            rho_u += mv.reduced_universe(u)
        tr = np.real(np.trace(rho_u))
        if tr > 1e-12:
            rho_u /= tr
        kl = quantum_kl(rho_u, rho_pref)
        energy = float(np.real(np.trace(rho @ H_joint)))
        return energy + kl - info_gain

    energy = float(np.real(np.trace(rho @ H_joint)))

    U = expm(-1j * H_joint)
    rho_a = U @ rho @ U.conj().T

    d_u = mv.d_u
    rho_u = np.zeros((d_u, d_u), dtype=complex)
    for u in range(mv.num_branches):
        s = u * d_u
        rho_u += rho_a[s:s + d_u, s:s + d_u]
    tr = np.real(np.trace(rho_u))
    if tr > 1e-12:
        rho_u /= tr

    kl = quantum_kl(rho_u, rho_pref)
    return energy + kl - info_gain


# ---------------------------------------------------------------------------
# 4. Multiverse EFE agent (integrates Tatha perceptual coding)
# ---------------------------------------------------------------------------
class MultiverseEFEAgent:
    """
    Active inference agent for the multiverse grid world.

    Uses a Tatha perceptual agent for predictive coding alongside
    quantum EFE over the multiverse Hilbert space.
    """
    def __init__(self, mv: MultiverseState, action_space,
                 H_actions: dict, rho_pref: np.ndarray,
                 perceptual_agent=None, grid_size: int = 10):
        self.mv = mv
        self.actions = list(action_space)
        self.H_actions = H_actions
        self.rho_pref = rho_pref
        self.grid_size = grid_size

        # Tatha perceptual agent for predictive coding
        if perceptual_agent is not None:
            self.perceptual = perceptual_agent
        else:
            obs_dim = grid_size * grid_size * 3  # prob + cos(phase) + sin(phase)
            self.perceptual = Tatha(input_size=obs_dim, hidden_size=32,
                                    belief_size=8, lr=0.005)
        self.tracker = CoordinateTracker(grid_size)

    def expected_free_energy(self, a) -> float:
        return multiverse_efe(self.mv, self.H_actions[a], self.rho_pref)

    def act(self):
        Gs = [self.expected_free_energy(a) for a in self.actions]
        return self.actions[int(np.argmin(Gs))]

    def act_softmax(self, beta: float = 1.0):
        Gs = np.array([self.expected_free_energy(a) for a in self.actions])
        p = np.exp(-beta * (Gs - Gs.min()))
        p /= p.sum()
        return np.random.choice(self.actions, p=p)

    def perceive(self, obs: np.ndarray):
        self.perceptual.perceive(obs)
        self.perceptual.settle(n_iter=10)

    def learn(self):
        self.perceptual.learn()

    def get_belief(self) -> np.ndarray:
        return self.perceptual.belief.activity.copy()


# ---------------------------------------------------------------------------
# 5. Multiverse director
# ---------------------------------------------------------------------------
class MultiverseDirector:
    PARAM_NAMES = (
        "branch_rate", "decoherence", "entanglement",
        "exploration_red", "exploration_blue", "barrier_decay",
        "max_barriers", "gravity", "measurement_bias", "meta_lr",
    )

    def __init__(self, n_params: int = 10, lr: float = 1e-2):
        self.params = np.array([
            0.10, 0.010, 0.50, 0.30, 0.30,
            0.05, 5.00, 0.50, 0.50, 0.01,
        ])
        self.lr = lr
        self.history: list[np.ndarray] = []

    def observe(self, mv: MultiverseState, reward: float) -> np.ndarray:
        return np.array([mv.num_branches,
                         mv.entanglement_entropy(),
                         reward], dtype=float)

    def act(self, obs: np.ndarray) -> np.ndarray:
        _, entropy, reward = obs
        grad = np.zeros_like(self.params)
        grad[0] =  0.010 * (reward - 0.5)
        grad[1] = -0.001 * entropy
        grad[2] =  0.010 * entropy
        grad[3] =  0.010 * (reward - 0.5)
        grad[4] = -0.010 * (reward - 0.5)
        self.params += self.lr * grad
        self.params[0] = np.clip(self.params[0], 0.00, 0.50)
        self.params[1] = np.clip(self.params[1], 0.00, 0.20)
        self.params[2] = np.clip(self.params[2], 0.00, 1.00)
        self.history.append(self.params.copy())
        return self.params.copy()

    def save(self, path: str) -> None:
        np.save(path, self.params)

    def load(self, path: str) -> None:
        self.params = np.load(path)

    def to_meta_params(self) -> dict:
        """Convert director params to MetaParameterSpace-compatible dict."""
        return {
            'red_entanglement': float(self.params[2]),
            'blue_entanglement': float(self.params[2] * 0.6),
            'red_exploration': float(self.params[3]),
            'blue_exploration': float(self.params[4]),
            'barrier_decay': float(self.params[5]),
            'barrier_max': int(self.params[6]),
            'gravity_scale': float(self.params[7]),
            'decoherence_rate': float(self.params[1]),
            'coherence_bonus_red': float(self.params[2]),
            'intercept_bonus_blue': float(self.params[8]),
        }


# ---------------------------------------------------------------------------
# 6. Multiverse grid world (bridges quantum_realm patterns)
# ---------------------------------------------------------------------------
class MultiverseGridWorld:
    """
    A grid environment where each cell can exist in multiple universe branches.

    Combines quantum_realm's wavepacket approach with multiverse branching:
      - Agent has a probability field over grid positions
      - Branching creates parallel universe copies
      - Decoherence reduces branch amplitudes
      - Waypoints trigger measurements that collapse branches
    """
    def __init__(self, size: int = 10, max_branches: int = 8,
                 dt: float = 0.1, decoherence_rate: float = 0.01):
        self.size = size
        self.max_branches = max_branches
        self.dt = dt
        self.decoherence_rate = decoherence_rate

        self.actions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        # Multiverse state
        self.mv = MultiverseState(universe_dim=size * size,
                                  max_branches=max_branches)

        # Environment features
        self.waypoints = self._generate_waypoints()
        self.waypoint_idx = 0

        # Telemetry
        self.telemetry = {
            'branch_history': [],
            'entropy_history': [],
            'reward_history': [],
            'waypoint_reaches': [],
        }

    def _generate_waypoints(self):
        ws = self.size
        wps = [[ws - 1, ws - 1], [ws // 2, ws // 2], [0, ws - 1], [ws - 1, 0]]
        np.random.shuffle(wps)
        return wps

    def reset(self):
        self.mv = MultiverseState(universe_dim=self.size * self.size,
                                  max_branches=self.max_branches)
        self.waypoint_idx = 0
        self.telemetry = {
            'branch_history': [],
            'entropy_history': [],
            'reward_history': [],
            'waypoint_reaches': [],
        }
        return self.observe()

    def observe(self) -> np.ndarray:
        """Observation = probability density per branch + phase + branch amplitudes."""
        obs = np.zeros(self.size * self.size * 3)
        for u in range(self.mv.num_branches):
            probs = np.abs(self.mv.psi[u]) ** 2
            offset = u * self.size * self.size
            # Probability density
            obs[:self.size * self.size] += np.abs(self.mv.c[u]) ** 2 * probs
            # Phase (cos/sin)
            obs[self.size * self.size:2 * self.size * self.size] += \
                np.abs(self.mv.c[u]) ** 2 * np.cos(np.angle(self.mv.psi[u]))
            obs[2 * self.size * self.size:] += \
                np.abs(self.mv.c[u]) ** 2 * np.sin(np.angle(self.mv.psi[u]))
        return obs

    def step(self, action: int):
        """Step: evolve, branch, decohere, check waypoints."""
        # Build action Hamiltonian (displacement in grid space)
        H_action = np.zeros((self.size * self.size, self.size * self.size),
                            dtype=complex)
        d = self.size
        for r in range(d):
            for c in range(d):
                idx = r * d + c
                if action == 0 and r > 0:      # UP
                    H_action[idx, (r - 1) * d + c] = 1.0
                elif action == 1 and r < d - 1:  # DOWN
                    H_action[idx, (r + 1) * d + c] = 1.0
                elif action == 2 and c > 0:      # LEFT
                    H_action[idx, r * d + (c - 1)] = 1.0
                elif action == 3 and c < d - 1:  # RIGHT
                    H_action[idx, r * d + (c + 1)] = 1.0

        # Evolve
        self.mv.evolve(dt=self.dt, H_universe=H_action)

        # Branching
        if self.mv.num_branches < self.max_branches and np.random.random() < 0.1:
            target = np.random.randint(0, self.mv.num_branches)
            apply_branching(self.mv, target, theta=np.pi / 4)

        # Decoherence
        decohere(self.mv, gamma=self.decoherence_rate)

        # Check waypoint measurement
        reward = 0.0
        done = False
        if self.waypoint_idx < len(self.waypoints):
            wr, wc = self.waypoints[self.waypoint_idx]
            prob_at_wp = 0.0
            for u in range(self.mv.num_branches):
                prob_at_wp += np.abs(self.mv.c[u]) ** 2 * \
                    np.abs(self.mv.psi[u, wr * self.size + wc]) ** 2
            if prob_at_wp > 0.3:
                reward = 1.0
                self.waypoint_idx += 1
                self.telemetry['waypoint_reaches'].append({
                    'step': len(self.telemetry['branch_history']),
                    'pos': [wr, wc],
                    'prob': prob_at_wp,
                })
                if self.waypoint_idx >= len(self.waypoints):
                    done = True

        # Record telemetry
        self.telemetry['branch_history'].append(self.mv.num_branches)
        self.telemetry['entropy_history'].append(self.mv.entanglement_entropy())
        self.telemetry['reward_history'].append(reward)

        return self.observe(), reward, done

    def get_classical_position(self) -> list:
        """Most probable position across all branches."""
        probs = np.zeros(self.size * self.size)
        for u in range(self.mv.num_branches):
            probs += np.abs(self.mv.c[u]) ** 2 * np.abs(self.mv.psi[u]) ** 2
        max_idx = np.argmax(probs)
        return [max_idx // self.size, max_idx % self.size]

    def print_realm(self):
        """ASCII visualization of probability density."""
        probs = np.zeros((self.size, self.size))
        for u in range(self.mv.num_branches):
            p_u = np.abs(self.mv.c[u]) ** 2 * np.abs(self.mv.psi[u]) ** 2
            probs += p_u.reshape(self.size, self.size)
        max_prob = np.max(probs)
        chars = [' ', '.', 'o', 'O', '@', '#', '*']
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if self.waypoint_idx < len(self.waypoints) and \
                   [r, c] == self.waypoints[self.waypoint_idx]:
                    row += "W"
                else:
                    p = probs[r, c] / max_prob if max_prob > 0 else 0
                    idx = min(int(p * (len(chars) - 1)), len(chars) - 1)
                    row += chars[idx]
            print("  " + row)


# ---------------------------------------------------------------------------
# 7. Multiverse visualizer (matches project conventions)
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MULTIVERSE_VIZ = True
except ImportError:
    MULTIVERSE_VIZ = False


class MultiverseVisualizer:
    """Visualization tools for multiverse dynamics."""

    def plot_branch_evolution(self, world: MultiverseGridWorld,
                              save_path=None):
        """Plot branch count and entropy over time."""
        if not MULTIVERSE_VIZ or len(world.telemetry['branch_history']) == 0:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        ax = axes[0]
        ax.plot(world.telemetry['branch_history'], 'b-', linewidth=2)
        ax.set_ylabel('Number of Branches')
        ax.set_title('Multiverse Branch Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(world.telemetry['entropy_history'], 'r-', linewidth=2)
        ax.set_ylabel('Entanglement Entropy')
        ax.set_xlabel('Step')
        ax.set_title('Inter-Branch Entropy', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    def plot_probability_field(self, world: MultiverseGridWorld,
                               save_path=None):
        """Plot aggregate probability density across branches."""
        if not MULTIVERSE_VIZ:
            return None

        probs = np.zeros((world.size, world.size))
        for u in range(world.mv.num_branches):
            p_u = np.abs(world.mv.c[u]) ** 2 * np.abs(world.mv.psi[u]) ** 2
            probs += p_u.reshape(world.size, world.size)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(probs, cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title('Aggregate Probability Density', fontweight='bold')

        # Overlay waypoints
        for i, wp in enumerate(world.waypoints):
            color = 'green' if i < world.waypoint_idx else 'yellow'
            ax.plot(wp[1], wp[0], 'o', color=color, markersize=10,
                    markeredgecolor='black')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig


# ---------------------------------------------------------------------------
# 8. Driver
# ---------------------------------------------------------------------------
def _random_hermitian(dim: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return 0.5 * (H + H.conj().T)


def run(steps: int = 200, grid_size: int = 10, max_branches: int = 8,
        seed: int = 0, verbose: bool = True):
    """
    End-to-end driver using MultiverseGridWorld + Tatha perceptual agent.

    Returns (world, agent, director).
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    world = MultiverseGridWorld(size=grid_size, max_branches=max_branches,
                                dt=0.1, decoherence_rate=0.01)

    # Tatha perceptual agent
    obs_dim = grid_size * grid_size * 3
    perceptual = Tatha(input_size=obs_dim, hidden_size=32, belief_size=8, lr=0.005)

    # EFE agent with Tatha
    actions = list(world.actions.keys())
    d_u = grid_size * grid_size
    H_actions = {a: _random_hermitian(d_u, seed=seed + a) for a in actions}
    rho_pref = np.eye(d_u) / d_u
    agent = MultiverseEFEAgent(world.mv, actions, H_actions, rho_pref,
                               perceptual_agent=perceptual, grid_size=grid_size)

    director = MultiverseDirector()

    obs = world.reset()
    agent.perceive(obs)

    reward = 0.0
    for step in range(steps):
        params = director.act(director.observe(world.mv, reward))

        a = agent.act()
        obs, reward, done = world.step(a)

        agent.perceive(obs)
        agent.learn()

        if verbose and step % 20 == 0:
            pos = world.get_classical_position()
            print(f"step={step:4d}  branches={world.mv.num_branches:3d}  "
                  f"entropy={world.mv.entanglement_entropy():.4f}  "
                  f"reward={reward:.2f}  pos={pos}")

        if done:
            if verbose:
                print(f"\n>>> ALL WAYPOINTS REACHED in {step+1} steps!")
            break

    return world, agent, director


def run_realtime(port: int = 8765, grid_size: int = 10, load_checkpoint: str = None):
    """
    Launch real-time multiverse agent server.

    Uses TathaRealTime with MetaDirector for cross-module parameter control.
    Returns (agent, server) for programmatic use.
    """
    from tatha_realtime import TathaRealTime, TathaConfig, TathaWSServer

    config = TathaConfig(grid_size=grid_size, ws_port=port)
    agent = TathaRealTime(config)

    if load_checkpoint:
        agent.checkpoint_load(load_checkpoint)

    # Initialize MetaDirector for parameter control
    agent.init_meta_director(n_params=10, hidden_size=32, belief_size=8, lr=0.01)

    server = TathaWSServer(agent, port=port)
    print(f"[MULTIVERSE] Real-time server starting on ws://0.0.0.0:{port}")
    print(f"[MULTIVERSE] MetaDirector active for cross-module parameter control")
    server.start()
    return agent, server


# ---------------------------------------------------------------------------
# 7. Built-in tests
# ---------------------------------------------------------------------------
def run_tests() -> None:
    print("Running built-in tests...")

    # normalization
    mv = MultiverseState(universe_dim=4, max_branches=8)
    mv.evolve(dt=0.1, H_universe=_random_hermitian(4, seed=1))
    assert abs(np.linalg.norm(mv.joint_vector()) - 1.0) < 1e-6, "normalize"
    print("  [OK] normalization")

    # branching
    mv = MultiverseState(universe_dim=4, max_branches=8)
    before = mv.num_branches
    ok = apply_branching(mv, 0, theta=np.pi / 4)
    assert ok and mv.num_branches == before + 1, "branch count"
    assert abs(np.linalg.norm(mv.joint_vector()) - 1.0) < 1e-6, "branch norm"
    print("  [OK] branching")

    # entropy
    mv = MultiverseState(universe_dim=4, max_branches=8)
    apply_branching(mv, 0)
    apply_branching(mv, 1)
    assert mv.entanglement_entropy() > 0, "entropy positive"
    print("  [OK] entanglement entropy")

    # decoherence
    mv = MultiverseState(universe_dim=4, max_branches=8)
    apply_branching(mv, 0)
    decohere(mv, gamma=0.5)
    assert abs(np.linalg.norm(mv.joint_vector()) - 1.0) < 1e-6, "decohere"
    print("  [OK] decoherence")

    # EFE finite
    mv = MultiverseState(universe_dim=4, max_branches=4)
    H = {a: _random_hermitian(4, seed=10 + a) for a in [0, 1]}
    agent = MultiverseEFEAgent(mv, [0, 1], H, np.eye(4) / 4)
    for a in [0, 1]:
        g = agent.expected_free_energy(a)
        assert np.isfinite(g), f"EFE not finite for {a}"
    print("  [OK] EFE finite")

    # agent action
    a = agent.act()
    assert a in [0, 1], "action in space"
    print("  [OK] agent act")

    # director update
    d = MultiverseDirector()
    obs = d.observe(mv, reward=1.0)
    p = d.act(obs)
    assert p.shape == (10,), "director shape"
    print("  [OK] director")

    # director to_meta_params
    meta = d.to_meta_params()
    assert 'red_entanglement' in meta, "director meta_params"
    print("  [OK] director meta_params")

    # MultiverseGridWorld
    world = MultiverseGridWorld(size=6, max_branches=4)
    obs = world.reset()
    assert obs.shape[0] > 0, "world observe"
    obs2, reward, done = world.step(0)
    assert obs2.shape == obs.shape, "world step shape"
    print("  [OK] MultiverseGridWorld")

    # agent with Tatha perceptual
    obs_dim = 6 * 6 * 3
    perc = Tatha(input_size=obs_dim, hidden_size=16, belief_size=4, lr=0.005)
    H2 = {a: _random_hermitian(36, seed=20 + a) for a in [0, 1]}
    agent2 = MultiverseEFEAgent(world.mv, [0, 1], H2, np.eye(36) / 36,
                                perceptual_agent=perc, grid_size=6)
    agent2.perceive(obs)
    a2 = agent2.act()
    assert a2 in [0, 1], "agent with Tatha act"
    print("  [OK] MultiverseEFEAgent with Tatha")

    # real-time import works (lazy to avoid circular)
    try:
        from tatha_realtime import TathaRealTime, TathaConfig
        rt_config = TathaConfig(GRID_SIZE=4)
        rt_agent = TathaRealTime(rt_config)
        rt_obs = np.random.randn(4 * 4 * 3).tolist()
        result = rt_agent.observe(rt_obs, reward=0.5)
        assert 'action' in result, "realtime observe returns action"
        assert 'belief' in result, "realtime observe returns belief"
        print("  [OK] tatha_realtime import and observe")
    except ImportError:
        print("  [SKIP] tatha_realtime (circular import in test context)")

    # meta_director import works
    from meta_director import MetaParameterSpace, MetaDirector
    md = MetaDirector(n_params=MetaParameterSpace.N_PARAMS)
    assert md.n_params == MetaParameterSpace.N_PARAMS, "meta_director params"
    print("  [OK] meta_director import")

    print("All tests passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="run built-in unit tests")
    parser.add_argument("--realtime", action="store_true",
                        help="launch real-time WebSocket server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--load", type=str, default=None,
                        help="checkpoint to load")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--grid", type=int, default=10,
                        help="grid size")
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.realtime:
        run_realtime(port=args.port, grid_size=args.grid,
                     load_checkpoint=args.load)
    else:
        run(steps=args.steps, grid_size=args.grid,
            max_branches=args.branches, seed=args.seed)