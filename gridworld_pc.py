"""
Grid World & Space-Scale Astrotechnology with Pure Predictive Coding & Active Inference

Integrates the fixed Tatha perceptual model with forward dynamics
for principled action selection via Expected Free Energy minimization.

Features:
- CoordinateTracker for internal SLAM-like mapping
- SpaceGridWorld: 20x20 solar system with planets, gravity wells, asteroid belts
- Tabular, Neural, and Quantum forward models with coordinate bottleneck
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unified_pc_space import (Tatha, QuantumTatha, ForwardModel, QuantumForwardModel,
                              CoordinateNeuralForwardModel, CoordinateQuantumForwardModel,
                              CoordinateTracker)


# ---------------------------------------------------------------------------
# Classic 5x5 Grid World
# ---------------------------------------------------------------------------
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.agent_pos = [0, 0]
        self.target_pos = [4, 4]
        self.walls = [(2, 2), (2, 3), (3, 2)]
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def reset(self):
        self.agent_pos = [0, 0]
        while True:
            self.target_pos = [
                np.random.randint(0, self.size),
                np.random.randint(0, self.size)
            ]
            if tuple(self.target_pos) not in self.walls and self.target_pos != self.agent_pos:
                break
        return self.observe()

    def observe(self):
        grid = np.zeros(self.size * self.size)
        idx = lambda r, c: r * self.size + c
        for w in self.walls:
            grid[idx(*w)] = 0.33
        grid[idx(*self.agent_pos)] = 0.66
        grid[idx(*self.target_pos)] = 1.0
        return grid

    def step(self, action):
        dr, dc = self.actions[action]
        new_r = np.clip(self.agent_pos[0] + dr, 0, self.size - 1)
        new_c = np.clip(self.agent_pos[1] + dc, 0, self.size - 1)
        if (new_r, new_c) in self.walls:
            new_r, new_c = self.agent_pos
        self.agent_pos = [new_r, new_c]
        reward = 1.0 if self.agent_pos == self.target_pos else 0.0
        done = reward > 0
        return self.observe(), reward, done


# ---------------------------------------------------------------------------
# Space-Scale Astrotechnology Environment
# ---------------------------------------------------------------------------
class SpaceGridWorld:
    """
    20x20 space environment with astrophysical dynamics.

    Features:
    - Planets: circular gravity wells that perturb spacecraft trajectory
    - Asteroid belts: dense obstacle clusters
    - Solar wind: random drift perturbations
    - Waypoints: multiple targets to visit in sequence
    """
    def __init__(self, size=20, n_planets=3, n_asteroids=4):
        self.size = size
        self.n_planets = n_planets
        self.n_asteroids = n_asteroids
        self.agent_pos = [0, 0]
        self.target_pos = [size - 1, size - 1]
        self.planets = []  # [(r, c, radius, gravity_strength), ...]
        self.asteroids = set()  # set of (r, c)
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.waypoints = []
        self.waypoint_idx = 0
        self._generate_space()

    def _generate_space(self):
        """Generate solar system layout."""
        np.random.seed(None)  # Allow different layouts per instance
        self.planets = []
        self.asteroids = set()

        # Place planets (gravity wells) avoiding start and target corners
        for _ in range(self.n_planets):
            while True:
                pr = np.random.randint(3, self.size - 3)
                pc = np.random.randint(3, self.size - 3)
                radius = np.random.randint(1, 3)
                gravity = np.random.uniform(0.3, 0.8)
                # Avoid too close to start (0,0) or target
                if np.linalg.norm([pr, pc]) > 4 and np.linalg.norm([pr - self.size + 1, pc - self.size + 1]) > 4:
                    self.planets.append((pr, pc, radius, gravity))
                    break

        # Place asteroid belts (clusters of small obstacles)
        for _ in range(self.n_asteroids):
            center_r = np.random.randint(2, self.size - 2)
            center_c = np.random.randint(2, self.size - 2)
            for _ in range(np.random.randint(3, 8)):
                ar = np.clip(center_r + np.random.randint(-2, 3), 0, self.size - 1)
                ac = np.clip(center_c + np.random.randint(-2, 3), 0, self.size - 1)
                if (ar, ac) != (0, 0):
                    self.asteroids.add((ar, ac))

        # Generate waypoints (visit in sequence)
        self.waypoints = [
            [self.size - 1, self.size - 1],
            [self.size // 2, self.size // 2],
            [self.size - 1, 0],
            [0, self.size - 1],
        ]
        np.random.shuffle(self.waypoints)
        self.waypoint_idx = 0
        self.target_pos = self.waypoints[0]

    def reset(self):
        self.agent_pos = [0, 0]
        self.waypoint_idx = 0
        self.target_pos = self.waypoints[0]
        return self.observe()

    def observe(self):
        grid = np.zeros(self.size * self.size)
        idx = lambda r, c: r * self.size + c
        # Asteroids
        for ar, ac in self.asteroids:
            grid[idx(ar, ac)] = 0.33
        # Planets (gravity wells) - brighter
        for pr, pc, radius, _ in self.planets:
            grid[idx(pr, pc)] = 0.5
        # Agent
        grid[idx(*self.agent_pos)] = 0.66
        # Target waypoint
        grid[idx(*self.target_pos)] = 1.0
        return grid

    def _gravity_effect(self, pos):
        """Apply gravity perturbation from nearby planets."""
        r, c = pos
        for pr, pc, radius, gravity in self.planets:
            dist = np.sqrt((r - pr)**2 + (c - pc)**2)
            if dist < radius + 2 and dist > 0:
                # Pull toward planet center
                pull = gravity / (dist + 1)
                dr = (pr - r) / dist * pull
                dc = (pc - c) / dist * pull
                r += dr
                c += dc
        return [np.clip(r, 0, self.size - 1), np.clip(c, 0, self.size - 1)]

    def step(self, action):
        dr, dc = self.actions[action]
        new_r = np.clip(self.agent_pos[0] + dr, 0, self.size - 1)
        new_c = np.clip(self.agent_pos[1] + dc, 0, self.size - 1)

        # Asteroid collision check
        if (int(round(new_r)), int(round(new_c))) in self.asteroids:
            new_r, new_c = self.agent_pos  # Bounce back

        # Apply gravity perturbation
        self.agent_pos = self._gravity_effect([new_r, new_c])
        self.agent_pos = [int(round(self.agent_pos[0])), int(round(self.agent_pos[1]))]

        # Check waypoint reached
        reward = 0.0
        done = False
        if self.agent_pos == self.target_pos:
            reward = 1.0
            self.waypoint_idx += 1
            if self.waypoint_idx < len(self.waypoints):
                self.target_pos = self.waypoints[self.waypoint_idx]
            else:
                done = True

        return self.observe(), reward, done

    def print_space(self, flat_grid=None):
        if flat_grid is None:
            grid = self.observe().reshape(self.size, self.size)
        else:
            grid = flat_grid.reshape(self.size, self.size)
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                val = grid[r, c]
                if val < 0.1:
                    row += ". "
                elif val < 0.4:
                    row += "* "  # asteroid
                elif val < 0.6:
                    row += "O "  # planet
                elif val < 0.85:
                    row += "S "  # spacecraft
                else:
                    row += "W "  # waypoint
            print("  " + row)


# ---------------------------------------------------------------------------
# Tabular Forward Model
# ---------------------------------------------------------------------------
class TabularForwardModel:
    def __init__(self, world):
        self.memory = {}
        self.world = world

    def _to_int(self, action):
        arr = np.asarray(action)
        if arr.ndim == 0:
            return int(arr)
        if arr.size == 1:
            return int(arr.item())
        return int(np.argmax(arr))

    def _key(self, obs, action):
        grid = np.asarray(obs).reshape(self.world.size, self.world.size)
        agent_pos = np.argwhere(np.abs(grid - 0.66) < 0.1)
        target_pos = np.argwhere(np.abs(grid - 1.0) < 0.1)
        if agent_pos.size == 0:
            ar, ac = 0, 0
        else:
            ar, ac = int(agent_pos.flat[0]), int(agent_pos.flat[1])
        if target_pos.size == 0:
            tr, tc = self.world.size - 1, self.world.size - 1
        else:
            tr, tc = int(target_pos.flat[0]), int(target_pos.flat[1])
        return (ar, ac, tr, tc, self._to_int(action))

    def _physics_predict(self, obs, action):
        grid = np.asarray(obs).reshape(self.world.size, self.world.size)
        agent_pos = np.argwhere(np.abs(grid - 0.66) < 0.1)
        target_pos = np.argwhere(np.abs(grid - 1.0) < 0.1)
        if agent_pos.size == 0:
            ar, ac = 0, 0
        else:
            ar, ac = int(agent_pos.flat[0]), int(agent_pos.flat[1])
        if target_pos.size == 0:
            tr, tc = self.world.size - 1, self.world.size - 1
        else:
            tr, tc = int(target_pos.flat[0]), int(target_pos.flat[1])

        dr, dc = self.world.actions[self._to_int(action)]
        new_r = np.clip(ar + dr, 0, self.world.size - 1)
        new_c = np.clip(ac + dc, 0, self.world.size - 1)

        # Handle walls/asteroids
        if hasattr(self.world, 'walls') and (new_r, new_c) in self.world.walls:
            new_r, new_c = ar, ac
        if hasattr(self.world, 'asteroids') and (new_r, new_c) in self.world.asteroids:
            new_r, new_c = ar, ac

        # Handle gravity (simplified: just stay if near planet)
        if hasattr(self.world, 'planets'):
            for pr, pc, radius, _ in self.world.planets:
                if np.sqrt((new_r - pr)**2 + (new_c - pc)**2) < radius + 1:
                    new_r = int(np.clip(new_r + 0.3 * (pr - new_r), 0, self.world.size - 1))
                    new_c = int(np.clip(new_c + 0.3 * (pc - new_c), 0, self.world.size - 1))

        pred = np.zeros(self.world.size * self.world.size)
        idx = lambda r, c: r * self.world.size + c
        if hasattr(self.world, 'walls'):
            for w in self.world.walls:
                pred[idx(*w)] = 0.33
        if hasattr(self.world, 'asteroids'):
            for ar_ast, ac_ast in self.world.asteroids:
                pred[idx(ar_ast, ac_ast)] = 0.33
        if hasattr(self.world, 'planets'):
            for pr, pc, _, _ in self.world.planets:
                pred[idx(pr, pc)] = 0.5

        if new_r == tr and new_c == tc:
            pred[idx(new_r, new_c)] = 1.0
        else:
            pred[idx(tr, tc)] = 1.0
            pred[idx(new_r, new_c)] = 0.66
        return pred

    def predict(self, obs, action):
        key = self._key(obs, action)
        if key in self.memory:
            return self.memory[key]
        return self._physics_predict(obs, action)

    def train(self, obs, action, next_obs):
        key = self._key(obs, action)
        self.memory[key] = next_obs.copy()
        return 0.0

    def coverage(self):
        return len(self.memory)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def encode_action(action, n_actions=4):
    vec = np.zeros(n_actions)
    vec[action] = 1.0
    return vec


def print_grid(flat_grid, size=5):
    grid = flat_grid.reshape(size, size)
    for r in range(size):
        row = ""
        for c in range(size):
            val = grid[r, c]
            if val < 0.15:
                row += ". "
            elif val < 0.45:
                row += "# "
            elif val < 0.85:
                row += "A "
            else:
                row += "T "
        print("  " + row)


def _extract_agent_pos(obs, size):
    grid = np.asarray(obs).reshape(size, size)
    pos = np.argwhere(np.abs(grid - 0.66) < 0.1)
    if pos.size > 0:
        return pos[0].astype(int)
    target_pos = np.argwhere(np.abs(grid - 1.0) < 0.1)
    if target_pos.size > 0:
        return target_pos[0].astype(int)
    return np.array([0, 0])


def _extract_target_pos(obs, size):
    grid = np.asarray(obs).reshape(size, size)
    pos = np.argwhere(np.abs(grid - 1.0) < 0.1)
    if pos.size == 0:
        return np.array([size - 1, size - 1])
    return pos[0].astype(int)


# ---------------------------------------------------------------------------
# Predictive Coding Grid Agent with Coordinate Tracking
# ---------------------------------------------------------------------------
class PCGridAgent:
    def __init__(self, world, perceptual_agent, forward_model,
                 preference_type='target', exploration_coef=0.15):
        self.world = world
        self.perceptual = perceptual_agent
        self.fm = forward_model
        self.preference_type = preference_type
        self.exploration_coef = exploration_coef
        self.tracker = CoordinateTracker(world.size)

    def get_preferred_observation(self):
        grid = np.zeros(self.world.size * self.world.size)
        idx = lambda r, c: r * self.world.size + c
        if self.preference_type == 'target':
            grid[idx(*self.world.target_pos)] = 1.0
        return grid

    def expected_free_energy(self, obs, action):
        action_vec = encode_action(action)
        pred_next = self.fm.predict(obs, action_vec)
        preferred = self.get_preferred_observation()

        # 1. Expected surprise
        surprise = np.mean((pred_next - preferred) ** 2)

        # 2. Predicted distance penalty
        pred_agent = _extract_agent_pos(pred_next, self.world.size)
        target = _extract_target_pos(obs, self.world.size)
        pred_dist = np.linalg.norm(pred_agent - target)
        max_dist = np.sqrt(2) * (self.world.size - 1)
        distance_penalty = 0.5 * (pred_dist / max_dist)

        # 3. Ambiguity
        p = np.clip(pred_next, 1e-8, 1 - 1e-8)
        ambiguity = -np.sum(p * np.log(p) + (1 - p) * np.log(1 - p))

        # 4. Information gain from tracker + model
        agent_now = _extract_agent_pos(obs, self.world.size)
        tracker_ig = self.tracker.info_gain(agent_now, action)
        if hasattr(self.fm, 'tracker'):
            model_ig = 0.0 if self.fm.tracker.is_known(agent_now, action) else 0.3
        elif hasattr(self.fm, 'memory'):
            key = self.fm._key(obs, action)
            model_ig = 0.0 if key in self.fm.memory else 0.3
        else:
            model_ig = 0.0
        info_gain = max(tracker_ig, model_ig)

        efe = surprise + distance_penalty + 0.005 * ambiguity - self.exploration_coef * info_gain
        return efe, pred_next

    def select_action(self, obs):
        best_action = 0
        best_efe = float('inf')
        candidates = []
        for a in range(4):
            efe, pred = self.expected_free_energy(obs, a)
            candidates.append((a, efe, pred))
            if efe < best_efe:
                best_efe = efe
                best_action = a
        return best_action, candidates

    def navigate(self, max_steps=20, train=True, verbose=True):
        obs = self.world.reset()
        self.perceptual.perceive(obs)
        self.perceptual.settle(n_iter=15)

        for step in range(max_steps):
            action, candidates = self.select_action(obs)

            if verbose:
                print(f"\nStep {step+1} | Pos: {list(self.world.agent_pos)} | Target: {self.world.target_pos}")
                print("  EFE candidates:")
                for a, efe, _ in candidates:
                    marker = " <-- CHOSEN" if a == action else ""
                    print(f"    {self.world.action_names[a]:5s} | EFE={efe:.4f}{marker}")
                if hasattr(self.world, 'print_space'):
                    self.world.print_space(obs)
                else:
                    print_grid(obs, self.world.size)

            next_obs, reward, done = self.world.step(action)
            action_vec = encode_action(action)

            # Update coordinate tracker
            agent_now = _extract_agent_pos(obs, self.world.size)
            agent_next = _extract_agent_pos(next_obs, self.world.size)
            self.tracker.update(agent_now, action, agent_next)

            self.fm.train(obs, action_vec, next_obs)

            self.perceptual.perceive(next_obs)
            self.perceptual.settle(n_iter=10)
            if train:
                self.perceptual.learn()

            obs = next_obs

            if done:
                if verbose:
                    print(f"  >>> Target reached in {step+1} steps!")
                return True, step + 1

        if verbose:
            print(f"  >>> Max steps reached.")
        return False, max_steps


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def train_forward_model(world, fm, episodes=2000, steps_per_ep=20):
    for ep in range(episodes):
        obs = world.reset()
        for _ in range(steps_per_ep):
            a = np.random.randint(0, 4)
            a_vec = encode_action(a)
            next_obs, _, done = world.step(a)
            fm.train(obs, a_vec, next_obs)
            obs = next_obs
            if done:
                break


def train_perceptual_agent(agent, world, steps=1000):
    for _ in range(steps):
        obs = world.observe()
        agent.perceive(obs)
        agent.settle(n_iter=10)
        agent.learn()
        if np.random.rand() < 0.15:
            world.reset()
        else:
            a = np.random.randint(0, 4)
            world.step(a)


def evaluate_agent(agent, n_episodes=20, max_steps=20):
    successes = 0
    total_steps = 0
    for _ in range(n_episodes):
        success, steps = agent.navigate(max_steps=max_steps, train=False, verbose=False)
        if success:
            successes += 1
        total_steps += steps
    return successes / n_episodes, total_steps / n_episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # =====================================================================
    # PART 1: Classic 5x5 Grid World (verify 100% success)
    # =====================================================================
    print("=" * 70)
    print("PART 1: Classic 5x5 Grid World — Pure Predictive Coding")
    print("=" * 70)

    world = GridWorld()

    print("\n[1] Training Forward Models...")
    tabular_fm = TabularForwardModel(world)
    train_forward_model(world, tabular_fm, episodes=3000, steps_per_ep=30)
    print(f"  Tabular FM    | Coverage: {tabular_fm.coverage()} transitions")

    neural_fm = CoordinateNeuralForwardModel(grid_size=5, lr=0.05)
    neural_fm.walls = world.walls
    train_forward_model(world, neural_fm, episodes=3000, steps_per_ep=30)
    print(f"  Neural FM     | Tracker coverage: {len(neural_fm.tracker.transition_map)} transitions")

    quantum_fm = CoordinateQuantumForwardModel(grid_size=5, quantum_dim=16, lr=0.05)
    quantum_fm.walls = world.walls
    train_forward_model(world, quantum_fm, episodes=3000, steps_per_ep=30)
    print(f"  Quantum FM    | Tracker coverage: {len(quantum_fm.tracker.transition_map)} transitions")

    print("\n[2] Pre-training Perceptual Agents...")
    classical_perceptual = Tatha(input_size=25, hidden_size=16, belief_size=8, lr=0.01)
    train_perceptual_agent(classical_perceptual, world, steps=800)
    print("  Classical Tatha | Pre-trained")

    quantum_perceptual = QuantumTatha(input_size=25, hidden_size=16, belief_size=8, lr=0.01)
    train_perceptual_agent(quantum_perceptual, world, steps=800)
    print("  Quantum Tatha   | Pre-trained")

    print("\n[3] Demonstration: Classical Tatha + Tabular Forward Model")
    demo_agent = PCGridAgent(world, classical_perceptual, tabular_fm)
    demo_agent.navigate(max_steps=15, train=True, verbose=True)

    print("\n[4] Evaluation (20 episodes each, max 20 steps)")
    configs = [
        ("Classical Tatha + Tabular FM", Tatha, tabular_fm),
        ("Classical Tatha + Coordinate Neural FM", Tatha, neural_fm),
        ("Quantum Tatha + Coordinate Quantum FM", QuantumTatha, quantum_fm),
    ]

    for name, PerceptualClass, fm in configs:
        p = PerceptualClass(input_size=25, hidden_size=16, belief_size=8, lr=0.01)
        train_perceptual_agent(p, world, steps=800)
        agent = PCGridAgent(world, p, fm)
        sr, avg_steps = evaluate_agent(agent, n_episodes=20, max_steps=20)
        print(f"\n{name}")
        print(f"  Success Rate: {sr:.1%}")
        print(f"  Avg Steps:    {avg_steps:.1f}")

    # =====================================================================
    # PART 2: Space-Scale Astrotechnology (20x20 Solar System)
    # =====================================================================
    print("\n" + "=" * 70)
    print("PART 2: Space-Scale Astrotechnology — 20x20 Solar System")
    print("=" * 70)

    space_world = SpaceGridWorld(size=20, n_planets=3, n_asteroids=4)
    print("\nSolar System Layout:")
    space_world.print_space()
    print(f"\nWaypoints to visit: {space_world.waypoints}")

    print("\n[1] Training Space Forward Models...")
    space_tabular = TabularForwardModel(space_world)
    train_forward_model(space_world, space_tabular, episodes=5000, steps_per_ep=50)
    print(f"  Tabular FM    | Coverage: {space_tabular.coverage()} transitions")

    space_neural = CoordinateNeuralForwardModel(grid_size=20, lr=0.05)
    space_neural.walls = space_world.asteroids
    train_forward_model(space_world, space_neural, episodes=5000, steps_per_ep=50)
    print(f"  Neural FM     | Tracker coverage: {len(space_neural.tracker.transition_map)} transitions")

    space_quantum = CoordinateQuantumForwardModel(grid_size=20, quantum_dim=16, lr=0.05)
    space_quantum.walls = space_world.asteroids
    train_forward_model(space_world, space_quantum, episodes=5000, steps_per_ep=50)
    print(f"  Quantum FM    | Tracker coverage: {len(space_quantum.tracker.transition_map)} transitions")

    print("\n[2] Pre-training Space Perceptual Agents...")
    space_classical = Tatha(input_size=400, hidden_size=32, belief_size=16, lr=0.005)
    train_perceptual_agent(space_classical, space_world, steps=1500)
    print("  Classical Tatha | Pre-trained on space observations")

    space_quantum_perceptual = QuantumTatha(input_size=400, hidden_size=32, belief_size=16, lr=0.005)
    train_perceptual_agent(space_quantum_perceptual, space_world, steps=1500)
    print("  Quantum Tatha   | Pre-trained on space observations")

    print("\n[3] Space Navigation Demonstration")
    space_demo = PCGridAgent(space_world, space_classical, space_tabular)
    space_demo.navigate(max_steps=60, train=True, verbose=True)

    print("\n[4] Space Evaluation (10 episodes each, max 60 steps)")
    space_configs = [
        ("Space: Classical + Tabular", Tatha, space_tabular, 400),
        ("Space: Classical + Neural", Tatha, space_neural, 400),
        ("Space: Quantum + Quantum", QuantumTatha, space_quantum, 400),
    ]

    for name, PerceptualClass, fm, input_size in space_configs:
        p = PerceptualClass(input_size=input_size, hidden_size=32, belief_size=16, lr=0.005)
        train_perceptual_agent(p, space_world, steps=1500)
        agent = PCGridAgent(space_world, p, fm)
        sr, avg_steps = evaluate_agent(agent, n_episodes=10, max_steps=60)
        print(f"\n{name}")
        print(f"  Success Rate: {sr:.1%}")
        print(f"  Avg Steps:    {avg_steps:.1f}")

    print("\n" + "=" * 70)
    print("Done. Coordinate tracking + space-scale astrotechnology complete.")
    print("=" * 70)
