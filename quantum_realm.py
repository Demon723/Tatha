"""
================================================================================
Quantum Realm Extension for Tathā Unified
================================================================================

A fully quantum grid world where:
  • The agent IS a wavefunction (complex amplitude field over the grid)
  • Each cell is a basis state |r,c⟩ in a finite-dimensional Hilbert space
  • Planets = potential wells (attractive Hamiltonian terms)
  • Asteroids = infinite potential barriers (with tunneling probability)
  • Waypoints = measurement operators that collapse the wavepacket
  • Navigation = Schrödinger evolution + unitary action operators + collapse
  • EFE = quantum expectation value over action superpositions

Integrates with: unified_pc_space.py
"""

import numpy as np

# Import core predictive coding agents from unified system
try:
    from unified_pc_space import Tatha, QuantumTatha, AdaptiveQuantumTatha
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unified_pc_space import Tatha, QuantumTatha, AdaptiveQuantumTatha

# ==============================================================================
# PART Q1: Quantum Wavepacket Agent State
# ==============================================================================

class QuantumWavepacket:
    """
    A 2D quantum wavepacket living on the grid.
    psi[r, c] = amplitude at cell (r, c).
    Normalization: sum |psi|^2 = 1.
    """
    def __init__(self, grid_size, initial_pos=None, sigma=1.0):
        self.grid_size = grid_size
        self.sigma = sigma
        self.psi = np.zeros((grid_size, grid_size), dtype=complex)

        if initial_pos is None:
            initial_pos = (0, 0)
        self._initialize_gaussian(initial_pos, sigma)
        self._normalize()

        # Momentum space for kinetic evolution
        self.kx = 2 * np.pi * np.fft.fftfreq(grid_size)
        self.ky = 2 * np.pi * np.fft.fftfreq(grid_size)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')

    def _initialize_gaussian(self, center, sigma):
        """Initialize as Gaussian wavepacket."""
        cr, cc = center
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                dr, dc = r - cr, c - cc
                self.psi[r, c] = np.exp(-(dr**2 + dc**2) / (4 * sigma**2))

    def _normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.psi)**2))
        if norm > 1e-12:
            self.psi /= norm

    def probability_density(self):
        """Born rule: P(r,c) = |psi(r,c)|^2"""
        return np.abs(self.psi)**2

    def phase_field(self):
        """Phase arg(psi) at each cell."""
        return np.angle(self.psi)

    def expected_position(self):
        """⟨r⟩, ⟨c⟩ expectation values."""
        probs = self.probability_density()
        r_vals = np.arange(self.grid_size)
        c_vals = np.arange(self.grid_size)
        R, C = np.meshgrid(r_vals, c_vals, indexing='ij')
        exp_r = np.sum(R * probs)
        exp_c = np.sum(C * probs)
        return exp_r, exp_c

    def position_variance(self):
        """Variance of position distribution."""
        probs = self.probability_density()
        r_vals = np.arange(self.grid_size)
        c_vals = np.arange(self.grid_size)
        R, C = np.meshgrid(r_vals, c_vals, indexing='ij')
        exp_r = np.sum(R * probs)
        exp_c = np.sum(C * probs)
        var_r = np.sum((R - exp_r)**2 * probs)
        var_c = np.sum((C - exp_c)**2 * probs)
        return var_r, var_c

    def collapse_to_cell(self, r, c):
        """Measurement collapse: psi -> delta(r,c)."""
        self.psi = np.zeros_like(self.psi)
        self.psi[r, c] = 1.0 + 0j

    def collapse_near(self, pos, sigma=0.5):
        """Partial collapse: Gaussian centered at pos."""
        self._initialize_gaussian(pos, sigma)
        self._normalize()

    def copy(self):
        """Deep copy of wavepacket."""
        wp = QuantumWavepacket(self.grid_size, sigma=self.sigma)
        wp.psi = self.psi.copy()
        return wp


# ==============================================================================
# PART Q2: Quantum Environment Hamiltonian
# ==============================================================================

class QuantumHamiltonian:
    """
    Environment Hamiltonian H = T + V where:
      T = kinetic energy (momentum operator, -∇²/2m)
      V = potential energy (planets = wells, asteroids = barriers, walls = ∞)
    """
    def __init__(self, grid_size, mass=1.0, hbar=0.5):
        self.grid_size = grid_size
        self.mass = mass
        self.hbar = hbar
        self.V = np.zeros((grid_size, grid_size))
        # Momentum space grids for kinetic evolution
        self.kx = 2 * np.pi * np.fft.fftfreq(grid_size)
        self.ky = 2 * np.pi * np.fft.fftfreq(grid_size)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing="ij")

    def set_planet_potential(self, planets):
        """
        Planets create attractive potential wells.
        V(r,c) = -gravity_strength / (distance + epsilon)
        """
        for pr, pc, radius, gravity in planets:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    dist = np.sqrt((r - pr)**2 + (c - pc)**2)
                    if dist < radius + 3:
                        self.V[r, c] -= gravity / (dist + 0.5)

    def set_barrier_potential(self, obstacles, height=50.0):
        """
        Asteroids/walls create high potential barriers.
        Tunneling probability ~ exp(-2 * barrier_height * width)
        """
        for r, c in obstacles:
            self.V[r, c] += height
            # Thick barrier: also affect neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        self.V[nr, nc] += height * 0.3

    def set_waypoint_potential(self, waypoints, depth=10.0):
        """
        Waypoints create slight attractive bias (measurement attracts).
        """
        for wr, wc in waypoints:
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    dist = np.sqrt((r - wr)**2 + (c - wc)**2)
                    if dist < 5:
                        self.V[r, c] -= depth / (dist + 1.0)

    def get_potential(self):
        return self.V

    def kinetic_evolution(self, psi, dt):
        """
        Evolve under kinetic term using FFT (spectral method).
        T = p²/2m -> evolution in k-space is multiplication by exp(-i * k² * dt / 2m)
        """
        psi_k = np.fft.fft2(psi)
        k2 = self.KX**2 + self.KY**2
        psi_k *= np.exp(-1j * self.hbar * k2 * dt / (2 * self.mass))
        return np.fft.ifft2(psi_k)

    def potential_evolution(self, psi, dt):
        """
        Evolve under potential term: multiplication by exp(-i * V * dt / hbar)
        """
        return psi * np.exp(-1j * self.V * dt / self.hbar)

    def full_step(self, psi, dt):
        """
        Split-operator method: one full step = T(dt/2) * V(dt) * T(dt/2)
        2nd order accurate.
        """
        psi = self.kinetic_evolution(psi, dt / 2)
        psi = self.potential_evolution(psi, dt)
        psi = self.kinetic_evolution(psi, dt / 2)
        return psi

    def tunneling_probability(self, psi, barrier_cells):
        """
        Probability that the wavepacket tunnels through barriers.
        = sum of |psi|² on the far side of barriers.
        """
        probs = np.abs(psi)**2
        # Simple heuristic: probability on barrier cells themselves
        tunnel_prob = 0.0
        for r, c in barrier_cells:
            tunnel_prob += probs[r, c]
        return tunnel_prob


# ==============================================================================
# PART Q3: Quantum Action Operators
# ==============================================================================

class QuantumActionOperators:
    """
    Unitary operators for each action (UP, DOWN, LEFT, RIGHT).
    Each action applies a momentum kick / displacement to the wavepacket.
    """
    def __init__(self, grid_size, hbar=0.5, kick_strength=1.0):
        self.grid_size = grid_size
        self.hbar = hbar
        self.kick_strength = kick_strength

    def _displacement_operator(self, psi, dr, dc):
        """
        Apply displacement: shift wavepacket by (dr, dc) with momentum kick.
        Uses FFT-based translation operator.
        """
        psi_k = np.fft.fft2(psi)
        kx = 2 * np.pi * np.fft.fftfreq(self.grid_size)
        ky = 2 * np.pi * np.fft.fftfreq(self.grid_size)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        # Translation: psi'(r) = psi(r - dr) -> multiply by exp(-i * k · dr) in k-space
        phase = np.exp(-1j * (KX * dr + KY * dc) * self.kick_strength)
        psi_k *= phase
        return np.fft.ifft2(psi_k)

    def up(self, psi):
        return self._displacement_operator(psi, -1, 0)

    def down(self, psi):
        return self._displacement_operator(psi, +1, 0)

    def left(self, psi):
        return self._displacement_operator(psi, 0, -1)

    def right(self, psi):
        return self._displacement_operator(psi, 0, +1)

    def apply_action(self, psi, action_idx):
        ops = [self.up, self.down, self.left, self.right]
        return ops[action_idx](psi)

    def action_superposition(self, psi, action_weights):
        """
        Apply all actions in superposition with given weights.
        Result = sum_i weight_i * U_i |psi>
        """
        result = np.zeros_like(psi)
        for i, weight in enumerate(action_weights):
            if abs(weight) > 1e-10:
                result += weight * self.apply_action(psi, i)
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(result)**2))
        if norm > 1e-12:
            result /= norm
        return result


# ==============================================================================
# PART Q4: Quantum Realm Grid World
# ==============================================================================

class QuantumRealmGridWorld:
    """
    A fully quantum grid environment.

    The agent is a wavepacket that evolves under the environment Hamiltonian.
    Actions apply unitary displacement operators.
    Observations are measurements that partially collapse the wavepacket.
    Waypoints trigger measurement collapse when probability at waypoint is high.
    """
    def __init__(self, size=20, n_planets=3, n_asteroids=4, 
                 dt=0.1, mass=1.0, hbar=0.5, decoherence_rate=0.01):
        self.size = size
        self.n_planets = n_planets
        self.n_asteroids = n_asteroids
        self.dt = dt
        self.mass = mass
        self.hbar = hbar
        self.decoherence_rate = decoherence_rate

        self.actions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        # Quantum state
        self.wavepacket = QuantumWavepacket(size, initial_pos=(0, 0), sigma=1.5)

        # Environment features
        self.planets = []
        self.asteroids = set()
        self.walls = set()
        self.waypoints = []
        self.waypoint_idx = 0

        # Hamiltonian
        self.hamiltonian = QuantumHamiltonian(size, mass=mass, hbar=hbar)
        self.action_ops = QuantumActionOperators(size, hbar=hbar, kick_strength=1.0)

        # Telemetry
        self.telemetry = {
            'path': [],
            'prob_history': [],
            'phase_history': [],
            'energy_history': [],
            'tunnel_events': [],
            'collapse_events': [],
            'waypoint_reaches': []
        }

        self._generate_realm()

    def _generate_realm(self):
        """Generate the quantum realm layout."""
        self.planets = []
        self.asteroids = set()
        self.walls = set()

        # Planets (potential wells)
        for _ in range(self.n_planets):
            while True:
                pr = np.random.randint(3, self.size - 3)
                pc = np.random.randint(3, self.size - 3)
                radius = np.random.randint(2, 4)
                gravity = np.random.uniform(0.5, 1.5)
                if np.linalg.norm([pr, pc]) > 4:
                    self.planets.append((pr, pc, radius, gravity))
                    break

        # Asteroids (barriers with tunneling)
        for _ in range(self.n_asteroids):
            cr = np.random.randint(2, self.size - 2)
            cc = np.random.randint(2, self.size - 2)
            for _ in range(np.random.randint(2, 6)):
                ar = np.clip(cr + np.random.randint(-1, 2), 0, self.size - 1)
                ac = np.clip(cc + np.random.randint(-1, 2), 0, self.size - 1)
                if (ar, ac) != (0, 0):
                    self.asteroids.add((ar, ac))
                    self.walls.add((ar, ac))

        # Waypoints (measurement targets)
        self.waypoints = [
            [self.size - 1, self.size - 1],
            [self.size // 2, self.size // 2],
            [self.size - 1, 0],
            [0, self.size - 1],
        ]
        np.random.shuffle(self.waypoints)
        self.waypoint_idx = 0

        # Build Hamiltonian
        self.hamiltonian.set_planet_potential(self.planets)
        self.hamiltonian.set_barrier_potential(self.asteroids, height=30.0)
        self.hamiltonian.set_waypoint_potential(self.waypoints, depth=5.0)

    def reset(self):
        self.wavepacket = QuantumWavepacket(self.size, initial_pos=(0, 0), sigma=1.5)
        self.waypoint_idx = 0
        self.telemetry = {
            'path': [],
            'prob_history': [],
            'phase_history': [],
            'energy_history': [],
            'tunnel_events': [],
            'collapse_events': [],
            'waypoint_reaches': []
        }
        return self.observe()

    def observe(self):
        """
        Observation = probability density field + phase field + environment features.
        Flattened for the predictive coding agent.
        """
        probs = self.wavepacket.probability_density()
        phases = self.wavepacket.phase_field()

        # Encode as 4 channels flattened:
        # Channel 0: probability density
        # Channel 1: cos(phase)
        # Channel 2: sin(phase)
        # Channel 3: environment potential V
        obs = np.zeros(self.size * self.size * 4)

        for r in range(self.size):
            for c in range(self.size):
                idx = r * self.size + c
                obs[idx] = probs[r, c]
                obs[idx + self.size*self.size] = np.cos(phases[r, c])
                obs[idx + 2*self.size*self.size] = np.sin(phases[r, c])
                obs[idx + 3*self.size*self.size] = self.hamiltonian.V[r, c]

        return obs

    def step(self, action):
        """
        Quantum step:
        1. Apply action unitary
        2. Evolve under Hamiltonian for time dt
        3. Apply decoherence (environmental noise)
        4. Check for measurement (waypoint detection)
        5. Check for tunneling events
        """
        # 1. Action unitary
        self.wavepacket.psi = self.action_ops.apply_action(self.wavepacket.psi, action)

        # 2. Hamiltonian evolution
        self.wavepacket.psi = self.hamiltonian.full_step(self.wavepacket.psi, self.dt)

        # 3. Decoherence: random phase noise
        if self.decoherence_rate > 0:
            noise = np.random.randn(self.size, self.size) * np.sqrt(self.decoherence_rate * self.dt)
            self.wavepacket.psi *= np.exp(1j * noise)

        # 4. Boundary reflection (infinite walls at edges)
        self.wavepacket.psi[0, :] *= 0.5
        self.wavepacket.psi[-1, :] *= 0.5
        self.wavepacket.psi[:, 0] *= 0.5
        self.wavepacket.psi[:, -1] *= 0.5

        self.wavepacket._normalize()

        # 5. Check tunneling
        tunnel_prob = self.hamiltonian.tunneling_probability(self.wavepacket.psi, self.asteroids)
        if tunnel_prob > 0.05:
            self.telemetry['tunnel_events'].append({
                'step': len(self.telemetry['path']),
                'probability': tunnel_prob
            })

        # 6. Check waypoint measurement (collapse if high probability)
        reward = 0.0
        done = False
        if self.waypoint_idx < len(self.waypoints):
            wr, wc = self.waypoints[self.waypoint_idx]
            prob_at_waypoint = np.abs(self.wavepacket.psi[wr, wc])**2

            if prob_at_waypoint > 0.3:  # Measurement threshold
                # Partial collapse toward waypoint
                self.wavepacket.collapse_near((wr, wc), sigma=0.8)
                reward = 1.0
                self.waypoint_idx += 1
                self.telemetry['waypoint_reaches'].append({
                    'step': len(self.telemetry['path']),
                    'pos': [wr, wc],
                    'waypoint_idx': self.waypoint_idx - 1,
                    'collapse_probability': prob_at_waypoint
                })
                self.telemetry['collapse_events'].append({
                    'step': len(self.telemetry['path']),
                    'type': 'waypoint',
                    'pos': (wr, wc),
                    'probability': prob_at_waypoint
                })

                if self.waypoint_idx >= len(self.waypoints):
                    done = True

        # Record telemetry
        self.telemetry['path'].append(self.wavepacket.expected_position())
        self.telemetry['prob_history'].append(self.wavepacket.probability_density().copy())
        self.telemetry['phase_history'].append(self.wavepacket.phase_field().copy())

        # Energy expectation
        H_psi = self.hamiltonian.potential_evolution(
            self.hamiltonian.kinetic_evolution(self.wavepacket.psi, self.dt), self.dt
        )
        energy = np.real(np.vdot(self.wavepacket.psi, H_psi))
        self.telemetry['energy_history'].append(energy)

        return self.observe(), reward, done

    def get_classical_position(self):
        """Most likely position (mode of probability density)."""
        probs = self.wavepacket.probability_density()
        max_idx = np.unravel_index(np.argmax(probs), probs.shape)
        return list(max_idx)

    def get_position_uncertainty(self):
        """Standard deviation of position."""
        var_r, var_c = self.wavepacket.position_variance()
        return np.sqrt(var_r), np.sqrt(var_c)

    def print_realm(self):
        """ASCII visualization of probability density."""
        probs = self.wavepacket.probability_density()
        max_prob = np.max(probs)
        chars = [' ', '.', 'o', 'O', '@', '#', '*']
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) in self.asteroids:
                    row += "X"
                elif any((r, c) == (pr, pc) for pr, pc, _, _ in self.planets):
                    row += "P"
                elif self.waypoint_idx < len(self.waypoints) and [r, c] == self.waypoints[self.waypoint_idx]:
                    row += "W"
                else:
                    p = probs[r, c] / max_prob if max_prob > 0 else 0
                    idx = min(int(p * (len(chars) - 1)), len(chars) - 1)
                    row += chars[idx]
            print("  " + row)


# ==============================================================================
# PART Q5: Quantum EFE Agent
# ==============================================================================

class QuantumEFEAgent:
    """
    Active inference agent for the quantum realm.

    EFE = Expected Free Energy computed as quantum expectation value:
      G(a) = ⟨ψ| H_a |ψ⟩ + D_KL(p_a || p_preferred) - info_gain
    where H_a is the action-dependent Hamiltonian and p_a is the predicted
    probability distribution after action a.
    """
    def __init__(self, world, perceptual_agent, exploration_coef=0.2):
        self.world = world
        self.perceptual = perceptual_agent
        self.exploration_coef = exploration_coef
        self.tracker = CoordinateTracker(world.size)

    def get_preferred_observation(self):
        """Preferred state: probability concentrated at current waypoint."""
        obs = np.zeros(self.world.size * self.world.size * 4)
        if self.world.waypoint_idx < len(self.world.waypoints):
            wr, wc = self.world.waypoints[self.world.waypoint_idx]
            idx = wr * self.world.size + wc
            obs[idx] = 1.0  # High probability at waypoint
        return obs

    def _simulate_action(self, action):
        """
        Simulate taking an action without modifying actual state.
        Returns predicted wavepacket after action + Hamiltonian step.
        """
        psi_copy = self.world.wavepacket.copy()
        psi_copy.psi = self.world.action_ops.apply_action(psi_copy.psi, action)
        psi_copy.psi = self.world.hamiltonian.full_step(psi_copy.psi, self.world.dt)
        psi_copy._normalize()
        return psi_copy

    def expected_free_energy(self, action):
        """
        Quantum EFE:
        1. Predicted surprise (KL from preferred)
        2. Predicted energy (Hamiltonian expectation)
        3. Position uncertainty penalty
        4. Information gain (exploration bonus)
        """
        # Simulate action
        predicted_wp = self._simulate_action(action)
        pred_probs = predicted_wp.probability_density()

        # 1. Predicted surprise (distance from preferred waypoint)
        preferred = self.get_preferred_observation()
        pref_probs = preferred[:self.world.size * self.world.size].reshape(self.world.size, self.world.size)
        # KL divergence (approximate)
        kl = 0.0
        for r in range(self.world.size):
            for c in range(self.world.size):
                p = pred_probs[r, c] + 1e-10
                q = pref_probs[r, c] + 1e-10
                kl += p * np.log(p / q)

        # 2. Predicted energy
        H_psi = self.world.hamiltonian.potential_evolution(
            self.world.hamiltonian.kinetic_evolution(predicted_wp.psi, self.world.dt), self.world.dt
        )
        pred_energy = np.real(np.vdot(predicted_wp.psi, H_psi))

        # 3. Position uncertainty penalty (don't spread too thin)
        var_r, var_c = predicted_wp.position_variance()
        uncertainty_penalty = 0.1 * (var_r + var_c)

        # 4. Information gain (probability mass in unvisited regions)
        info_gain = 0.0
        for r in range(self.world.size):
            for c in range(self.world.size):
                if (r, c) not in self.tracker.visited:
                    info_gain += pred_probs[r, c]

        efe = kl + 0.01 * pred_energy + uncertainty_penalty - self.exploration_coef * info_gain
        return efe, predicted_wp

    def select_action(self):
        """Select action minimizing quantum EFE."""
        best_action = 0
        best_efe = float('inf')
        candidates = []

        for a in range(4):
            efe, pred = self.expected_free_energy(a)
            candidates.append((a, efe, pred))
            if efe < best_efe:
                best_efe = efe
                best_action = a

        return best_action, candidates

    def navigate(self, max_steps=100, train=True, verbose=True):
        """Navigate the quantum realm."""
        obs = self.world.reset()
        self.perceptual.perceive(obs)
        self.perceptual.settle(n_iter=15)

        for step in range(max_steps):
            action, candidates = self.select_action()

            if verbose and step % 10 == 0:
                pos = self.world.get_classical_position()
                unc = self.world.get_position_uncertainty()
                print(f"\nStep {step+1} | Classical pos: {pos} | σ=({unc[0]:.2f}, {unc[1]:.2f})")
                print(f"  Target waypoint: {self.world.waypoints[self.world.waypoint_idx] if self.world.waypoint_idx < len(self.world.waypoints) else 'ALL COMPLETE'}")
                print("  EFE candidates:")
                for a, efe, _ in candidates:
                    marker = " <-- CHOSEN" if a == action else ""
                    print(f"    {self.world.action_names[a]:5s} | EFE={efe:.4f}{marker}")
                self.world.print_realm()

            next_obs, reward, done = self.world.step(action)

            # Update tracker with classical position
            class_pos = self.world.get_classical_position()
            self.tracker.visited.add(tuple(class_pos))

            self.perceptual.perceive(next_obs)
            self.perceptual.settle(n_iter=10)
            if train:
                self.perceptual.learn()

            if done:
                if verbose:
                    print(f"\n>>> ALL WAYPOINTS REACHED in {step+1} steps!")
                    print(f">>> Tunneling events: {len(self.world.telemetry['tunnel_events'])}")
                    print(f">>> Collapse events: {len(self.world.telemetry['collapse_events'])}")
                return True, step + 1

        if verbose:
            print(f"\n>>> Max steps ({max_steps}) reached.")
            print(f">>> Waypoints reached: {self.world.waypoint_idx}/{len(self.world.waypoints)}")
        return False, max_steps


# ==============================================================================
# PART Q6: Quantum Visualization
# ==============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    QUANTUM_VIZ = True
except ImportError:
    QUANTUM_VIZ = False


class QuantumRealmVisualizer:
    """Visualization tools for quantum realm dynamics."""

    def __init__(self):
        pass

    def plot_wavepacket(self, world, save_path=None):
        """Plot probability density, phase, real, and imaginary parts."""
        if not QUANTUM_VIZ:
            print("Matplotlib not available")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        probs = world.wavepacket.probability_density()
        phases = world.wavepacket.phase_field()
        real = np.real(world.wavepacket.psi)
        imag = np.imag(world.wavepacket.psi)

        # Probability density
        ax = axes[0, 0]
        im = ax.imshow(probs, cmap='hot', interpolation='nearest')
        ax.set_title('Probability Density |ψ|²', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        self._overlay_features(ax, world)

        # Phase
        ax = axes[0, 1]
        im = ax.imshow(phases, cmap='hsv', interpolation='nearest')
        ax.set_title('Phase arg(ψ)', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        self._overlay_features(ax, world)

        # Real part
        ax = axes[1, 0]
        vmax = np.max(np.abs(real))
        im = ax.imshow(real, cmap='RdBu', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title('Re(ψ)', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        self._overlay_features(ax, world)

        # Imaginary part
        ax = axes[1, 1]
        vmax = np.max(np.abs(imag))
        im = ax.imshow(imag, cmap='RdBu', vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_title('Im(ψ)', fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
        self._overlay_features(ax, world)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def _overlay_features(self, ax, world):
        """Overlay planets, asteroids, waypoints on the plot."""
        for pr, pc, radius, _ in world.planets:
            circle = plt.Circle((pc, pr), radius, color='blue', fill=False, linewidth=2)
            ax.add_patch(circle)
        for ar, ac in world.asteroids:
            ax.plot(ac, ar, 'kx', markersize=8)
        for i, wp in enumerate(world.waypoints):
            color = 'green' if i < world.waypoint_idx else 'yellow'
            ax.plot(wp[1], wp[0], 'o', color=color, markersize=10, markeredgecolor='black')

    def animate_probabilities(self, world, save_path=None):
        """Plot probability evolution over time."""
        if not QUANTUM_VIZ or len(world.telemetry['prob_history']) == 0:
            return None

        n_frames = min(16, len(world.telemetry['prob_history']))
        indices = np.linspace(0, len(world.telemetry['prob_history'])-1, n_frames, dtype=int)

        rows = int(np.ceil(np.sqrt(n_frames)))
        cols = int(np.ceil(n_frames / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = np.array(axes).flatten()

        for idx, ax_idx in zip(indices, range(n_frames)):
            ax = axes[ax_idx]
            probs = world.telemetry['prob_history'][idx]
            im = ax.imshow(probs, cmap='hot', interpolation='nearest')
            ax.set_title(f'Step {idx}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide extra axes
        for ax_idx in range(n_frames, len(axes)):
            axes[ax_idx].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def plot_energy_trajectory(self, world, save_path=None):
        """Plot energy and position expectation over time."""
        if not QUANTUM_VIZ or len(world.telemetry['energy_history']) == 0:
            return None

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Energy
        ax = axes[0]
        energies = world.telemetry['energy_history']
        ax.plot(energies, 'b-', linewidth=1)
        ax.set_ylabel('Energy Expectation ⟨H⟩')
        ax.set_title('Energy Evolution', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark collapse events
        for event in world.telemetry['collapse_events']:
            ax.axvline(x=event['step'], color='red', linestyle='--', alpha=0.5)

        # Position trajectory
        ax = axes[1]
        path = np.array(world.telemetry['path'])
        ax.plot(path[:, 1], path[:, 0], 'b-', alpha=0.5, label='⟨position⟩')
        ax.plot(path[0, 1], path[0, 0], 'go', markersize=10, label='Start')
        ax.plot(path[-1, 1], path[-1, 0], 'ro', markersize=10, label='End')

        # Waypoints
        for i, wp in enumerate(world.waypoints):
            color = 'green' if i < world.waypoint_idx else 'yellow'
            ax.plot(wp[1], wp[0], 'o', color=color, markersize=12, markeredgecolor='black')

        ax.set_xlim(0, world.size - 1)
        ax.set_ylim(world.size - 1, 0)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title('Position Expectation Trajectory', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def plot_tunneling_events(self, world, save_path=None):
        """Visualize tunneling events."""
        if not QUANTUM_VIZ or len(world.telemetry['tunnel_events']) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 6))
        events = world.telemetry['tunnel_events']
        steps = [e['step'] for e in events]
        probs = [e['probability'] for e in events]

        ax.scatter(steps, probs, c='purple', s=100, alpha=0.7, edgecolors='black')
        ax.set_xlabel('Step')
        ax.set_ylabel('Tunneling Probability')
        ax.set_title('Quantum Tunneling Events', fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig


# ==============================================================================
# PART Q7: Main Demo
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("QUANTUM REALM — Full Quantum Grid World Demo")
    print("=" * 70)

    # Create quantum realm
    realm = QuantumRealmGridWorld(size=20, n_planets=3, n_asteroids=4,
                                   dt=0.15, mass=1.0, hbar=0.5, decoherence_rate=0.005)

    print(f"\n[1] Quantum Realm Created")
    print(f"    Grid: {realm.size}x{realm.size}")
    print(f"    Planets: {len(realm.planets)} (potential wells)")
    print(f"    Asteroids: {len(realm.asteroids)} (barriers with tunneling)")
    print(f"    Waypoints: {realm.waypoints}")
    print(f"    ħ: {realm.hbar} | m: {realm.mass} | dt: {realm.dt}")

    print(f"\n[2] Initial Wavepacket State")
    pos = realm.get_classical_position()
    unc = realm.get_position_uncertainty()
    print(f"    Classical position: {pos}")
    print(f"    Position uncertainty: σ=({unc[0]:.2f}, {unc[1]:.2f})")
    print(f"    Initial realm:")
    realm.print_realm()

    # Create perceptual agent (input = 4 channels x 400 cells = 1600)
    print(f"\n[3] Creating Quantum EFE Agent")
    p = Tatha(input_size=1600, hidden_size=64, belief_size=16, lr=0.005)
    qagent = QuantumEFEAgent(realm, p, exploration_coef=0.25)

    # Navigate
    print(f"\n[4] Quantum Navigation (max 80 steps)")
    success, steps = qagent.navigate(max_steps=80, train=True, verbose=True)

    print(f"\n[5] Results")
    print(f"    Success: {success}")
    print(f"    Steps taken: {steps}")
    print(f"    Waypoints reached: {realm.waypoint_idx}/{len(realm.waypoints)}")
    print(f"    Tunneling events: {len(realm.telemetry['tunnel_events'])}")
    print(f"    Collapse events: {len(realm.telemetry['collapse_events'])}")

    if realm.telemetry['tunnel_events']:
        print(f"    Tunneling probabilities:")
        for e in realm.telemetry['tunnel_events'][:5]:
            print(f"      Step {e['step']}: P_tunnel={e['probability']:.4f}")

    # Visualization
    if QUANTUM_VIZ:
        print(f"\n[6] Generating Quantum Visualizations")
        viz = QuantumRealmVisualizer()

        viz.plot_wavepacket(realm, save_path="/mnt/agents/output/quantum_wavepacket.png")
        viz.animate_probabilities(realm, save_path="/mnt/agents/output/quantum_prob_evolution.png")
        viz.plot_energy_trajectory(realm, save_path="/mnt/agents/output/quantum_energy.png")
        viz.plot_tunneling_events(realm, save_path="/mnt/agents/output/quantum_tunneling.png")

        print("    Generated:")
        print("      - quantum_wavepacket.png")
        print("      - quantum_prob_evolution.png")
        print("      - quantum_energy.png")
        print("      - quantum_tunneling.png")
    else:
        print(f"\n[6] Matplotlib not available — skipping visualization")

    print("\n" + "=" * 70)
    print("Quantum Realm demo complete.")
    print("=" * 70)
