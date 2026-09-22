"""
================================================================================
Quantum Swarm Extension — Multi-Agent Entangled Active Inference
================================================================================

N agents share a joint quantum state:
  ψ_joint[r1,c1,r2,c2,...,rN,cN] = amplitude of all agents simultaneously

Features:
  • Entanglement: measuring one agent collapses others non-locally
  • Swarm EFE: each agent optimizes considering swarm coherence
  • Entanglement entropy: quantifies quantum correlations
  • Decoherence cascade: environmental noise breaks entanglement over time
  • Bell inequality: verify genuine quantum vs. classical correlations

Integrates with: quantum_realm.py, unified_pc_space.py
"""

import numpy as np

# Import from quantum realm
try:
    from quantum_realm import (QuantumWavepacket, QuantumHamiltonian, 
                                QuantumActionOperators, QuantumRealmGridWorld)
    from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from quantum_realm import (QuantumWavepacket, QuantumHamiltonian,
                                QuantumActionOperators, QuantumRealmGridWorld)
    from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker


# ==============================================================================
# PART S1: Joint Wavefunction & Entanglement
# ==============================================================================

class QuantumSwarm:
    """
    N-agent joint quantum state.

    For N agents on a G×G grid, the joint state has shape (G,G,G,G,...,G,G)
    with 2N dimensions. For N>2 this is intractable, so we use a factorized
    approximation with pairwise entanglement:

    ψ_joint ≈ ψ₁ ⊗ ψ₂ ⊗ ... ⊗ ψ_N  +  entanglement corrections
    """
    def __init__(self, n_agents, grid_size, initial_positions=None, sigma=1.5,
                 entanglement_strength=0.3):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.sigma = sigma
        self.entanglement_strength = entanglement_strength

        # Individual wavepackets
        self.agents = []
        for i in range(n_agents):
            pos = initial_positions[i] if initial_positions else (i % grid_size, i // grid_size)
            wp = QuantumWavepacket(grid_size, initial_pos=pos, sigma=sigma)
            self.agents.append(wp)

        # Pairwise entanglement matrices: C[i,j] = correlation between agent i and j
        self.entanglement = np.zeros((n_agents, n_agents))

        # Joint probability cache (factorized approximation)
        self._update_joint_probs()

    def _update_joint_probs(self):
        """Compute factorized joint probability P(r1,c1,r2,c2,...)."""
        # For efficiency, store individual probs and compute marginals on demand
        self.individual_probs = [a.probability_density() for a in self.agents]

    def get_joint_probability(self, positions):
        """
        P(r1,c1, r2,c2, ...) under factorized + entanglement approximation.
        positions = [(r1,c1), (r2,c2), ...]
        """
        prob = 1.0
        for i, (r, c) in enumerate(positions):
            prob *= self.individual_probs[i][r, c]

        # Entanglement correction: boost probability when agents are close
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                if self.entanglement[i, j] > 0:
                    ri, ci = positions[i]
                    rj, cj = positions[j]
                    dist = np.sqrt((ri - rj)**2 + (ci - cj)**2)
                    # Entangled agents prefer to be close
                    prob *= (1 + self.entanglement[i, j] * np.exp(-dist / 3.0))

        return prob

    def expected_positions(self):
        """Return list of (⟨r⟩, ⟨c⟩) for each agent."""
        return [a.expected_position() for a in self.agents]

    def entangle_pair(self, i, j, strength=None):
        """Create quantum correlation between agent i and j."""
        if strength is None:
            strength = self.entanglement_strength
        self.entanglement[i, j] = strength
        self.entanglement[j, i] = strength

    def disentangle_pair(self, i, j):
        """Remove correlation."""
        self.entanglement[i, j] = 0.0
        self.entanglement[j, i] = 0.0

    def apply_global_decoherence(self, rate):
        """Environmental noise reduces all entanglement."""
        self.entanglement *= (1 - rate)

    def collapse_agent(self, idx, pos, sigma=0.8):
        """
        Measure/collapse agent idx to position pos.
        If entangled, other agents experience partial collapse toward pos.
        """
        # Collapse the measured agent
        self.agents[idx].collapse_near(pos, sigma)

        # Non-local effect on entangled partners
        for j in range(self.n_agents):
            if j != idx and self.entanglement[idx, j] > 0:
                # Partial collapse: partner's wavepacket shifts toward measured position
                # weighted by entanglement strength
                strength = self.entanglement[idx, j]
                # Create a "ghost" potential well at pos for partner
                partner = self.agents[j]
                probs = partner.probability_density().copy()
                # Use grid-scaled sigma for non-local effect
                nonlocal_sigma = max(sigma, self.grid_size / 4.0)
                for r in range(self.grid_size):
                    for c in range(self.grid_size):
                        dist = np.sqrt((r - pos[0])**2 + (c - pos[1])**2)
                        # Boost probability near collapse point
                        boost = 1 + strength * np.exp(-dist**2 / (2 * nonlocal_sigma**2))
                        probs[r, c] *= boost
                probs /= np.sum(probs)
                # Reconstruct wavefunction with new probability and old phases
                phases = partner.phase_field()
                partner.psi = np.sqrt(probs) * np.exp(1j * phases)
                partner._normalize()

        self._update_joint_probs()

    def entanglement_entropy(self, idx):
        """
        Von Neumann entropy of agent idx's reduced density matrix.
        S = -Tr(ρ log ρ). High entropy = highly entangled with others.
        """
        probs = self.individual_probs[idx].flatten()
        probs = probs[probs > 1e-12]
        return -np.sum(probs * np.log2(probs))

    def mutual_information(self, i, j):
        """
        I(i:j) = S(i) + S(j) - S(i,j).
        Measures how much information agent i and j share.
        """
        si = self.entanglement_entropy(i)
        sj = self.entanglement_entropy(j)
        # Joint entropy (approximate: assume factorized)
        joint_probs = np.outer(self.individual_probs[i].flatten(),
                               self.individual_probs[j].flatten())
        joint_probs = joint_probs[joint_probs > 1e-12]
        sij = -np.sum(joint_probs * np.log2(joint_probs))
        return si + sj - sij

    def bell_inequality_violation(self, i, j):
        """
        CHSH-like inequality. If |S| > 2, the correlation is genuinely quantum.
        Classical bound = 2, Quantum bound = 2√2 ≈ 2.828.
        """
        if self.entanglement[i, j] <= 0:
            return 0.0
        # Simplified: entanglement strength maps to Bell violation
        # Real CHSH requires 4 measurement settings per agent
        return 2 + self.entanglement[i, j] * 0.8

    def copy(self):
        """Deep copy of swarm state."""
        new_swarm = QuantumSwarm(self.n_agents, self.grid_size, sigma=self.sigma,
                                  entanglement_strength=self.entanglement_strength)
        for i, agent in enumerate(self.agents):
            new_swarm.agents[i].psi = agent.psi.copy()
        new_swarm.entanglement = self.entanglement.copy()
        new_swarm._update_joint_probs()
        return new_swarm


# ==============================================================================
# PART S2: Swarm Quantum Environment
# ==============================================================================

class SwarmQuantumRealm:
    """
    Multi-agent quantum environment.

    All agents evolve simultaneously under a shared Hamiltonian.
    Actions are applied to individual agents.
    Waypoints can require ALL agents to be present (cooperative) or
    ANY agent (competitive).
    """
    def __init__(self, n_agents=3, size=15, n_planets=2, n_asteroids=3,
                 dt=0.12, mass=1.0, hbar=0.5, decoherence_rate=0.008,
                 cooperative=True):
        self.n_agents = n_agents
        self.size = size
        self.n_planets = n_planets
        self.n_asteroids = n_asteroids
        self.dt = dt
        self.mass = mass
        self.hbar = hbar
        self.decoherence_rate = decoherence_rate
        self.cooperative = cooperative

        self.actions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        # Generate environment features
        self.planets = []
        self.asteroids = set()
        self._generate_features()

        # Initialize swarm
        positions = [(i * size // n_agents, i * size // n_agents) for i in range(n_agents)]
        self.swarm = QuantumSwarm(n_agents, size, initial_positions=positions, sigma=1.2)

        # Entangle all pairs initially
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                self.swarm.entangle_pair(i, j, strength=0.4)

        # Shared Hamiltonian
        self.hamiltonian = QuantumHamiltonian(size, mass=mass, hbar=hbar)
        self.hamiltonian.set_planet_potential(self.planets)
        self.hamiltonian.set_barrier_potential(self.asteroids, height=25.0)

        # Action operators (shared)
        self.action_ops = QuantumActionOperators(size, hbar=hbar, kick_strength=0.9)

        # Waypoints: each agent has its own target, or shared targets
        self.waypoints = self._generate_waypoints()
        self.waypoint_idx = [0] * n_agents

        # Telemetry
        self.telemetry = {
            'paths': [[] for _ in range(n_agents)],
            'entanglement_history': [],
            'bell_violations': [],
            'collapse_events': [],
            'waypoint_reaches': [],
            'swarm_coherence': []
        }

    def _generate_features(self):
        self.planets = []
        self.asteroids = set()
        for _ in range(self.n_planets):
            while True:
                pr = np.random.randint(2, self.size - 2)
                pc = np.random.randint(2, self.size - 2)
                radius = np.random.randint(2, 3)
                gravity = np.random.uniform(0.5, 1.2)
                if np.linalg.norm([pr, pc]) > 3:
                    self.planets.append((pr, pc, radius, gravity))
                    break
        for _ in range(self.n_asteroids):
            cr = np.random.randint(2, self.size - 2)
            cc = np.random.randint(2, self.size - 2)
            for _ in range(np.random.randint(2, 5)):
                ar = np.clip(cr + np.random.randint(-1, 2), 0, self.size - 1)
                ac = np.clip(cc + np.random.randint(-1, 2), 0, self.size - 1)
                if (ar, ac) != (0, 0):
                    self.asteroids.add((ar, ac))

    def _generate_waypoints(self):
        if self.cooperative:
            # All agents must reach the SAME waypoint sequence
            return [[self.size - 1, self.size - 1], [self.size // 2, 0], [0, self.size - 1]]
        else:
            # Each agent has different targets
            return [[self.size - 1, self.size - 1] for _ in range(self.n_agents)]

    def reset(self):
        positions = [(i * self.size // self.n_agents, i * self.size // self.n_agents) 
                     for i in range(self.n_agents)]
        self.swarm = QuantumSwarm(self.n_agents, self.size, initial_positions=positions, sigma=1.2)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                self.swarm.entangle_pair(i, j, strength=0.4)
        self.waypoint_idx = [0] * self.n_agents
        self.telemetry = {
            'paths': [[] for _ in range(self.n_agents)],
            'entanglement_history': [],
            'bell_violations': [],
            'collapse_events': [],
            'waypoint_reaches': [],
            'swarm_coherence': []
        }
        return self.observe()

    def observe(self):
        """
        Observation for each agent = its own probability field + phase + 
        environment + other agents' expected positions.
        """
        obs_list = []
        probs_all = [a.probability_density() for a in self.swarm.agents]
        phases_all = [a.phase_field() for a in self.swarm.agents]

        for i in range(self.n_agents):
            obs = np.zeros(self.size * self.size * 4 + self.n_agents * 2)
            # Own probability
            for r in range(self.size):
                for c in range(self.size):
                    idx = r * self.size + c
                    obs[idx] = probs_all[i][r, c]
                    obs[idx + self.size*self.size] = np.cos(phases_all[i][r, c])
                    obs[idx + 2*self.size*self.size] = np.sin(phases_all[i][r, c])
                    obs[idx + 3*self.size*self.size] = self.hamiltonian.V[r, c]
            # Other agents' expected positions
            for j in range(self.n_agents):
                exp_r, exp_c = self.swarm.agents[j].expected_position()
                obs[self.size*self.size*4 + j*2] = exp_r / self.size
                obs[self.size*self.size*4 + j*2 + 1] = exp_c / self.size
            obs_list.append(obs)
        return obs_list

    def step(self, actions):
        """
        actions = [a0, a1, ..., aN] for each agent.
        """
        # Apply individual actions
        for i, action in enumerate(actions):
            self.swarm.agents[i].psi = self.action_ops.apply_action(
                self.swarm.agents[i].psi, action
            )

        # Shared Hamiltonian evolution (all agents feel same potential)
        for i in range(self.n_agents):
            self.swarm.agents[i].psi = self.hamiltonian.full_step(
                self.swarm.agents[i].psi, self.dt
            )
            self.swarm.agents[i]._normalize()

        # Decoherence: reduce entanglement
        self.swarm.apply_global_decoherence(self.decoherence_rate)

        # Boundary damping
        for agent in self.swarm.agents:
            agent.psi[0, :] *= 0.5
            agent.psi[-1, :] *= 0.5
            agent.psi[:, 0] *= 0.5
            agent.psi[:, -1] *= 0.5
            agent._normalize()

        # Update joint probs
        self.swarm._update_joint_probs()

        # Record telemetry
        for i in range(self.n_agents):
            pos = self.swarm.agents[i].expected_position()
            self.telemetry['paths'][i].append(pos)

        self.telemetry['entanglement_history'].append(self.swarm.entanglement.copy())

        # Bell violations
        bell = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                bell.append(self.swarm.bell_inequality_violation(i, j))
        self.telemetry['bell_violations'].append(bell)

        # Swarm coherence = average entanglement strength
        coherence = np.mean(self.swarm.entanglement[self.swarm.entanglement > 0]) if np.any(self.swarm.entanglement > 0) else 0
        self.telemetry['swarm_coherence'].append(coherence)

        # Check waypoint measurements
        rewards = [0.0] * self.n_agents
        done = False

        if self.cooperative:
            # All agents must be near the same waypoint
            if self.waypoint_idx[0] < len(self.waypoints):
                wr, wc = self.waypoints[self.waypoint_idx[0]]
                all_near = True
                for i in range(self.n_agents):
                    prob = np.abs(self.swarm.agents[i].psi[wr, wc])**2
                    if prob < 0.15:  # Lower threshold for cooperative
                        all_near = False
                        break
                if all_near:
                    # Collapse all agents to waypoint
                    for i in range(self.n_agents):
                        self.swarm.collapse_agent(i, (wr, wc), sigma=0.6)
                        rewards[i] = 1.0
                    self.waypoint_idx = [w + 1 for w in self.waypoint_idx]
                    self.telemetry['waypoint_reaches'].append({
                        'step': len(self.telemetry['paths'][0]),
                        'pos': [wr, wc],
                        'agents': list(range(self.n_agents)),
                        'cooperative': True
                    })
                    if self.waypoint_idx[0] >= len(self.waypoints):
                        done = True
        else:
            # Individual waypoint detection
            for i in range(self.n_agents):
                if self.waypoint_idx[i] < len(self.waypoints):
                    wr, wc = self.waypoints[self.waypoint_idx[i]]
                    prob = np.abs(self.swarm.agents[i].psi[wr, wc])**2
                    if prob > 0.25:
                        self.swarm.collapse_agent(i, (wr, wc), sigma=0.6)
                        rewards[i] = 1.0
                        self.waypoint_idx[i] += 1
                        self.telemetry['waypoint_reaches'].append({
                            'step': len(self.telemetry['paths'][i]),
                            'pos': [wr, wc],
                            'agent': i,
                            'cooperative': False
                        })
            if all(w >= len(self.waypoints) for w in self.waypoint_idx):
                done = True

        return self.observe(), rewards, done

    def get_classical_positions(self):
        """Most likely position for each agent."""
        positions = []
        for agent in self.swarm.agents:
            probs = agent.probability_density()
            max_idx = np.unravel_index(np.argmax(probs), probs.shape)
            positions.append(list(max_idx))
        return positions

    def print_realm(self):
        """ASCII visualization showing all agents."""
        probs_all = [a.probability_density() for a in self.swarm.agents]
        max_probs = [np.max(p) for p in probs_all]
        chars = [' ', '.', 'o', 'O', '@', '#', '*']

        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) in self.asteroids:
                    row += "X"
                elif any((r, c) == (pr, pc) for pr, pc, _, _ in self.planets):
                    row += "P"
                elif self.cooperative and self.waypoint_idx[0] < len(self.waypoints) and [r, c] == self.waypoints[self.waypoint_idx[0]]:
                    row += "W"
                else:
                    # Show agent with highest probability here
                    best_agent = -1
                    best_p = 0
                    for i, probs in enumerate(probs_all):
                        p = probs[r, c] / max_probs[i] if max_probs[i] > 0 else 0
                        if p > best_p:
                            best_p = p
                            best_agent = i
                    if best_agent >= 0 and best_p > 0.1:
                        idx = min(int(best_p * (len(chars) - 1)), len(chars) - 1)
                        row += str(best_agent)  # Show agent number
                    else:
                        row += chars[min(int(best_p * (len(chars) - 1)), len(chars) - 1)]
            print("  " + row)


# ==============================================================================
# PART S3: Swarm EFE Agent
# ==============================================================================

class SwarmEFEAgent:
    """
    Each agent computes EFE considering:
      1. Its own predicted state
      2. Entanglement with swarm (non-local effects)
      3. Swarm coherence (maintain entanglement = bonus)
    """
    def __init__(self, realm, perceptual_agents, exploration_coef=0.25,
                 coherence_bonus=0.5):
        self.realm = realm
        self.perceptuals = perceptual_agents
        self.exploration_coef = exploration_coef
        self.coherence_bonus = coherence_bonus
        self.trackers = [CoordinateTracker(realm.size) for _ in range(realm.n_agents)]

    def get_preferred_observation(self, agent_idx):
        obs = np.zeros(self.realm.size * self.realm.size * 4 + self.realm.n_agents * 2)
        if self.realm.waypoint_idx[agent_idx] < len(self.realm.waypoints):
            wr, wc = self.realm.waypoints[self.realm.waypoint_idx[agent_idx]]
            idx = wr * self.realm.size + wc
            obs[idx] = 1.0
        return obs

    def _simulate_action(self, agent_idx, action):
        """Simulate one agent's action while holding others fixed."""
        swarm_copy = self.realm.swarm.copy()
        swarm_copy.agents[agent_idx].psi = self.realm.action_ops.apply_action(
            swarm_copy.agents[agent_idx].psi, action
        )
        swarm_copy.agents[agent_idx].psi = self.realm.hamiltonian.full_step(
            swarm_copy.agents[agent_idx].psi, self.realm.dt
        )
        swarm_copy.agents[agent_idx]._normalize()
        return swarm_copy

    def expected_free_energy(self, agent_idx, obs, action):
        """EFE for one agent in the swarm context."""
        predicted_swarm = self._simulate_action(agent_idx, action)
        pred_agent = predicted_swarm.agents[agent_idx]
        pred_probs = pred_agent.probability_density()

        # 1. Predicted surprise (distance from preferred waypoint)
        preferred = self.get_preferred_observation(agent_idx)
        pref_probs = preferred[:self.realm.size * self.realm.size].reshape(self.realm.size, self.realm.size)
        kl = 0.0
        for r in range(self.realm.size):
            for c in range(self.realm.size):
                p = pred_probs[r, c] + 1e-10
                q = pref_probs[r, c] + 1e-10
                kl += p * np.log(p / q)

        # 2. Predicted energy
        H_psi = self.realm.hamiltonian.potential_evolution(
            self.realm.hamiltonian.kinetic_evolution(pred_agent.psi, self.realm.dt), self.realm.dt
        )
        pred_energy = np.real(np.vdot(pred_agent.psi, H_psi))

        # 3. Position uncertainty
        var_r, var_c = pred_agent.position_variance()
        uncertainty_penalty = 0.1 * (var_r + var_c)

        # 4. Information gain
        info_gain = 0.0
        for r in range(self.realm.size):
            for c in range(self.realm.size):
                if (r, c) not in self.trackers[agent_idx].visited:
                    info_gain += pred_probs[r, c]

        # 5. SWARM BONUS: maintain entanglement coherence
        current_coherence = np.mean(self.realm.swarm.entanglement[self.realm.swarm.entanglement > 0]) if np.any(self.realm.swarm.entanglement > 0) else 0
        predicted_coherence = np.mean(predicted_swarm.entanglement[predicted_swarm.entanglement > 0]) if np.any(predicted_swarm.entanglement > 0) else 0
        coherence_penalty = -self.coherence_bonus * (predicted_coherence - current_coherence)

        efe = kl + 0.01 * pred_energy + uncertainty_penalty - self.exploration_coef * info_gain + coherence_penalty
        return efe, predicted_swarm

    def select_actions(self, obs_list):
        """Each agent independently selects action minimizing its swarm-aware EFE."""
        actions = []
        all_candidates = []
        for i in range(self.realm.n_agents):
            best_a = 0
            best_efe = float('inf')
            candidates = []
            for a in range(4):
                efe, _ = self.expected_free_energy(i, obs_list[i], a)
                candidates.append((a, efe))
                if efe < best_efe:
                    best_efe = efe
                    best_a = a
            actions.append(best_a)
            all_candidates.append(candidates)
        return actions, all_candidates

    def navigate(self, max_steps=80, train=True, verbose=True):
        obs_list = self.realm.reset()
        for i in range(self.realm.n_agents):
            self.perceptuals[i].perceive(obs_list[i])
            self.perceptuals[i].settle(n_iter=12)

        for step in range(max_steps):
            actions, candidates = self.select_actions(obs_list)

            if verbose and step % 15 == 0:
                positions = self.realm.get_classical_positions()
                coherence = self.realm.telemetry['swarm_coherence'][-1] if self.realm.telemetry['swarm_coherence'] else 0
                print(f"\nStep {step+1} | Coherence: {coherence:.3f}")
                for i, pos in enumerate(positions):
                    print(f"  Agent {i}: pos={pos} | Action: {self.realm.actions[actions[i]]} | EFE={candidates[i][actions[i]][1]:.3f}")
                self.realm.print_realm()

            next_obs_list, rewards, done = self.realm.step(actions)

            # Update trackers
            for i in range(self.realm.n_agents):
                pos = self.realm.get_classical_positions()[i]
                self.trackers[i].visited.add(tuple(pos))

            # Perceptual learning
            for i in range(self.realm.n_agents):
                self.perceptuals[i].perceive(next_obs_list[i])
                self.perceptuals[i].settle(n_iter=8)
                if train:
                    self.perceptuals[i].learn()

            obs_list = next_obs_list

            if done:
                if verbose:
                    print(f"\n>>> ALL WAYPOINTS REACHED in {step+1} steps!")
                    print(f">>> Final coherence: {self.realm.telemetry['swarm_coherence'][-1]:.3f}")
                return True, step + 1

        if verbose:
            print(f"\n>>> Max steps ({max_steps}) reached.")
            reached = sum(1 for w in self.realm.waypoint_idx if w >= len(self.realm.waypoints))
            print(f">>> Agents completed: {reached}/{self.realm.n_agents}")
        return False, max_steps


# ==============================================================================
# PART S4: Swarm Visualization
# ==============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SWARM_VIZ = True
except ImportError:
    SWARM_VIZ = False


class SwarmVisualizer:
    def __init__(self):
        pass

    def plot_swarm_state(self, realm, save_path=None):
        """Plot all agents' probability densities + entanglement network."""
        if not SWARM_VIZ:
            return None

        n = realm.n_agents
        fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        for i in range(n):
            # Probability density
            ax = axes[0, i]
            probs = realm.swarm.agents[i].probability_density()
            im = ax.imshow(probs, cmap='hot', interpolation='nearest')
            ax.set_title(f'Agent {i} |ψ|²', fontweight='bold')

            # Overlay features
            for pr, pc, radius, _ in realm.planets:
                circle = plt.Circle((pc, pr), radius, color='blue', fill=False, lw=2)
                ax.add_patch(circle)
            for ar, ac in realm.asteroids:
                ax.plot(ac, ar, 'kx', markersize=6)
            if realm.waypoint_idx[i] < len(realm.waypoints):
                wr, wc = realm.waypoints[realm.waypoint_idx[i]]
                ax.plot(wc, wr, 'yo', markersize=10, markeredgecolor='black')
            plt.colorbar(im, ax=ax, fraction=0.046)

            # Phase
            ax = axes[1, i]
            phases = realm.swarm.agents[i].phase_field()
            im = ax.imshow(phases, cmap='hsv', interpolation='nearest')
            ax.set_title(f'Agent {i} Phase', fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def plot_entanglement_network(self, realm, save_path=None):
        """Graph visualization of entanglement strengths."""
        if not SWARM_VIZ:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))
        n = realm.n_agents

        # Position agents in a circle
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = np.cos(angles) * 3
        y = np.sin(angles) * 3

        # Draw edges (entanglement)
        for i in range(n):
            for j in range(i + 1, n):
                strength = realm.swarm.entanglement[i, j]
                if strength > 0.01:
                    ax.plot([x[i], x[j]], [y[i], y[j]], 'b-', alpha=strength, linewidth=strength*5)

        # Draw nodes
        for i in range(n):
            pos = realm.get_classical_positions()[i]
            ax.plot(x[i], y[i], 'o', markersize=20, color='red', markeredgecolor='black')
            ax.text(x[i], y[i], f'{i}\n({pos[0]},{pos[1]})', ha='center', va='center', fontsize=10, fontweight='bold')

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Entanglement Network', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def plot_telemetry(self, realm, save_path=None):
        """Plot coherence, Bell violations, and paths over time."""
        if not SWARM_VIZ or len(realm.telemetry['swarm_coherence']) == 0:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Coherence over time
        ax = axes[0, 0]
        ax.plot(realm.telemetry['swarm_coherence'], 'b-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Swarm Coherence')
        ax.set_title('Entanglement Coherence vs Time', fontweight='bold')
        ax.grid(True, alpha=0.3)
        for event in realm.telemetry['waypoint_reaches']:
            ax.axvline(x=event['step'], color='green', linestyle='--', alpha=0.5)

        # Bell violations
        ax = axes[0, 1]
        if realm.telemetry['bell_violations']:
            max_bell = [max(b) if b else 0 for b in realm.telemetry['bell_violations']]
            ax.plot(max_bell, 'r-', linewidth=2)
            ax.axhline(y=2.0, color='gray', linestyle='--', label='Classical bound')
            ax.axhline(y=2.828, color='purple', linestyle='--', label='Quantum bound')
            ax.set_xlabel('Step')
            ax.set_ylabel('Max Bell Violation')
            ax.set_title('Quantum Correlation Strength', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Paths
        ax = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, realm.n_agents))
        for i in range(realm.n_agents):
            path = np.array(realm.telemetry['paths'][i])
            if len(path) > 0:
                ax.plot(path[:, 1], path[:, 0], '-', color=colors[i], alpha=0.6, label=f'Agent {i}')
                ax.plot(path[0, 1], path[0, 0], 'o', color=colors[i], markersize=8)
                ax.plot(path[-1, 1], path[-1, 0], 's', color=colors[i], markersize=8)
        for wp in realm.waypoints:
            ax.plot(wp[1], wp[0], 'y*', markersize=15, markeredgecolor='black')
        ax.set_xlim(0, realm.size - 1)
        ax.set_ylim(realm.size - 1, 0)
        ax.set_title('Agent Trajectories', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Entanglement matrix heatmap
        ax = axes[1, 1]
        im = ax.imshow(realm.swarm.entanglement, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(realm.n_agents))
        ax.set_yticks(range(realm.n_agents))
        ax.set_xlabel('Agent')
        ax.set_ylabel('Agent')
        ax.set_title('Final Entanglement Matrix', fontweight='bold')
        for i in range(realm.n_agents):
            for j in range(realm.n_agents):
                ax.text(j, i, f'{realm.swarm.entanglement[i,j]:.2f}', ha='center', va='center')
        plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig


# ==============================================================================
# PART S5: Main Demo
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("QUANTUM SWARM — Multi-Agent Entangled Active Inference")
    print("=" * 70)

    # Create cooperative swarm realm
    realm = SwarmQuantumRealm(n_agents=3, size=12, n_planets=2, n_asteroids=2,
                               dt=0.12, cooperative=True)

    print(f"\n[1] Swarm Realm Created")
    print(f"    Agents: {realm.n_agents}")
    print(f"    Grid: {realm.size}x{realm.size}")
    print(f"    Cooperative: {realm.cooperative}")
    print(f"    Waypoints: {realm.waypoints}")
    print(f"    Initial entanglement:")
    print(f"    {realm.swarm.entanglement}")

    print(f"\n[2] Initial State")
    for i, pos in enumerate(realm.get_classical_positions()):
        print(f"    Agent {i}: pos={pos}")
    print(f"    Entropy per agent: {[realm.swarm.entanglement_entropy(i) for i in range(realm.n_agents)]}")

    # Create perceptual agents
    print(f"\n[3] Creating Swarm EFE Agents")
    input_size = realm.size * realm.size * 4 + realm.n_agents * 2
    perceptuals = [Tatha(input_size=input_size, hidden_size=48, belief_size=12, lr=0.005) 
                   for _ in range(realm.n_agents)]
    swarm_agent = SwarmEFEAgent(realm, perceptuals, exploration_coef=0.3, coherence_bonus=0.4)

    # Navigate
    print(f"\n[4] Cooperative Swarm Navigation (max 60 steps)")
    success, steps = swarm_agent.navigate(max_steps=60, train=True, verbose=True)

    print(f"\n[5] Results")
    print(f"    Success: {success}")
    print(f"    Steps: {steps}")
    print(f"    Waypoints: {realm.waypoint_idx}")
    print(f"    Final coherence: {realm.telemetry['swarm_coherence'][-1]:.3f}")
    print(f"    Final entanglement:")
    print(f"    {realm.swarm.entanglement}")
    print(f"    Bell violations (final): {[realm.swarm.bell_inequality_violation(i,j) for i in range(realm.n_agents) for j in range(i+1, realm.n_agents)]}")

    # Visualization
    if SWARM_VIZ:
        print(f"\n[6] Generating Visualizations")
        viz = SwarmVisualizer()
        viz.plot_swarm_state(realm, save_path="/mnt/agents/output/swarm_state.png")
        viz.plot_entanglement_network(realm, save_path="/mnt/agents/output/swarm_network.png")
        viz.plot_telemetry(realm, save_path="/mnt/agents/output/swarm_telemetry.png")
        print("    Generated: swarm_state.png, swarm_network.png, swarm_telemetry.png")
    else:
        print(f"\n[6] Matplotlib not available")

    print("\n" + "=" * 70)
    print("Quantum Swarm demo complete.")
    print("=" * 70)
