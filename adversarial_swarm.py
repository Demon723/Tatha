"""
================================================================================
Adversarial Swarm — Zero-Sum Quantum Active Inference
================================================================================

Two competing quantum swarms:
  • RED SWARM: Navigates to waypoints (cooperative within swarm)
  • BLUE SWARM: Predicts RED's trajectory and places barriers to block

Both swarms run predictive coding + quantum EFE.
RED minimizes distance-to-waypoint.
BLUE minimizes RED's success probability (maximizes RED's EFE).

Game-theoretic active inference: each swarm is a "player" in a
partially-observable stochastic game. The Nash equilibrium is approximated
via iterative best-response EFE minimization.

Integrates with: quantum_swarm.py, quantum_realm.py, unified_pc_space.py
"""

import numpy as np

import sys, os
sys.path.insert(0, '/mnt/agents/output')
from quantum_swarm import (QuantumSwarm, SwarmQuantumRealm, SwarmEFEAgent, SwarmVisualizer,
                            QuantumWavepacket, QuantumHamiltonian, QuantumActionOperators)
from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker


# ==============================================================================
# PART A1: Adversarial Swarm Environment
# ==============================================================================

class AdversarialSwarmRealm:
    """
    Two-team quantum arena.

    RED (offense): n_red agents, must reach waypoints cooperatively.
    BLUE (defense): n_blue agents, place temporary barriers to block RED.

    Turn structure:
      1. RED moves all agents (Schrödinger evolution + actions)
      2. BLUE observes RED's probability field, places barriers
      3. Environment updates (barriers decay over time)
      4. Check waypoint measurements
    """
    def __init__(self, n_red=3, n_blue=2, size=14, n_planets=2,
                 dt=0.12, barrier_decay=0.15, max_barriers=8):
        self.n_red = n_red
        self.n_blue = n_blue
        self.size = size
        self.dt = dt
        self.barrier_decay = barrier_decay
        self.max_barriers = max_barriers

        self.actions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PLACE_BARRIER"}
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PLACE"}

        # Static features
        self.planets = []
        self._generate_planets(n_planets)

        # Dynamic barriers (placed by BLUE)
        self.barriers = set()
        self.barrier_lifetime = {}  # (r,c) -> remaining strength

        # RED swarm
        red_positions = [(i * size // (n_red + 1), 0) for i in range(n_red)]
        self.red_swarm = QuantumSwarm(n_red, size, initial_positions=red_positions, sigma=1.2)
        for i in range(n_red):
            for j in range(i + 1, n_red):
                self.red_swarm.entangle_pair(i, j, strength=0.35)

        # BLUE swarm
        blue_positions = [(size // 2, size - 1 - i * 3) for i in range(n_blue)]
        self.blue_swarm = QuantumSwarm(n_blue, size, initial_positions=blue_positions, sigma=1.0)
        for i in range(n_blue):
            for j in range(i + 1, n_blue):
                self.blue_swarm.entangle_pair(i, j, strength=0.25)

        # Waypoints for RED
        self.waypoints = [
            [size - 2, size - 2],
            [size // 2, size - 2],
            [size - 2, size // 2]
        ]
        self.waypoint_idx = 0

        # Shared Hamiltonian (both swarms feel same potential)
        self.hamiltonian = QuantumHamiltonian(size, mass=1.0, hbar=0.5)
        self.hamiltonian.set_planet_potential(self.planets)

        # Action operators
        self.red_ops = QuantumActionOperators(size, hbar=0.5, kick_strength=0.9)
        self.blue_ops = QuantumActionOperators(size, hbar=0.5, kick_strength=0.7)

        # Telemetry
        self.telemetry = {
            'red_paths': [[] for _ in range(n_red)],
            'blue_paths': [[] for _ in range(n_blue)],
            'barrier_history': [],
            'red_coherence': [],
            'blue_coherence': [],
            'waypoint_reaches': [],
            'red_surprise': [],  # How hard RED is finding it
            'blue_success': []   # How well BLUE is blocking
        }

    def _generate_planets(self, n_planets):
        self.planets = []
        for _ in range(n_planets):
            while True:
                pr = np.random.randint(3, self.size - 3)
                pc = np.random.randint(3, self.size - 3)
                radius = np.random.randint(2, 3)
                gravity = np.random.uniform(0.5, 1.0)
                if np.linalg.norm([pr, pc]) > 3 and np.linalg.norm([pr - self.size, pc - self.size]) > 3:
                    self.planets.append((pr, pc, radius, gravity))
                    break

    def _update_hamiltonian(self):
        """Recompute potential with current barriers."""
        self.hamiltonian.V = np.zeros((self.size, self.size))
        self.hamiltonian.set_planet_potential(self.planets)
        if self.barriers:
            self.hamiltonian.set_barrier_potential(self.barriers, height=40.0)

    def reset(self):
        self.barriers = set()
        self.barrier_lifetime = {}
        self.waypoint_idx = 0

        red_positions = [(i * self.size // (self.n_red + 1), 0) for i in range(self.n_red)]
        self.red_swarm = QuantumSwarm(self.n_red, self.size, initial_positions=red_positions, sigma=1.2)
        for i in range(self.n_red):
            for j in range(i + 1, self.n_red):
                self.red_swarm.entangle_pair(i, j, strength=0.35)

        blue_positions = [(self.size // 2, self.size - 1 - i * 3) for i in range(self.n_blue)]
        self.blue_swarm = QuantumSwarm(self.n_blue, self.size, initial_positions=blue_positions, sigma=1.0)
        for i in range(self.n_blue):
            for j in range(i + 1, self.n_blue):
                self.blue_swarm.entangle_pair(i, j, strength=0.25)

        self.telemetry = {
            'red_paths': [[] for _ in range(self.n_red)],
            'blue_paths': [[] for _ in range(self.n_blue)],
            'barrier_history': [],
            'red_coherence': [],
            'blue_coherence': [],
            'waypoint_reaches': [],
            'red_surprise': [],
            'blue_success': []
        }
        self._update_hamiltonian()
        return self.observe()

    def observe(self):
        """
        RED observation: own state + BLUE positions + barrier map + waypoint.
        BLUE observation: RED probability field + own state + barrier map.
        """
        # RED obs: probability + phase + barriers + waypoint + blue positions
        red_obs = []
        red_probs = [a.probability_density() for a in self.red_swarm.agents]
        red_phases = [a.phase_field() for a in self.red_swarm.agents]

        for i in range(self.n_red):
            obs = np.zeros(self.size * self.size * 5 + self.n_blue * 2)
            for r in range(self.size):
                for c in range(self.size):
                    idx = r * self.size + c
                    obs[idx] = red_probs[i][r, c]
                    obs[idx + self.size*self.size] = np.cos(red_phases[i][r, c])
                    obs[idx + 2*self.size*self.size] = np.sin(red_phases[i][r, c])
                    obs[idx + 3*self.size*self.size] = self.hamiltonian.V[r, c]
                    obs[idx + 4*self.size*self.size] = 1.0 if (r, c) in self.barriers else 0.0
            # Blue agent positions
            for j in range(self.n_blue):
                exp_r, exp_c = self.blue_swarm.agents[j].expected_position()
                obs[self.size*self.size*5 + j*2] = exp_r / self.size
                obs[self.size*self.size*5 + j*2 + 1] = exp_c / self.size
            red_obs.append(obs)

        # BLUE obs: RED probability field (inferred from tracking) + own state + barriers
        blue_obs = []
        blue_probs = [a.probability_density() for a in self.blue_swarm.agents]
        blue_phases = [a.phase_field() for a in self.blue_swarm.agents]

        # Aggregate RED probability field (what BLUE sees)
        red_aggregate = np.zeros((self.size, self.size))
        for rp in red_probs:
            red_aggregate += rp
        red_aggregate /= self.n_red

        for i in range(self.n_blue):
            obs = np.zeros(self.size * self.size * 5)
            for r in range(self.size):
                for c in range(self.size):
                    idx = r * self.size + c
                    obs[idx] = blue_probs[i][r, c]
                    obs[idx + self.size*self.size] = np.cos(blue_phases[i][r, c])
                    obs[idx + 2*self.size*self.size] = np.sin(blue_phases[i][r, c])
                    obs[idx + 3*self.size*self.size] = red_aggregate[r, c]  # RED threat field
                    obs[idx + 4*self.size*self.size] = 1.0 if (r, c) in self.barriers else 0.0
            blue_obs.append(obs)

        return red_obs, blue_obs

    def step_red(self, red_actions):
        """RED moves. Actions 0-3 = movement."""
        for i, action in enumerate(red_actions):
            self.red_swarm.agents[i].psi = self.red_ops.apply_action(
                self.red_swarm.agents[i].psi, action
            )
            self.red_swarm.agents[i].psi = self.hamiltonian.full_step(
                self.red_swarm.agents[i].psi, self.dt
            )
            self.red_swarm.agents[i]._normalize()

        # Boundary + decoherence
        for agent in self.red_swarm.agents:
            agent.psi[0, :] *= 0.5
            agent.psi[-1, :] *= 0.5
            agent.psi[:, 0] *= 0.5
            agent.psi[:, -1] *= 0.5
            agent._normalize()

        self.red_swarm.apply_global_decoherence(0.008)
        self.red_swarm._update_joint_probs()

        for i in range(self.n_red):
            pos = self.red_swarm.agents[i].expected_position()
            self.telemetry['red_paths'][i].append(pos)

        coherence = np.mean(self.red_swarm.entanglement[self.red_swarm.entanglement > 0]) if np.any(self.red_swarm.entanglement > 0) else 0
        self.telemetry['red_coherence'].append(coherence)

    def step_blue(self, blue_actions):
        """BLUE moves and places barriers. Action 4 = place barrier at current position."""
        for i, action in enumerate(blue_actions):
            if action == 4:
                # Place barrier at BLUE's most likely position
                probs = self.blue_swarm.agents[i].probability_density()
                max_r, max_c = np.unravel_index(np.argmax(probs), probs.shape)
                if len(self.barriers) < self.max_barriers:
                    self.barriers.add((int(max_r), int(max_c)))
                    self.barrier_lifetime[(int(max_r), int(max_c))] = 1.0
            else:
                self.blue_swarm.agents[i].psi = self.blue_ops.apply_action(
                    self.blue_swarm.agents[i].psi, action
                )
                self.blue_swarm.agents[i].psi = self.hamiltonian.full_step(
                    self.blue_swarm.agents[i].psi, self.dt
                )
                self.blue_swarm.agents[i]._normalize()

        # Boundary
        for agent in self.blue_swarm.agents:
            agent.psi[0, :] *= 0.5
            agent.psi[-1, :] *= 0.5
            agent.psi[:, 0] *= 0.5
            agent.psi[:, -1] *= 0.5
            agent._normalize()

        self.blue_swarm.apply_global_decoherence(0.01)
        self.blue_swarm._update_joint_probs()

        for i in range(self.n_blue):
            pos = self.blue_swarm.agents[i].expected_position()
            self.telemetry['blue_paths'][i].append(pos)

        coherence = np.mean(self.blue_swarm.entanglement[self.blue_swarm.entanglement > 0]) if np.any(self.blue_swarm.entanglement > 0) else 0
        self.telemetry['blue_coherence'].append(coherence)

        # Decay barriers
        to_remove = []
        for pos, life in self.barrier_lifetime.items():
            self.barrier_lifetime[pos] -= self.barrier_decay
            if self.barrier_lifetime[pos] <= 0:
                to_remove.append(pos)
        for pos in to_remove:
            self.barriers.discard(pos)
            del self.barrier_lifetime[pos]

        self.telemetry['barrier_history'].append(self.barriers.copy())
        self._update_hamiltonian()

    def check_waypoints(self):
        """Check if RED reached current waypoint cooperatively."""
        reward_red = 0.0
        reward_blue = 0.0
        done = False

        if self.waypoint_idx < len(self.waypoints):
            wr, wc = self.waypoints[self.waypoint_idx]
            all_near = True
            for i in range(self.n_red):
                prob = np.abs(self.red_swarm.agents[i].psi[wr, wc])**2
                if prob < 0.12:
                    all_near = False
                    break
            if all_near:
                for i in range(self.n_red):
                    self.red_swarm.collapse_agent(i, (wr, wc), sigma=0.6)
                reward_red = 1.0
                self.waypoint_idx += 1
                self.telemetry['waypoint_reaches'].append({
                    'step': len(self.telemetry['red_paths'][0]),
                    'pos': [wr, wc]
                })
                if self.waypoint_idx >= len(self.waypoints):
                    done = True
                    reward_blue = -1.0  # BLUE loses
            else:
                # BLUE gets reward proportional to RED's difficulty
                red_surprise = self._compute_red_surprise()
                reward_blue = min(0.5, red_surprise * 0.1)
                self.telemetry['red_surprise'].append(red_surprise)
                self.telemetry['blue_success'].append(reward_blue)

        return reward_red, reward_blue, done

    def _compute_red_surprise(self):
        """How far is RED from its waypoint? Higher = more surprise = BLUE winning."""
        if self.waypoint_idx >= len(self.waypoints):
            return 0.0
        wr, wc = self.waypoints[self.waypoint_idx]
        total_dist = 0.0
        for i in range(self.n_red):
            er, ec = self.red_swarm.agents[i].expected_position()
            dist = np.sqrt((er - wr)**2 + (ec - wc)**2)
            total_dist += dist
        return total_dist / self.n_red

    def get_classical_positions(self, team='red'):
        swarm = self.red_swarm if team == 'red' else self.blue_swarm
        n = self.n_red if team == 'red' else self.n_blue
        positions = []
        for i in range(n):
            probs = swarm.agents[i].probability_density()
            max_idx = np.unravel_index(np.argmax(probs), probs.shape)
            positions.append(list(max_idx))
        return positions

    def print_arena(self):
        """ASCII visualization."""
        red_probs = [a.probability_density() for a in self.red_swarm.agents]
        blue_probs = [a.probability_density() for a in self.blue_swarm.agents]

        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) in self.barriers:
                    row += "B"
                elif any((r, c) == (pr, pc) for pr, pc, _, _ in self.planets):
                    row += "P"
                elif self.waypoint_idx < len(self.waypoints) and [r, c] == self.waypoints[self.waypoint_idx]:
                    row += "W"
                else:
                    # Check RED
                    best_red = -1
                    best_rp = 0
                    for i, rp in enumerate(red_probs):
                        p = rp[r, c]
                        if p > best_rp:
                            best_rp = p
                            best_red = i
                    # Check BLUE
                    best_blue = -1
                    best_bp = 0
                    for i, bp in enumerate(blue_probs):
                        p = bp[r, c]
                        if p > best_bp:
                            best_bp = p
                            best_blue = i

                    if best_red >= 0 and best_rp > 0.15:
                        row += f"\033[91m{best_red}\033[0m"  # Red color
                    elif best_blue >= 0 and best_bp > 0.15:
                        row += f"\033[94m{best_blue}\033[0m"  # Blue color
                    else:
                        row += "."
            print("  " + row)


# ==============================================================================
# PART A2: Adversarial EFE Agents
# ==============================================================================

class RedSwarmAgent:
    """RED: Minimize EFE toward waypoints while avoiding barriers."""
    def __init__(self, realm, perceptual_agents, exploration_coef=0.25):
        self.realm = realm
        self.perceptuals = perceptual_agents
        self.exploration_coef = exploration_coef
        self.coherence_bonus = 0.4
        self.trackers = [CoordinateTracker(realm.size) for _ in range(realm.n_red)]

    def select_actions(self, obs_list):
        """Each agent independently selects action minimizing EFE."""
        actions = []
        for i in range(self.realm.n_red):
            best_a = 0
            best_efe = float('inf')
            for a in range(4):
                efe, _ = self.expected_free_energy(i, obs_list[i], a)
                if efe < best_efe:
                    best_efe = efe
                    best_a = a
            actions.append(best_a)
        return actions, []

    def expected_free_energy(self, agent_idx, obs, action):
        """RED EFE: distance to waypoint + barrier penalty - info_gain."""
        # Simulate RED action
        swarm_copy = self.realm.red_swarm.copy()
        swarm_copy.agents[agent_idx].psi = self.realm.red_ops.apply_action(
            swarm_copy.agents[agent_idx].psi, action
        )
        # Temporarily update Hamiltonian
        old_V = self.realm.hamiltonian.V.copy()
        self.realm._update_hamiltonian()
        swarm_copy.agents[agent_idx].psi = self.realm.hamiltonian.full_step(
            swarm_copy.agents[agent_idx].psi, self.realm.dt
        )
        self.realm.hamiltonian.V = old_V
        swarm_copy.agents[agent_idx]._normalize()

        pred_agent = swarm_copy.agents[agent_idx]
        pred_probs = pred_agent.probability_density()

        # Surprise (distance from waypoint)
        if self.realm.waypoint_idx < len(self.realm.waypoints):
            wr, wc = self.realm.waypoints[self.realm.waypoint_idx]
            kl = 0.0
            for r in range(self.realm.size):
                for c in range(self.realm.size):
                    p = pred_probs[r, c] + 1e-10
                    target = 1.0 if (r == wr and c == wc) else 1e-10
                    kl += p * np.log(p / target)
        else:
            kl = 0.0

        # Barrier penalty
        barrier_penalty = 0.0
        for r in range(self.realm.size):
            for c in range(self.realm.size):
                if (r, c) in self.realm.barriers:
                    barrier_penalty += pred_probs[r, c] * 5.0

        # Uncertainty
        var_r, var_c = pred_agent.position_variance()
        uncertainty = 0.1 * (var_r + var_c)

        # Info gain
        info_gain = 0.0
        for r in range(self.realm.size):
            for c in range(self.realm.size):
                if (r, c) not in self.trackers[agent_idx].visited:
                    info_gain += pred_probs[r, c]

        # Coherence bonus
        current_coherence = np.mean(self.realm.red_swarm.entanglement[self.realm.red_swarm.entanglement > 0]) if np.any(self.realm.red_swarm.entanglement > 0) else 0
        new_coherence = np.mean(swarm_copy.entanglement[swarm_copy.entanglement > 0]) if np.any(swarm_copy.entanglement > 0) else 0
        coherence_penalty = -0.3 * (new_coherence - current_coherence)

        efe = kl + barrier_penalty + uncertainty - self.exploration_coef * info_gain + coherence_penalty
        return efe, swarm_copy


class BlueSwarmAgent:
    """BLUE: Maximize RED's difficulty (minimize RED's success)."""
    def __init__(self, realm, perceptual_agents, exploration_coef=0.2):
        self.realm = realm
        self.perceptuals = perceptual_agents
        self.exploration_coef = exploration_coef
        self.trackers = [CoordinateTracker(realm.size) for _ in range(realm.n_blue)]

    def get_preferred_observation(self, agent_idx):
        """BLUE wants to be where RED is going."""
        obs = np.zeros(self.realm.size * self.size * 5)
        # Preferred = high RED probability regions (intercept)
        red_aggregate = np.zeros((self.realm.size, self.realm.size))
        for agent in self.realm.red_swarm.agents:
            red_aggregate += agent.probability_density()
        red_aggregate /= self.realm.n_red
        for r in range(self.realm.size):
            for c in range(self.realm.size):
                idx = r * self.realm.size + c
                obs[idx + 3 * self.realm.size * self.realm.size] = red_aggregate[r, c]
        return obs

    def expected_free_energy(self, agent_idx, obs, action):
        """
        BLUE EFE: 
          - Negative of RED's predicted success (want RED to fail)
          - Barrier placement bonus
          - Position uncertainty (don't spread too thin)
        """
        if action == 4:
            # PLACE BARRIER: evaluate where barrier hurts RED most
            probs = self.realm.blue_swarm.agents[agent_idx].probability_density()
            max_r, max_c = np.unravel_index(np.argmax(probs), probs.shape)

            # How much does a barrier here hurt RED?
            red_pain = 0.0
            for i in range(self.realm.n_red):
                red_prob = self.realm.red_swarm.agents[i].probability_density()
                red_pain += red_prob[max_r, max_c]

            # Cost: using up barrier slot
            slot_cost = 0.5 if len(self.realm.barriers) >= self.realm.max_barriers - 1 else 0.0

            efe = -red_pain * 3.0 + slot_cost
            return efe, None
        else:
            # MOVE: get closer to RED's predicted path
            swarm_copy = self.realm.blue_swarm.copy()
            swarm_copy.agents[agent_idx].psi = self.realm.blue_ops.apply_action(
                swarm_copy.agents[agent_idx].psi, action
            )
            old_V = self.realm.hamiltonian.V.copy()
            self.realm._update_hamiltonian()
            swarm_copy.agents[agent_idx].psi = self.realm.hamiltonian.full_step(
                swarm_copy.agents[agent_idx].psi, self.realm.dt
            )
            self.realm.hamiltonian.V = old_V
            swarm_copy.agents[agent_idx]._normalize()

            pred_agent = swarm_copy.agents[agent_idx]
            pred_probs = pred_agent.probability_density()

            # Want to overlap with RED's probability field
            red_aggregate = np.zeros((self.realm.size, self.realm.size))
            for agent in self.realm.red_swarm.agents:
                red_aggregate += agent.probability_density()
            red_aggregate /= self.realm.n_red

            overlap = np.sum(pred_probs * red_aggregate)

            # Penalty for being near existing barriers (waste)
            barrier_overlap = 0.0
            for r in range(self.realm.size):
                for c in range(self.realm.size):
                    if (r, c) in self.realm.barriers:
                        barrier_overlap += pred_probs[r, c]

            var_r, var_c = pred_agent.position_variance()
            uncertainty = 0.05 * (var_r + var_c)

            efe = -overlap * 2.0 + barrier_overlap + uncertainty
            return efe, swarm_copy

    def select_actions(self, obs_list):
        actions = []
        for i in range(self.realm.n_blue):
            best_a = 0
            best_efe = float('inf')
            for a in range(5):  # 0-3 move, 4 place
                efe, _ = self.expected_free_energy(i, obs_list[i], a)
                if efe < best_efe:
                    best_efe = efe
                    best_a = a
            actions.append(best_a)
        return actions


# ==============================================================================
# PART A3: Game Controller
# ==============================================================================

class AdversarialGame:
    """Runs the zero-sum quantum game."""
    def __init__(self, realm, red_agent, blue_agent):
        self.realm = realm
        self.red = red_agent
        self.blue = blue_agent
        self.red_score = 0.0
        self.blue_score = 0.0

    def play_round(self, verbose=True):
        """One full turn: RED moves, BLUE responds, environment updates."""
        red_obs, blue_obs = self.realm.observe()

        # RED moves
        red_actions, _ = self.red.select_actions(red_obs)
        self.realm.step_red(red_actions)

        # BLUE observes and responds
        red_obs, blue_obs = self.realm.observe()
        blue_actions = self.blue.select_actions(blue_obs)
        self.realm.step_blue(blue_actions)

        # Check waypoints
        r_reward, b_reward, done = self.realm.check_waypoints()
        self.red_score += r_reward
        self.blue_score += b_reward

        if verbose:
            red_pos = self.realm.get_classical_positions('red')
            blue_pos = self.realm.get_classical_positions('blue')
            print(f"  RED: {red_pos} | BLUE: {blue_pos} | Barriers: {len(self.realm.barriers)} | "
                  f"Score: R={self.red_score:.1f} B={self.blue_score:.1f}")

        return done

    def play(self, max_rounds=100, verbose=True):
        if verbose:
            print(f"\nAdversarial Quantum Game — {self.realm.n_red} RED vs {self.realm.n_blue} BLUE")
            print(f"Waypoints: {self.realm.waypoints}")
            print(f"Max barriers: {self.realm.max_barriers}, Decay: {self.realm.barrier_decay}")

        for round in range(max_rounds):
            if verbose and round % 10 == 0:
                print(f"\nRound {round+1}")
                self.realm.print_arena()

            done = self.play_round(verbose=(verbose and round % 10 == 0))

            if done:
                if verbose:
                    print(f"\n>>> RED WINS in {round+1} rounds!")
                return 'red', round + 1

        if verbose:
            print(f"\n>>> TIMEOUT. RED reached {self.realm.waypoint_idx}/{len(self.realm.waypoints)} waypoints.")
        return 'blue', max_rounds


# ==============================================================================
# PART A4: Main Demo
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("ADVERSARIAL SWARM — Zero-Sum Quantum Game")
    print("=" * 70)

    realm = AdversarialSwarmRealm(n_red=3, n_blue=2, size=14, n_planets=2,
                                   dt=0.12, barrier_decay=0.12, max_barriers=8)

    # RED agents
    red_input_size = realm.size * realm.size * 5 + realm.n_blue * 2
    red_perceptuals = [Tatha(input_size=red_input_size, hidden_size=48, belief_size=12, lr=0.005)
                       for _ in range(realm.n_red)]
    red_agent = RedSwarmAgent(realm, red_perceptuals, exploration_coef=0.25)

    # BLUE agents
    blue_input_size = realm.size * realm.size * 5
    blue_perceptuals = [Tatha(input_size=blue_input_size, hidden_size=32, belief_size=10, lr=0.005)
                        for _ in range(realm.n_blue)]
    blue_agent = BlueSwarmAgent(realm, blue_perceptuals, exploration_coef=0.2)

    # Play
    game = AdversarialGame(realm, red_agent, blue_agent)
    winner, rounds = game.play(max_rounds=80, verbose=True)

    print(f"\nFinal Score: RED={game.red_score:.1f} | BLUE={game.blue_score:.1f}")
    print(f"Winner: {winner.upper()} in {rounds} rounds")
    print(f"Waypoints reached: {realm.waypoint_idx}/{len(realm.waypoints)}")
    print(f"Total barriers placed: {len(realm.telemetry['barrier_history'])}")

    print("\n" + "=" * 70)
    print("Adversarial Swarm demo complete.")
    print("=" * 70)
