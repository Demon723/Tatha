"""
================================================================================
Meta-Director — Hierarchical Active Inference for Swarm Parameter Control
================================================================================

A third-level agent (Director) observes the RED-vs-BLUE adversarial game
and modulates meta-parameters to optimize mission outcomes:

  Director observes: game state → predicts outcome → modulates parameters

Modulated parameters:
  • RED entanglement strength (cohesion vs. independence)
  • BLUE barrier decay rate (persistence vs. transience)
  • RED exploration coefficient (explore vs. exploit)
  • BLUE exploration coefficient (aggressive vs. passive blocking)
  • Hamiltonian potential scaling (gravity well strength)

The Director is itself a predictive coding agent with a slow timescale,
learning which parameter vectors lead to successful waypoint completion.

This maps to:
  • Prefrontal cortex modulating striatal dopamine (neuroscience)
  • Meta-learning / learning-to-learn (machine learning)
  • Optimal control with adaptive gains (control theory)

Integrates with: adversarial_swarm.py, quantum_swarm.py, unified_pc_space.py
"""

import numpy as np

import sys, os
sys.path.insert(0, '/mnt/agents/output')
from adversarial_swarm import (AdversarialSwarmRealm, RedSwarmAgent, BlueSwarmAgent, 
                                AdversarialGame)
from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker


# ==============================================================================
# PART M1: Meta-Parameter Space
# ==============================================================================

class MetaParameterSpace:
    """
    Continuous parameter space for the Director to control.
    Each parameter is bounded [0, 1] and mapped to physical ranges.
    """
    PARAM_NAMES = [
        'red_entanglement',      # 0.0-1.0 → entanglement strength 0.0-0.8
        'blue_entanglement',     # 0.0-1.0 → entanglement strength 0.0-0.6
        'red_exploration',       # 0.0-1.0 → exploration_coef 0.0-0.5
        'blue_exploration',      # 0.0-1.0 → exploration_coef 0.0-0.4
        'barrier_decay',         # 0.0-1.0 → decay rate 0.05-0.25
        'barrier_max',           # 0.0-1.0 → max barriers 3-12
        'gravity_scale',         # 0.0-1.0 → planet gravity 0.2-2.0
        'decoherence_rate',      # 0.0-1.0 → decoherence 0.0-0.02
        'coherence_bonus_red',   # 0.0-1.0 → bonus 0.0-1.0
        'intercept_bonus_blue',  # 0.0-1.0 → overlap weight 0.0-3.0
    ]
    N_PARAMS = len(PARAM_NAMES)

    # Physical ranges
    RANGES = {
        'red_entanglement': (0.0, 0.8),
        'blue_entanglement': (0.0, 0.6),
        'red_exploration': (0.0, 0.5),
        'blue_exploration': (0.0, 0.4),
        'barrier_decay': (0.05, 0.25),
        'barrier_max': (3, 12),
        'gravity_scale': (0.2, 2.0),
        'decoherence_rate': (0.0, 0.02),
        'coherence_bonus_red': (0.0, 1.0),
        'intercept_bonus_blue': (0.0, 3.0),
    }

    @classmethod
    def normalize(cls, params_dict):
        """Convert physical parameters to [0,1] vector."""
        vec = np.zeros(cls.N_PARAMS)
        for i, name in enumerate(cls.PARAM_NAMES):
            lo, hi = cls.RANGES[name]
            val = params_dict.get(name, (lo + hi) / 2)
            vec[i] = np.clip((val - lo) / (hi - lo), 0, 1)
        return vec

    @classmethod
    def denormalize(cls, vec):
        """Convert [0,1] vector to physical parameters."""
        params = {}
        for i, name in enumerate(cls.PARAM_NAMES):
            lo, hi = cls.RANGES[name]
            params[name] = lo + vec[i] * (hi - lo)
        return params

    @classmethod
    def apply_to_realm(cls, realm, vec):
        """Apply parameter vector to an AdversarialSwarmRealm."""
        params = cls.denormalize(vec)
        # RED entanglement
        for i in range(realm.n_red):
            for j in range(i + 1, realm.n_red):
                realm.red_swarm.entanglement[i, j] = params['red_entanglement']
                realm.red_swarm.entanglement[j, i] = params['red_entanglement']
        # BLUE entanglement
        for i in range(realm.n_blue):
            for j in range(i + 1, realm.n_blue):
                realm.blue_swarm.entanglement[i, j] = params['blue_entanglement']
                realm.blue_swarm.entanglement[j, i] = params['blue_entanglement']
        # Barrier params
        realm.barrier_decay = params['barrier_decay']
        realm.max_barriers = int(params['barrier_max'])
        # Decoherence
        realm.decoherence_rate = params['decoherence_rate']
        # Gravity
        for idx, (pr, pc, radius, _) in enumerate(realm.planets):
            realm.planets[idx] = (pr, pc, radius, params['gravity_scale'])
        return params


# ==============================================================================
# PART M2: Director Agent
# ==============================================================================

class MetaDirector:
    """
    The Director observes game telemetry and outputs parameter vectors.

    Input: game state features (compressed telemetry)
    Hidden: predictive coding layer
    Output: parameter vector + confidence

    Learns via predictive coding: predicts game outcome from parameters,
    then updates parameters to minimize predicted failure.
    """
    def __init__(self, n_params=10, hidden_size=32, belief_size=8, lr=0.01,
                 memory_size=100):
        self.n_params = n_params
        self.memory_size = memory_size

        # Predictive coding stack for meta-cognition
        self.perceptual = Tatha(input_size=n_params + 5,  # params + outcome features
                                hidden_size=hidden_size,
                                belief_size=belief_size,
                                lr=lr)

        # Parameter policy: belief layer → parameter vector
        self.W_policy = np.random.randn(n_params, belief_size) * 0.1
        self.b_policy = np.zeros(n_params)

        # Outcome predictor: parameters → predicted success probability
        self.W_outcome = np.random.randn(1, n_params) * 0.1
        self.b_outcome = np.zeros(1)

        # Episodic memory: (params, outcome) pairs
        self.memory = []
        self.memory_idx = 0

        # Current parameter vector
        self.current_params = np.ones(n_params) * 0.5

    def encode_game_state(self, realm, red_score, blue_score, rounds):
        """Compress game telemetry into a feature vector."""
        features = np.zeros(5)
        # Feature 0: RED progress (waypoints reached / total)
        features[0] = realm.waypoint_idx / max(len(realm.waypoints), 1)
        # Feature 1: RED coherence
        red_coh = np.mean(realm.red_swarm.entanglement[realm.red_swarm.entanglement > 0]) if np.any(realm.red_swarm.entanglement > 0) else 0
        features[1] = red_coh
        # Feature 2: BLUE coherence
        blue_coh = np.mean(realm.blue_swarm.entanglement[realm.blue_swarm.entanglement > 0]) if np.any(realm.blue_swarm.entanglement > 0) else 0
        features[2] = blue_coh
        # Feature 3: Score differential
        features[3] = (red_score - blue_score + 5) / 10  # Normalize to [0,1]
        # Feature 4: Game progress (rounds / max)
        features[4] = min(rounds / 100, 1.0)
        return features

    def predict_outcome(self, params):
        """Predict success probability given parameters."""
        z = self.W_outcome @ params + self.b_outcome
        return 1 / (1 + np.exp(-np.clip(z[0], -10, 10)))

    def select_parameters(self, game_state, explore=True):
        """
        Use predictive coding to generate parameters.

        1. Perceive current state
        2. Settle to prediction
        3. Policy maps belief to parameters
        4. Add exploration noise
        """
        # Construct observation: current params + game state
        obs = np.concatenate([self.current_params, game_state])

        self.perceptual.perceive(obs)
        self.perceptual.settle(n_iter=15)

        # Policy: belief → parameters
        belief = self.perceptual.belief.activity
        params_raw = self.W_policy @ belief + self.b_policy

        # Sigmoid to [0,1]
        params = 1 / (1 + np.exp(-np.clip(params_raw, -10, 10)))

        if explore:
            # Ornstein-Uhlenbeck-like noise for smooth exploration
            noise = np.random.randn(self.n_params) * 0.1
            params = np.clip(params + noise, 0.05, 0.95)

        self.current_params = params
        return params

    def learn_from_outcome(self, params, outcome, game_state):
        """
        outcome = 1.0 if RED won, 0.0 if BLUE won.
        Update policy to increase P(success) for good parameters.
        """
        # Store in episodic memory
        if len(self.memory) < self.memory_size:
            self.memory.append(None)
        self.memory[self.memory_idx] = (params.copy(), outcome, game_state.copy())
        self.memory_idx = (self.memory_idx + 1) % self.memory_size

        # Predictive update: construct observation with actual outcome
        obs = np.concatenate([params, game_state])
        obs[-1] = outcome  # Replace predicted outcome with actual

        self.perceptual.perceive(obs)
        self.perceptual.settle(n_iter=10)
        self.perceptual.learn()

        # Policy gradient: if outcome was good, move params toward this direction
        if len(self.memory) >= 10:
            # Sample from memory
            batch = np.random.choice(len(self.memory), min(10, len(self.memory)), replace=False)
            for idx in batch:
                p, o, gs = self.memory[idx]
                # Outcome prediction error
                pred = self.predict_outcome(p)
                error = o - pred
                # Update outcome predictor
                d_W = error * p.reshape(1, -1)
                d_b = np.array([error])
                self.W_outcome += 0.01 * d_W
                self.b_outcome += 0.01 * d_b
                # Update policy (reinforce good params)
                if o > 0.5:
                    self.W_policy += 0.005 * np.outer(p - 0.5, self.perceptual.belief.activity)

    def get_parameter_report(self):
        """Human-readable parameter report."""
        params = MetaParameterSpace.denormalize(self.current_params)
        lines = ["Current Director Parameters:"]
        for name, val in params.items():
            lines.append(f"  {name:25s}: {val:.3f}")
        return "\n".join(lines)


# ==============================================================================
# PART M3: Meta-Learning Game Loop
# ==============================================================================

class MetaLearningGame:
    """
    Outer loop: Director plays many adversarial games,
    learning which parameters lead to RED success.
    """
    def __init__(self, director, n_episodes=50, rounds_per_game=60):
        self.director = director
        self.n_episodes = n_episodes
        self.rounds_per_game = rounds_per_game
        self.outcomes = []
        self.param_history = []

    def run_episode(self, episode_idx, verbose=True):
        """One full adversarial game with Director-chosen parameters."""
        # Create fresh realm
        realm = AdversarialSwarmRealm(n_red=3, n_blue=2, size=14, n_planets=2,
                                       dt=0.12, barrier_decay=0.12, max_barriers=8)

        # Director selects parameters
        game_state = self.director.encode_game_state(realm, 0, 0, 0)
        explore = episode_idx < self.n_episodes * 0.7  # 70% exploration
        params_vec = self.director.select_parameters(game_state, explore=explore)
        params_phys = MetaParameterSpace.apply_to_realm(realm, params_vec)

        if verbose:
            print(f"\nEpisode {episode_idx+1} | Explore={explore}")
            print(f"  Parameters: red_ent={params_phys['red_entanglement']:.2f}, "
                  f"red_expl={params_phys['red_exploration']:.2f}, "
                  f"bar_decay={params_phys['barrier_decay']:.3f}, "
                  f"grav={params_phys['gravity_scale']:.2f}")

        # Create agents with Director-modulated exploration
        red_input = realm.size * realm.size * 5 + realm.n_blue * 2
        red_perceptuals = [Tatha(input_size=red_input, hidden_size=48, belief_size=12, lr=0.005)
                           for _ in range(realm.n_red)]
        red_agent = RedSwarmAgent(realm, red_perceptuals, 
                                   exploration_coef=params_phys['red_exploration'])

        blue_input = realm.size * realm.size * 5
        blue_perceptuals = [Tatha(input_size=blue_input, hidden_size=32, belief_size=10, lr=0.005)
                            for _ in range(realm.n_blue)]
        blue_agent = BlueSwarmAgent(realm, blue_perceptuals,
                                     exploration_coef=params_phys['blue_exploration'])

        # Play game
        game = AdversarialGame(realm, red_agent, blue_agent)
        winner, rounds = game.play(max_rounds=self.rounds_per_game, verbose=False)

        outcome = 1.0 if winner == 'red' else 0.0

        # Director learns
        final_state = self.director.encode_game_state(realm, game.red_score, game.blue_score, rounds)
        self.director.learn_from_outcome(params_vec, outcome, final_state)

        self.outcomes.append(outcome)
        self.param_history.append(params_vec.copy())

        if verbose:
            print(f"  Result: {winner.upper()} wins in {rounds} rounds")
            print(f"  Score: RED={game.red_score:.1f} BLUE={game.blue_score:.1f}")
            print(f"  Cumulative RED win rate: {np.mean(self.outcomes):.1%}")

        return outcome, rounds, params_vec

    def run(self, verbose=True):
        """Run all episodes."""
        print("=" * 70)
        print("META-LEARNING DIRECTOR — Training")
        print("=" * 70)
        print(f"Episodes: {self.n_episodes} | Rounds/game: {self.rounds_per_game}")
        print(f"Parameter space: {self.director.n_params} dimensions")

        for ep in range(self.n_episodes):
            self.run_episode(ep, verbose=(verbose and ep % 5 == 0))

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final RED win rate: {np.mean(self.outcomes):.1%}")
        print(f"Final 10 episodes: {np.mean(self.outcomes[-10:]):.1%}")
        print(f"\nBest parameters (last episode):")
        print(self.director.get_parameter_report())

        return self.outcomes, self.param_history

    def get_learning_curves(self):
        """Return smoothed win rate over episodes."""
        window = min(10, len(self.outcomes))
        smoothed = np.convolve(self.outcomes, np.ones(window)/window, mode='valid')
        return smoothed


# ==============================================================================
# PART M4: Main Demo
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Create Director
    director = MetaDirector(n_params=MetaParameterSpace.N_PARAMS,
                            hidden_size=32, belief_size=8, lr=0.01,
                            memory_size=50)

    # Run meta-learning
    meta_game = MetaLearningGame(director, n_episodes=30, rounds_per_game=50)
    outcomes, param_history = meta_game.run(verbose=True)

    # Test with best parameters (no exploration)
    print("\n" + "=" * 70)
    print("TESTING BEST PARAMETERS (no exploration)")
    print("=" * 70)

    realm = AdversarialSwarmRealm(n_red=3, n_blue=2, size=14, n_planets=2)
    game_state = director.encode_game_state(realm, 0, 0, 0)
    best_params = director.select_parameters(game_state, explore=False)
    best_phys = MetaParameterSpace.apply_to_realm(realm, best_params)

    print(f"\nBest parameters:")
    for name, val in best_phys.items():
        print(f"  {name:25s}: {val:.3f}")

    red_input = realm.size * realm.size * 5 + realm.n_blue * 2
    red_p = [Tatha(input_size=red_input, hidden_size=48, belief_size=12, lr=0.005) 
             for _ in range(realm.n_red)]
    blue_input = realm.size * realm.size * 5
    blue_p = [Tatha(input_size=blue_input, hidden_size=32, belief_size=10, lr=0.005) 
              for _ in range(realm.n_blue)]
    red = RedSwarmAgent(realm, red_p, exploration_coef=best_phys['red_exploration'])
    blue = BlueSwarmAgent(realm, blue_p, exploration_coef=best_phys['blue_exploration'])
    game = AdversarialGame(realm, red, blue)
    winner, rounds = game.play(max_rounds=60, verbose=True)

    print(f"\nTest result: {winner.upper()} wins in {rounds} rounds")

    print("\n" + "=" * 70)
    print("Meta-Director demo complete.")
    print("=" * 70)
