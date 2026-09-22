"""
================================================================================
Tathā Unified — Predictive Coding + Active Inference + Space Navigation
================================================================================

A complete, self-contained system integrating:
  • Classical / Quantum / Adaptive Quantum predictive coding agents
  • Coordinate-based forward models with SLAM-like tracking
  • 5×5 Grid World and 20×20 Space-Scale Astrotechnology environments
  • Adaptive waypoint sequencing (nearest, info_gain, tsp_greedy, adaptive_eig)
  • Curriculum training with experience replay
  • Matplotlib visualization suite

Dependencies: numpy, matplotlib
"""

import numpy as np

# ==============================================================================
# PART A: Core Predictive Coding Stack
# ==============================================================================

def _safe(arr, fallback=0.0):
    return np.nan_to_num(arr, nan=fallback, posinf=fallback, neginf=fallback)


class PrecisionGate:
    def __init__(self, size, lr=0.001):
        self.size = size
        self.precision = np.ones(size) * 0.5
        self.lr = lr

    def apply(self, error):
        return self.precision * error

    def adapt(self, error):
        error_sq = error ** 2
        self.precision += self.lr * (error_sq - self.precision)
        self.precision = np.clip(self.precision, 0.1, 5.0)


class Neuromodulator:
    def __init__(self, n_layers, window=20, boost=2.0, decay=0.5):
        self.n_layers = n_layers
        self.window = window
        self.boost = boost
        self.decay = decay
        self.history = [[] for _ in range(n_layers)]
        self.expected = [1.0] * n_layers
        self.lr_multipliers = [1.0] * n_layers
        self.precision_multipliers = [1.0] * n_layers
        self.step_count = 0

    def update(self, layer_errors):
        self.step_count += 1
        for i, err in enumerate(layer_errors):
            self.history[i].append(err)
            if len(self.history[i]) > self.window:
                self.history[i].pop(0)
            if len(self.history[i]) >= 5:
                self.expected[i] = np.mean(self.history[i])
            dopamine = err - self.expected[i]
            if dopamine > 0.01:
                self.lr_multipliers[i] = min(self.lr_multipliers[i] * 1.1, self.boost)
                self.precision_multipliers[i] = min(self.precision_multipliers[i] * 1.05, 1.5)
            else:
                self.lr_multipliers[i] = max(self.lr_multipliers[i] * 0.98, self.decay)
                self.precision_multipliers[i] = max(self.precision_multipliers[i] * 0.99, 0.5)
        return self.lr_multipliers, self.precision_multipliers

    def report(self):
        lines = []
        for i in range(self.n_layers):
            lines.append(
                f"  Layer {i}: LR={self.lr_multipliers[i]:.2f}x | "
                f"Precision={self.precision_multipliers[i]:.2f}x | "
                f"ExpectedErr={self.expected[i]:.4f}"
            )
        return "\n".join(lines)


class QuantumBeliefLayer:
    def __init__(self, size, lr=0.005, dt=0.1, decay=0.01):
        self.size = size
        self.base_lr = lr
        self.lr = lr
        self.dt = dt
        self.decay = decay
        self.psi = np.ones(size, dtype=complex) / np.sqrt(size)
        self.psi_prev = self.psi.copy()
        self.activity = np.abs(self.psi)**2
        self.prediction = np.zeros(size)
        self.error = np.zeros(size)
        A = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        self.W_rec, _ = np.linalg.qr(A)
        self.bias = 0.01 * (np.random.randn(size) + 1j * np.random.randn(size))
        self.gate = PrecisionGate(size)

    def forward_predict(self):
        psi_raw = self.W_rec @ self.psi_prev + self.bias
        norm = np.linalg.norm(psi_raw)
        if norm > 1e-8:
            psi_raw = psi_raw / norm
        self.prediction = np.abs(psi_raw)**2
        return self.prediction

    def compute_error(self, target=None):
        target = target if target is not None else np.abs(self.psi_prev)**2
        raw_error = target - self.prediction
        self.error = self.gate.apply(raw_error)
        return self.error

    def settle_step(self, below_error=None):
        p_target = self.prediction + self.error
        if below_error is not None:
            p_target += 0.05 * below_error
        p_target = np.clip(p_target, 0.001, 1.0)
        p_target = p_target / np.sum(p_target)
        p_current = np.abs(self.psi)**2
        new_p = p_current + self.dt * (p_target - p_current)
        new_p = np.clip(new_p, 0.001, 1.0)
        new_p = new_p / np.sum(new_p)
        psi_pred_raw = self.W_rec @ self.psi_prev + self.bias
        mags = np.abs(psi_pred_raw)
        phases = np.where(mags > 1e-8, np.angle(psi_pred_raw), np.angle(self.psi))
        self.psi = np.sqrt(new_p) * np.exp(1j * phases)
        self.psi /= np.linalg.norm(self.psi)
        self.activity = new_p

    def settle(self, below_error=None, n_iter=20):
        for _ in range(n_iter):
            self.settle_step(below_error)

    def learn(self):
        psi_pred_raw = self.W_rec @ self.psi_prev + self.bias
        mags = np.abs(psi_pred_raw)
        phases_pred = np.where(mags > 1e-8, np.angle(psi_pred_raw), np.angle(self.psi))
        target_psi = np.sqrt(self.activity) * np.exp(1j * phases_pred)
        delta = target_psi - psi_pred_raw
        dW = np.outer(delta, np.conj(self.psi_prev))
        db = delta
        max_grad = 0.5
        dW_mag = np.clip(np.abs(dW), 0, max_grad)
        dW = dW_mag * np.exp(1j * np.angle(dW))
        db_mag = np.clip(np.abs(db), 0, max_grad)
        db = db_mag * np.exp(1j * np.angle(db))
        self.W_rec += self.lr * dW - self.decay * self.W_rec
        self.bias += self.lr * db - self.decay * self.bias
        reg = 0.01
        self.W_rec -= reg * (self.W_rec @ np.conj(self.W_rec.T) - np.eye(self.size)) @ self.W_rec
        self.gate.adapt(self.error)

    def tick(self):
        self.psi_prev = self.psi.copy()


class PredictiveLayer:
    def __init__(self, size, above_size, lr=0.001, dt=0.1, decay=0.01,
                 max_grad=0.5, is_top=False):
        self.size = size
        self.base_lr = lr
        self.lr = lr
        self.dt = dt
        self.decay = decay
        self.max_grad = max_grad
        self.is_top = is_top
        self.activity = np.zeros(size)
        self.prediction = np.zeros(size)
        self.error = np.zeros(size)
        if is_top:
            self.W_topdown = np.zeros((size, above_size))
        else:
            self.W_topdown = np.random.randn(size, above_size) * np.sqrt(
                2.0 / (size + above_size)
            )
        self.W_recurrent = np.random.randn(size, size) * np.sqrt(
            2.0 / (size + size)
        )
        self.bias = np.zeros(size)
        self.gate = PrecisionGate(size)
        self.memory_current = np.zeros(size)
        self.memory_previous = np.zeros(size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward_predict(self, above_activity=None):
        if self.is_top:
            td = np.zeros(self.size)
        else:
            above = _safe(above_activity) if above_activity is not None else np.zeros(self.W_topdown.shape[1])
            td = self.W_topdown @ above
        mem = _safe(self.memory_previous)
        rec = self.W_recurrent @ mem
        self.prediction = self.sigmoid(td + rec + self.bias)
        return self.prediction

    def compute_error(self, target=None):
        target = _safe(target) if target is not None else _safe(self.memory_previous)
        raw_error = target - self.prediction
        self.error = self.gate.apply(raw_error)
        return self.error

    def settle_step(self, below_error=None):
        delta = self.dt * (self.prediction - self.activity + self.error)
        if below_error is not None:
            delta += self.dt * 0.1 * below_error
        self.activity += delta
        self.activity = np.clip(self.activity, 0, 1)
        self.activity = _safe(self.activity, fallback=0.5)

    def settle(self, below_error=None, n_iter=20):
        for _ in range(n_iter):
            self.settle_step(below_error)

    def learn(self, above_activity):
        above = _safe(above_activity)
        mem = _safe(self.memory_previous)
        if not self.is_top:
            dW_td = np.outer(self.error, above)
            dW_td = self._clip_norm(dW_td, self.max_grad)
            self.W_topdown += self.lr * dW_td - self.decay * self.W_topdown
        dW_rec = np.outer(self.error, mem)
        dW_rec = self._clip_norm(dW_rec, self.max_grad)
        db = np.clip(self.error, -self.max_grad, self.max_grad)
        self.W_recurrent += self.lr * dW_rec - self.decay * self.W_recurrent
        self.bias += self.lr * db - self.decay * self.bias
        if not np.isfinite(self.W_recurrent).all():
            self.W_recurrent = np.random.randn(*self.W_recurrent.shape) * 0.01
        if not self.is_top and not np.isfinite(self.W_topdown).all():
            self.W_topdown = np.random.randn(*self.W_topdown.shape) * 0.01
        self.gate.adapt(self.error)

    def _clip_norm(self, grad, max_norm):
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / norm)
        return grad

    def tick(self):
        self.memory_previous = self.memory_current.copy()
        self.memory_current = self.activity.copy()


class Tatha:
    def __init__(self, input_size, hidden_size, belief_size, lr=0.001):
        self.belief = PredictiveLayer(belief_size, belief_size, lr, is_top=True)
        self.hidden = PredictiveLayer(hidden_size, belief_size, lr)
        self.input_layer = PredictiveLayer(input_size, hidden_size, lr)
        self.step_count = 0
        self.observation = None

    def perceive(self, observation):
        self.observation = observation.copy()

    def _infer_step(self):
        self.belief.forward_predict()
        self.hidden.forward_predict(self.belief.activity)
        self.input_layer.forward_predict(self.hidden.activity)
        self.input_layer.compute_error(self.observation)
        hidden_backprop = self.input_layer.W_topdown.T @ self.input_layer.error
        self.hidden.compute_error(self.hidden.prediction + hidden_backprop)
        belief_backprop = self.hidden.W_topdown.T @ self.hidden.error
        self.belief.compute_error(self.belief.prediction + belief_backprop)
        self.belief.settle_step()
        self.hidden.settle_step()
        self.input_layer.settle_step()

    def settle(self, n_iter=30):
        for _ in range(n_iter):
            self._infer_step()
        return self.input_layer.prediction

    def learn(self):
        self._infer_step()
        self.belief.learn(np.zeros(self.belief.size))
        self.hidden.learn(self.belief.activity)
        self.input_layer.learn(self.hidden.activity)
        self.belief.tick()
        self.hidden.tick()
        self.input_layer.tick()
        self.step_count += 1

    def total_surprise(self):
        e_in = np.mean(self.input_layer.error ** 2)
        e_hid = np.mean(self.hidden.error ** 2)
        e_bel = np.mean(self.belief.error ** 2)
        return e_in + e_hid + e_bel


class QuantumTatha:
    def __init__(self, input_size, hidden_size, belief_size, lr=0.005):
        self.belief = QuantumBeliefLayer(belief_size, lr)
        self.hidden = PredictiveLayer(hidden_size, belief_size, lr)
        self.input_layer = PredictiveLayer(input_size, hidden_size, lr)
        self.step_count = 0
        self.observation = None

    def perceive(self, observation):
        self.observation = observation.copy()

    def _infer_step(self):
        self.belief.forward_predict()
        self.hidden.forward_predict(self.belief.activity)
        self.input_layer.forward_predict(self.hidden.activity)
        self.input_layer.compute_error(self.observation)
        hidden_backprop = self.input_layer.W_topdown.T @ self.input_layer.error
        self.hidden.compute_error(self.hidden.prediction + hidden_backprop)
        belief_backprop = self.hidden.W_topdown.T @ self.hidden.error
        self.belief.compute_error(self.belief.prediction + belief_backprop)
        self.belief.settle_step()
        self.hidden.settle_step()
        self.input_layer.settle_step()

    def settle(self, n_iter=30):
        for _ in range(n_iter):
            self._infer_step()
        return self.input_layer.prediction

    def learn(self):
        self._infer_step()
        self.belief.learn()
        self.hidden.learn(self.belief.activity)
        self.input_layer.learn(self.hidden.activity)
        self.belief.tick()
        self.hidden.tick()
        self.input_layer.tick()
        self.step_count += 1

    def total_surprise(self):
        e_in = np.mean(self.input_layer.error ** 2)
        e_hid = np.mean(self.hidden.error ** 2)
        e_bel = np.mean(self.belief.error ** 2)
        return e_in + e_hid + e_bel


class AdaptiveQuantumTatha:
    def __init__(self, input_size, hidden_size, belief_size, lr=0.005):
        self.belief = QuantumBeliefLayer(belief_size, lr)
        self.hidden = PredictiveLayer(hidden_size, belief_size, lr)
        self.input_layer = PredictiveLayer(input_size, hidden_size, lr)
        self.neuro = Neuromodulator(n_layers=3, window=20, boost=2.0, decay=0.5)
        self.step_count = 0
        self.observation = None
        self.base_lrs = [lr, lr, lr]

    def perceive(self, observation):
        self.observation = observation.copy()

    def _infer_step(self):
        self.belief.forward_predict()
        self.hidden.forward_predict(self.belief.activity)
        self.input_layer.forward_predict(self.hidden.activity)
        self.input_layer.compute_error(self.observation)
        hidden_backprop = self.input_layer.W_topdown.T @ self.input_layer.error
        self.hidden.compute_error(self.hidden.prediction + hidden_backprop)
        belief_backprop = self.hidden.W_topdown.T @ self.hidden.error
        self.belief.compute_error(self.belief.prediction + belief_backprop)
        self.belief.settle_step()
        self.hidden.settle_step()
        self.input_layer.settle_step()

    def settle(self, n_iter=30):
        for _ in range(n_iter):
            self._infer_step()
        return self.input_layer.prediction

    def learn(self):
        self._infer_step()
        layer_errors = [
            np.mean(self.belief.error ** 2),
            np.mean(self.hidden.error ** 2),
            np.mean(self.input_layer.error ** 2)
        ]
        lr_mults, prec_mults = self.neuro.update(layer_errors)
        self.belief.lr = self.base_lrs[0] * lr_mults[0]
        self.hidden.lr = self.base_lrs[1] * lr_mults[1]
        self.input_layer.lr = self.base_lrs[2] * lr_mults[2]
        self.belief.gate.precision *= prec_mults[0]
        self.belief.gate.precision = np.clip(self.belief.gate.precision, 0.1, 5.0)
        self.hidden.gate.precision *= prec_mults[1]
        self.hidden.gate.precision = np.clip(self.hidden.gate.precision, 0.1, 5.0)
        self.input_layer.gate.precision *= prec_mults[2]
        self.input_layer.gate.precision = np.clip(self.input_layer.gate.precision, 0.1, 5.0)
        self.belief.learn()
        self.hidden.learn(self.belief.activity)
        self.input_layer.learn(self.hidden.activity)
        self.belief.tick()
        self.hidden.tick()
        self.input_layer.tick()
        self.step_count += 1

    def total_surprise(self):
        e_in = np.mean(self.input_layer.error ** 2)
        e_hid = np.mean(self.hidden.error ** 2)
        e_bel = np.mean(self.belief.error ** 2)
        return e_in + e_hid + e_bel


# ==============================================================================
# PART B: Forward Models & Coordinate Tracker
# ==============================================================================

class ForwardModel:
    def __init__(self, state_dim, action_dim, hidden_dim=32, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.W1 = np.random.randn(hidden_dim, state_dim + action_dim) * np.sqrt(
            2.0 / (state_dim + action_dim + hidden_dim)
        )
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(state_dim, hidden_dim) * np.sqrt(
            2.0 / (hidden_dim + state_dim)
        )
        self.b2 = np.zeros(state_dim)
        self.gate = PrecisionGate(state_dim, lr=0.01)

    def predict(self, state, action_onehot):
        x = np.concatenate([state, action_onehot])
        self.h = np.tanh(self.W1 @ x + self.b1)
        self.out = self.sigmoid(self.W2 @ self.h + self.b2)
        return self.out

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def train(self, state, action_onehot, next_state):
        pred = self.predict(state, action_onehot)
        raw_error = next_state - pred
        error = self.gate.apply(raw_error)
        d_out = error * pred * (1 - pred)
        d_W2 = np.outer(d_out, self.h)
        d_b2 = d_out
        d_h = self.W2.T @ d_out * (1 - self.h ** 2)
        x = np.concatenate([state, action_onehot])
        d_W1 = np.outer(d_h, x)
        d_b1 = d_h
        max_grad = 1.0
        for grad in [d_W1, d_W2]:
            norm = np.linalg.norm(grad)
            if norm > max_grad:
                grad[:] = grad * (max_grad / norm)
        d_b1 = np.clip(d_b1, -max_grad, max_grad)
        d_b2 = np.clip(d_b2, -max_grad, max_grad)
        decay = 0.001
        self.W1 += self.lr * d_W1 - decay * self.W1
        self.b1 += self.lr * d_b1 - decay * self.b1
        self.W2 += self.lr * d_W2 - decay * self.W2
        self.b2 += self.lr * d_b2 - decay * self.b2
        self.gate.adapt(raw_error)
        return np.mean(error ** 2)


class QuantumForwardModel:
    def __init__(self, state_dim, action_dim, hidden_dim=32, quantum_dim=16, lr=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.quantum_dim = quantum_dim
        self.lr = lr
        self.W1 = np.random.randn(hidden_dim, state_dim + action_dim) * 0.05
        self.b1 = np.zeros(hidden_dim)
        self.Wq = (
            np.random.randn(quantum_dim, hidden_dim) * 0.001
            + 1j * np.random.randn(quantum_dim, hidden_dim) * 0.001
        )
        self.W_proj = np.random.randn(state_dim, quantum_dim) * 0.05
        self.b_proj = np.zeros(state_dim)
        self.gate = PrecisionGate(state_dim, lr=0.01)
        self._step_count = 0

    def _check_finite(self):
        for attr in ['W1', 'Wq', 'W_proj', 'b1', 'b_proj']:
            arr = getattr(self, attr)
            if not np.isfinite(arr).all():
                if np.iscomplexobj(arr):
                    setattr(self, attr, (np.random.randn(*arr.shape) + 1j*np.random.randn(*arr.shape)) * 0.001)
                else:
                    setattr(self, attr, np.random.randn(*arr.shape) * 0.01)

    def predict(self, state, action_onehot):
        self._check_finite()
        x = np.concatenate([state, action_onehot])
        z1 = self.W1 @ x + self.b1
        z1 = np.clip(z1, -20, 20)
        self.h = np.tanh(z1)
        self.psi_raw = self.Wq @ self.h
        if not np.isfinite(self.psi_raw).all():
            self.psi_raw = np.ones(self.quantum_dim, dtype=complex) / np.sqrt(self.quantum_dim)
        self.norm = np.linalg.norm(self.psi_raw)
        if self.norm > 1e-8:
            self.psi = self.psi_raw / self.norm
        else:
            self.psi = np.ones(self.quantum_dim, dtype=complex) / np.sqrt(self.quantum_dim)
        self.probs = np.abs(self.psi)**2
        z = self.W_proj @ self.probs + self.b_proj
        z = np.clip(z, -20, 20)
        self.out = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return self.out

    def train(self, state, action_onehot, next_state):
        self._step_count += 1
        pred = self.predict(state, action_onehot)
        raw_error = next_state - pred
        error = self.gate.apply(raw_error)
        d_out = error * pred * (1 - pred)
        d_Wproj = np.outer(d_out, self.probs)
        d_bproj = d_out
        d_probs = self.W_proj.T @ d_out
        d_psi = np.conj(self.psi) * d_probs
        inner = np.vdot(self.psi, d_psi)
        norm_safe = max(self.norm, 1e-8)
        d_psi_raw = (d_psi - self.psi * inner) / norm_safe
        d_Wq = np.outer(d_psi_raw, self.h)
        d_h_from_q = np.real(np.conj(self.Wq).T @ d_psi_raw)
        d_h = d_h_from_q * (1 - self.h ** 2)
        x = np.concatenate([state, action_onehot])
        d_W1 = np.outer(d_h, x)
        d_b1 = d_h
        max_grad = 0.5
        for grad in [d_W1, d_Wproj]:
            norm_g = np.linalg.norm(grad)
            if norm_g > max_grad:
                grad[:] = grad * (max_grad / norm_g)
        d_Wq_norm = np.linalg.norm(d_Wq)
        if d_Wq_norm > max_grad:
            d_Wq = d_Wq * (max_grad / d_Wq_norm)
        d_b1 = np.clip(d_b1, -max_grad, max_grad)
        d_bproj = np.clip(d_bproj, -max_grad, max_grad)
        decay = 0.01
        self.W1 += self.lr * d_W1 - decay * self.W1
        self.b1 += self.lr * d_b1 - decay * self.b1
        self.Wq += self.lr * d_Wq - decay * self.Wq
        self.W_proj += self.lr * d_Wproj - decay * self.W_proj
        self.b_proj += self.lr * d_bproj - decay * self.b_proj
        if self._step_count % 100 == 0:
            self._check_finite()
        self.gate.adapt(raw_error)
        return np.mean(error ** 2)


class CoordinateTracker:
    """SLAM-like internal mapping of visited positions and transitions."""
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.visited = set()
        self.transition_map = {}
        self.visit_count = np.zeros((grid_size, grid_size))
        self.coord_uncertainty = np.ones((grid_size, grid_size))

    def update(self, agent_pos, action, next_pos):
        r, c = int(agent_pos[0]), int(agent_pos[1])
        nr, nc = int(next_pos[0]), int(next_pos[1])
        self.visited.add((r, c))
        self.visited.add((nr, nc))
        self.transition_map[(r, c, int(action))] = (nr, nc)
        self.visit_count[r, c] += 1
        self.visit_count[nr, nc] += 1
        self.coord_uncertainty[r, c] = max(0.1, 1.0 / (1 + self.visit_count[r, c]))
        self.coord_uncertainty[nr, nc] = max(0.1, 1.0 / (1 + self.visit_count[nr, nc]))

    def is_known(self, agent_pos, action):
        r, c = int(agent_pos[0]), int(agent_pos[1])
        return (r, c, int(action)) in self.transition_map

    def predict_next(self, agent_pos, action):
        r, c = int(agent_pos[0]), int(agent_pos[1])
        key = (r, c, int(action))
        if key in self.transition_map:
            return self.transition_map[key]
        return None

    def info_gain(self, agent_pos, action):
        r, c = int(agent_pos[0]), int(agent_pos[1])
        if (r, c, int(action)) in self.transition_map:
            return 0.0
        return self.coord_uncertainty[r, c]

    def get_map(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        for r, c in self.visited:
            grid[r, c] = 1.0
        return grid


class CoordinateNeuralForwardModel:
    def __init__(self, grid_size, action_dim=4, lr=0.05):
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.lr = lr
        n_classes = grid_size
        self.W = np.random.randn(2 * n_classes, 6) * 0.01
        self.b = np.zeros(2 * n_classes)
        self.walls = None
        self.tracker = CoordinateTracker(grid_size)

    def _extract_agent_pos(self, state):
        grid = np.asarray(state).reshape(self.grid_size, self.grid_size)
        agent = np.argwhere(np.abs(grid - 0.66) < 0.1)
        if agent.size > 0:
            return int(agent[0, 0]), int(agent[0, 1])
        target = np.argwhere(np.abs(grid - 1.0) < 0.1)
        if target.size > 0:
            return int(target[0, 0]), int(target[0, 1])
        return 0, 0

    def _reconstruct_grid(self, agent_pos, target_pos, grid_size=None):
        if grid_size is None:
            grid_size = self.grid_size
        grid = np.zeros(grid_size * grid_size)
        if self.walls is not None:
            for w in self.walls:
                grid[w[0]*grid_size + w[1]] = 0.33
        tr, tc = int(target_pos[0]), int(target_pos[1])
        ar, ac = int(agent_pos[0]), int(agent_pos[1])
        grid[tr*grid_size + tc] = 1.0
        if (ar, ac) != (tr, tc):
            grid[ar*grid_size + ac] = 0.66
        else:
            grid[ar*grid_size + ac] = 1.0
        return grid

    def predict(self, state, action_onehot):
        ar, ac = self._extract_agent_pos(state)
        self.ar, self.ac = ar, ac
        grid = np.asarray(state).reshape(self.grid_size, self.grid_size)
        target = np.argwhere(np.abs(grid - 1.0) < 0.1)
        if target.size > 0:
            self.tr, self.tc = int(target[0, 0]), int(target[0, 1])
        else:
            self.tr, self.tc = self.grid_size - 1, self.grid_size - 1

        tracked = self.tracker.predict_next((ar, ac), np.argmax(action_onehot))
        if tracked is not None:
            self.pred_coords = np.array(tracked, dtype=float)
            self.row_probs = np.zeros(self.grid_size)
            self.col_probs = np.zeros(self.grid_size)
            self.row_probs[tracked[0]] = 1.0
            self.col_probs[tracked[1]] = 1.0
            self.out = self._reconstruct_grid(self.pred_coords, (self.tr, self.tc))
            return self.out

        x = np.concatenate([np.array([ar, ac]) / (self.grid_size - 1), action_onehot])
        logits = self.W @ x + self.b
        row_logits = logits[:self.grid_size]
        col_logits = logits[self.grid_size:]
        self.row_probs = np.exp(row_logits - np.max(row_logits))
        self.row_probs /= np.sum(self.row_probs)
        self.col_probs = np.exp(col_logits - np.max(col_logits))
        self.col_probs /= np.sum(self.col_probs)
        next_ar = np.argmax(self.row_probs)
        next_ac = np.argmax(self.col_probs)
        if self.walls is not None:
            if (int(next_ar), int(next_ac)) in self.walls:
                next_ar, next_ac = ar, ac
        self.pred_coords = np.array([next_ar, next_ac], dtype=float)
        self.out = self._reconstruct_grid(self.pred_coords, (self.tr, self.tc))
        return self.out

    def train(self, state, action_onehot, next_state):
        pred = self.predict(state, action_onehot)
        ar, ac = self._extract_agent_pos(state)
        next_ar, next_ac = self._extract_agent_pos(next_state)
        self.tracker.update((ar, ac), np.argmax(action_onehot), (next_ar, next_ac))
        target_row = next_ar
        target_col = next_ac
        row_loss = -np.log(self.row_probs[target_row] + 1e-8)
        col_loss = -np.log(self.col_probs[target_col] + 1e-8)
        loss = row_loss + col_loss
        d_row_logits = self.row_probs.copy()
        d_row_logits[target_row] -= 1.0
        d_col_logits = self.col_probs.copy()
        d_col_logits[target_col] -= 1.0
        d_logits = np.concatenate([d_row_logits, d_col_logits])
        x = np.concatenate([np.array([ar, ac]) / (self.grid_size - 1), action_onehot])
        d_W = np.outer(d_logits, x)
        d_b = d_logits
        max_grad = 2.0
        grad_norm = np.linalg.norm(d_W)
        if grad_norm > max_grad:
            d_W = d_W * (max_grad / grad_norm)
        d_b = np.clip(d_b, -max_grad, max_grad)
        self.W += self.lr * d_W
        self.b += self.lr * d_b
        return loss


class CoordinateQuantumForwardModel:
    def __init__(self, grid_size, action_dim=4, quantum_dim=16, lr=0.05):
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.quantum_dim = quantum_dim
        self.lr = lr
        n_classes = grid_size
        self.Wq = (
            np.random.randn(quantum_dim, 6) * 0.005
            + 1j * np.random.randn(quantum_dim, 6) * 0.005
        )
        self.W_proj = np.random.randn(2 * n_classes, quantum_dim) * 0.01
        self.b_proj = np.zeros(2 * n_classes)
        self.walls = None
        self.tracker = CoordinateTracker(grid_size)

    def _extract_agent_pos(self, state):
        grid = np.asarray(state).reshape(self.grid_size, self.grid_size)
        agent = np.argwhere(np.abs(grid - 0.66) < 0.1)
        if agent.size > 0:
            return int(agent[0, 0]), int(agent[0, 1])
        target = np.argwhere(np.abs(grid - 1.0) < 0.1)
        if target.size > 0:
            return int(target[0, 0]), int(target[0, 1])
        return 0, 0

    def _reconstruct_grid(self, agent_pos, target_pos, grid_size=None):
        if grid_size is None:
            grid_size = self.grid_size
        grid = np.zeros(grid_size * grid_size)
        if self.walls is not None:
            for w in self.walls:
                grid[w[0]*grid_size + w[1]] = 0.33
        tr, tc = int(target_pos[0]), int(target_pos[1])
        ar, ac = int(agent_pos[0]), int(agent_pos[1])
        grid[tr*grid_size + tc] = 1.0
        if (ar, ac) != (tr, tc):
            grid[ar*grid_size + ac] = 0.66
        else:
            grid[ar*grid_size + ac] = 1.0
        return grid

    def predict(self, state, action_onehot):
        ar, ac = self._extract_agent_pos(state)
        self.ar, self.ac = ar, ac
        grid = np.asarray(state).reshape(self.grid_size, self.grid_size)
        target = np.argwhere(np.abs(grid - 1.0) < 0.1)
        if target.size > 0:
            self.tr, self.tc = int(target[0, 0]), int(target[0, 1])
        else:
            self.tr, self.tc = self.grid_size - 1, self.grid_size - 1

        tracked = self.tracker.predict_next((ar, ac), np.argmax(action_onehot))
        if tracked is not None:
            self.pred_coords = np.array(tracked, dtype=float)
            self.row_probs = np.zeros(self.grid_size)
            self.col_probs = np.zeros(self.grid_size)
            self.row_probs[tracked[0]] = 1.0
            self.col_probs[tracked[1]] = 1.0
            self.out = self._reconstruct_grid(self.pred_coords, (self.tr, self.tc))
            return self.out

        x = np.concatenate([np.array([ar, ac]) / (self.grid_size - 1), action_onehot])
        psi_raw = self.Wq @ x
        norm = np.linalg.norm(psi_raw)
        if norm > 1e-8:
            psi = psi_raw / norm
        else:
            psi = np.ones(self.quantum_dim, dtype=complex) / np.sqrt(self.quantum_dim)
        self.psi = psi
        self.norm = norm
        probs = np.abs(psi)**2
        logits = self.W_proj @ probs + self.b_proj
        row_logits = logits[:self.grid_size]
        col_logits = logits[self.grid_size:]
        self.row_probs = np.exp(row_logits - np.max(row_logits))
        self.row_probs /= np.sum(self.row_probs)
        self.col_probs = np.exp(col_logits - np.max(col_logits))
        self.col_probs /= np.sum(self.col_probs)
        next_ar = np.argmax(self.row_probs)
        next_ac = np.argmax(self.col_probs)
        if self.walls is not None:
            if (int(next_ar), int(next_ac)) in self.walls:
                next_ar, next_ac = ar, ac
        self.pred_coords = np.array([next_ar, next_ac], dtype=float)
        self.out = self._reconstruct_grid(self.pred_coords, (self.tr, self.tc))
        return self.out

    def train(self, state, action_onehot, next_state):
        pred = self.predict(state, action_onehot)
        ar, ac = self._extract_agent_pos(state)
        next_ar, next_ac = self._extract_agent_pos(next_state)
        self.tracker.update((ar, ac), np.argmax(action_onehot), (next_ar, next_ac))
        target_row = next_ar
        target_col = next_ac
        row_loss = -np.log(self.row_probs[target_row] + 1e-8)
        col_loss = -np.log(self.col_probs[target_col] + 1e-8)
        loss = row_loss + col_loss
        d_row_logits = self.row_probs.copy()
        d_row_logits[target_row] -= 1.0
        d_col_logits = self.col_probs.copy()
        d_col_logits[target_col] -= 1.0
        d_logits = np.concatenate([d_row_logits, d_col_logits])
        d_Wproj = np.outer(d_logits, np.abs(self.psi)**2)
        d_bproj = d_logits
        d_probs = self.W_proj.T @ d_logits
        d_psi = np.conj(self.psi) * d_probs
        inner = np.vdot(self.psi, d_psi)
        norm_safe = max(self.norm, 1e-8)
        d_psi_raw = (d_psi - self.psi * inner) / norm_safe
        x = np.concatenate([np.array([ar, ac]) / (self.grid_size - 1), action_onehot])
        d_Wq = np.outer(d_psi_raw, x)
        max_grad = 2.0
        d_Wq_norm = np.linalg.norm(d_Wq)
        if d_Wq_norm > max_grad:
            d_Wq = d_Wq * (max_grad / d_Wq_norm)
        d_Wproj_norm = np.linalg.norm(d_Wproj)
        if d_Wproj_norm > max_grad:
            d_Wproj = d_Wproj * (max_grad / d_Wproj_norm)
        d_bproj = np.clip(d_bproj, -max_grad, max_grad)
        self.Wq += self.lr * d_Wq
        self.W_proj += self.lr * d_Wproj
        self.b_proj += self.lr * d_bproj
        return loss


# ==============================================================================
# PART C: Environments
# ==============================================================================

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


class SpaceGridWorld:
    def __init__(self, size=20, n_planets=3, n_asteroids=4):
        self.size = size
        self.n_planets = n_planets
        self.n_asteroids = n_asteroids
        self.agent_pos = [0, 0]
        self.target_pos = [size - 1, size - 1]
        self.planets = []
        self.asteroids = set()
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.waypoints = []
        self.waypoint_idx = 0
        self._generate_space()

    def _generate_space(self):
        self.planets = []
        self.asteroids = set()
        for _ in range(self.n_planets):
            while True:
                pr = np.random.randint(3, self.size - 3)
                pc = np.random.randint(3, self.size - 3)
                radius = np.random.randint(1, 3)
                gravity = np.random.uniform(0.3, 0.8)
                if np.linalg.norm([pr, pc]) > 4 and np.linalg.norm([pr - self.size + 1, pc - self.size + 1]) > 4:
                    self.planets.append((pr, pc, radius, gravity))
                    break
        for _ in range(self.n_asteroids):
            center_r = np.random.randint(2, self.size - 2)
            center_c = np.random.randint(2, self.size - 2)
            for _ in range(np.random.randint(3, 8)):
                ar = np.clip(center_r + np.random.randint(-2, 3), 0, self.size - 1)
                ac = np.clip(center_c + np.random.randint(-2, 3), 0, self.size - 1)
                if (ar, ac) != (0, 0):
                    self.asteroids.add((ar, ac))
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
        for ar, ac in self.asteroids:
            grid[idx(ar, ac)] = 0.33
        for pr, pc, radius, _ in self.planets:
            grid[idx(pr, pc)] = 0.5
        grid[idx(*self.agent_pos)] = 0.66
        grid[idx(*self.target_pos)] = 1.0
        return grid

    def _gravity_effect(self, pos):
        r, c = pos
        for pr, pc, radius, gravity in self.planets:
            dist = np.sqrt((r - pr)**2 + (c - pc)**2)
            if dist < radius + 2 and dist > 0:
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
        if (int(round(new_r)), int(round(new_c))) in self.asteroids:
            new_r, new_c = self.agent_pos
        self.agent_pos = self._gravity_effect([new_r, new_c])
        self.agent_pos = [int(round(self.agent_pos[0])), int(round(self.agent_pos[1]))]
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
                    row += "* "
                elif val < 0.6:
                    row += "O "
                elif val < 0.85:
                    row += "S "
                else:
                    row += "W "
            print("  " + row)


class AdaptiveWaypointSequencer:
    """Dynamic waypoint ordering with obstacle-aware path planning."""
    def __init__(self, waypoints, strategy='adaptive_eig', grid_size=20):
        self.original_waypoints = [tuple(w) for w in waypoints]
        self.waypoints = [tuple(w) for w in waypoints]
        self.strategy = strategy
        self.grid_size = grid_size
        self.visited = []
        self.visit_order = []

    def reset(self):
        self.visited = []
        self.visit_order = []

    def _euclidean(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _obstacle_penalty(self, agent_pos, waypoint, obstacles, planets):
        base_dist = self._euclidean(agent_pos, waypoint)
        penalty = 0.0
        for obs in obstacles:
            if self._euclidean(waypoint, obs) < 3:
                penalty += 2.0
        for pr, pc, radius, _ in planets:
            if self._euclidean(waypoint, (pr, pc)) < radius + 3:
                penalty += 1.5
        return base_dist + penalty

    def _bresenham_line(self, start, end):
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def get_next_target(self, agent_pos, tracker=None, obstacles=None, planets=None):
        unvisited = [w for w in self.waypoints if w not in self.visited]
        if not unvisited:
            return None
        agent_pos = tuple(agent_pos)
        obstacles = obstacles or []
        planets = planets or []

        if self.strategy == 'nearest':
            scores = [self._euclidean(agent_pos, w) for w in unvisited]
        elif self.strategy == 'info_gain':
            scores = []
            for w in unvisited:
                dist = self._euclidean(agent_pos, w)
                ig = 0
                if tracker is not None:
                    for cell in self._bresenham_line(agent_pos, w):
                        r, c = cell
                        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                            ig += tracker.coord_uncertainty[r, c]
                scores.append(dist - 0.5 * ig)
        elif self.strategy == 'tsp_greedy':
            scores = [self._obstacle_penalty(agent_pos, w, obstacles, planets) for w in unvisited]
        elif self.strategy == 'adaptive_eig':
            scores = []
            for w in unvisited:
                dist = self._euclidean(agent_pos, w)
                obs_pen = self._obstacle_penalty(agent_pos, w, obstacles, planets) - dist
                info = 0.0
                if tracker is not None:
                    for cell in self._bresenham_line(agent_pos, w):
                        r, c = cell
                        if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                            info += tracker.coord_uncertainty[r, c]
                scores.append(dist + obs_pen - 0.3 * info)
        else:
            scores = [self._euclidean(agent_pos, w) for w in unvisited]
        return list(unvisited[np.argmin(scores)])

    def mark_visited(self, waypoint):
        w = tuple(waypoint)
        if w not in self.visited:
            self.visited.append(w)
            self.visit_order.append(w)

    def is_complete(self):
        return len(self.visited) >= len(self.waypoints)


class EnhancedSpaceGridWorld:
    """20x20 space with adaptive waypoint sequencing and telemetry."""
    def __init__(self, size=20, n_planets=3, n_asteroids=4,
                 waypoint_strategy='adaptive_eig', difficulty=1.0):
        self.size = size
        self.n_planets = int(n_planets * difficulty)
        self.n_asteroids = int(n_asteroids * difficulty)
        self.difficulty = difficulty
        self.waypoint_strategy = waypoint_strategy
        self.agent_pos = [0, 0]
        self.planets = []
        self.asteroids = set()
        self.actions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        self.sequencer = None
        self.telemetry = {
            'path': [], 'efe_history': [], 'action_history': [],
            'reward_history': [], 'waypoint_reaches': []
        }
        self._generate_space()

    def _generate_space(self):
        self.planets = []
        self.asteroids = set()
        for _ in range(self.n_planets):
            while True:
                pr = np.random.randint(3, self.size - 3)
                pc = np.random.randint(3, self.size - 3)
                radius = np.random.randint(1, 3)
                gravity = np.random.uniform(0.3, 0.8)
                if np.linalg.norm([pr, pc]) > 4 and np.linalg.norm([pr - self.size + 1, pc - self.size + 1]) > 4:
                    self.planets.append((pr, pc, radius, gravity))
                    break
        for _ in range(self.n_asteroids):
            cr = np.random.randint(2, self.size - 2)
            cc = np.random.randint(2, self.size - 2)
            for _ in range(np.random.randint(3, 8)):
                ar = np.clip(cr + np.random.randint(-2, 3), 0, self.size - 1)
                ac = np.clip(cc + np.random.randint(-2, 3), 0, self.size - 1)
                if (ar, ac) != (0, 0):
                    self.asteroids.add((ar, ac))
        base = [[self.size-1, self.size-1], [self.size//2, self.size//2],
                [self.size-1, 0], [0, self.size-1]]
        np.random.shuffle(base)
        self.sequencer = AdaptiveWaypointSequencer(base, self.waypoint_strategy, self.size)
        self.target_pos = self.sequencer.get_next_target([0, 0], obstacles=list(self.asteroids),
                                                          planets=self.planets)

    def reset(self):
        self.agent_pos = [0, 0]
        self.sequencer.reset()
        self.target_pos = self.sequencer.get_next_target([0, 0], obstacles=list(self.asteroids),
                                                          planets=self.planets)
        self.telemetry = {
            'path': [[0, 0]], 'efe_history': [], 'action_history': [],
            'reward_history': [], 'waypoint_reaches': []
        }
        return self.observe()

    def observe(self):
        grid = np.zeros(self.size * self.size)
        idx = lambda r, c: r * self.size + c
        for ar, ac in self.asteroids:
            grid[idx(ar, ac)] = 0.33
        for pr, pc, radius, _ in self.planets:
            grid[idx(pr, pc)] = 0.5
        grid[idx(*self.agent_pos)] = 0.66
        grid[idx(*self.target_pos)] = 1.0
        return grid

    def _gravity_effect(self, pos):
        r, c = pos
        for pr, pc, radius, gravity in self.planets:
            dist = np.sqrt((r - pr)**2 + (c - pc)**2)
            if dist < radius + 2 and dist > 0:
                pull = gravity / (dist + 1)
                r += (pr - r) / dist * pull
                c += (pc - c) / dist * pull
        return [np.clip(r, 0, self.size - 1), np.clip(c, 0, self.size - 1)]

    def step(self, action):
        dr, dc = self.actions[action]
        new_r = np.clip(self.agent_pos[0] + dr, 0, self.size - 1)
        new_c = np.clip(self.agent_pos[1] + dc, 0, self.size - 1)
        if (int(round(new_r)), int(round(new_c))) in self.asteroids:
            new_r, new_c = self.agent_pos
        self.agent_pos = self._gravity_effect([new_r, new_c])
        self.agent_pos = [int(round(self.agent_pos[0])), int(round(self.agent_pos[1]))]
        self.telemetry['path'].append(self.agent_pos.copy())
        self.telemetry['action_history'].append(action)
        reward = 0.0
        done = False
        if self.agent_pos == self.target_pos:
            reward = 1.0
            self.telemetry['reward_history'].append(reward)
            self.telemetry['waypoint_reaches'].append({
                'step': len(self.telemetry['path']) - 1,
                'pos': self.agent_pos.copy(),
                'waypoint_idx': len(self.sequencer.visited)
            })
            self.sequencer.mark_visited(self.target_pos)
            nxt = self.sequencer.get_next_target(self.agent_pos,
                                                  obstacles=list(self.asteroids),
                                                  planets=self.planets)
            if nxt is None:
                done = True
                self.target_pos = self.agent_pos
            else:
                self.target_pos = nxt
        else:
            self.telemetry['reward_history'].append(0.0)
        return self.observe(), reward, done


# ==============================================================================
# PART D: Agents, Training & Utilities
# ==============================================================================

def encode_action(action, n_actions=4):
    vec = np.zeros(n_actions)
    vec[action] = 1.0
    return vec


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

        if hasattr(self.world, 'walls') and (new_r, new_c) in self.world.walls:
            new_r, new_c = ar, ac
        if hasattr(self.world, 'asteroids') and (new_r, new_c) in self.world.asteroids:
            new_r, new_c = ar, ac
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
        surprise = np.mean((pred_next - preferred) ** 2)
        pred_agent = _extract_agent_pos(pred_next, self.world.size)
        target = _extract_target_pos(obs, self.world.size)
        pred_dist = np.linalg.norm(pred_agent - target)
        max_dist = np.sqrt(2) * (self.world.size - 1)
        distance_penalty = 0.5 * (pred_dist / max_dist)
        p = np.clip(pred_next, 1e-8, 1 - 1e-8)
        ambiguity = -np.sum(p * np.log(p) + (1 - p) * np.log(1 - p))
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
            next_obs, reward, done = self.world.step(action)
            action_vec = encode_action(action)
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


class ExperienceReplay:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0

    def push(self, state, action, next_state, reward):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (state.copy(), action.copy(), next_state.copy(), reward)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), n, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class CurriculumTrainer:
    def __init__(self, base_size=20, max_difficulty=1.0, stages=5):
        self.base_size = base_size
        self.max_difficulty = max_difficulty
        self.stages = stages
        self.difficulties = np.linspace(0.2, max_difficulty, stages)
        self.stage_rewards = [[] for _ in range(stages)]
        self.stage_successes = [[] for _ in range(stages)]

    def create_world(self, stage_idx, strategy='adaptive_eig'):
        diff = self.difficulties[stage_idx]
        return EnhancedSpaceGridWorld(
            size=self.base_size,
            n_planets=max(1, int(3 * diff)),
            n_asteroids=max(1, int(4 * diff)),
            waypoint_strategy=strategy,
            difficulty=1.0
        )

    def train_stage(self, stage_idx, forward_model, perceptual_class,
                    episodes=500, steps_per_ep=60, replay=None,
                    input_size=400, hidden_size=32, belief_size=16,
                    lr=0.005, verbose=True):
        world = self.create_world(stage_idx)
        if hasattr(forward_model, 'walls'):
            forward_model.walls = world.asteroids

        p = perceptual_class(input_size=input_size, hidden_size=hidden_size,
                            belief_size=belief_size, lr=lr)
        for _ in range(200):
            obs = world.observe()
            p.perceive(obs)
            p.settle(n_iter=5)
            p.learn()
            if np.random.rand() < 0.15:
                world.reset()
            else:
                world.step(np.random.randint(0, 4))

        agent = PCGridAgent(world, p, forward_model, exploration_coef=0.15)
        success_count = 0
        total_reward = 0

        for ep in range(episodes):
            epsilon = max(0.1, 1.0 - ep / (episodes * 0.7))
            obs = world.reset()
            p.perceive(obs)
            p.settle(n_iter=8)
            ep_reward = 0

            for step in range(steps_per_ep):
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action, _ = agent.select_action(obs)

                action_vec = encode_action(action)
                next_obs, reward, done = world.step(action)
                ep_reward += reward

                agent_now = _extract_agent_pos(obs, world.size)
                agent_next = _extract_agent_pos(next_obs, world.size)
                agent.tracker.update(agent_now, action, agent_next)
                forward_model.train(obs, action_vec, next_obs)

                if replay is not None:
                    replay.push(obs, action_vec, next_obs, reward)
                    if len(replay) >= 32 and step % 4 == 0:
                        for s, a, ns, _ in replay.sample(16):
                            forward_model.train(s, a, ns)

                p.perceive(next_obs)
                p.settle(n_iter=5)
                p.learn()
                obs = next_obs
                if done:
                    success_count += 1
                    break

            total_reward += ep_reward
            self.stage_rewards[stage_idx].append(ep_reward)
            self.stage_successes[stage_idx].append(1 if ep_reward > 0 else 0)

            if verbose and ep % max(1, episodes // 10) == 0:
                print(f"  Stage {stage_idx} Ep {ep:4d} | Reward={ep_reward:.1f} | ε={epsilon:.2f}")

        avg_reward = total_reward / episodes
        success_rate = success_count / episodes
        if verbose:
            print(f"  Stage {stage_idx} DONE | AvgReward={avg_reward:.3f} | Success={success_rate:.1%}")
        return success_rate, avg_reward

    def get_progression_stats(self):
        stats = []
        for i in range(self.stages):
            if self.stage_rewards[i]:
                stats.append({
                    'stage': i,
                    'difficulty': self.difficulties[i],
                    'avg_reward': np.mean(self.stage_rewards[i]),
                    'success_rate': np.mean(self.stage_successes[i]),
                    'final_50_success': np.mean(self.stage_successes[i][-50:]) if len(self.stage_successes[i]) >= 50 else np.mean(self.stage_successes[i])
                })
        return stats


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


# ==============================================================================
# PART E: Visualization Suite
# ==============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization disabled.")


class PCVisualizer:
    """Comprehensive visualization for predictive coding space navigation."""

    def __init__(self, figsize=(14, 10)):
        self.figsize = figsize

    def render_grid(self, world, path=None, title="Space Environment", ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig = ax.figure

        size = world.size
        grid = np.zeros((size, size, 3))
        grid[:, :, :] = 0.05

        for ar, ac in world.asteroids:
            grid[ar, ac] = [0.4, 0.3, 0.2]

        for pr, pc, radius, gravity in world.planets:
            for dr in range(-radius-1, radius+2):
                for dc in range(-radius-1, radius+2):
                    r, c = pr + dr, pc + dc
                    if 0 <= r < size and 0 <= c < size:
                        dist = np.sqrt(dr**2 + dc**2)
                        if dist <= radius + 1:
                            intensity = max(0, 1 - dist / (radius + 1.5))
                            grid[r, c] = [0.2 + 0.3 * intensity,
                                          0.1 + 0.2 * intensity,
                                          0.4 + 0.4 * intensity]

        for i, wp in enumerate(world.sequencer.visited):
            wr, wc = wp
            grid[wr, wc] = [0.0, 0.8, 0.2]

        tr, tc = world.target_pos
        grid[tr, tc] = [0.9, 0.9, 0.1]

        ar, ac = world.agent_pos
        grid[ar, ac] = [0.9, 0.2, 0.2]

        ax.imshow(grid, interpolation='nearest')

        if path is not None and len(path) > 1:
            path_arr = np.array(path)
            ax.plot(path_arr[:, 1], path_arr[:, 0], 'w-', linewidth=1.5, alpha=0.6)
            ax.plot(path_arr[0, 1], path_arr[0, 0], 'go', markersize=10, label='Start')
            ax.plot(path_arr[-1, 1], path_arr[-1, 0], 'ro', markersize=10, label='End')

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=[0.9, 0.2, 0.2], markersize=8, label='Agent'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=[0.9, 0.9, 0.1], markersize=8, label='Target'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=[0.0, 0.8, 0.2], markersize=8, label='Visited'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=[0.4, 0.3, 0.2], markersize=8, label='Asteroid'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=[0.3, 0.2, 0.6], markersize=8, label='Planet'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot_trajectory(self, world, title="Agent Trajectory", save_path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        path = world.telemetry['path']
        self.render_grid(world, path=path, title=title, ax=ax)

        for wp_info in world.telemetry['waypoint_reaches']:
            step = wp_info['step']
            pos = wp_info['pos']
            ax.annotate(f'WP{wp_info["waypoint_idx"]+1}\n@{step}',
                       xy=(pos[1], pos[0]), xytext=(pos[1]+2, pos[0]-2),
                       fontsize=8, color='lime', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='lime', lw=1))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def plot_training_curves(self, trainer, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        stats = trainer.get_progression_stats()
        stages = [s['stage'] for s in stats]
        difficulties = [s['difficulty'] for s in stats]

        ax = axes[0, 0]
        success_rates = [s['success_rate'] for s in stats]
        ax.bar(stages, success_rates, color='steelblue', alpha=0.7)
        ax.set_xlabel('Curriculum Stage')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Difficulty Stage')
        ax.set_ylim(0, 1)
        for i, (st, sr) in enumerate(zip(stages, success_rates)):
            ax.text(st, sr + 0.02, f'{sr:.1%}', ha='center', fontsize=9)

        ax = axes[0, 1]
        for i, rewards in enumerate(trainer.stage_rewards):
            if rewards:
                window = min(50, len(rewards))
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(len(smoothed)), smoothed, label=f'Stage {i} (d={trainer.difficulties[i]:.1f})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Smoothed Reward')
        ax.set_title('Reward Curves per Stage')
        ax.legend(fontsize=8)

        ax = axes[1, 0]
        final_success = [s['final_50_success'] for s in stats]
        ax.plot(difficulties, final_success, 'o-', color='darkgreen', linewidth=2, markersize=8)
        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Final 50-Episode Success Rate')
        ax.set_title('Generalization: Difficulty vs Performance')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.axis('off')
        table_data = []
        for s in stats:
            table_data.append([
                f"Stage {s['stage']}",
                f"{s['difficulty']:.1f}",
                f"{s['avg_reward']:.2f}",
                f"{s['success_rate']:.1%}",
                f"{s['final_50_success']:.1%}"
            ])
        table = ax.table(cellText=table_data,
                        colLabels=['Stage', 'Difficulty', 'Avg Reward', 'Success', 'Final 50'],
                        cellLoc='center', loc='center',
                        colColours=['#4472C4']*5)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        for key, cell in table.get_celld().items():
            if key[0] == 0:
                cell.set_text_props(color='white', fontweight='bold')
        ax.set_title('Curriculum Summary', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def plot_efe_landscape(self, agent, world, obs, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax = axes[0]
        self.render_grid(world, title="Current State", ax=ax)

        ax = axes[1]
        actions = []
        efes = []
        for a in range(4):
            efe, _ = agent.expected_free_energy(obs, a)
            actions.append(world.action_names[a])
            efes.append(efe)

        colors = ['#e74c3c' if e == min(efes) else '#3498db' for e in efes]
        bars = ax.bar(actions, efes, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Expected Free Energy (lower = better)')
        ax.set_title('EFE by Action')
        ax.axhline(y=min(efes), color='red', linestyle='--', alpha=0.5,
                   label=f'Chosen: {actions[np.argmin(efes)]}')
        for bar, val in zip(bars, efes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig

    def plot_belief_state(self, perceptual_agent, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        layers = [
            ('Input Layer', perceptual_agent.input_layer.activity),
            ('Hidden Layer', perceptual_agent.hidden.activity),
            ('Belief Layer', perceptual_agent.belief.activity if hasattr(perceptual_agent.belief, 'activity')
             else np.abs(perceptual_agent.belief.psi)**2)
        ]
        for ax, (name, activity) in zip(axes, layers):
            activity = np.asarray(activity).flatten()
            n = len(activity)
            side = int(np.ceil(np.sqrt(n)))
            padded = np.zeros(side * side)
            padded[:n] = activity
            img = padded.reshape(side, side)
            im = ax.imshow(img, cmap='viridis', aspect='auto')
            ax.set_title(f'{name}\n({n} units)', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        return fig


# ==============================================================================
# PART F: Main Demo
# ==============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 70)
    print("TATHA UNIFIED — Full System Demo")
    print("=" * 70)

    # --- 1. Core Agent Verification ---
    print("\n[1] Core Predictive Coding Verification")
    print("-" * 50)
    agent = Tatha(input_size=10, hidden_size=6, belief_size=3, lr=0.02)
    surprises = []
    for t in range(100):
        obs = np.sin(np.linspace(0, 2*np.pi, 10) + t*0.2) * 0.4 + 0.5
        agent.perceive(obs)
        agent.settle(n_iter=25)
        agent.learn()
        surprises.append(agent.total_surprise())
    print(f"  Classical Tatha | Surprise reduction: {(1 - surprises[-1]/surprises[0])*100:.1f}%")

    qagent = QuantumTatha(input_size=10, hidden_size=16, belief_size=8, lr=0.01)
    qsurprises = []
    for t in range(100):
        obs = np.sin(np.linspace(0, 2*np.pi, 10) + t*0.2) * 0.4 + 0.5
        qagent.perceive(obs)
        qagent.settle(n_iter=25)
        qagent.learn()
        qsurprises.append(qagent.total_surprise())
    print(f"  Quantum Tatha   | Surprise reduction: {(1 - qsurprises[-1]/qsurprises[0])*100:.1f}%")

    # --- 2. 5x5 Grid World ---
    print("\n[2] Classic 5x5 Grid World")
    print("-" * 50)
    world = GridWorld()
    tabular_fm = TabularForwardModel(world)
    train_forward_model(world, tabular_fm, episodes=1500, steps_per_ep=30)
    print(f"  Tabular FM | Coverage: {tabular_fm.coverage()} transitions")

    neural_fm = CoordinateNeuralForwardModel(grid_size=5, lr=0.05)
    neural_fm.walls = world.walls
    train_forward_model(world, neural_fm, episodes=1500, steps_per_ep=30)
    print(f"  Neural FM  | Tracker: {len(neural_fm.tracker.transition_map)} transitions")

    p = Tatha(input_size=25, hidden_size=16, belief_size=8, lr=0.01)
    train_perceptual_agent(p, world, steps=500)
    test_agent = PCGridAgent(world, p, tabular_fm)
    successes = 0
    for _ in range(10):
        s, st = test_agent.navigate(max_steps=15, train=False, verbose=False)
        if s:
            successes += 1
    print(f"  Evaluation | {successes}/10 success (Classical + Tabular)")

    # --- 3. Enhanced Space World ---
    print("\n[3] Enhanced 20x20 Space World")
    print("-" * 50)
    space_world = EnhancedSpaceGridWorld(size=20, n_planets=3, n_asteroids=4,
                                          waypoint_strategy='adaptive_eig')
    print(f"  Waypoints: {space_world.sequencer.original_waypoints}")
    print(f"  Strategy: {space_world.waypoint_strategy}")
    print(f"  Planets: {len(space_world.planets)} | Asteroids: {len(space_world.asteroids)}")

    space_fm = CoordinateNeuralForwardModel(grid_size=20, lr=0.03)
    space_fm.walls = space_world.asteroids

    # Light curriculum demo
    trainer = CurriculumTrainer(base_size=20, max_difficulty=1.0, stages=3)
    replay = ExperienceReplay(capacity=5000)
    print("\n  Running light curriculum (3 stages x 100 episodes)...")
    for stage in range(3):
        trainer.train_stage(stage, space_fm, Tatha, episodes=100, steps_per_ep=60,
                           replay=replay, verbose=False)

    stats = trainer.get_progression_stats()
    print("\n  Curriculum Results:")
    for s in stats:
        print(f"    Stage {s['stage']}: Success={s['success_rate']:.1%}, "
              f"AvgReward={s['avg_reward']:.2f}")

    # --- 4. Visualization Demo ---
    if MATPLOTLIB_AVAILABLE:
        print("\n[4] Generating Visualizations")
        print("-" * 50)
        viz = PCVisualizer()

        # Trajectory plot
        p_space = Tatha(input_size=400, hidden_size=32, belief_size=16, lr=0.005)
        train_perceptual_agent(p_space, space_world, steps=200)
        demo_agent = PCGridAgent(space_world, p_space, space_fm)
        demo_agent.navigate(max_steps=40, train=False, verbose=False)

        fig = viz.plot_trajectory(space_world, title="Space Navigation Trajectory",
                                   save_path="/mnt/agents/output/space_trajectory.png")
        plt.close(fig)

        # Training curves
        fig = viz.plot_training_curves(trainer, save_path="/mnt/agents/output/training_curves.png")
        plt.close(fig)

        # Belief state
        fig = viz.plot_belief_state(p_space, save_path="/mnt/agents/output/belief_state.png")
        plt.close(fig)

        print("  Generated:")
        print("    - space_trajectory.png")
        print("    - training_curves.png")
        print("    - belief_state.png")
    else:
        print("\n[4] Matplotlib not available — skipping visualization")

    print("\n" + "=" * 70)
    print("Demo complete.")
    print("=" * 70)
