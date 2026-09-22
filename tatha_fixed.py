"""
Tathā — Temporal Active Inference Agent (FIXED)
Predictive coding network with quantum belief layer, neuromodulation,
active inference action selection, and coordinate tracking.

FIXES:
1. Coordinated settling with bottom-up error propagation.
2. Stale-gradient fix in learn().
3. Top-layer isolation for belief/quantum layers.
4. QuantumForwardModel: correct Wirtinger backprop through unit sphere.
5. Coordinate-based forward models with agent-only input (no target noise).
6. CoordinateTracker for internal SLAM-like mapping.
"""

import numpy as np


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

    def snapshot(self):
        return {
            'belief_psi': self.belief.psi.copy(),
            'belief_psi_prev': self.belief.psi_prev.copy(),
            'belief_activity': self.belief.activity.copy(),
            'hidden_activity': self.hidden.activity.copy(),
            'hidden_memory_current': self.hidden.memory_current.copy(),
            'hidden_memory_previous': self.hidden.memory_previous.copy(),
            'input_activity': self.input_layer.activity.copy(),
            'input_memory_current': self.input_layer.memory_current.copy(),
            'input_memory_previous': self.input_layer.memory_previous.copy(),
        }

    def restore(self, snap):
        self.belief.psi = snap['belief_psi']
        self.belief.psi_prev = snap['belief_psi_prev']
        self.belief.activity = snap['belief_activity']
        self.hidden.activity = snap['hidden_activity']
        self.hidden.memory_current = snap['hidden_memory_current']
        self.hidden.memory_previous = snap['hidden_memory_previous']
        self.input_layer.activity = snap['input_activity']
        self.input_layer.memory_current = snap['input_memory_current']
        self.input_layer.memory_previous = snap['input_memory_previous']


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


# ---------------------------------------------------------------------------
# Coordinate Tracker — SLAM-like internal mapping
# ---------------------------------------------------------------------------
class CoordinateTracker:
    """
    Maintains an internal coordinate map of visited positions and known transitions.
    Provides uncertainty estimates for information gain computation.
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.visited = set()
        self.transition_map = {}  # (r, c, action) -> (next_r, next_c)
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
        # Uncertainty decreases with visits
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
        """Expected information gain for taking action from agent_pos."""
        r, c = int(agent_pos[0]), int(agent_pos[1])
        if (r, c, int(action)) in self.transition_map:
            return 0.0
        return self.coord_uncertainty[r, c]

    def get_map(self):
        """Return a grid visualization of visited cells."""
        grid = np.zeros((self.grid_size, self.grid_size))
        for r, c in self.visited:
            grid[r, c] = 1.0
        return grid


# ---------------------------------------------------------------------------
# Coordinate-based Forward Models (agent-only input, no target noise)
# ---------------------------------------------------------------------------
class CoordinateNeuralForwardModel:
    """
    Linear neural forward model with coordinate bottleneck.
    Input: agent position + action only (target is irrelevant for dynamics).
    Output: 5-class softmax for next row and column.
    """
    def __init__(self, grid_size, action_dim=4, lr=0.05):
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.lr = lr
        n_classes = grid_size
        # Input: 2 coords + action = 2 + 4 = 6D
        self.W = np.random.randn(2 * n_classes, 6) * 0.01
        self.b = np.zeros(2 * n_classes)
        self.walls = None
        self.tracker = CoordinateTracker(grid_size)

    def _extract_agent_pos(self, state):
        grid = np.asarray(state).reshape(self.grid_size, self.grid_size)
        agent = np.argwhere(np.abs(grid - 0.66) < 0.1)
        if agent.size > 0:
            return int(agent[0, 0]), int(agent[0, 1])
        # Overlap case
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
        # Extract target from state
        grid = np.asarray(state).reshape(self.grid_size, self.grid_size)
        target = np.argwhere(np.abs(grid - 1.0) < 0.1)
        if target.size > 0:
            self.tr, self.tc = int(target[0, 0]), int(target[0, 1])
        else:
            self.tr, self.tc = self.grid_size - 1, self.grid_size - 1

        # Check tracker first
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

        # Update tracker
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
    """
    Quantum forward model with coordinate bottleneck.
    Direct input -> quantum amplitudes -> 5-class softmax for row and column.
    """
    def __init__(self, grid_size, action_dim=4, quantum_dim=16, lr=0.05):
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.quantum_dim = quantum_dim
        self.lr = lr
        n_classes = grid_size
        # Input: 2 coords + 4 action = 6D -> quantum -> 2*n_classes logits
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

        # Check tracker first
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

        # Update tracker
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


if __name__ == "__main__":
    print("=" * 60)
    print("Tathā — Temporal Predictive Coding Agent (FIXED)")
    print("=" * 60)

    print("\n--- Classical Tathā ---")
    agent = Tatha(input_size=10, hidden_size=6, belief_size=3, lr=0.02)
    surprises = []
    for t in range(200):
        obs = np.sin(np.linspace(0, 2 * np.pi, 10) + t * 0.2) * 0.4 + 0.5
        agent.perceive(obs)
        agent.settle(n_iter=25)
        agent.learn()
        surprise = agent.total_surprise()
        surprises.append(surprise)
        if t % 40 == 0:
            print(f"Step {t:3d} | Surprise = {surprise:.6f}")
    print(f"Final Surprise: {surprises[-1]:.6f}")
    print(f"Reduction: {(1 - surprises[-1]/surprises[0])*100:.1f}%")

    print("\n--- Quantum Tathā ---")
    qagent = QuantumTatha(input_size=10, hidden_size=16, belief_size=8, lr=0.01)
    qsurprises = []
    for t in range(200):
        obs = np.sin(np.linspace(0, 2 * np.pi, 10) + t * 0.2) * 0.4 + 0.5
        qagent.perceive(obs)
        qagent.settle(n_iter=25)
        qagent.learn()
        surprise = qagent.total_surprise()
        qsurprises.append(surprise)
        if t % 40 == 0:
            print(f"Step {t:3d} | Surprise={surprise:.6f}")
    print(f"Final Surprise: {qsurprises[-1]:.6f}")
    print(f"Reduction: {(1 - qsurprises[-1]/qsurprises[0])*100:.1f}%")

    print("\n--- Adaptive Quantum Tathā ---")
    aagent = AdaptiveQuantumTatha(input_size=10, hidden_size=16, belief_size=8, lr=0.01)
    asurprises = []
    for t in range(300):
        obs = np.sin(np.linspace(0, 2 * np.pi, 10) + t * 0.2) * 0.4 + 0.5
        aagent.perceive(obs)
        aagent.settle(n_iter=25)
        aagent.learn()
        surprise = aagent.total_surprise()
        asurprises.append(surprise)
        if t % 50 == 0:
            print(f"\nStep {t:3d} | Surprise={surprise:.6f}")
            print(aagent.neuro.report())
    print(f"\nFinal Surprise: {asurprises[-1]:.6f}")
    print(f"Reduction: {(1 - asurprises[-1]/asurprises[0])*100:.1f}%")
    print("\nTathā is learning to predict.")
