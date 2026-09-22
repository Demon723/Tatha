"""Core tests for Tatha agent system."""
import numpy as np
import pytest


def test_tatha_import():
    from unified_pc_space import Tatha, QuantumTatha, CoordinateTracker
    t = Tatha(input_size=100, hidden_size=32, belief_size=8, lr=0.005)
    assert t is not None
    assert t.input_layer.size == 100


def test_quantum_tatha():
    from unified_pc_space import QuantumTatha
    t = QuantumTatha(input_size=100, hidden_size=32, belief_size=8, lr=0.005)
    assert t is not None


def test_coordinate_tracker():
    from unified_pc_space import CoordinateTracker
    ct = CoordinateTracker(grid_size=10)
    assert ct is not None
    assert ct.grid_size == 10


def test_ppo_agent():
    from tatha_ppo import PPOAgent, PPOConfig
    config = PPOConfig()
    agent = PPOAgent(obs_dim=100, n_actions=4, config=config)
    assert agent is not None


def test_replay_buffer():
    from tatha_replay import PrioritizedReplayBuffer, ReplayBuffer, CurriculumScheduler, RewardShaper
    buf = PrioritizedReplayBuffer(capacity=100)
    buf.push(np.zeros(300), 0, 0.0, np.zeros(300), False)
    assert len(buf) == 1
    batch, indices, weights = buf.sample(1)
    assert len(batch) == 1

    uniform = ReplayBuffer(capacity=100)
    uniform.push(np.zeros(300), 0, 0.0, np.zeros(300), False)
    assert len(uniform) == 1


def test_curriculum_scheduler():
    from tatha_replay import CurriculumScheduler
    cs = CurriculumScheduler(n_levels=5)
    assert cs.get_difficulty() == 0
    cs.update(0.95)
    assert cs.get_difficulty() > 0


def test_attention():
    from tatha_attention import AttentionConfig, MultiHeadAttention, TemporalAttentionWrapper, PositionalEncoding
    config = AttentionConfig()
    attn = MultiHeadAttention(config)
    x = np.random.randn(10, config.attention_dim * config.num_heads) * 0.01
    out, weights = attn.forward(x)
    assert out.shape == x.shape


def test_config():
    from tatha_utils import TathaConfig
    config = TathaConfig()
    assert config.GRID_SIZE == 10
    d = config.to_dict()
    assert "GRID_SIZE" in d or "grid_size" in d


def test_benchmark_suite():
    from tatha_benchmarks import BenchmarkSuite, BenchmarkResult
    suite = BenchmarkSuite()
    result = BenchmarkResult("test", 0.95, 1.5, 100)
    assert str(result) != ""
    suite.results.append(result)
    assert len(suite.results) == 1


def test_observation_validator():
    from tatha_utils import ObservationValidator
    validator = ObservationValidator(expected_size=300)
    valid, error = validator.validate([0.0] * 300)
    assert valid is True


def test_error_recovery():
    from tatha_utils import ErrorRecovery
    er = ErrorRecovery(max_retries=3)
    result = er.execute_with_recovery(lambda: 42)
    assert result == 42


def test_neuroscience_metrics():
    from tatha_utils import NeuroscienceMetrics
    nm = NeuroscienceMetrics()
    nm.record_free_energy(1.0)
    nm.record_precision(0.5)
    summary = nm.summary()
    assert "free_energy" in summary or len(summary) > 0


def test_meta_director():
    from meta_director import MetaParameterSpace, MetaDirector
    n = MetaParameterSpace.N_PARAMS
    md = MetaDirector(n_params=n)
    assert md.n_params == n
    params = md.select_parameters(np.zeros(5), explore=False)
    assert params.shape[0] == n


def test_multiverse():
    from multiverse_era import MultiverseState, branching_operator, decohere
    mv = MultiverseState(universe_dim=4, max_branches=8)
    mv.evolve(dt=0.1)
    assert abs(np.linalg.norm(mv.joint_vector()) - 1.0) < 1e-6


def test_tatha_realtime():
    from tatha_realtime import TathaRealTime, TathaConfig, TathaWSServer, TathaREPL, ActionExecutor
    config = TathaConfig()
    config.GRID_SIZE = 10
    agent = TathaRealTime(config)
    assert agent is not None
    assert agent.episode == 0
    obs = agent.reset()
    assert obs is not None
    result = agent.observe(obs, reward=0.0, done=False)
    assert "action" in result
    assert "belief" in result
    assert "efe" in result
    assert "branches" in result


def test_action_executor():
    from tatha_realtime import ActionExecutor
    executor = ActionExecutor()
    result = executor.execute(0, {"obs": None})
    assert result["status"] == "executed"
    assert result["action"] == 0
    history = executor.get_history(5)
    assert len(history) == 1


def test_checkpoint():
    from tatha_realtime import TathaRealTime, TathaConfig
    config = TathaConfig()
    config.GRID_SIZE = 10
    agent = TathaRealTime(config)
    path = agent.checkpoint_save("test_checkpoint")
    assert path is not None and path.endswith(".pkl")
    ok = agent.checkpoint_load("test_checkpoint")
    assert ok is True


def test_parity_with_quantum_realm():
    """Ensure quantum realm can be imported and used."""
    from quantum_realm import QuantumWavepacket, QuantumHamiltonian, QuantumRealmGridWorld
    realm = QuantumRealmGridWorld(size=10, n_planets=2, dt=0.1)
    assert realm.size == 10
    wp = QuantumWavepacket(realm.size)
    assert wp is not None
