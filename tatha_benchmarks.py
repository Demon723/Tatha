"""
tatha_benchmarks.py
===================
Standardized benchmarks across all Tatha environments.

Provides:
  - Grid navigation benchmark
  - Quantum realm benchmark
  - Swarm coordination benchmark
  - Adversarial game benchmark
  - Multiverse branching benchmark
  - Throughput and latency tests
"""
from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    score: float
    time_seconds: float
    steps: int
    metadata: dict = None

    def __str__(self):
        return f"{self.name}: {self.score:.4f} ({self.time_seconds:.2f}s, {self.steps} steps)"


class BenchmarkSuite:
    """Run standardized benchmarks across environments."""

    def __init__(self, output_dir: str = "./benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def benchmark_grid_navigation(self, env, agent, max_steps: int = 100) -> BenchmarkResult:
        """Benchmark grid world navigation."""
        start = time.time()
        obs = env.reset()
        success = False
        steps = 0

        for step in range(max_steps):
            action = agent.act() if hasattr(agent, 'act') else 0
            obs, reward, done = env.step(action)
            if hasattr(agent, 'perceive'):
                agent.perceive(obs)
                agent.learn()
            steps += 1
            if done:
                success = True
                break

        elapsed = time.time() - start
        score = 1.0 if success else 0.0
        result = BenchmarkResult("grid_navigation", score, elapsed, steps,
                                  {'success': success})
        self.results.append(result)
        return result

    def benchmark_quantum_realm(self, env, agent, max_steps: int = 80) -> BenchmarkResult:
        """Benchmark quantum realm waypoint completion."""
        start = time.time()
        obs = env.reset()
        waypoints_reached = 0
        total_steps = 0

        for step in range(max_steps):
            action = agent.act() if hasattr(agent, 'act') else 0
            obs, reward, done = env.step(action)
            if hasattr(agent, 'perceive'):
                agent.perceive(obs)
                agent.learn()
            total_steps += 1
            if done:
                waypoints_reached += 1

        elapsed = time.time() - start
        score = waypoints_reached / len(env.waypoints) if env.waypoints else 0.0
        result = BenchmarkResult("quantum_realm", score, elapsed, total_steps,
                                  {'waypoints': waypoints_reached})
        self.results.append(result)
        return result

    def benchmark_swarm_coordination(self, env, agents, max_steps: int = 60) -> BenchmarkResult:
        """Benchmark multi-agent swarm coordination."""
        start = time.time()
        obs_list = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            actions = [a.act() for a in agents]
            obs_list, rewards, done = env.step(actions)
            for i, agent in enumerate(agents):
                agent.perceive(obs_list[i])
                agent.learn()
            total_reward += sum(rewards)
            if done:
                break

        elapsed = time.time() - start
        avg_reward = total_reward / max_steps
        result = BenchmarkResult("swarm_coordination", avg_reward, elapsed,
                                  max_steps, {'total_reward': total_reward})
        self.results.append(result)
        return result

    def benchmark_adversarial(self, env, red_agents, blue_agents,
                                 max_rounds: int = 50) -> BenchmarkResult:
        """Benchmark adversarial RED vs BLUE game."""
        start = time.time()
        env.reset()
        red_score = 0.0
        rounds = 0

        for round_num in range(max_rounds):
            # RED moves
            red_actions = [a.select_action() for a in red_agents]
            # BLUE responds
            blue_actions = [a.select_action() for a in blue_agents]
            rewards = env.step(red_actions + blue_actions)
            red_score += rewards[0]
            rounds += 1

        elapsed = time.time() - start
        score = red_score / max_rounds
        result = BenchmarkResult("adversarial", score, elapsed, rounds,
                                  {'red_score': red_score})
        self.results.append(result)
        return result

    def benchmark_multiverse(self, env, agent, max_steps: int = 200) -> BenchmarkResult:
        """Benchmark multiverse branching."""
        start = time.time()
        obs = env.reset()
        max_branches = 0

        for step in range(max_steps):
            action = agent.act()
            obs, reward, done = env.step(action)
            if hasattr(agent, 'perceive'):
                agent.perceive(obs)
                agent.learn()
            max_branches = max(max_branches, env.mv.num_branches)
            if done:
                break

        elapsed = time.time() - start
        score = max_branches
        result = BenchmarkResult("multiverse", score, elapsed, max_steps,
                                  {'max_branches': max_branches})
        self.results.append(result)
        return result

    def benchmark_throughput(self, agent, obs_size: int = 300,
                                n_iterations: int = 1000) -> BenchmarkResult:
        """Benchmark inference throughput."""
        obs = np.random.randn(obs_size)
        start = time.time()

        for _ in range(n_iterations):
            if hasattr(agent, 'perceive'):
                agent.perceive(obs)
            if hasattr(agent, 'act'):
                agent.act()

        elapsed = time.time() - start
        throughput = n_iterations / elapsed
        result = BenchmarkResult("throughput", throughput, elapsed, n_iterations,
                                  {'throughput_per_sec': throughput})
        self.results.append(result)
        return result

    def benchmark_latency(self, agent, obs_size: int = 300,
                             n_samples: int = 100) -> BenchmarkResult:
        """Benchmark inference latency."""
        obs = np.random.randn(obs_size)
        latencies = []

        for _ in range(n_samples):
            start = time.perf_counter()
            if hasattr(agent, 'perceive'):
                agent.perceive(obs)
            if hasattr(agent, 'act'):
                agent.act()
            latencies.append(time.perf_counter() - start)

        avg_latency = float(np.mean(latencies)) * 1000  # ms
        p99_latency = float(np.percentile(latencies, 99)) * 1000
        result = BenchmarkResult("latency", avg_latency, sum(latencies), n_samples,
                                  {'avg_ms': avg_latency, 'p99_ms': p99_latency})
        self.results.append(result)
        return result

    def run_all(self, envs: dict, agents: dict) -> list[BenchmarkResult]:
        """Run all benchmarks."""
        self.results = []

        if 'grid' in envs and 'grid_agent' in agents:
            self.benchmark_grid_navigation(envs['grid'], agents['grid_agent'])

        if 'quantum' in envs and 'quantum_agent' in agents:
            self.benchmark_quantum_realm(envs['quantum'], agents['quantum_agent'])

        if 'swarm' in envs and 'swarm_agents' in agents:
            self.benchmark_swarm_coordination(envs['swarm'], agents['swarm_agents'])

        if 'adversarial' in envs and 'red_agents' in agents and 'blue_agents' in agents:
            self.benchmark_adversarial(envs['adversarial'],
                                        agents['red_agents'], agents['blue_agents'])

        if 'multiverse' in envs and 'multiverse_agent' in agents:
            self.benchmark_multiverse(envs['multiverse'], agents['multiverse_agent'])

        if 'agent' in agents:
            self.benchmark_throughput(agents['agent'])
            self.benchmark_latency(agents['agent'])

        # Save results
        self.save_results()
        return self.results

    def save_results(self):
        """Save benchmark results to file."""
        import json
        data = [result.__dict__ for result in self.results]
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Benchmark results saved to {self.output_dir / 'results.json'}")

    def print_report(self):
        """Print formatted benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        for r in self.results:
            print(f"  {r}")
        print("=" * 60)
