"""
tatha_web.py
============
Web UI dashboard for Tatha agent monitoring.

Provides:
  - Live branch evolution chart
  - Probability field visualization
  - Diagnostics panel
  - Real-time telemetry
"""
from __future__ import annotations
import json
import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec
    WEB_VIZ = True
except ImportError:
    WEB_VIZ = False


class TathaDashboard:
    """Web dashboard for agent monitoring."""
    def __init__(self, agent=None, port: int = 8080):
        self.agent = agent
        self.port = port
        self._plots_generated = False
        self._html_path = None

    def generate_dashboard(self, world, save_path: str = "dashboard.html"):
        """Generate self-contained HTML dashboard."""
        if not WEB_VIZ:
            print("Matplotlib not available")
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from io import BytesIO
        import base64

        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig)

        # Branch evolution
        ax1 = fig.add_subplot(gs[0, 0])
        if hasattr(world, 'telemetry') and world.telemetry['branch_history']:
            ax1.plot(world.telemetry['branch_history'], 'b-', linewidth=2)
            ax1.set_title('Branches Over Time', fontweight='bold')
            ax1.set_ylabel('Number of Branches')
            ax1.grid(True, alpha=0.3)

        # Entropy
        ax2 = fig.add_subplot(gs[0, 1])
        if world.telemetry['entropy_history']:
            ax2.plot(world.telemetry['entropy_history'], 'r-', linewidth=2)
            ax2.set_title('Entanglement Entropy', fontweight='bold')
            ax2.set_xlabel('Step')
            ax2.grid(True, alpha=0.3)

        # Reward
        ax3 = fig.add_subplot(gs[0, 2])
        if world.telemetry['reward_history']:
            rewards = world.telemetry['reward_history']
            cumulative = np.cumsum(rewards)
            ax3.plot(cumulative, 'g-', linewidth=2)
            ax3.set_title('Cumulative Reward', fontweight='bold')
            ax3.set_xlabel('Step')
            ax3.grid(True, alpha=0.3)

        # Probability field
        ax4 = fig.add_subplot(gs[1, :])
        if world:
            probs = np.zeros((world.size, world.size))
            for u in range(world.mv.num_branches):
                p_u = np.abs(world.mv.c[u]) ** 2 * np.abs(world.mv.psi[u]) ** 2
                probs += p_u.reshape(world.size, world.size)
            im = ax4.imshow(probs, cmap='hot', interpolation='nearest')
            plt.colorbar(im, ax=ax4, fraction=0.046)
            ax4.set_title('Aggregate Probability Density', fontweight='bold')

        # Diagnostics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        if self.agent:
            diag = self.agent.get_diagnostics()
            table_data = [[k, f"{v:.4f}" if isinstance(v, float) else str(v)]
                          for k, v in diag.items()]
            table = ax5.table(
                cellText=table_data,
                colLabels=['Metric', 'Value'],
                cellLoc='left',
                loc='center',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            ax5.set_title('Agent Diagnostics', fontweight='bold')

        plt.tight_layout()
        img_path = Path(save_path).with_suffix('.png')
        plt.savefig(img_path, dpi=150, bbox_inches='tight')

        # Generate HTML
        with open(Path(save_path)) as f:
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Tatha Agent Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d4ff; }}
        .metric {{ background: #16213e; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .metric-value {{ color: #00d4ff; font-size: 24px; font-weight: bold; }}
        img {{ max-width: 100%; border-radius: 8px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Tatha Agent Dashboard</h1>
        <div class="grid">
            <div class="metric"><h3>Branches</h3><div class="metric-value">{diag.get('branches', 0)}</div></div>
            <div class="metric"><h3>Entropy</h3><div class="metric-value">{diag.get('entropy', 0):.4f}</div></div>
            <div class="metric"><h3>Episode</h3><div class="metric-value">{diag.get('episode', 0)}</div></div>
            <div class="metric"><h3>Steps</h3><div class="metric-value">{diag.get('total_steps', 0)}</div></div>
            <div class="metric"><h3>Replay Size</h3><div class="metric-value">{diag.get('replay_size', 0)}</div></div>
            <div class="metric"><h3>Meta Director</h3><div class="metric-value">{'Active' if diag.get('meta_director_active') else 'Inactive'}</div></div>
        </div>
        <img src="{img_path.name}" alt="Dashboard Charts">
    </div>
</body>
</html>"""

        html_path = Path(save_path)
        html_path.write_text(html)
        print(f"Dashboard saved to {html_path}")
        return html_path

    def start_server(self, world, agent):
        """Start HTTP server for dashboard (placeholder)."""
        print(f"[WEB] Dashboard available at http://localhost:{self.port}")
        print(f"[WEB] Generate dashboard: dashboard.generate_dashboard(world, 'dashboard.html')")
