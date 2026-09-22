"""
tatha_rest.py
=============
FastAPI REST server for Tatha predictive-coding agent.

Provides:
  - REST endpoints alongside WebSocket server
  - Health checks
  - Agent state queries
  - Batch inference
  - Training control

Run:
    python tatha_rest.py --port 8766
"""
from __future__ import annotations
import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    REST_AVAILABLE = True
except ImportError:
    REST_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tatha_utils import ObservationValidator, ErrorRecovery, setup_logging

logger = setup_logging("tatha-rest")

if TYPE_CHECKING:
    from tatha_realtime import TathaRealTime


class TathaRESTServer:
    """REST API server for Tatha agent."""
    def __init__(self, agent, port: int = 8766):
        self.agent = agent
        self.port = port
        self.logger = setup_logging("tatha-rest")
        self.validator = ObservationValidator(
            expected_size=agent.config.GRID_SIZE ** 2 * 3,
        )
        self.recovery = ErrorRecovery()
        if REST_AVAILABLE:
            self._build_app()
        else:
            self.app = None

    def _build_app(self):
        self.app = FastAPI(title="Tatha Agent API", version="1.0.0")

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/health")
        def health():
            return {"status": "healthy", "agent": "tatha"}

        @self.app.post("/observe")
        def observe(request: dict):
            try:
                data = request.get("data", [])
                reward = float(request.get("reward", 0.0))
                done = request.get("done", False)

                # Validate
                valid, error = self.validator.validate(data)
                if not valid:
                    raise HTTPException(status_code=400, detail=error)

                # Execute with recovery
                result = self.recovery.execute_with_recovery(
                    self.agent.observe, data, reward, done
                )
                return {"status": "ok", **result}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/learn")
        def learn(request: dict = None):
            try:
                batch_size = request.get("batch_size", self.agent.config.BATCH_SIZE) if request else self.agent.config.BATCH_SIZE
                self.agent.learn_from_replay(batch_size=batch_size)
                return {"status": "ok", "replay_size": len(self.agent.replay)}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/reset")
        def reset():
            obs = self.agent.reset()
            return {"status": "ok", "observation": obs.tolist()}

        @self.app.post("/checkpoint")
        def checkpoint(request: dict):
            name = request.get("name", None)
            path = self.agent.checkpoint_save(name)
            return {"status": "ok", "path": path}

        @self.app.post("/load")
        def load(request: dict):
            name = request.get("name", "latest")
            ok = self.agent.checkpoint_load(name)
            return {"status": "ok", "loaded": ok}

        @self.app.get("/diagnostics")
        def diagnostics():
            return {"status": "ok", **self.agent.get_diagnostics()}

        @self.app.get("/history")
        def history(n: int = 20):
            return {"status": "ok", "actions": self.agent.executor.get_history(n)}

        @self.app.post("/batch")
        def batch_observe(request: dict):
            """Process multiple observations in batch."""
            observations = request.get("observations", [])
            results = []
            for obs_data in observations:
                data = obs_data.get("data", [])
                reward = float(obs_data.get("reward", 0.0))
                done = obs_data.get("done", False)
                result = self.agent.observe(data, reward, done)
                results.append(result)
            return {"status": "ok", "batch_size": len(results), "results": results}

        @self.app.post("/meta-director/observe")
        def meta_observe(request: dict):
            """Get MetaDirector parameter modulation."""
            reward = float(request.get("reward", 0.0))
            result = self.agent.meta_director_observe(reward)
            return {"status": "ok", **result}

        @self.app.post("/meta-director/learn")
        def meta_learn(request: dict):
            """Let MetaDirector learn from outcome."""
            outcome = float(request.get("outcome", 0.0))
            self.agent.meta_director_learn(outcome)
            return {"status": "ok"}

        @self.app.get("/config")
        def config():
            return {"status": "ok", "config": self.agent.config.to_dict()}

    def run(self):
        """Start the REST server."""
        if not REST_AVAILABLE:
            print("ERROR: fastapi not installed. Run: pip install fastapi uvicorn")
            sys.exit(1)

        print(f"[REST] Starting on http://0.0.0.0:{self.port}")
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")
