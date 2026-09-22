"""
Local LLM module - uses Ollama for local inference.

Provides drop-in replacement for anthropic.Anthropic client.
Works with any local model served via Ollama (http://localhost:11434).

Usage:
    from llm_local import LocalLLM, TathaLocalAgent
    agent = TathaLocalAgent(model="granite3.3:8b")
    plan = agent.run("Add rate limiting to the API")

Dependencies:
    required: requests
    optional: ollama (preferred, falls back to HTTP)

Environment:
    OLLAMA_HOST  Ollama server URL (default: http://localhost:11434)
    LOCAL_LLM_MODEL  Model name (default: granite3.3:8b)
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("LOCAL_LLM_MODEL", "granite3.3:8b")
REQUEST_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "120"))
MAX_RETRIES = int(os.environ.get("LLM_RETRIES", "3"))
RETRY_DELAY = float(os.environ.get("LLM_RETRY_DELAY", "2.0"))


def _default_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


class LocalLLM:
    """Local LLM client using Ollama HTTP API."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: int = REQUEST_TIMEOUT,
    ):
        self.model = model or DEFAULT_MODEL
        self.host = host or OLLAMA_HOST
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(_default_headers())

    def _ensure_model(self) -> bool:
        """Check if the model is available locally."""
        try:
            resp = self._session.get(
                f"{self.host}/api/tags", timeout=5
            )
            models = resp.json().get("models", [])
            return any(m["name"] == self.model for m in models)
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> str:
        """Send a chat completion request. Returns text response."""
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": stream,
        }
        if system:
            payload["system"] = system

        for attempt in range(MAX_RETRIES):
            try:
                resp = self._session.post(
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )
                resp.raise_for_status()
                data = resp.json()
                if "response" in data:
                    return data["response"]
                if "message" in data:
                    return data["message"].get("content", "")
                return json.dumps(data)
            except requests.exceptions.Timeout:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"LLM request timed out after {self.timeout}s"
                )
            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.host}. "
                    f"Run: ollama serve"
                )
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                raise RuntimeError(f"LLM request failed: {exc}") from exc

        raise RuntimeError("Max retries exceeded")

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Streaming chat completion."""
        payload = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": True,
        }
        if system:
            payload["system"] = system

        full_text = ""
        try:
            resp = self._session.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        full_text += data["response"]
            return full_text
        except Exception as exc:
            raise RuntimeError(f"LLM stream failed: {exc}") from exc

    def embed(self, text: str) -> list[float]:
        """Get embeddings for text."""
        payload = {"model": self.model, "input": text}
        try:
            resp = self._session.post(
                f"{self.host}/api/embed",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as exc:
            raise RuntimeError(f"Embedding failed: {exc}") from exc

    def list_models(self) -> list[str]:
        """List available local models."""
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=5)
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def health(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            resp = self._session.get(
                f"{self.host}/api/tags", timeout=3
            )
            return resp.status_code == 200
        except Exception:
            return False


# ============================================================================
# LLM-BASED TATHA AGENT
# ============================================================================

_SYSTEM_PROMPT = """You are a repository-aware developer agent working on Tatha.

Grounding protocol (mandatory):
  - Verify every fact about the codebase with a tool call before stating it.
  - Mark unverifiable claims with [UNVERIFIED].
  - Never invent file paths, symbols, or signatures.

Process:
  1. Inspect the repository with list_project_files, read_project_file, grep_project.
  2. State the minimal set of files that must change and why.
  3. Propose a concrete diff-level plan.
  4. Before returning, run the self-verification checklist.

When asked for JSON, return only the JSON object."""


class TathaLocalAgent:
    """Tatha agent using a local LLM instead of the Anthropic API."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        max_turns: int = 5,
        max_tokens: int = 4096,
    ):
        self.llm = LocalLLM(model=model, host=host)
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.model_name = model or DEFAULT_MODEL
        if not self.llm._ensure_model():
            available = self.llm.list_models()
            raise RuntimeError(
                f"Model '{self.model_name}' not found locally. "
                f"Available: {available}"
            )

    def _chat(self, messages: list[dict[str, str]], system: bool = True) -> str:
        sys_prompt = _SYSTEM_PROMPT if system else None
        return self.llm.chat(
            messages=messages,
            system=sys_prompt,
            max_tokens=self.max_tokens,
        )

    def investigate(self, task: str) -> list[dict[str, Any]]:
        """Run investigation loop with local LLM."""
        messages = [{"role": "user", "content": task}]
        for turn in range(self.max_turns):
            response = self._chat(messages, system=(turn == 0))
            messages.append({"role": "assistant", "content": response})

            # Check if LLM provided a structured plan (JSON)
            if "```json" in response or '{' in response and '}' in response:
                try:
                    json_start = response.index("{")
                    json_end = response.rindex("}") + 1
                    plan = json.loads(response[json_start:json_end])
                    if "files_to_change" in plan:
                        return messages
                except (json.JSONDecodeError, ValueError):
                    pass

            # If no more tool use requested, stop
            if "Final answer" in response or len(messages) > self.max_turns:
                break

        return messages

    def plan(self, task: str) -> dict[str, Any]:
        """Generate a structured plan for a task."""
        messages = [{"role": "user", "content": task}]
        response = self._chat(messages, system=True)

        # Try to extract JSON from response
        try:
            json_start = response.index("{")
            json_end = response.rindex("}") + 1
            plan = json.loads(response[json_start:json_end])
            return plan
        except (json.JSONDecodeError, ValueError, IndexError):
            return {
                "summary": response[:500],
                "files_to_change": [],
                "steps": [],
                "test_strategy": "Run existing tests",
                "risks": [],
            }

    def run(self, task: str) -> dict[str, Any]:
        """Run full agent pipeline: investigate -> plan -> verify."""
        print(f"[LOCAL LLM] Using model: {self.model_name}")
        print(f"[LOCAL LLM] Agent starting investigation...")

        context = self.investigate(task)
        plan = self.plan(task)

        # Verification pass
        verify_prompt = (
            f"Verify this plan for task: {task}\n\n"
            f"Plan:\n{json.dumps(plan, indent=2)}\n\n"
            "Check for missing tests, wrong file paths, "
            "unverified claims. Return corrected plan if needed."
        )
        verify_response = self._chat(
            [{"role": "user", "content": verify_prompt}],
            system=False,
        )

        try:
            json_start = verify_response.index("{")
            json_end = verify_response.rindex("}") + 1
            verified_plan = json.loads(verify_response[json_start:json_end])
            if verified_plan.get("files_to_change"):
                plan = verified_plan
        except (json.JSONDecodeError, ValueError, IndexError):
            pass

        plan["model_used"] = self.model_name
        plan["status"] = "complete"
        return plan


def run_local_agent(task: str, model: Optional[str] = None) -> dict[str, Any]:
    """Convenience function to run a local LLM agent."""
    agent = TathaLocalAgent(model=model)
    return agent.run(task)


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage:")
        print("  python llm_local.py status")
        print("  python llm_local.py chat '<message>'")
        print("  python llm_local.py agent '<task>'")
        print("  python llm_local.py models")
        print("  python llm_local.py embed '<text>'")
        return 0

    mode = argv[1]

    if mode == "status":
        llm = LocalLLM()
        healthy = llm.health()
        models = llm.list_models()
        print(f"Ollama server: {'connected' if healthy else 'not connected'}")
        print(f"Model: {DEFAULT_MODEL}")
        print(f"Available models: {models}")
        print(f"Server: {OLLAMA_HOST}")

    elif mode == "models":
        llm = LocalLLM()
        models = llm.list_models()
        for m in models:
            print(f"  {m}")

    elif mode == "chat":
        message = argv[2] if len(argv) > 2 else "Hello"
        llm = LocalLLM()
        response = llm.chat(
            messages=[{"role": "user", "content": message}],
            system=True,
        )
        print(response)

    elif mode == "agent":
        task = " ".join(argv[2:]) if len(argv) > 2 else "Add rate limiting"
        result = run_local_agent(task)
        print(json.dumps(result, indent=2))

    elif mode == "embed":
        text = argv[2] if len(argv) > 2 else "test"
        llm = LocalLLM()
        embedding = llm.embed(text)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

    else:
        print(f"unknown mode: {mode}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
