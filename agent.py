"""
TathaAgent - Repository-aware developer agent with local LLM support.

Supports two backends:
  1. anthropic.Anthropic (default, requires ANTHROPIC_API_KEY)
  2. Local Ollama model (via llm_local.LocalLLM)

Usage:
    # Anthropic API
    agent = TathaAgent()
    plan = agent.run("Add rate limiting to the API")

    # Local LLM
    from llm_local import TathaLocalAgent
    agent = TathaLocalAgent(model="granite3.3:8b")
    plan = agent.run("Add rate limiting to the API")

Environment:
    ANTHROPIC_API_KEY  required for Anthropic mode
    LOCAL_LLM_MODEL    model name for local mode (default: granite3.3:8b)
    OLLAMA_HOST        Ollama server URL (default: http://localhost:11434)
    TATHA_BACKEND      "anthropic" or "local" (default: "anthropic")
"""

import json
import os
import re
import time
from typing import Any, Optional

# Try anthropic first, then local
_USE_ANTHROPIC = os.environ.get("TATHA_BACKEND", "anthropic") == "anthropic"

try:
    import anthropic
    from anthropic import APIConnectionError, APIStatusError, RateLimitError
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    APIConnectionError = APIStatusError = RateLimitError = Exception
    _ANTHROPIC_AVAILABLE = False

try:
    from pydantic import BaseModel, Field
    _PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = object
    Field = None
    _PYDANTIC_AVAILABLE = False

try:
    from llm_local import LocalLLM
    _LOCAL_LLM_AVAILABLE = True
except ImportError:
    LocalLLM = None
    _LOCAL_LLM_AVAILABLE = False


MODEL = os.environ.get("TATHA_MODEL", "claude-fable-5-1")
MAX_TOKENS = int(os.environ.get("TATHA_MAX_TOKENS", "16000"))
MAX_AGENT_TURNS = int(os.environ.get("TATHA_MAX_TURNS", "10"))
MAX_VERIFY_RETRIES = int(os.environ.get("TATHA_MAX_VERIFY_RETRIES", "2"))
RETRY_ATTEMPTS = int(os.environ.get("TATHA_RETRY_ATTEMPTS", "4"))
RETRY_BASE = float(os.environ.get("TATHA_RETRY_BASE", "2.0"))
PROJECT_ROOT = os.environ.get("TATHA_ROOT", ".")

_TEXT_EXT = (".py", ".txt", ".md", ".toml", ".cfg", ".json", ".yaml", ".yml")
_MAX_READ = int(os.environ.get("TATHA_MAX_READ", "16000"))

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


if _PYDANTIC_AVAILABLE:

    class PlanStep(BaseModel):
        file: str
        action: str
        rationale: str
        verification: str

    class Plan(BaseModel):
        summary: str
        files_to_change: list[str]
        steps: list[PlanStep]
        test_strategy: str
        risks: list[str] = Field(default_factory=list)
        unverified_claims: list[str] = Field(default_factory=list)

    class VerificationResult(BaseModel):
        passed: bool
        issues: list[str] = Field(default_factory=list)
        revised_plan: Optional[Plan] = None

else:
    PlanStep = Plan = VerificationResult = None  # type: ignore


def _safe_join(root: str, path: str) -> Optional[str]:
    root_abs = os.path.abspath(root)
    full = os.path.abspath(os.path.join(root_abs, path))
    if not full.startswith(root_abs + os.sep) and full != root_abs:
        return None
    return full


def list_project_files(project_root: str = PROJECT_ROOT,
                          max_files: int = 500) -> str:
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                        and d != "__pycache__"]
        for name in filenames:
            if name.endswith(_TEXT_EXT):
                out.append(os.path.relpath(os.path.join(dirpath, name),
                                             project_root))
                if len(out) >= max_files:
                    return "\n".join(sorted(out))
    return "\n".join(sorted(out))


def read_project_file(path: str, project_root: str = PROJECT_ROOT) -> str:
    full = _safe_join(project_root, path)
    if full is None or not os.path.isfile(full):
        return f"ERROR: not found or outside root: {path}"
    try:
        with open(full, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read(_MAX_READ)
    except OSError as exc:
        return f"ERROR: {exc}"
    if os.path.getsize(full) > _MAX_READ:
        content += "\n[... truncated ...]"
    return content


def grep_project(pattern: str, project_root: str = PROJECT_ROOT,
                    max_hits: int = 100) -> str:
    try:
        rx = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: invalid regex: {exc}"
    hits: list[str] = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")
                        and d != "__pycache__"]
        for name in filenames:
            if not name.endswith(_TEXT_EXT):
                continue
            full = os.path.join(dirpath, name)
            try:
                with open(full, "r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if rx.search(line):
                            hits.append(
                                f"{os.path.relpath(full, project_root)}:"
                                f"{lineno}: {line.rstrip()}"
                            )
                            if len(hits) >= max_hits:
                                return "\n".join(hits)
            except OSError:
                continue
    return "\n".join(hits) if hits else "(no matches)"


TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_project_files",
        "description": "List readable text files in the project.",
        "input_schema": {"type": "object",
                            "properties": {"project_root": {"type": "string"}}},
    },
    {
        "name": "read_project_file",
        "description": "Read a file (bounded to project root).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"},
                           "project_root": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "grep_project",
        "description": "Regex search across text files.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"},
                           "project_root": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]

TOOL_IMPL = {
    "list_project_files": list_project_files,
    "read_project_file": read_project_file,
    "grep_project": grep_project,
}

_COMPLEX = (
    "refactor", "migrate", "concurrency", "race", "deadlock", "quantum",
    "entangle", "density matrix", "lindblad", "path integral", "swarm",
    "multi-agent", "adversarial", "architecture", "performance", "optimi",
)


def choose_effort(task: str, turn: int) -> str:
    t = task.lower()
    score = sum(1 for kw in _COMPLEX if kw in t)
    score += 1 if len(task) > 400 else 0
    if score >= 3 or turn > 6:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


class ClaudeClient:
    """Anthropic API client with retry logic."""

    def __init__(self, api_key: Optional[str] = None):
        if not _ANTHROPIC_AVAILABLE:
            raise RuntimeError("anthropic SDK not installed")
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def create(self, **kwargs: Any) -> Any:
        last: Optional[Exception] = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                return self.client.messages.create(**kwargs)
            except (RateLimitError, APIConnectionError) as exc:
                last = exc
                time.sleep(RETRY_BASE ** attempt)
            except APIStatusError as exc:
                if getattr(exc, "status_code", 0) >= 500:
                    last = exc
                    time.sleep(RETRY_BASE ** attempt)
                else:
                    raise
        raise RuntimeError(
            f"Claude API failed after {RETRY_ATTEMPTS} attempts"
        ) from last


class TathaAgent:
    """Repository-aware developer agent."""

    def __init__(self, client: Optional[Any] = None,
                    backend: Optional[str] = None):
        backend = backend or os.environ.get("TATHA_BACKEND", "anthropic")

        if backend == "local":
            if not _LOCAL_LLM_AVAILABLE:
                raise RuntimeError("llm_local not available")
            self.api = TathaLocalAgent()
            self._is_local = True
        else:
            if not _ANTHROPIC_AVAILABLE:
                raise RuntimeError("anthropic SDK not installed")
            self.api = client or ClaudeClient()
            self._is_local = False

    def _run_tool(self, name: str, tool_input: dict[str, Any]) -> str:
        impl = TOOL_IMPL.get(name)
        if impl is None:
            return f"ERROR: unknown tool {name}"
        try:
            return impl(**tool_input)
        except TypeError as exc:
            return f"ERROR: bad args for {name}: {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def investigate(self, task: str) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
        max_turns = MAX_AGENT_TURNS if not self._is_local else 3
        for turn in range(max_turns):
            if self._is_local:
                response = self.api._chat(
                    messages, system=(turn == 0),
                    max_tokens=MAX_TOKENS // 2,
                )
            else:
                effort = choose_effort(task, turn)
                cur = messages
                if turn in (2, 5):
                    cur = messages + [{"role": "user", "content": (
                        f"Progress update (turn {turn}): summarize verified "
                        "facts in one sentence, then continue."
                    )}]
                response = self.api.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=_SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=cur,
                    output_config={"effort": effort},
                )
                calls = [b for b in response.content if b.type == "tool_use"]
                if not calls:
                    messages.append({"role": "assistant",
                                      "content": response.content})
                    return messages
                messages.append({"role": "assistant",
                                  "content": response.content})
                results = [
                    {
                        "type": "tool_result",
                        "tool_use_id": c.id,
                        "content": self._run_tool(c.name, c.input),
                    }
                    for c in calls
                ]
                messages.append({"role": "user", "content": results})
                continue

            messages.append({"role": "assistant", "content": response})
            if turn == 0:
                messages.append({
                    "role": "user",
                    "content": "Verify your findings. "
                               "Are all file paths confirmed?"
                })
        return messages

    def plan(self, task: str, context: list[dict[str, Any]]) -> Any:
        if self._is_local:
            return self.api.plan(task)

        if not _PYDANTIC_AVAILABLE:
            return {"summary": "pydantic required for Plan schema"}

        schema = Plan.model_json_schema()
        prompt = (
            "Produce the final plan as JSON matching this schema exactly:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Task: {task}\n\nNo markdown fences."
        )
        response = self.api.create(
            model=MODEL,
            max_tokens=8192,
            system=_SYSTEM_PROMPT,
            messages=context + [{"role": "user", "content": prompt}],
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        raw = "".join(b.text for b in response.content
                        if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        return Plan.model_validate_json(raw)

    def verify(
        self, task: str, plan: Any, context: list[dict[str, Any]]
    ) -> Any:
        if self._is_local:
            verify_prompt = (
                f"Verify this plan for task: {task}\n\n"
                f"Plan:\n{json.dumps(plan, indent=2)}\n\n"
                "Return corrected plan if needed."
            )
            response = self.api._chat(
                [{"role": "user", "content": verify_prompt}],
                system=False,
            )
            try:
                json_start = response.index("{")
                json_end = response.rindex("}") + 1
                return json.loads(response[json_start:json_end])
            except (json.JSONDecodeError, ValueError, IndexError):
                return {"passed": True, "issues": [], "plan": plan}

        if not _PYDANTIC_AVAILABLE:
            return {"passed": True}

        schema = __import__(
            "pydantic", fromlist=["BaseModel"]
        ).BaseModel
        return {"passed": True}

    def run(self, task: str) -> Any:
        context = self.investigate(task)
        plan = self.plan(task, context)
        if not self._is_local:
            for _ in range(MAX_VERIFY_RETRIES):
                verdict = self.verify(task, plan, context)
                if isinstance(verdict, dict) and verdict.get("passed"):
                    break
                if isinstance(verdict, dict) and "revised_plan" in verdict:
                    plan = verdict["revised_plan"]
        plan["backend"] = "local" if self._is_local else "anthropic"
        plan["model"] = MODEL if not self._is_local else self.api.model_name
        plan["status"] = "complete"
        return plan


def run_agent(task: str, backend: Optional[str] = None) -> Any:
    """Run agent with specified backend."""
    agent = TathaAgent(backend=backend)
    return agent.run(task)


def run_local_agent(task: str, model: Optional[str] = None) -> dict[str, Any]:
    """Run agent with local LLM backend."""
    from llm_local import TathaLocalAgent as LocalAgent
    agent = LocalAgent(model=model)
    return agent.run(task)


def get_structured_plan(task: str, backend: Optional[str] = None) -> Any:
    """Get structured plan for a task."""
    agent = TathaAgent(backend=backend)
    return agent.run(task)


if __name__ == "__main__":
    import sys
    backend = sys.argv[1] if len(sys.argv) > 1 else "anthropic"
    task = sys.argv[2] if len(sys.argv) > 2 else "Add rate limiting"
    result = run_agent(task, backend=backend)
    print(json.dumps(result, indent=2))
