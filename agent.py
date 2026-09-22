import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import os


load_dotenv()


client = anthropic.Anthropic()
MODEL = "claude-fable-5-1"
MAX_AGENT_TURNS = 8


SYSTEM_PROMPT = """You are a repository-aware developer agent.
Use the provided tools to inspect the project before proposing changes.
Return a structured plan with: files_to_change, rationale, and test_strategy."""


# --- Tool definitions ---
def list_project_files(project_root: str = ".") -> str:
    """List readable text files in the project."""
    files = []
    for root, _, filenames in os.walk(project_root):
        for f in filenames:
            if f.endswith((".py", ".txt", ".md", ".cfg", ".toml", ".json")):
                rel = os.path.relpath(os.path.join(root, f), project_root)
                files.append(rel)
    return "\n".join(sorted(files))


def read_project_file(path: str, project_root: str = ".") -> str:
    """Read a project file, bounded to the project root."""
    full = os.path.normpath(os.path.join(project_root, path))
    if not full.startswith(os.path.normpath(project_root)):
        return "ERROR: path escapes project root"
    with open(full, "r", encoding="utf-8") as f:
        return f.read()[:8000]


tools = [
    {
        "name": "list_project_files",
        "description": "List readable text files in the project.",
        "input_schema": {
            "type": "object",
            "properties": {
                "project_root": {"type": "string", "default": "."}
            },
        },
    },
    {
        "name": "read_project_file",
        "description": "Read a project file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "project_root": {"type": "string", "default": "."},
            },
            "required": ["path"],
        },
    },
]


# --- Structured output schema ---
class Plan(BaseModel):
    files_to_change: list[str]
    rationale: str
    test_strategy: str


# --- Agent loop ---
def run_agent(feature_request: str):
    messages = [{"role": "user", "content": feature_request}]

    for turn in range(MAX_AGENT_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
            output_config={"effort": "high"},
        )

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_calls:
            print("Final response:")
            for b in text_blocks:
                print(b.text)
            return

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for call in tool_calls:
            if call.name == "list_project_files":
                result = list_project_files(**call.input)
            elif call.name == "read_project_file":
                result = read_project_file(**call.input)
            else:
                result = f"Unknown tool: {call.name}"

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    print("Max turns reached.")


# --- Structured final plan ---
def get_structured_plan(feature_request: str) -> Plan:
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=[{"role": "user", "content": feature_request}],
        output_config={"effort": "high"},
    )

    final = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system="Extract the plan as structured JSON.",
        messages=[
            {"role": "user", "content": f"Feature request: {feature_request}"},
            {"role": "assistant", "content": response.content},
        ],
        output_config={
            "effort": "low",
            "format": {
                "type": "json_schema",
                "schema": Plan.model_json_schema(),
            },
        },
    )

    return Plan.model_validate_json(final.content[0].text)


if __name__ == "__main__":
    run_agent("Add rate limiting to the bookmark API.")
