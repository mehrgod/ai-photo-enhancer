"""
Agent orchestration module.
Builds prompts, calls Ollama locally, and parses JSON enhancement pipelines.
"""

import json
import subprocess
from typing import Any, Dict

# Fallback pipelines for each goal (used if Ollama fails or returns invalid JSON)
GOAL_MAP = {
    "Professional Headshot": [
        "brightness_adjustment",
        "contrast_adjustment",
        "face_sharpen",
    ],
    "Improve Lighting": ["brightness_adjustment", "contrast_adjustment"],
    "Sharpen Face": ["face_sharpen", "contrast_adjustment"],
    "Social Media Photo": ["auto_adjust", "face_sharpen"],
    "Auto Enhance": ["brightness_adjustment", "contrast_adjustment", "noise_reduction"],
}

# List of operations the agent can choose from
AVAILABLE_OPERATIONS = [
    "brightness_adjustment",
    "contrast_adjustment",
    "face_sharpen",
    "noise_reduction",
    "auto_adjust",
]


def build_agent_prompt(goal: str, analysis: Dict) -> str:
    """
    Build a prompt for the local LLM agent.
    The agent receives the user's goal and image analysis metrics,
    then decides which enhancement operations to apply.
    """
    instructions = [
        "You are a lightweight local enhancement agent.",
        "The user selected a portrait enhancement goal and provided image analysis metrics.",
        "Choose a short ordered pipeline from the available operations.",
        "",
        "IMPORTANT: Return ONLY a JSON object, with NO explanations, NO text before or after.",
        "",
        f"Goal: {goal}",
        "",
        "Image analysis:",
        f"- brightness: {analysis.get('brightness')}",
        f"- blur: {analysis.get('blur')}",
        f"- noise: {analysis.get('noise')}",
        "",
        "Available operations:",
    ]
    for operation in AVAILABLE_OPERATIONS:
        instructions.append(f"- {operation}")

    instructions.extend(
        [
            "",
            "Example response:",
            '{',
            '  "pipeline": [',
            '    "brightness_adjustment",',
            '    "face_sharpen",',
            '    "contrast_adjustment"',
            '  ]',
            "}",
        ]
    )
    return "\n".join(instructions)


def call_ollama(prompt: str, model: str = "phi3") -> str:
    """
    Call Ollama locally to run the agent.
    Requires: `ollama run <model>` to be available on the system.
    """
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=180,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Ollama failed with return code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Ollama call timed out after 180 seconds") from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Ollama not found. Make sure Ollama is installed and in PATH."
        ) from exc


def parse_agent_response(response_text: str) -> Dict[str, Any]:
    """
    Parse the agent's JSON response.
    If the response contains extra text, try to extract the JSON block.
    """
    cleaned = response_text.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError("Could not parse JSON from agent response") from exc
        raise


def run_agent_plan(goal: str, analysis: Dict, model: str = "phi3") -> Dict:
    """
    Run the agent to generate an enhancement pipeline.
    Returns a dict with a 'pipeline' list of operation names.

    If Ollama fails or returns invalid JSON, falls back to GOAL_MAP.
    """
    prompt = build_agent_prompt(goal, analysis)

    try:
        response_text = call_ollama(prompt, model=model)
        response = parse_agent_response(response_text)
        pipeline = response.get("pipeline", [])

        # Validate that pipeline is a list
        if not isinstance(pipeline, list):
            raise ValueError("Pipeline is not a list")

    except Exception as exc:
        # Fall back to predefined pipeline for the goal
        print(f"Agent call failed ({exc}), using fallback pipeline.")
        pipeline = GOAL_MAP.get(goal, [])

    return {"pipeline": pipeline}
