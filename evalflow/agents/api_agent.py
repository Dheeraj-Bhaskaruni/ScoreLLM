"""
evalflow.agents.api_agent — HF Inference API agent (OpenAI-compatible).

Supports both synchronous and asynchronous operation for use with
SimulationEngine and AsyncSimulationEngine respectively.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from ..core import Agent, AsyncAgent, StepResult, ToolCall

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment, misc]
    AsyncOpenAI = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Shared system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise function-calling agent.
Only use the provided tools. Do not hallucinate arguments.

TOOLS:
1. search(query: str) -> str — Find information.
   Example: Action: search {"query": "AAPL stock price"}

2. calculate(expression: str) -> str — Do math.
   Example: Action: calculate {"expression": "100 * 5"}

3. writer(topic: str) -> str — Write a final report.
   Example: Action: writer {"topic": "Q4 Forecast"}

4. done(answer: str) -> None — Return the final answer.
   Example: Action: done {"answer": "The price is $150"}

RULES:
- Include a brief 'Thought:' before Action to explain reasoning.
- Response must contain 'Action: tool_name {"arg": "value"}'
- If search returns data sufficient to answer, call 'done' immediately.
"""


def _parse_action(raw_text: str) -> ToolCall:
    """Parse an LLM response into a structured ToolCall."""
    clean = re.sub(r"\[(ASS|USER|ENV|INST)\]", "", raw_text).strip()

    action_match = re.search(r"Action:\s*(\w+)", clean, re.IGNORECASE)
    if not action_match:
        return ToolCall(tool_name="done", arguments={"answer": clean[:300]}, raw_output=clean)

    raw_name = action_match.group(1).lower()
    tool_name = _normalize_tool_name(raw_name)

    # Extract JSON block
    args = _extract_json_args(clean, action_match.end())

    # Enforce schema
    args = _enforce_schema(tool_name, args, clean)

    return ToolCall(tool_name=tool_name, arguments=args, raw_output=clean)


def _normalize_tool_name(name: str) -> str:
    if "search" in name:
        return "search"
    if "calc" in name:
        return "calculate"
    if "write" in name:
        return "writer"
    if "done" in name:
        return "done"
    return "search"


def _extract_json_args(text: str, start_idx: int) -> Dict[str, Any]:
    open_idx = text.find("{", start_idx)
    if open_idx == -1:
        return {}

    brace_count = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
        if brace_count == 0:
            json_text = text[open_idx : i + 1]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                try:
                    import ast

                    return ast.literal_eval(json_text)
                except (ValueError, SyntaxError):
                    return {}
    return {}


def _enforce_schema(tool_name: str, args: Dict[str, Any], raw: str) -> Dict[str, Any]:
    """Ensure the args dict has the correct key for the given tool."""
    expected_key = {
        "search": "query",
        "calculate": "expression",
        "writer": "topic",
        "done": "answer",
    }.get(tool_name, "value")

    if expected_key in args:
        return args

    # Grab first string value as fallback
    for v in args.values():
        if isinstance(v, str):
            return {expected_key: v}

    return {expected_key: raw[:200]}


def _build_messages(history: List[StepResult], current_observation: str) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for step in history[-5:]:  # Bound context to last 5 steps
        messages.append({"role": "user", "content": f"Observation: {step.input_state[:300]}"})
        prev_args = json.dumps(step.action.arguments)
        messages.append({"role": "assistant", "content": f"Action: {step.action.tool_name} {prev_args}"})
        messages.append({"role": "user", "content": f"Result: {step.output_observation[:300]}"})
    messages.append({"role": "user", "content": f"Observation: {current_observation[:500]}\nWhat is your next Action?"})
    return messages


# ---------------------------------------------------------------------------
# Synchronous agent
# ---------------------------------------------------------------------------


class HFApiAgent(Agent):
    """Synchronous agent using HF Inference API via OpenAI-compatible client."""

    def __init__(
        self,
        model_id: str,
        api_token: str,
        base_url: str = "https://router.huggingface.co/v1/",
        temperature: float = 0.1,
    ):
        if OpenAI is None:
            raise ImportError("Install 'openai' package: pip install openai")
        self.client = OpenAI(base_url=base_url, api_key=api_token)
        self.model_id = model_id
        self.temperature = temperature

    @property
    def agent_id(self) -> str:
        return f"HFApiAgent({self.model_id})"

    def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        messages = _build_messages(history, current_observation)
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=150,
                temperature=self.temperature,
                stop=["\nObservation:", "Observation:"],
            )
            raw = completion.choices[0].message.content.strip()
            return _parse_action(raw)
        except Exception as e:
            logger.error("API call failed: %s", e)
            return ToolCall(tool_name="error", arguments={"msg": str(e)}, raw_output=str(e))


# ---------------------------------------------------------------------------
# Async agent
# ---------------------------------------------------------------------------


class AsyncHFApiAgent(AsyncAgent):
    """Async agent for use with AsyncSimulationEngine."""

    def __init__(
        self,
        model_id: str,
        api_token: str,
        base_url: str = "https://router.huggingface.co/v1/",
        temperature: float = 0.1,
    ):
        if AsyncOpenAI is None:
            raise ImportError("Install 'openai' package: pip install openai")
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_token)
        self.model_id = model_id
        self.temperature = temperature

    @property
    def agent_id(self) -> str:
        return f"AsyncHFApiAgent({self.model_id})"

    async def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        messages = _build_messages(history, current_observation)
        try:
            completion = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=150,
                temperature=self.temperature,
                stop=["\nObservation:", "Observation:"],
            )
            raw = completion.choices[0].message.content.strip()
            return _parse_action(raw)
        except Exception as e:
            logger.error("Async API call failed: %s", e)
            return ToolCall(tool_name="error", arguments={"msg": str(e)}, raw_output=str(e))
