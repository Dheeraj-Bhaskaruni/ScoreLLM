"""Shared fixtures for EvalFlow tests."""
from __future__ import annotations

from typing import List

import pytest

from evalflow.core import Agent, Scenario, StepResult, ToolCall
from evalflow.environments import MockEnvironment


@pytest.fixture
def simple_scenario() -> Scenario:
    return Scenario(
        id="test-001",
        name="Find Apple stock price",
        description="Agent should find Apple stock price",
        initial_context="What is the current price of Apple stock?",
        expected_tool_sequence=["search"],
        metadata={"difficulty": "easy", "domain": "finance", "category": "standard"},
    )


@pytest.fixture
def multi_step_scenario() -> Scenario:
    return Scenario(
        id="test-002",
        name="Calculate GDP per capita",
        description="Find GDP and population, then calculate per-capita GDP",
        initial_context="What is the per-capita GDP of France?",
        expected_tool_sequence=["search", "search", "calculate"],
        metadata={"difficulty": "hard", "domain": "finance", "category": "multi_hop"},
    )


@pytest.fixture
def edge_case_scenario() -> Scenario:
    return Scenario(
        id="test-003",
        name="Empty input",
        description="Agent receives an empty user message",
        initial_context="",
        expected_tool_sequence=["done"],
        metadata={"difficulty": "hard", "domain": "edge_case", "category": "edge_case"},
    )


class DeterministicAgent(Agent):
    """Agent that follows a fixed tool sequence for testing."""

    def __init__(self, tool_sequence: List[str]):
        self._sequence = tool_sequence
        self._step = 0

    def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        if self._step >= len(self._sequence):
            return ToolCall(tool_name="done", arguments={"answer": "final answer"})
        tool = self._sequence[self._step]
        self._step += 1
        if tool == "done":
            return ToolCall(tool_name="done", arguments={"answer": "deterministic answer"})
        elif tool == "search":
            return ToolCall(tool_name="search", arguments={"query": "test query"})
        elif tool == "calculate":
            return ToolCall(tool_name="calculate", arguments={"expression": "2 + 2"})
        elif tool == "writer":
            return ToolCall(tool_name="writer", arguments={"topic": "Test Report"})
        return ToolCall(tool_name=tool, arguments={})


class CrashingAgent(Agent):
    """Agent that crashes on a specific step for testing error handling."""

    def __init__(self, crash_on_step: int = 0):
        self._crash_step = crash_on_step

    def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        if len(history) == self._crash_step:
            raise RuntimeError("Simulated agent crash")
        return ToolCall(tool_name="done", arguments={"answer": "ok"})


@pytest.fixture
def mock_env() -> MockEnvironment:
    return MockEnvironment(seed=42)


@pytest.fixture
def deterministic_agent() -> DeterministicAgent:
    return DeterministicAgent(["search", "calculate", "done"])


@pytest.fixture
def crashing_agent() -> CrashingAgent:
    return CrashingAgent(crash_on_step=0)
