"""
evalflow.core — Domain models and abstract interfaces for the EvalFlow framework.

All models use Pydantic for validation, serialization, and type safety.
"""
from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Core value objects
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """A single tool invocation by an agent, capturing name, args, and raw LLM output."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    raw_output: Optional[str] = None


class StepResult(BaseModel):
    """One step in an agent-environment interaction loop."""
    step_id: int
    timestamp: float = Field(default_factory=time.time)
    input_state: str
    action: ToolCall
    output_observation: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Scenario(BaseModel):
    """A test-case definition that drives a single simulation."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    description: str
    initial_context: str
    expected_tool_sequence: Optional[List[str]] = None
    expected_final_answer: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Convenience helpers --------------------------------------------------
    @property
    def difficulty(self) -> str:
        return self.metadata.get("difficulty", "unknown")

    @property
    def domain(self) -> str:
        return self.metadata.get("domain", "unknown")

    @property
    def category(self) -> str:
        return self.metadata.get("category", "standard")


class SimulationTrace(BaseModel):
    """Complete record of a single agent-environment simulation run."""
    scenario_id: str
    agent_id: str
    start_time: float = Field(default_factory=time.time)
    end_time: float = 0.0
    steps: List[StepResult] = Field(default_factory=list)
    final_output: Optional[str] = None
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def tool_sequence(self) -> List[str]:
        return [s.action.tool_name for s in self.steps if s.action.tool_name.lower() != "done"]


class MetricResult(BaseModel):
    """Result of a single metric evaluation."""
    name: str
    score: float
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Aggregated evaluation for one scenario run."""
    scenario: Scenario
    trace: SimulationTrace
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)


class RunConfig(BaseModel):
    """Configuration for an evaluation run — captures the 'what' and 'how'."""
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_id: str = "unknown"
    model_name: Optional[str] = None
    agent_config: Dict[str, Any] = Field(default_factory=dict)
    max_steps: int = 10
    concurrency: int = 5
    dataset_size: int = 50
    seed: Optional[int] = None
    created_at: float = Field(default_factory=time.time)


class RunSummary(BaseModel):
    """Top-level summary persisted after each evaluation run."""
    config: RunConfig
    status: RunStatus = RunStatus.PENDING
    total_scenarios: int = 0
    completed: int = 0
    failed: int = 0
    aggregate_metrics: Dict[str, float] = Field(default_factory=dict)
    duration_seconds: float = 0.0
    results: List[EvaluationResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class Agent(ABC):
    """Abstract base class for the system under test (SUT)."""

    @abstractmethod
    def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        """Decide on the next action based on history and current state."""
        ...

    @property
    def agent_id(self) -> str:
        return self.__class__.__name__


class AsyncAgent(ABC):
    """Async variant for agents backed by API calls."""

    @abstractmethod
    async def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        ...

    @property
    def agent_id(self) -> str:
        return self.__class__.__name__


class Environment(ABC):
    """Abstract base class for the simulation environment."""

    @abstractmethod
    def reset(self, scenario: Scenario) -> str:
        """Initialize environment for a scenario, return initial observation."""
        ...

    @abstractmethod
    def execute(self, action: ToolCall) -> str:
        """Execute the tool call and return observation string."""
        ...
