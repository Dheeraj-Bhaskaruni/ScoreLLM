"""
evalflow.simulator — Synchronous and asynchronous simulation engines.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, List, Optional

from .core import (
    Agent,
    AsyncAgent,
    Environment,
    Scenario,
    SimulationTrace,
    StepResult,
    ToolCall,
)

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[int, int, str], None]]


class SimulationEngine:
    """Synchronous simulation engine — runs one scenario at a time."""

    def __init__(self, environment: Environment, max_steps: int = 10):
        self.environment = environment
        self.max_steps = max_steps

    def run_scenario(self, agent: Agent, scenario: Scenario) -> SimulationTrace:
        trace = SimulationTrace(
            scenario_id=scenario.id,
            agent_id=agent.agent_id,
            start_time=time.time(),
        )

        try:
            current_observation = self.environment.reset(scenario)

            for i in range(self.max_steps):
                step_start = time.time()

                try:
                    action: ToolCall = agent.act(trace.steps, current_observation)
                except Exception as e:
                    trace.error = f"Agent Crash: {e}"
                    logger.error("Agent crashed on scenario %s at step %d: %s", scenario.id, i, e)
                    break

                if action.tool_name.lower() == "done":
                    trace.final_output = action.arguments.get("answer", "")
                    trace.steps.append(
                        StepResult(
                            step_id=i,
                            timestamp=step_start,
                            input_state=current_observation,
                            action=action,
                            output_observation="<TERMINATED>",
                        )
                    )
                    break

                try:
                    observation = self.environment.execute(action)
                except Exception as e:
                    observation = f"Tool Execution Error: {e}"

                trace.steps.append(
                    StepResult(
                        step_id=i,
                        timestamp=step_start,
                        input_state=current_observation,
                        action=action,
                        output_observation=observation,
                    )
                )
                current_observation = observation

        except Exception as e:
            trace.error = f"Simulation Error: {e}"
        finally:
            trace.end_time = time.time()

        return trace

    def run_batch(
        self,
        agent: Agent,
        scenarios: List[Scenario],
        on_progress: ProgressCallback = None,
    ) -> List[SimulationTrace]:
        traces: List[SimulationTrace] = []
        for idx, scenario in enumerate(scenarios):
            if on_progress:
                on_progress(idx, len(scenarios), scenario.name)
            traces.append(self.run_scenario(agent, scenario))
        return traces


class AsyncSimulationEngine:
    """
    Async simulation engine with controlled concurrency via semaphore.

    Designed for agents that call external APIs (HF Inference, OpenAI, etc.)
    where parallelism dramatically reduces wall-clock time.
    """

    def __init__(self, environment: Environment, max_steps: int = 10, concurrency: int = 5):
        self.environment = environment
        self.max_steps = max_steps
        self.concurrency = concurrency

    async def run_scenario(self, agent: AsyncAgent, scenario: Scenario) -> SimulationTrace:
        trace = SimulationTrace(
            scenario_id=scenario.id,
            agent_id=agent.agent_id,
            start_time=time.time(),
        )

        try:
            current_observation = self.environment.reset(scenario)

            for i in range(self.max_steps):
                step_start = time.time()

                try:
                    action: ToolCall = await agent.act(trace.steps, current_observation)
                except Exception as e:
                    trace.error = f"Agent Crash: {e}"
                    logger.error("Async agent crashed on %s step %d: %s", scenario.id, i, e)
                    break

                if action.tool_name.lower() == "done":
                    trace.final_output = action.arguments.get("answer", "")
                    trace.steps.append(
                        StepResult(
                            step_id=i,
                            timestamp=step_start,
                            input_state=current_observation,
                            action=action,
                            output_observation="<TERMINATED>",
                        )
                    )
                    break

                try:
                    observation = self.environment.execute(action)
                except Exception as e:
                    observation = f"Tool Execution Error: {e}"

                trace.steps.append(
                    StepResult(
                        step_id=i,
                        timestamp=step_start,
                        input_state=current_observation,
                        action=action,
                        output_observation=observation,
                    )
                )
                current_observation = observation

        except Exception as e:
            trace.error = f"Simulation Error: {e}"
        finally:
            trace.end_time = time.time()

        return trace

    async def run_batch(
        self,
        agent: AsyncAgent,
        scenarios: List[Scenario],
        on_progress: ProgressCallback = None,
    ) -> List[SimulationTrace]:
        semaphore = asyncio.Semaphore(self.concurrency)
        completed = 0

        async def _run_one(scenario: Scenario) -> SimulationTrace:
            nonlocal completed
            async with semaphore:
                trace = await self.run_scenario(agent, scenario)
                completed += 1
                if on_progress:
                    on_progress(completed, len(scenarios), scenario.name)
                return trace

        tasks = [_run_one(s) for s in scenarios]
        return await asyncio.gather(*tasks)
