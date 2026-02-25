"""
demo.py — Quick demo of the EvalFlow pipeline.

Runs a single scenario end-to-end: agent -> environment -> metrics.
"""
from typing import List

from evalflow.core import Agent, Scenario, StepResult, ToolCall
from evalflow.environments import MockEnvironment
from evalflow.metrics.metrics import ExpectedToolUsage, MetricEngine, StepCount, SuccessRate
from evalflow.simulator import SimulationEngine


class MockSearchAgent(Agent):
    """Simple agent: search -> calculate -> done."""

    def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        if len(history) == 0:
            return ToolCall(tool_name="search", arguments={"query": "Apple stock price"})
        elif len(history) == 1:
            return ToolCall(tool_name="calculate", arguments={"expression": "150 * 10"})
        else:
            return ToolCall(tool_name="done", arguments={"answer": "The total value of 10 Apple shares is $1,500."})


def main():
    print("EvalFlow Demo")
    print("=" * 40)

    scenario = Scenario(
        id="demo-001",
        name="Get Stock Value",
        description="Search for stock and calculate total value",
        initial_context="User wants to know value of 10 Apple shares.",
        expected_tool_sequence=["search", "calculate"],
        metadata={"difficulty": "hard", "domain": "finance"},
    )

    agent = MockSearchAgent()
    env = MockEnvironment()
    engine = SimulationEngine(environment=env)

    print(f"Running: {scenario.name}")
    trace = engine.run_scenario(agent, scenario)

    print(f"Steps: {len(trace.steps)}")
    for step in trace.steps:
        print(f"  [{step.action.tool_name}] -> {step.output_observation[:80]}")
    print(f"Final Output: {trace.final_output}")

    evaluator = MetricEngine([SuccessRate(), StepCount(), ExpectedToolUsage()])
    results = evaluator.evaluate_trace(trace, scenario)

    print("\nMetrics:")
    for name, score in results.items():
        print(f"  {name}: {score}")


if __name__ == "__main__":
    main()
