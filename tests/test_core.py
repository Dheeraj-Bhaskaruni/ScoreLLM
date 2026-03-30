"""Tests for evalflow.core Pydantic models."""

import pytest

from evalflow.core import (
    EvaluationResult,
    MetricResult,
    RunConfig,
    RunStatus,
    RunSummary,
    Scenario,
    SimulationTrace,
    StepResult,
    ToolCall,
)


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(tool_name="search", arguments={"query": "test"})
        assert tc.tool_name == "search"
        assert tc.arguments["query"] == "test"
        assert tc.raw_output is None

    def test_serialization_roundtrip(self):
        tc = ToolCall(tool_name="calculate", arguments={"expression": "2+2"}, raw_output="raw")
        data = tc.model_dump()
        tc2 = ToolCall.model_validate(data)
        assert tc == tc2

    def test_json_roundtrip(self):
        tc = ToolCall(tool_name="done", arguments={"answer": "42"})
        json_str = tc.model_dump_json()
        tc2 = ToolCall.model_validate_json(json_str)
        assert tc == tc2


class TestScenario:
    def test_properties(self, simple_scenario):
        assert simple_scenario.difficulty == "easy"
        assert simple_scenario.domain == "finance"
        assert simple_scenario.category == "standard"

    def test_auto_id_generation(self):
        s = Scenario(name="Test", description="Test", initial_context="Test")
        assert len(s.id) == 8

    def test_serialization(self, simple_scenario):
        data = simple_scenario.model_dump()
        s2 = Scenario.model_validate(data)
        assert s2.name == simple_scenario.name
        assert s2.expected_tool_sequence == ["search"]


class TestSimulationTrace:
    def test_duration(self):
        trace = SimulationTrace(scenario_id="t1", agent_id="a1", start_time=100.0, end_time=105.5)
        assert trace.duration == pytest.approx(5.5)

    def test_tool_sequence_excludes_done(self):
        trace = SimulationTrace(
            scenario_id="t1",
            agent_id="a1",
            steps=[
                StepResult(
                    step_id=0,
                    input_state="obs",
                    action=ToolCall(tool_name="search", arguments={}),
                    output_observation="result",
                ),
                StepResult(
                    step_id=1,
                    input_state="result",
                    action=ToolCall(tool_name="calculate", arguments={}),
                    output_observation="4",
                ),
                StepResult(
                    step_id=2,
                    input_state="4",
                    action=ToolCall(tool_name="done", arguments={"answer": "4"}),
                    output_observation="<TERMINATED>",
                ),
            ],
        )
        assert trace.tool_sequence == ["search", "calculate"]

    def test_empty_trace(self):
        trace = SimulationTrace(scenario_id="t1", agent_id="a1")
        assert trace.tool_sequence == []
        assert trace.final_output is None


class TestRunConfig:
    def test_auto_fields(self):
        config = RunConfig(agent_id="test")
        assert len(config.run_id) == 12
        assert config.created_at > 0
        assert config.concurrency == 5

    def test_custom_values(self):
        config = RunConfig(
            run_id="custom-id",
            agent_id="myagent",
            model_name="gpt-4",
            max_steps=20,
            concurrency=10,
            dataset_size=100,
            seed=42,
        )
        assert config.run_id == "custom-id"
        assert config.seed == 42


class TestRunSummary:
    def test_default_status(self):
        config = RunConfig(agent_id="test")
        summary = RunSummary(config=config)
        assert summary.status == RunStatus.PENDING

    def test_full_summary(self):
        config = RunConfig(agent_id="test")
        summary = RunSummary(
            config=config,
            status=RunStatus.COMPLETED,
            total_scenarios=50,
            completed=45,
            failed=5,
            aggregate_metrics={"avg_SuccessRate": 0.9},
            duration_seconds=12.5,
        )
        assert summary.failed == 5
        assert summary.aggregate_metrics["avg_SuccessRate"] == 0.9


class TestEvaluationResult:
    def test_creation(self, simple_scenario):
        trace = SimulationTrace(scenario_id=simple_scenario.id, agent_id="test")
        result = EvaluationResult(
            scenario=simple_scenario,
            trace=trace,
            metrics={"SuccessRate": MetricResult(name="SuccessRate", score=1.0)},
        )
        assert result.metrics["SuccessRate"].score == 1.0
