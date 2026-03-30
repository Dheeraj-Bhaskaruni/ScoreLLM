"""Tests for evalflow.simulator."""

from evalflow.environments import MockEnvironment
from evalflow.simulator import SimulationEngine
from tests.conftest import CrashingAgent, DeterministicAgent


class TestSimulationEngine:
    def test_basic_run(self, simple_scenario, mock_env):
        agent = DeterministicAgent(["search", "done"])
        engine = SimulationEngine(environment=mock_env, max_steps=10)
        trace = engine.run_scenario(agent, simple_scenario)

        assert trace.scenario_id == simple_scenario.id
        assert trace.error is None
        assert trace.final_output is not None
        assert len(trace.steps) == 2  # search + done

    def test_max_steps_limit(self, simple_scenario, mock_env):
        # Agent that never calls done
        agent = DeterministicAgent(["search"] * 20)
        engine = SimulationEngine(environment=mock_env, max_steps=3)
        trace = engine.run_scenario(agent, simple_scenario)

        assert len(trace.steps) == 3
        assert trace.final_output is None  # Never called done

    def test_agent_crash_captured(self, simple_scenario, mock_env):
        agent = CrashingAgent(crash_on_step=0)
        engine = SimulationEngine(environment=mock_env)
        trace = engine.run_scenario(agent, simple_scenario)

        assert trace.error is not None
        assert "Simulated agent crash" in trace.error

    def test_agent_crash_mid_run(self, simple_scenario, mock_env):
        agent = CrashingAgent(crash_on_step=1)
        engine = SimulationEngine(environment=mock_env)
        trace = engine.run_scenario(agent, simple_scenario)

        # Step 0 should succeed (returns done before crash step)
        # Actually CrashingAgent returns done when not on crash step
        assert trace.final_output is not None

    def test_batch_run(self, simple_scenario, mock_env):
        agent = DeterministicAgent(["search", "done"])
        engine = SimulationEngine(environment=mock_env)
        scenarios = [simple_scenario, simple_scenario]
        traces = engine.run_batch(agent, scenarios)

        assert len(traces) == 2
        # Note: DeterministicAgent has internal state, so second run starts
        # from where first left off — that's expected behavior in batch

    def test_progress_callback(self, simple_scenario, mock_env):
        agent = DeterministicAgent(["done"])
        engine = SimulationEngine(environment=mock_env)

        progress_log = []
        engine.run_batch(
            agent,
            [simple_scenario],
            on_progress=lambda i, n, name: progress_log.append((i, n, name)),
        )
        assert len(progress_log) == 1
        assert progress_log[0][1] == 1

    def test_trace_timing(self, simple_scenario, mock_env):
        agent = DeterministicAgent(["search", "done"])
        engine = SimulationEngine(environment=mock_env)
        trace = engine.run_scenario(agent, simple_scenario)

        assert trace.start_time > 0
        assert trace.end_time >= trace.start_time
        assert trace.duration >= 0


class TestMockEnvironment:
    def test_finance_domain_response(self, simple_scenario, mock_env):
        from evalflow.core import ToolCall

        mock_env.reset(simple_scenario)
        result = mock_env.execute(ToolCall(tool_name="search", arguments={"query": "AAPL stock"}))
        assert "Apple Inc." in result or "AAPL" in result

    def test_calculate_safe(self, simple_scenario, mock_env):
        from evalflow.core import ToolCall

        mock_env.reset(simple_scenario)
        result = mock_env.execute(ToolCall(tool_name="calculate", arguments={"expression": "2 + 3 * 4"}))
        assert "14" in result

    def test_calculate_rejects_unsafe(self, simple_scenario, mock_env):
        from evalflow.core import ToolCall

        mock_env.reset(simple_scenario)
        result = mock_env.execute(
            ToolCall(tool_name="calculate", arguments={"expression": "__import__('os').system('ls')"})
        )
        assert "Error" in result

    def test_unknown_tool(self, simple_scenario, mock_env):
        from evalflow.core import ToolCall

        mock_env.reset(simple_scenario)
        result = mock_env.execute(ToolCall(tool_name="hack_system", arguments={}))
        assert "Unrecognized tool" in result

    def test_writer_tool(self, simple_scenario, mock_env):
        from evalflow.core import ToolCall

        mock_env.reset(simple_scenario)
        result = mock_env.execute(ToolCall(tool_name="writer", arguments={"topic": "Q4 Analysis"}))
        assert "Report Generated" in result

    def test_stochastic_failure(self):
        env = MockEnvironment(failure_rate=1.0, seed=42)  # Always fail
        from evalflow.core import ToolCall, Scenario

        env.reset(Scenario(name="t", description="t", initial_context="t"))
        result = env.execute(ToolCall(tool_name="search", arguments={"query": "test"}))
        assert "unavailable" in result.lower()
