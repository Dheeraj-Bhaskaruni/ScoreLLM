"""Tests for evalflow.storage — SQLite storage backend."""

import os
import tempfile

import pytest

from evalflow.storage import StorageBackend


@pytest.fixture
def storage():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    backend = StorageBackend(db_path=path)
    yield backend
    os.unlink(path)


class TestRunLifecycle:
    def test_insert_and_get_run(self, storage):
        storage.insert_run(
            run_id="test-001",
            agent_id="TestAgent",
            model_name="test-model",
            config={"seed": 42, "size": 10},
            status="running",
        )
        run = storage.get_run("test-001")
        assert run is not None
        assert run["run_id"] == "test-001"
        assert run["agent_id"] == "TestAgent"
        assert run["model_name"] == "test-model"
        assert run["status"] == "running"

    def test_update_run(self, storage):
        storage.insert_run("run-002", "Agent", "model", {})
        storage.update_run(
            "run-002",
            status="completed",
            total_scenarios=10,
            completed=9,
            failed=1,
            duration_seconds=42.5,
            aggregate_metrics={"avg_helpfulness": 3.5},
        )
        run = storage.get_run("run-002")
        assert run["status"] == "completed"
        assert run["completed"] == 9
        assert run["failed"] == 1
        assert run["metrics"]["avg_helpfulness"] == 3.5

    def test_list_runs(self, storage):
        for i in range(5):
            storage.insert_run(f"run-{i}", "Agent", "model", {})
        runs = storage.list_runs()
        assert len(runs) == 5

    def test_list_runs_by_model(self, storage):
        storage.insert_run("r1", "A", "model-a", {})
        storage.insert_run("r2", "A", "model-b", {})
        storage.insert_run("r3", "A", "model-a", {})
        runs = storage.list_runs(model_name="model-a")
        assert len(runs) == 2

    def test_delete_run(self, storage):
        storage.insert_run("del-001", "Agent", "model", {})
        assert storage.delete_run("del-001")
        assert storage.get_run("del-001") is None

    def test_get_nonexistent_run(self, storage):
        assert storage.get_run("nonexistent") is None

    def test_count_runs(self, storage):
        assert storage.count_runs() == 0
        storage.insert_run("c1", "A", "m", {})
        storage.insert_run("c2", "A", "m", {})
        assert storage.count_runs() == 2


class TestResults:
    def test_insert_and_get_results(self, storage):
        storage.insert_run("res-001", "Agent", "model", {})
        results = [
            {
                "scenario": {"id": "s1", "name": "Test"},
                "trace": {"steps": []},
                "metrics": {"helpfulness": 4.0},
            },
            {
                "scenario": {"id": "s2", "name": "Test 2"},
                "trace": {"steps": [{"tool": "search"}]},
                "metrics": {"helpfulness": 3.0},
            },
        ]
        storage.insert_results("res-001", results)
        loaded = storage.get_results("res-001")
        assert len(loaded) == 2
        assert loaded[0]["scenario"]["id"] == "s1"
        assert loaded[1]["metrics"]["helpfulness"] == 3.0


class TestDatasets:
    def test_insert_and_get_dataset(self, storage):
        storage.insert_dataset(
            dataset_hash="abc123",
            size=10,
            seed=42,
            domains=["finance", "technology"],
            scenarios=[{"id": "s1"}, {"id": "s2"}],
        )
        ds = storage.get_dataset("abc123")
        assert ds is not None
        assert ds["size"] == 10
        assert ds["seed"] == 42
        assert "finance" in ds["domains"]

    def test_duplicate_dataset_ignored(self, storage):
        storage.insert_dataset("dup", 5, 42, ["general"], [])
        storage.insert_dataset("dup", 10, 99, ["finance"], [])  # should be ignored
        ds = storage.get_dataset("dup")
        assert ds["size"] == 5  # first insert wins


class TestModelHistory:
    def test_get_model_history(self, storage):
        for i in range(3):
            storage.insert_run(f"hist-{i}", "Agent", "test-model", {}, status="completed")
            storage.update_run(
                f"hist-{i}",
                status="completed",
                aggregate_metrics={"avg_helpfulness": 3.0 + i * 0.5},
            )
        history = storage.get_model_history("test-model")
        assert len(history) == 3
        assert history[0]["metrics"]["avg_helpfulness"] == 3.0
