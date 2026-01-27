"""
evalflow.storage — SQLite-backed persistence for evaluation runs and results.

Replaces flat JSON file storage with a proper relational backend. Uses Python's
built-in sqlite3 module (zero extra dependencies). Supports concurrent reads,
atomic writes, and efficient querying across thousands of runs.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "evalflow.db"


class StorageBackend:
    """
    SQLite storage for experiment runs, traces, and metrics.

    Schema:
    - runs: top-level run metadata (config, status, timing, aggregates)
    - results: per-scenario evaluation results (trace + metrics per run)
    - datasets: dataset version hashes for reproducibility tracking
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Thread-safe connection context manager."""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    model_name TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    config_json TEXT NOT NULL,
                    total_scenarios INTEGER DEFAULT 0,
                    completed INTEGER DEFAULT 0,
                    failed INTEGER DEFAULT 0,
                    duration_seconds REAL DEFAULT 0.0,
                    aggregate_metrics_json TEXT DEFAULT '{}',
                    dataset_hash TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    scenario_json TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_hash TEXT PRIMARY KEY,
                    size INTEGER NOT NULL,
                    seed INTEGER,
                    domains TEXT,
                    created_at REAL NOT NULL,
                    scenarios_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
                CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_name);
            """)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def insert_run(
        self,
        run_id: str,
        agent_id: str,
        model_name: str,
        config: Dict[str, Any],
        status: str = "running",
        dataset_hash: Optional[str] = None,
    ) -> None:
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO runs (run_id, agent_id, model_name, status, config_json,
                   dataset_hash, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, agent_id, model_name or "", status,
                 json.dumps(config), dataset_hash, now, now),
            )

    def update_run(
        self,
        run_id: str,
        status: str,
        total_scenarios: int = 0,
        completed: int = 0,
        failed: int = 0,
        duration_seconds: float = 0.0,
        aggregate_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE runs SET status=?, total_scenarios=?, completed=?, failed=?,
                   duration_seconds=?, aggregate_metrics_json=?, updated_at=?
                   WHERE run_id=?""",
                (status, total_scenarios, completed, failed, duration_seconds,
                 json.dumps(aggregate_metrics or {}), time.time(), run_id),
            )

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        return self._row_to_run(row)

    def list_runs(self, limit: int = 100, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            if model_name:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE model_name=? ORDER BY created_at DESC LIMIT ?",
                    (model_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._row_to_run(r) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def insert_results(self, run_id: str, results: List[Dict[str, Any]]) -> None:
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT INTO results (run_id, scenario_id, scenario_json, trace_json, metrics_json)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (
                        run_id,
                        r["scenario"]["id"],
                        json.dumps(r["scenario"]),
                        json.dumps(r["trace"]),
                        json.dumps(r["metrics"]),
                    )
                    for r in results
                ],
            )

    def get_results(self, run_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT scenario_json, trace_json, metrics_json FROM results WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            {
                "scenario": json.loads(r["scenario_json"]),
                "trace": json.loads(r["trace_json"]),
                "metrics": json.loads(r["metrics_json"]),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def insert_dataset(
        self,
        dataset_hash: str,
        size: int,
        seed: Optional[int],
        domains: List[str],
        scenarios: List[Dict[str, Any]],
    ) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO datasets (dataset_hash, size, seed, domains, created_at, scenarios_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (dataset_hash, size, seed, json.dumps(domains), time.time(), json.dumps(scenarios)),
            )

    def get_dataset(self, dataset_hash: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE dataset_hash=?", (dataset_hash,)).fetchone()
        if not row:
            return None
        return {
            "dataset_hash": row["dataset_hash"],
            "size": row["size"],
            "seed": row["seed"],
            "domains": json.loads(row["domains"]),
            "scenarios": json.loads(row["scenarios_json"]),
        }

    # ------------------------------------------------------------------
    # Aggregation queries
    # ------------------------------------------------------------------

    def get_model_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all runs for a model, sorted by time — for trend analysis."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT run_id, status, total_scenarios, completed, failed,
                          duration_seconds, aggregate_metrics_json, created_at
                   FROM runs WHERE model_name=? AND status='completed'
                   ORDER BY created_at""",
                (model_name,),
            ).fetchall()
        return [
            {
                "run_id": r["run_id"],
                "total": r["total_scenarios"],
                "completed": r["completed"],
                "duration_s": r["duration_seconds"],
                "metrics": json.loads(r["aggregate_metrics_json"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def count_runs(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM runs").fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "run_id": row["run_id"],
            "agent_id": row["agent_id"],
            "model_name": row["model_name"],
            "status": row["status"],
            "config": json.loads(row["config_json"]),
            "total": row["total_scenarios"],
            "completed": row["completed"],
            "failed": row["failed"],
            "duration_s": row["duration_seconds"],
            "metrics": json.loads(row["aggregate_metrics_json"]),
            "dataset_hash": row["dataset_hash"],
            "created_at": row["created_at"],
        }

__all__ = ["StorageBackend"]
