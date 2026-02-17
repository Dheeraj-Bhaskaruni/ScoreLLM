#!/usr/bin/env python3
"""
upload_dataset.py — Push EvalFlow simulation results to Hugging Face Hub as a Dataset.

Usage:
    # First, generate results
    PYTHONPATH=. python3 run_batch.py

    # Then upload (uses HF_TOKEN from .env or environment)
    python3 upload_dataset.py --repo-id <your-username>/evalflow-results

    # Or dry-run (local preview without pushing to Hub) to inspect the dataset locally
    python3 upload_dataset.py --dry-run (local preview without pushing to Hub)
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def load_results(path: str = "simulation_results.json") -> list:
    with open(path) as f:
        return json.load(f)


def flatten_results(raw_data: list) -> list:
    """Convert nested simulation results into a flat tabular format for HF Datasets."""
    rows = []
    for item in raw_data:
        scenario = item["scenario"]
        trace = item["trace"]
        metrics = item["metrics"]

        # Flatten tool calls from steps
        tool_sequence = [
            step["action"]["tool_name"]
            for step in trace.get("steps", [])
            if step["action"]["tool_name"].lower() != "done"
        ]

        rows.append({
            # Scenario fields
            "scenario_id": scenario["id"],
            "scenario_name": scenario["name"],
            "scenario_description": scenario["description"],
            "initial_context": scenario["initial_context"],
            "expected_tools": json.dumps(scenario.get("expected_tool_sequence", [])),
            "difficulty": scenario.get("metadata", {}).get("difficulty", "unknown"),
            "domain": scenario.get("metadata", {}).get("domain", "unknown"),
            "category": scenario.get("metadata", {}).get("category", "standard"),
            # Trace fields
            "agent_id": trace["agent_id"],
            "num_steps": len(trace.get("steps", [])),
            "actual_tools": json.dumps(tool_sequence),
            "final_output": trace.get("final_output", ""),
            "error": trace.get("error", ""),
            "duration_seconds": round(trace["end_time"] - trace["start_time"], 4),
            # Metric fields
            "success_rate": metrics.get("SuccessRate", 0.0),
            "step_count": metrics.get("StepCount", 0.0),
            "tool_accuracy": metrics.get("ExpectedToolUsage", 0.0),
            "helpfulness_score": metrics.get("Helpfulness Score", 0.0),
            "helpfulness_reason": metrics.get("Helpfulness Reason", ""),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Push EvalFlow results to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HF Hub repo ID, e.g. 'your-username/evalflow-results'",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="simulation_results.json",
        help="Path to simulation results JSON",
    )
    parser.add_argument(
        "--dry-run (local preview without pushing to Hub)",
        action="store_true",
        help="Build dataset locally without pushing to Hub",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the HF dataset repo private",
    )
    args = parser.parse_args()

    # ---- Load and flatten ----
    print(f"Loading results from {args.results_file}...")
    raw = load_results(args.results_file)
    rows = flatten_results(raw)
    print(f"Flattened {len(rows)} evaluation records.")

    # ---- Build HF Dataset ----
    try:
        from datasets import Dataset, Features, Value
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    features = Features({
        "scenario_id": Value("string"),
        "scenario_name": Value("string"),
        "scenario_description": Value("string"),
        "initial_context": Value("string"),
        "expected_tools": Value("string"),
        "difficulty": Value("string"),
        "domain": Value("string"),
        "category": Value("string"),
        "agent_id": Value("string"),
        "num_steps": Value("int32"),
        "actual_tools": Value("string"),
        "final_output": Value("string"),
        "error": Value("string"),
        "duration_seconds": Value("float64"),
        "success_rate": Value("float64"),
        "step_count": Value("float64"),
        "tool_accuracy": Value("float64"),
        "helpfulness_score": Value("float64"),
        "helpfulness_reason": Value("string"),
    })

    dataset = Dataset.from_list(rows, features=features)
    print(f"\nDataset created: {dataset}")
    print(dataset.to_pandas().describe())

    if args.dry_run:
        # Save locally
        out_dir = Path("evalflow_dataset")
        dataset.save_to_disk(str(out_dir))
        print(f"\nDry-run complete. Dataset saved locally to '{out_dir}/'")
        print("Preview:")
        print(dataset.to_pandas().head())
        return

    # ---- Push to Hub ----
    if not args.repo_id:
        print("ERROR: --repo-id is required to push (e.g. 'your-username/evalflow-results')")
        sys.exit(1)

    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in environment. Set it in .env or export it.")
        sys.exit(1)

    print(f"\nPushing to https://huggingface.co/datasets/{args.repo_id} ...")
    dataset.push_to_hub(
        args.repo_id,
        token=token,
        private=args.private,
    )
    print(f"Done! View at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
# Supports: --dry-run, --repo-id
