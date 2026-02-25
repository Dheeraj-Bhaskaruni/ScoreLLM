"""
diagnostic_test.py — Test a real HF Inference API agent end-to-end.

Requires HF_TOKEN in .env.
"""
import os
import time

from dotenv import load_dotenv

from evalflow.agents.api_agent import HFApiAgent
from evalflow.core import StepResult
from evalflow.environments import MockEnvironment, Scenario

load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    print("No HF_TOKEN found in .env")
    exit(1)

model_id = "Qwen/Qwen2.5-7B-Instruct:together"
print(f"\nDIAGNOSTIC: Testing {model_id}")

agent = HFApiAgent(model_id=model_id, api_token=token)
env = MockEnvironment()
scenario = Scenario(
    name="Financial Analysis",
    description="Analyze Apple financial performance",
    initial_context="Analyze the financial performance of Apple Inc. (AAPL).",
    expected_tool_sequence=["search", "calculate"],
    metadata={"domain": "finance", "difficulty": "hard"},
)

history = []
current_obs = env.reset(scenario)
print(f"  [USER] {current_obs}")

for i in range(4):
    print(f"\n--- STEP {i + 1} ---")
    try:
        action = agent.act(history, current_obs)
    except Exception as e:
        print(f"  AGENT ERROR: {e}")
        break

    print(f"  [RAW] {repr(action.raw_output)}")
    print(f"  [PARSED] Tool: {action.tool_name} | Args: {action.arguments}")

    if action.tool_name == "done":
        print(f"\nSUCCESS: {action.arguments.get('answer')}")
        break

    obs = env.execute(action)
    print(f"  [ENV] {obs}")
    history.append(StepResult(step_id=i, input_state=current_obs, action=action, output_observation=obs))
    current_obs = obs

print("\n--- DIAGNOSTIC COMPLETE ---")
# Diagnostic: validates all imports and basic functionality
