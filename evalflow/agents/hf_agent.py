"""
evalflow.agents.hf_agent — Local Hugging Face model agent.

Demonstrates PyTorch/Transformers integration for on-device inference
within the evaluation harness.
"""
from __future__ import annotations

import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..core import Agent, StepResult, ToolCall

logger = logging.getLogger(__name__)


class HuggingFaceAgent(Agent):
    """
    Agent backed by a local HF model (e.g., GPT-2, DistilGPT2).

    Since small models aren't fine-tuned for tool use, we use heuristic
    parsing to extract tool calls from freeform generation — demonstrating
    that the harness can wrangle raw model outputs.
    """

    def __init__(self, model_name: str = "distilgpt2", device: str = "cpu"):
        logger.info("Loading HF model: %s on %s", model_name, device)
        self.device = device
        self._model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def agent_id(self) -> str:
        return f"HuggingFaceAgent({self._model_name})"

    def act(self, history: List[StepResult], current_observation: str) -> ToolCall:
        prompt = self._build_prompt(history, current_observation)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt):].strip()

        return self._parse_output(new_text, history)

    def _build_prompt(self, history: List[StepResult], current_observation: str) -> str:
        prompt = "System: You are a helpful assistant. Tools: [search, calculate, done].\n"
        for step in history[-3:]:  # Keep context bounded
            prompt += f"Observation: {step.input_state[:100]}\n"
            prompt += f"Action: {step.action.tool_name} {step.action.arguments}\n"
            prompt += f"Result: {step.output_observation[:100]}\n"
        prompt += f"Observation: {current_observation[:200]}\nAction:"
        return prompt

    def _parse_output(self, text: str, history: List[StepResult]) -> ToolCall:
        lower = text.lower()

        if "search" in lower:
            query = text.replace("search", "").strip() or "general query"
            return ToolCall(tool_name="search", arguments={"query": query}, raw_output=text)
        elif "calculate" in lower:
            return ToolCall(tool_name="calculate", arguments={"expression": "100 * 5"}, raw_output=text)
        elif len(history) >= 2:
            return ToolCall(tool_name="done", arguments={"answer": text[:200]}, raw_output=text)
        else:
            return ToolCall(tool_name="search", arguments={"query": text[:100]}, raw_output=text)
