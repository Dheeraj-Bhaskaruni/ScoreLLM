"""
evalflow.environments — Simulation environments for agent evaluation.

Provides a domain-aware MockEnvironment that returns realistic responses
based on scenario metadata, with optional latency simulation and
stochastic failures for stress testing.
"""

from __future__ import annotations

import ast
import random
import time
from typing import Dict, Optional

from .core import Environment, Scenario, ToolCall

# ---------------------------------------------------------------------------
# Domain-aware response banks
# ---------------------------------------------------------------------------

SEARCH_RESPONSES: Dict[str, Dict[str, str]] = {
    "finance": {
        "aapl": "Apple Inc. (AAPL): Price $192.53, P/E 31.2, Market Cap $2.98T, Revenue (TTM) $383.3B.",
        "tsla": "Tesla Inc. (TSLA): Price $248.42, P/E 68.5, Market Cap $791B, Revenue (TTM) $96.8B.",
        "nvda": "NVIDIA Corp. (NVDA): Price $875.28, P/E 65.3, Market Cap $2.16T, Revenue (TTM) $79.8B.",
        "amzn": "Amazon.com (AMZN): Price $186.13, P/E 58.7, Market Cap $1.94T, Profit Margin 6.4%.",
        "s&p": "S&P 500: YTD Return +12.4%, Current Level 5,234.18, 52-Week High 5,341.88.",
        "gdp": "France GDP: $3.05 trillion (2024), Per-capita GDP $44,408, Population 68.7M.",
        "bitcoin": "Bitcoin (BTC): Price $67,432, Market Cap $1.33T, 24h Volume $28.4B.",
        "interest": "Federal Reserve: Federal Funds Rate 5.25%-5.50%, last changed July 2023.",
        "default": "Financial data: Index up 2.3% this quarter. Earnings season shows mixed results across sectors.",
    },
    "technology": {
        "iphone": 'iPhone 16 Pro Max: A18 Pro chip, 48MP main camera, 6.9" display, starts at $1,199.',
        "m4": "Apple M4 chip: 10-core CPU, 10-core GPU, 16-core Neural Engine. Geekbench 6 single-core: 3,810.",
        "vision": "Apple Vision Pro: Launched Feb 2024, estimated 500K units sold in first year, $3,499.",
        "ios": "iOS 19: Enhanced Siri with LLM, on-device AI features, redesigned Control Center, RCS support.",
        "swift": "Swift Concurrency: async/await, structured concurrency with TaskGroup, actors for data isolation.",
        "webkit": "WebKit: Multi-process architecture, JIT compilation via JavaScriptCore, GPU-accelerated compositing.",
        "coreml": "Core ML: On-device inference, supports Transformers, 40% faster on M4 vs M3 for LLM workloads.",
        "default": "Technology overview: Latest benchmarks show 15-20% improvement over previous generation.",
    },
    "healthcare": {
        "diabetes": "Type-2 Diabetes: Symptoms include increased thirst, frequent urination, fatigue, blurred vision. Affects 462M worldwide.",
        "covid": "COVID-19 Vaccine Efficacy: mRNA vaccines 95% effective at preventing symptomatic disease (initial trials), 70% against Omicron variants.",
        "ibuprofen": "Ibuprofen: Common side effects include stomach pain, nausea, dizziness. Max dose 1200mg/day (OTC). Risk of GI bleeding with prolonged use.",
        "ozempic": "Ozempic (semaglutide): Phase 3 trials showed 14.9% body weight reduction. HbA1c reduction of 1.5-1.8%. FDA approved for T2D.",
        "default": "Healthcare data: WHO recommends regular screening. Consult a healthcare provider for personalized advice.",
    },
    "science": {
        "light": "Speed of light in vacuum: 299,792,458 m/s (exact). Denoted c. Takes 8 min 20 sec to reach Earth from Sun.",
        "mars": "Earth-Mars distance: Min 54.6M km (opposition), Max 401M km (conjunction). Average ~225M km. Travel time 6-9 months.",
        "carbon": "Carbon-14 half-life: 5,730 +/- 40 years. Used in radiocarbon dating. Naturally produced in atmosphere by cosmic rays.",
        "photosynthesis": "Photosynthesis: 6CO2 + 6H2O + light -> C6H12O6 + 6O2. Occurs in chloroplasts. Light and dark reactions.",
        "gravitational": "Gravitational constant G: 6.674 x 10^-11 N⋅m²/kg². Measured by Cavendish experiment (1798).",
        "webb": "JWST Discovery (2024): Detected earliest known galaxies at z>14, 300M years after Big Bang. Found water vapor in exoplanet atmospheres.",
        "default": "Scientific data retrieved. Results consistent with current peer-reviewed literature.",
    },
    "general": {
        "weather": "Seattle Weather: Currently 58°F, partly cloudy. High 63°F, Low 48°F. 30% chance of rain this evening.",
        "population": "Tokyo population: 13.96M (city proper), 37.4M (Greater Tokyo Area). World's largest metropolitan area.",
        "restaurant": "Top Italian Restaurants: 1. Osteria Francescana (Modena) 2. Da Vittorio (Bergamo) 3. Local: Mario's Trattoria (4.8 stars, 0.3mi).",
        "flight": "Flight UA 123: SFO -> JFK, Departed 8:15 AM PST, ETA 4:45 PM EST. Status: On Time. Gate B22.",
        "news": "Top Headlines: 1. AI regulation bill advances 2. Markets hit record high 3. Climate summit reaches agreement 4. Space mission launch delayed.",
        "timezone": "NYC (EDT, UTC-4) to London (BST, UTC+1): London is 5 hours ahead. Current time NYC: 2:00 PM, London: 7:00 PM.",
        "default": "Search complete. General information retrieved successfully.",
    },
}

# Safe math functions exposed to the expression evaluator
CALCULATE_FUNCTIONS = {
    "mean": lambda *args: sum(args) / len(args) if args else 0,
    "average": lambda *args: sum(args) / len(args) if args else 0,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "round": round,
}


class MockEnvironment(Environment):
    """
    Domain-aware mock environment that returns realistic responses
    based on scenario metadata and query content.

    Features:
    - Domain-specific search responses (finance, tech, healthcare, science, general)
    - Safe arithmetic evaluation using AST parsing
    - Optional latency simulation and stochastic failure injection
    """

    def __init__(
        self,
        latency_ms: float = 0.0,
        failure_rate: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self._current_domain = "general"
        if seed is not None:
            random.seed(seed)

    def reset(self, scenario: Scenario) -> str:
        self._current_domain = scenario.metadata.get("domain", "general")
        return f"User Query: {scenario.initial_context}"

    def execute(self, action: ToolCall) -> str:
        # Optional latency simulation
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)

        # Stochastic failure
        if self.failure_rate > 0 and random.random() < self.failure_rate:
            return "Error: Service temporarily unavailable (simulated failure)."

        tool = action.tool_name.lower()

        if tool == "search":
            return self._handle_search(action)
        elif tool == "calculate":
            return self._handle_calculate(action)
        elif tool == "writer":
            return self._handle_writer(action)
        elif tool == "done":
            return "Task Completed."
        else:
            return f"Error: Unrecognized tool '{action.tool_name}'. Available tools: search, calculate, writer, done."

    def _handle_search(self, action: ToolCall) -> str:
        query = str(action.arguments.get("query", "")).lower()

        # Get domain-specific responses
        domain_responses = SEARCH_RESPONSES.get(self._current_domain, SEARCH_RESPONSES["general"])

        # Try to match query against known topics
        for keyword, response in domain_responses.items():
            if keyword != "default" and keyword in query:
                return f"Search Result: {response}"

        # Fall back to domain default
        default = domain_responses.get("default", "No results found.")
        return f"Search Result: {default}"

    def _handle_calculate(self, action: ToolCall) -> str:
        expr = str(action.arguments.get("expression", "0"))
        try:
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if not isinstance(
                    node,
                    (
                        ast.Expression,
                        ast.BinOp,
                        ast.UnaryOp,
                        ast.Constant,
                        ast.Add,
                        ast.Sub,
                        ast.Mult,
                        ast.Div,
                        ast.Mod,
                        ast.Pow,
                        ast.USub,
                        ast.UAdd,
                        ast.Call,
                        ast.Name,
                        ast.Tuple,
                    ),
                ):
                    return f"Error: Unsupported expression element: {type(node).__name__}"

            code = compile(tree, "<calc>", "eval")
            result = eval(code, {"__builtins__": {}}, CALCULATE_FUNCTIONS)
            return f"Calculation Result: {result}"
        except SyntaxError:
            return f"Error: Invalid expression syntax: '{expr[:100]}'"
        except Exception as e:
            return f"Error: Calculation failed — {e}"

    def _handle_writer(self, action: ToolCall) -> str:
        topic = action.arguments.get("topic", "General Analysis")
        return (
            f"Report Generated: '{topic}'\n"
            f"Summary: Analysis complete. Key findings documented. "
            f"Recommendations provided based on available data. "
            f"Report saved to output buffer."
        )
