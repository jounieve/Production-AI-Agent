"""
eval/benchmark.py - Cost, latency, tool-call distribution, and a
deliberate TokenBudget trigger, all required by rubric G.

Two parts:

  1. `run_benchmark()` - runs the full agent (src/agent.run_agent) on
     >=10 questions from data/eval_questions.json, and for each run
     records wall-clock latency, estimated USD cost (from token usage
     x the pricing of whichever model agent.py actually used), and
     which tools were called. Prints and saves an aggregate summary.

  2. `trigger_token_budget_deliberately()` - runs one call through
     reasoning.self_consistency_synthesis with an artificially tiny
     TokenBudget so it is GUARANTEED to raise TokenBudgetExceeded. This
     satisfies the rubric's explicit requirement that TokenBudget be
     triggered at least once during testing, with the trigger
     documented (not just implemented and never exercised).

Run with:
    python eval/benchmark.py

Pricing note: cost is looked up from agent.PRICING_USD_PER_MTOK using
agent.MODEL_NAME, so it automatically matches whichever provider/model
LLM_PROVIDER actually selected (OpenAI gpt-4o-mini by default, $0 for a
local Ollama model, etc.) instead of a single hardcoded price. If the
resolved model isn't in that table, PRICE_PER_MILLION_INPUT/OUTPUT below
are used as a fallback - update them before reporting cost figures in
REPORT.md if that happens.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import AGENT_VERSION, MODEL_NAME, PRICING_USD_PER_MTOK, get_monitor_summary, run_agent
from guardrails import TokenBudget, TokenBudgetExceeded
from reasoning import self_consistency_synthesis

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_QUESTIONS_PATH = _DATA_DIR / "eval_questions.json"
_OUTPUT_PATH = Path(__file__).resolve().parent / "benchmark_results.json"

# Fallback pricing (OpenAI gpt-4o-mini, the project's default) used only if
# MODEL_NAME isn't found in agent.PRICING_USD_PER_MTOK.
PRICE_PER_MILLION_INPUT = 0.15
PRICE_PER_MILLION_OUTPUT = 0.60

_PRICE_IN, _PRICE_OUT = PRICING_USD_PER_MTOK.get(
    MODEL_NAME, (PRICE_PER_MILLION_INPUT, PRICE_PER_MILLION_OUTPUT)
)

# Rough split assumption for cost estimation: TokenBudget currently logs
# combined input+output per call, not split. We approximate a 60/40
# input/output split based on typical synthesis+critic call shapes
# (long context in, short structured answer out) observed during manual
# testing. This is an approximation, documented as such - for exact
# cost, read response.usage directly per call, which is a known
# limitation noted in REPORT.md section 6.
ASSUMED_INPUT_FRACTION = 0.6
ASSUMED_OUTPUT_FRACTION = 0.4


def _load_questions() -> list[dict]:
    """Loads the same question set used by ragas_eval.py, truncated to num_runs."""
    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def _estimate_cost_usd(total_tokens: int) -> float:
    """Approximates USD cost from a combined token count using the assumed
    60/40 input/output split and MODEL_NAME's real per-token pricing."""
    input_tokens = total_tokens * ASSUMED_INPUT_FRACTION
    output_tokens = total_tokens * ASSUMED_OUTPUT_FRACTION
    cost = (
        (input_tokens / 1_000_000) * _PRICE_IN
        + (output_tokens / 1_000_000) * _PRICE_OUT
    )
    return cost


async def run_benchmark(num_runs: int = 10) -> dict:
    """Runs run_agent() on num_runs questions, recording latency, estimated
    cost, and tool-call distribution; returns the aggregate summary dict
    that main() saves to benchmark_results.json."""
    questions = _load_questions()[:num_runs]
    if len(questions) < num_runs:
        print(
            f"WARNING: only {len(questions)} questions available in "
            f"eval_questions.json, requested {num_runs}. Add more "
            f"questions for a fuller benchmark."
        )

    latencies: list[float] = []
    costs: list[float] = []
    tool_call_counter: Counter[str] = Counter()
    blocked_runs = 0
    per_run_records = []

    for item in questions:
        question = item["question"]
        print(f"Running: {question[:60]}...")

        start = time.perf_counter()
        result = await run_agent(question)
        elapsed = time.perf_counter() - start

        if result.blocked_reason:
            blocked_runs += 1
            print(f"  -> blocked: {result.blocked_reason}")
            continue

        total_tokens = result.token_usage.get("used_tokens", 0)
        cost = _estimate_cost_usd(total_tokens)

        latencies.append(elapsed)
        costs.append(cost)
        tool_call_counter.update(result.tool_calls_made)

        per_run_records.append(
            {
                "question": question,
                "latency_seconds": round(elapsed, 2),
                "estimated_cost_usd": round(cost, 5),
                "tools_called": result.tool_calls_made,
                "total_tokens": total_tokens,
            }
        )
        print(f"  -> {elapsed:.2f}s, ~${cost:.5f}, tools: {result.tool_calls_made}")

    summary = {
        "num_runs_attempted": len(questions),
        "num_runs_blocked": blocked_runs,
        "num_runs_completed": len(latencies),
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "avg_cost_usd": round(sum(costs) / len(costs), 5) if costs else None,
        "total_cost_usd": round(sum(costs), 5) if costs else None,
        "tool_call_distribution": dict(tool_call_counter),
        "pricing_used": {
            "model": MODEL_NAME,
            "input_per_million_usd": _PRICE_IN,
            "output_per_million_usd": _PRICE_OUT,
            "note": "cost is an ESTIMATE based on an assumed 60/40 input/output token split",
        },
        "per_run": per_run_records,
    }
    return summary


def trigger_token_budget_deliberately() -> dict:
    """
    Deliberately exhausts a TokenBudget to prove the mechanism actually
    stops a session, and to give REPORT.md a real, reproducible example
    to cite for rubric G ("TokenBudget triggered at least once during
    testing, documented").
    """
    print("\n=== Deliberately triggering TokenBudget ===")
    tiny_budget = TokenBudget(max_tokens=50)  # far below what one synthesis call needs

    demo_chunks = [
        "Vacancy rates below 3% are generally considered a tight housing market."
    ]
    demo_sources = ["receiving_city_capacity.md"]
    question = "What housing vacancy rate indicates a tight market?"

    try:
        self_consistency_synthesis(question, demo_chunks, demo_sources, tiny_budget, k=1)
        return {"triggered": False, "note": "budget was NOT exceeded - increase demo strictness"}
    except TokenBudgetExceeded as exc:
        print(f"TokenBudget correctly raised: {exc}")
        return {
            "triggered": True,
            "configured_max_tokens": tiny_budget.max_tokens,
            "tokens_used_at_trigger": tiny_budget.used_tokens,
            "exception_message": str(exc),
        }


def main():
    """Runs the deliberate TokenBudget trigger, then the full benchmark,
    prints a summary, and saves everything to benchmark_results.json."""
    budget_trigger_result = trigger_token_budget_deliberately()

    print("\n=== Running full agent benchmark (10 runs) ===")
    summary = asyncio.run(run_benchmark(num_runs=10))
    summary["token_budget_trigger_demo"] = budget_trigger_result
    summary["agent_version"] = AGENT_VERSION
    # AgentMonitor (lab_B4_production.ipynb) accumulates across every run_agent()
    # call in this process, so by now it has seen all `num_runs` benchmark runs
    # plus every MCP tool call they made - this is the "at least one monitoring
    # alert" evidence required by rubric E, distinct from the Langfuse spans.
    summary["monitor_summary"] = get_monitor_summary()

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Runs completed:        {summary['num_runs_completed']}/{summary['num_runs_attempted']}")
    print(f"Runs blocked (L1):     {summary['num_runs_blocked']}")
    print(f"Avg latency (s):       {summary['avg_latency_seconds']}")
    print(f"Avg cost (USD):        {summary['avg_cost_usd']}")
    print(f"Total cost (USD):      {summary['total_cost_usd']}")
    print(f"Tool call distribution: {summary['tool_call_distribution']}")
    print(f"TokenBudget triggered: {budget_trigger_result['triggered']}")
    print(f"\nFull results saved to {_OUTPUT_PATH}")
    print("Copy the summary numbers directly into REPORT.md section 3.")


if __name__ == "__main__":
    main()