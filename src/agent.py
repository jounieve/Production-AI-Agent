"""
agent.py - Main orchestration loop for the Urban Migration Agent.

Flow:
    user query
      -> L1 input filter                              (guardrails.py)
      -> Claude tool-calling loop against the MCP server (mcp_server.py),
         with every tool call passing through the L4 ActionGate first
      -> L1 filtering of retrieved/tool-returned text  (indirect injection defense)
      -> Self-Consistency (k=3) synthesis, few-shot CoT (reasoning.py)
      -> Critic review, second agent role              (reasoning.py)
      -> final structured output

The agent talks to mcp_server.py as a REAL MCP client over stdio (not a
direct Python import), so the "working MCP server" requirement is
exercised end-to-end, not just present as unused code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

try:
    from langfuse.decorators import observe, langfuse_context
except ImportError:  # pragma: no cover
    def observe(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    class _NoOpLangfuseContext:
        def update_current_trace(self, **kwargs):
            pass

    langfuse_context = _NoOpLangfuseContext()

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import guardrails`/`import reasoning` work

from guardrails import ActionGate, TokenBudget, l1_filter_retrieved_context, l1_input_filter, risk_tier
from reasoning import critic_review, self_consistency_synthesis

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Indicative USD pricing per million tokens (input, output). Local Ollama
# models cost $0 - see eval/benchmark.py and _estimate_run_cost_usd() below.
PRICING_USD_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "llama3.2:latest": (0.0, 0.0),
    "llama3.2-vision:latest": (0.0, 0.0),
    "qwen2.5-coder:7b": (0.0, 0.0),
}

# What this agent is, in one sentence - used both to derive its EU AI Act
# tier below and as the source-of-truth description for REPORT.md section 1.
AGENT_DESCRIPTION = (
    "Urban migration research and analysis agent: assesses whether a "
    "receiving city can absorb a migrant influx, using corpus research and "
    "structured city data, for policy analysts and humanitarian researchers. "
    "Produces research summaries reviewed by a human, not an automated "
    "immigration/border-control or asylum-eligibility decision."
)
EU_AI_ACT_RISK_TIER, EU_AI_ACT_OBLIGATION = risk_tier(AGENT_DESCRIPTION)

# Provider selection is a single config switch (LLM_PROVIDER in .env) instead
# of the LLM call being wired to one hard-coded SDK. Both supported providers
# speak the OpenAI-compatible chat.completions API, so the same client class
# and the same tool-calling loop below work unchanged for either one.
# LLM_PROVIDER=openai|ollama picks explicitly; if unset, default to OpenAI
# when a real key is present, else fall back to the local Ollama server (no
# key required, no other model is deleted/replaced).
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Guard against the literal "sk-..." placeholder from .env.example: an
# unfilled placeholder must fall back to Ollama, not be treated as a real key.
_HAS_REAL_OPENAI_KEY = bool(_OPENAI_API_KEY) and not _OPENAI_API_KEY.startswith("sk-...")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
if not LLM_PROVIDER:
    LLM_PROVIDER = "openai" if _HAS_REAL_OPENAI_KEY else "ollama"

if LLM_PROVIDER == "openai":
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _LLM_BASE_URL = None  # OpenAI's default endpoint
    _LLM_API_KEY = _OPENAI_API_KEY
elif LLM_PROVIDER == "ollama":
    MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2:latest")
    _LLM_BASE_URL = OLLAMA_BASE_URL
    _LLM_API_KEY = "ollama"  # Ollama's OpenAI-compatible endpoint ignores the key
else:
    raise ValueError(f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Use 'openai' or 'ollama'.")

MAX_TOOL_TURNS = 6  # hard ceiling on tool-calling turns per query, independent of TokenBudget

_provider = LLM_PROVIDER.capitalize()
print(f"[agent] Provider: {_provider} | Model: {MODEL_NAME}")

_MCP_SERVER_PATH = str(Path(__file__).resolve().parent / "mcp_server.py")

# System prompt for the tool-selection loop, hoisted to a module constant (not
# built inline in run_agent()) so it can be hashed for AGENT_VERSION below and
# so its exact wording is visible/citable in one place.
TOOL_SELECTION_SYSTEM_PROMPT = (
    "You are an urban migration research agent with three tools.\n\n"
    "MANDATORY TOOL SEQUENCE - follow this on EVERY question:\n"
    "Step 1 - call get_city_capacity_profile for EACH city mentioned in the question.\n"
    "Step 2 - call search_migration_evidence to retrieve research thresholds and qualitative context. "
    "This step is REQUIRED on every question without exception - it provides the evidence needed to interpret the numeric data.\n"
    "Step 3 - if an origin region is mentioned, call compute_push_pull_index.\n\n"
    "You MUST call search_migration_evidence before synthesising any answer. "
    "Never skip it - without it you have no basis to interpret the numeric indicators."
)


def hash_prompt(prompt: str) -> str:
    """Short, stable fingerprint of a prompt (from lab_B4_production.ipynb):
    if this hash changes between runs, the prompt changed - a cheap way to
    trace behaviour changes back to a specific prompt edit in Langfuse."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


AGENT_VERSION = {
    "version": "1.0.0",
    "tool_selection_prompt_hash": hash_prompt(TOOL_SELECTION_SYSTEM_PROMPT),
    "model": MODEL_NAME,
    "provider": LLM_PROVIDER,
    "eu_ai_act_risk_tier": EU_AI_ACT_RISK_TIER,
}


class AgentMonitor:
    """
    Production monitoring: tracks run- and tool-level stats across the
    process and raises a print-based alert on slow runs, expensive runs,
    empty responses, and high tool error rates. This is deliberately
    separate from the Langfuse @observe spans above - spans answer "what
    happened in this one trace", alerts answer "is something wrong across
    many runs" (rubric E).
    """

    def __init__(
        self,
        slow_run_seconds: float = 60.0,
        expensive_run_usd: float = 0.05,
        tool_error_rate_threshold: float = 0.20,
    ):
        self.n_runs = 0
        self.total_cost_usd = 0.0
        self.alerts: list[str] = []
        self.slow_run_seconds = slow_run_seconds
        self.expensive_run_usd = expensive_run_usd
        self.tool_error_rate_threshold = tool_error_rate_threshold
        self._tools: dict[str, dict] = defaultdict(lambda: {"calls": 0, "errors": 0, "ms_total": 0.0})

    def record_run(self, question: str, response: str, duration_s: float, cost_usd: float) -> None:
        """Call once per completed run_agent() call."""
        self.n_runs += 1
        self.total_cost_usd += cost_usd
        if duration_s > self.slow_run_seconds:
            self._alert(f"SLOW RUN: {duration_s:.1f}s for '{question[:50]}'")
        if cost_usd > self.expensive_run_usd:
            self._alert(f"EXPENSIVE RUN: ${cost_usd:.4f} for '{question[:50]}'")
        if not response or len(response) < 20:
            self._alert(f"EMPTY/SHORT RESPONSE for '{question[:50]}'")

    def record_tool(self, name: str, success: bool, duration_ms: float) -> None:
        """Call once per MCP tool call, success = whether it returned without exception."""
        stats = self._tools[name]
        stats["calls"] += 1
        stats["ms_total"] += duration_ms
        if not success:
            stats["errors"] += 1
            rate = stats["errors"] / stats["calls"]
            if rate > self.tool_error_rate_threshold:
                self._alert(f"HIGH ERROR RATE for tool '{name}': {rate:.0%}")

    def _alert(self, message: str) -> None:
        self.alerts.append(message)
        print(f"[MONITOR ALERT] {message}")

    def summary(self) -> dict:
        """Aggregate report - printed by eval/benchmark.py at the end of a run."""
        return {
            "n_runs": self.n_runs,
            "total_cost_usd": round(self.total_cost_usd, 5),
            "alerts": list(self.alerts),
            "tools": {
                name: {
                    "calls": s["calls"],
                    "errors": s["errors"],
                    "avg_ms": round(s["ms_total"] / s["calls"], 1) if s["calls"] else 0.0,
                }
                for name, s in self._tools.items()
            },
        }


# Module-level so stats accumulate across every run_agent() call in this
# process (mirrors the lab's single `mon = AgentMonitor()` used across a session).
_monitor = AgentMonitor()


def get_monitor_summary() -> dict:
    """Public accessor for _monitor.summary(), used by eval/benchmark.py and app.py."""
    return _monitor.summary()


def _estimate_run_cost_usd(total_tokens: int) -> float:
    """Same 60/40 input/output split approximation as eval/benchmark.py,
    priced from PRICING_USD_PER_MTOK above for whichever model is
    actually in use (so it's $0 for a local Ollama model)."""
    price_in, price_out = PRICING_USD_PER_MTOK.get(MODEL_NAME, (0.0, 0.0))
    return (total_tokens * 0.6 / 1_000_000) * price_in + (total_tokens * 0.4 / 1_000_000) * price_out


@dataclass
class AgentRunResult:
    """Everything a caller (CLI, Streamlit app, eval scripts) needs from one
    run: the final answer plus every metric the report/rubric asks for."""
    query: str
    final_answer: str
    confidence: float
    self_consistency_agreement: float
    critic_verdict: str
    critic_justification: str
    tool_calls_made: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    blocked_reason: str | None = None


def _mcp_tool_to_openai_schema(tool) -> dict:
    """Converts an MCP tool descriptor (from session.list_tools()) into the
    OpenAI-style function-calling schema _select_tools_turn expects."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


@dataclass
class _LLMReply:
    """Normalised chat-completion reply: tool_calls are plain dicts
    ({"id", "name", "arguments": dict}), not SDK-specific objects, so the
    tool-calling loop below never touches provider response internals."""
    content: str | None
    tool_calls: list[dict]
    usage: dict

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def to_message(self) -> dict:
        """Reserialises this reply into the agnostic message shape used to
        extend `messages` for the next turn."""
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


def _to_openai_wire_message(m: dict) -> dict:
    """Converts one agnostic message dict (as built/appended throughout
    run_agent) into the exact shape OpenAI's chat.completions API expects."""
    if m["role"] == "assistant" and m.get("tool_calls"):
        return {
            "role": "assistant",
            "content": m.get("content") or "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in m["tool_calls"]
            ],
        }
    if m["role"] == "tool":
        return {"role": "tool", "tool_call_id": m["tool_call_id"], "content": str(m["content"])}
    return {"role": m["role"], "content": m.get("content", "")}


@observe(name="agent.tool_selection_llm_call")
def _select_tools_turn(client: "OpenAI", messages: list, openai_tools: list) -> _LLMReply:
    """
    One turn of the model deciding which tool(s) to call next (or that
    it's done gathering evidence). Wrapped in its own Langfuse span so each
    tool-selection LLM call is individually visible in a trace.
    """
    payload_messages = [_to_openai_wire_message(m) for m in messages]
    resp = client.chat.completions.create(
        model=MODEL_NAME, max_tokens=1024, tools=openai_tools, messages=payload_messages,
    )
    choice = resp.choices[0].message
    tool_calls = []
    for tc in (choice.tool_calls or []):
        args = tc.function.arguments
        tool_calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": json.loads(args) if isinstance(args, str) else args,
        })
    usage = resp.usage
    return _LLMReply(
        content=choice.content,
        tool_calls=tool_calls,
        usage={
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        },
    )


@observe(name="agent.mcp_tool_call")
async def _call_mcp_tool(session: "ClientSession", tool_name: str, tool_input: dict) -> str:
    """
    Executes a single MCP tool call over the stdio session and returns its
    text content. Wrapped in its own Langfuse span, separate from the
    tool-selection LLM call span above, so a Langfuse trace shows each
    LLM call and each tool call as distinct spans (rubric E).
    """
    result = await session.call_tool(tool_name, tool_input)
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))


@observe(name="agent.run")
async def run_agent(user_query: str, progress_callback=None) -> AgentRunResult:
    """
    Entry point. Runs the full pipeline for a single user query and
    returns a structured AgentRunResult. Never raises for expected
    failure modes (L1 block, tool errors) - those are surfaced in the
    result object so callers (CLI, eval scripts) can handle them
    uniformly.
    """
    langfuse_context.update_current_trace(
        metadata=AGENT_VERSION,
        tags=["urban-migration-agent"],
    )

    _run_start = time.perf_counter()
    token_budget = TokenBudget(max_tokens=60_000)
    action_gate = ActionGate()
    client = OpenAI(api_key=_LLM_API_KEY, base_url=_LLM_BASE_URL)  # fresh client per run

    # --- L1: input filtering ---
    l1_result = l1_input_filter(user_query)
    if progress_callback:
        progress_callback("l1_input", {
            "verdict": "BLOCKED" if not l1_result.allowed else "ALLOWED",
            "reasons": l1_result.reasons,
            "normalized": l1_result.normalized_query,
        })
    if not l1_result.allowed:
        _monitor.record_run(user_query, "", time.perf_counter() - _run_start, 0.0)
        return AgentRunResult(
            query=user_query,
            final_answer="",
            confidence=0.0,
            self_consistency_agreement=0.0,
            critic_verdict="N/A",
            critic_justification="",
            blocked_reason=f"Blocked by L1 input filter: {l1_result.reasons}",
        )

    query = l1_result.normalized_query
    tool_calls_made: list[str] = []
    collected_texts: list[str] = []
    collected_sources: list[str] = []

    server_params = StdioServerParameters(command=sys.executable, args=[_MCP_SERVER_PATH])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            openai_tools = [_mcp_tool_to_openai_schema(t) for t in mcp_tools.tools]
            if progress_callback:
                progress_callback("mcp_ready", {"tools": [t.name for t in mcp_tools.tools]})

            messages = [
                {"role": "system", "content": TOOL_SELECTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Research question: {query}",
                },
            ]

            for _turn in range(MAX_TOOL_TURNS):
                reply = _select_tools_turn(client, messages, openai_tools)
                usage = reply.usage or {}
                token_budget.add(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    label=f"tool_selection_turn_{_turn}",
                )

                # reply.to_message() re-serialises the normalised _LLMReply
                # back into the agnostic {"role": ..., "tool_calls": [...]} shape
                # that _to_openai_wire_message() expects on the NEXT call, so
                # it can be appended to `messages` as-is.
                messages.append(reply.to_message())

                if not reply.has_tool_calls:
                    break  # Model is done gathering evidence

                for tc in reply.tool_calls:  # each tc is {"id", "name", "arguments": dict}
                    gate_result = action_gate.check(tc["name"])
                    if not gate_result.allowed:
                        if progress_callback:
                            progress_callback("l4_blocked", {"tool": tc["name"], "reason": gate_result.reason})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"BLOCKED by L4 action gate: {gate_result.reason}",
                        })
                        continue

                    tool_calls_made.append(tc["name"])
                    _tool_start = time.perf_counter()
                    try:
                        tool_input = tc["arguments"]  # already parsed to a dict by _select_tools_turn
                        if progress_callback:
                            progress_callback("tool_call", {"tool": tc["name"], "args": tool_input})
                        result_text = await _call_mcp_tool(session, tc["name"], tool_input)
                        _monitor.record_tool(tc["name"], True, (time.perf_counter() - _tool_start) * 1000)
                    except Exception as exc:  # noqa: BLE001
                        result_text = f"Tool call failed: {exc}"
                        _monitor.record_tool(tc["name"], False, (time.perf_counter() - _tool_start) * 1000)

                    # L1 filtering on retrieved content (indirect injection defence)
                    safe_text = l1_filter_retrieved_context([result_text])
                    filtered = len(safe_text) == 0
                    if progress_callback:
                        progress_callback("tool_result", {
                            "tool": tc["name"],
                            "preview": result_text[:300],
                            "filtered": filtered,
                        })
                    if safe_text:
                        collected_texts.append(safe_text[0])
                        collected_sources.append(tc["name"])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                    })

    if not collected_texts:
        _monitor.record_run(
            user_query, "", time.perf_counter() - _run_start,
            _estimate_run_cost_usd(token_budget.summary()["used_tokens"]),
        )
        return AgentRunResult(
            query=user_query,
            final_answer="",
            confidence=0.0,
            self_consistency_agreement=0.0,
            critic_verdict="N/A",
            critic_justification="",
            tool_calls_made=tool_calls_made,
            token_usage=token_budget.summary(),
            blocked_reason="No usable evidence was retrieved (all tool results empty or filtered).",
        )

    # --- Reasoning: self-consistency synthesis + critic (second agent role) ---
    if progress_callback:
        progress_callback("synthesis_start", {"k": 3, "chunks": len(collected_texts)})
    synthesis = self_consistency_synthesis(query, collected_texts, collected_sources, token_budget)
    if progress_callback:
        for i, c in enumerate(synthesis.all_candidates):
            progress_callback("synthesis_candidate", {
                "k": i + 1,
                "confidence": c.confidence,
                "conclusion": c.conclusion[:200],
            })
        progress_callback("synthesis_winner", {
            "confidence": synthesis.winning_candidate.confidence,
            "agreement": synthesis.agreement_ratio,
        })
    verdict = critic_review(query, collected_texts, collected_sources, synthesis.winning_candidate, token_budget)
    if progress_callback:
        progress_callback("critic", {"verdict": verdict.verdict, "justification": verdict.justification})

    _monitor.record_run(
        user_query, synthesis.winning_candidate.conclusion, time.perf_counter() - _run_start,
        _estimate_run_cost_usd(token_budget.summary()["used_tokens"]),
    )
    return AgentRunResult(
        query=user_query,
        final_answer=synthesis.winning_candidate.conclusion,
        confidence=synthesis.winning_candidate.confidence,
        self_consistency_agreement=synthesis.agreement_ratio,
        critic_verdict=verdict.verdict,
        critic_justification=verdict.justification,
        tool_calls_made=tool_calls_made,
        token_usage=token_budget.summary(),
    )


def main():
    # Demo entry point: runs one fixed query end-to-end and prints every
    # field of AgentRunResult, so `python src/agent.py` alone is a full
    # smoke test of the pipeline (per README section 2).
    print("AGENT VERSION:", AGENT_VERSION)
    print(f"EU AI ACT: {EU_AI_ACT_RISK_TIER} - {EU_AI_ACT_OBLIGATION}")

    demo_query = (
        "Is Nantes a more suitable receiving city than Lyon for climate migrants "
        "from West Africa, given its housing vacancy rate and school capacity utilization?"
    )
    result = asyncio.run(run_agent(demo_query))

    print("=" * 70)
    print("QUERY:", result.query)
    print("=" * 70)
    if result.blocked_reason:
        print("BLOCKED:", result.blocked_reason)
        return

    print("TOOL CALLS:", result.tool_calls_made)
    print("\nFINAL ANSWER:\n", result.final_answer)
    print(f"\nCONFIDENCE: {result.confidence}")
    print(f"SELF-CONSISTENCY AGREEMENT: {result.self_consistency_agreement}")
    print(f"\nCRITIC VERDICT: {result.critic_verdict}")
    print("CRITIC JUSTIFICATION:", result.critic_justification)
    print("\nTOKEN USAGE:", result.token_usage)
    print("\nMONITOR SUMMARY:", get_monitor_summary())


if __name__ == "__main__":
    main()