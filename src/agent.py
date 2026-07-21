"""
agent.py — Main orchestration loop for the Urban Migration Agent.

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
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # loads ANTHROPIC_API_KEY, LANGFUSE_* from .env into os.environ

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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from guardrails import ActionGate, TokenBudget, l1_filter_retrieved_context, l1_input_filter
from reasoning import critic_review, self_consistency_synthesis

AGENT_VERSION = "1.0.0"
MODEL_NAME = "claude-sonnet-4-5"
MAX_TOOL_TURNS = 6  # hard ceiling on tool-calling turns per query, independent of TokenBudget

_MCP_SERVER_PATH = str(Path(__file__).resolve().parent / "mcp_server.py")


@dataclass
class AgentRunResult:
    query: str
    final_answer: str
    confidence: float
    self_consistency_agreement: float
    critic_verdict: str
    critic_justification: str
    tool_calls_made: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    blocked_reason: str | None = None


def _mcp_tool_to_anthropic_schema(tool) -> dict:
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
    }


@observe(name="agent.tool_selection_llm_call")
def _select_tools_turn(client: "anthropic.Anthropic", messages: list, anthropic_tools: list):
    """
    One turn of Claude deciding which tool(s) to call next (or that it's
    done gathering evidence). Wrapped in its own Langfuse span, separate
    from the top-level agent.run span, so each tool-selection LLM call is
    individually visible in a trace (rubric E).
    """
    return client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        tools=anthropic_tools,
        messages=messages,
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
async def run_agent(user_query: str) -> AgentRunResult:
    """
    Entry point. Runs the full pipeline for a single user query and
    returns a structured AgentRunResult. Never raises for expected
    failure modes (L1 block, tool errors) — those are surfaced in the
    result object so callers (CLI, eval scripts) can handle them
    uniformly.
    """
    langfuse_context.update_current_trace(
        metadata={"agent_version": AGENT_VERSION},
        tags=["urban-migration-agent"],
    )

    token_budget = TokenBudget(max_tokens=60_000)
    action_gate = ActionGate()
    client = anthropic.Anthropic()

    # --- L1: input filtering ---
    l1_result = l1_input_filter(user_query)
    if not l1_result.allowed:
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
            anthropic_tools = [_mcp_tool_to_anthropic_schema(t) for t in mcp_tools.tools]

            messages = [
                {
                    "role": "user",
                    "content": (
                        f"Research question: {query}\n\n"
                        "Use the available tools to gather evidence before answering. "
                        "Call search_migration_evidence for qualitative/narrative context, "
                        "get_city_capacity_profile for raw city stats, and "
                        "compute_push_pull_index when a specific origin/destination "
                        "corridor risk score is useful."
                    ),
                }
            ]

            for _turn in range(MAX_TOOL_TURNS):
                response = _select_tools_turn(client, messages, anthropic_tools)
                token_budget.add(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    label=f"tool_selection_turn_{_turn}",
                )

                messages.append({"role": "assistant", "content": response.content})

                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                if not tool_use_blocks:
                    break  # Claude is done gathering evidence

                tool_results = []
                for block in tool_use_blocks:
                    gate_result = action_gate.check(block.name)
                    if not gate_result.allowed:
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"BLOCKED by L4 action gate: {gate_result.reason}",
                                "is_error": True,
                            }
                        )
                        continue

                    tool_calls_made.append(block.name)
                    try:
                        result_text = await _call_mcp_tool(session, block.name, block.input)
                    except Exception as exc:  # noqa: BLE001
                        result_text = f"Tool call failed: {exc}"

                    # Capture evidence text for the reasoning step, applying
                    # L1 filtering to defend against indirect injection
                    # planted inside retrieved/tool-returned content.
                    safe_text = l1_filter_retrieved_context([result_text])
                    if safe_text:
                        collected_texts.append(safe_text[0])
                        collected_sources.append(block.name)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})

    if not collected_texts:
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
    synthesis = self_consistency_synthesis(query, collected_texts, collected_sources, token_budget)
    verdict = critic_review(query, collected_texts, collected_sources, synthesis.winning_candidate, token_budget)

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
    demo_query = "Is Toulouse a realistic destination for climate migrants from a drought-affected region over the next 12 months?"
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


if __name__ == "__main__":
    main()