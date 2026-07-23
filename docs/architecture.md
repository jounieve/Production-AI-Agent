# Architecture - Urban Migration Research Agent

## Diagram

```
                                   ┌─────────────────────────┐
                                   │   User research query    │
                                   └────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌──────────────────────────┐
                                  │ L1 input filter            │  guardrails.py
                                  │ (unicode normalize +       │
                                  │  injection pattern match)  │
                                  └──────────────┬─────────────┘
                                     blocked │      │ allowed
                                             ▼      ▼
                              ┌───────────────┐   ┌──────────────────────────────┐
                              │ Return         │   │ Tool-calling loop (agent.py)  │
                              │ blocked_reason │   │ OpenAI-compatible client <->  │
                              │                │   │ MCP client (stdio); provider  │
                              │                │   │ = OpenAI or local Ollama, one │
                              │                │   │ LLM_PROVIDER config switch    │
                              └───────────────┘   └───────────────┬───────────────┘
                                                                   │ each tool call
                                                                   ▼
                                                     ┌──────────────────────────┐
                                                     │ L4 action gate            │ guardrails.py
                                                     │ (ACTION_RISK_MATRIX)      │
                                                     └─────────────┬─────────────┘
                                                    blocked │        │ allowed
                                                            ▼        ▼
                                              ┌───────────┐  ┌──────────────────────────┐
                                              │ Tool error │  │ mcp_server.py (3 tools)   │
                                              │ surfaced   │  │  1. search_migration_     │
                                              └───────────┘  │     evidence (retrieval.py)│
                                                              │  2. get_city_capacity_    │
                                                              │     profile               │
                                                              │  3. compute_push_pull_    │
                                                              │     index                 │
                                                              └─────────────┬─────────────┘
                                                                            │ tool result text
                                                                            ▼
                                                        ┌──────────────────────────────┐
                                                        │ L1 filter on retrieved content │  guardrails.py
                                                        │ (indirect injection defense)   │
                                                        └───────────────┬───────────────┘
                                                                        ▼
                                                        ┌──────────────────────────────┐
                                                        │ Self-Consistency synthesis     │  reasoning.py
                                                        │ (k=3 independent samples,       │
                                                        │  few-shot CoT: EVIDENCE /       │
                                                        │  ANALYSIS / CONCLUSION /        │
                                                        │  CONFIDENCE, majority vote)      │
                                                        └───────────────┬───────────────┘
                                                                        ▼
                                                        ┌──────────────────────────────┐
                                                        │ Critic agent (2nd agent role)  │  reasoning.py
                                                        │ APPROVED / REJECTED verdict     │
                                                        └───────────────┬───────────────┘
                                                                        ▼
                                                        ┌──────────────────────────────┐
                                                        │ AgentRunResult (final output)  │
                                                        └──────────────────────────────┘

  Cutting across every stage: TokenBudget (guardrails.py) tracks cumulative token
  usage and raises TokenBudgetExceeded if a session exceeds its configured cap -
  an independent circuit breaker regardless of what L1/L4 catch.

  Cutting across retrieval.py and reasoning.py: every stage-level function is
  wrapped in a Langfuse @observe span (bm25_search, dense_search, rerank,
  hybrid_search, llm_call, self_consistency_synthesis, critic_review), plus a
  top-level agent.run span with agent_version in its metadata.
```

## Components

| Component | File | Role |
|---|---|---|
| L1 input filter | `guardrails.py` | Normalizes Unicode, rejects oversized queries, pattern-matches known injection phrasings - runs before anything else touches the query. |
| L1 retrieved-content filter | `guardrails.py` | Same pattern check applied to tool/RAG output, defending against indirect injection planted inside documents. |
| L4 action gate | `guardrails.py` | Consults `ACTION_RISK_MATRIX` before every tool call; fail-closed for unknown tools; enforces per-session call limits; requires explicit allow for high-risk actions. |
| TokenBudget | `guardrails.py` | Cumulative token counter; raises past a configured ceiling - cost control and a second, independent defense against runaway tool-calling loops. |
| Hybrid retriever | `retrieval.py` | Parent-child chunking, BM25 + dense embeddings fused with Reciprocal Rank Fusion, cross-encoder reranking, parent expansion. Also exposes `basic_retrieval()` as the RAGAS baseline. |
| MCP server | `mcp_server.py` | Exposes `search_migration_evidence`, `get_city_capacity_profile`, `compute_push_pull_index` as MCP tools with full docstrings and error handling. |
| LLM provider switch | `agent.py`, `reasoning.py` | `LLM_PROVIDER` env var (`openai` or `ollama`) picks the backend; both speak the same OpenAI-compatible `chat.completions` API, so no per-provider branching is needed in the tool-calling loop itself. |
| Agent loop | `agent.py` | Orchestrates the full pipeline as a real MCP client over stdio; enforces L1/L4 at each step; assembles the final `AgentRunResult`. |
| Production versioning & monitoring | `agent.py` (`AGENT_VERSION`, `hash_prompt`, `AgentMonitor`) | `AGENT_VERSION` includes a SHA-256 hash of the tool-selection system prompt so a behaviour change can be traced to a prompt edit; `AgentMonitor` accumulates run/tool stats across a process and prints an alert on a slow run, expensive run, empty response, or high tool error rate - distinct from the per-call Langfuse spans. |
| EU AI Act classifier | `guardrails.py` (`risk_tier`) | Classifies `agent.AGENT_DESCRIPTION` into a risk tier + obligation from free text; checks for the actual decision-making use case (e.g. border control) rather than the bare topic word, so this research agent doesn't get misclassified as high-risk just for being about migration. |
| Synthesis (reasoning) | `reasoning.py` | Few-shot CoT prompt (`EVIDENCE/ANALYSIS/CONCLUSION/CONFIDENCE`), run k=3 times (Self-Consistency), aggregated by conclusion-similarity clustering. |
| Critic (2nd agent role) | `reasoning.py` | Separate LLM role that checks the winning synthesis against the evidence actually retrieved, flags hallucination/overconfidence, returns a verdict. |

## One design decision worth explaining: why RRF instead of a learned fusion weight

`retrieval.py` fuses the BM25 and dense-embedding rankings with **Reciprocal
Rank Fusion** (`rrf_score(d) = Σ 1 / (k + rank(d))`) rather than a weighted
sum of raw scores (e.g. `0.5 * bm25_score + 0.5 * cosine_score`).

**Trade-off:** a weighted raw-score sum would in principle allow finer
control (e.g. favor the lexical signal more on queries containing an exact
proper noun like a city name). But BM25 scores and cosine similarities live
on completely different, non-comparable scales - combining them with a
fixed weight requires either score normalization (fragile, sensitive to
corpus size and query length) or manual weight-tuning per corpus. RRF sidesteps
this entirely: it depends only on **rank position** in each list, not on the
raw score's magnitude, so it is scale-invariant by construction and requires
no tuning to combine two heterogeneous retrieval methods safely. The cost is
that RRF cannot express "trust the lexical signal 70% here" - it treats a
top-1 BM25 hit and a top-1 dense hit identically. For this project's corpus
size (a few dozen parent chunks) and query style (natural-language questions,
not exact-match lookups), that trade-off favors RRF: robustness against
scale mismatches matters more than fine-grained per-query weighting would.