"""
reasoning.py - Reasoning strategy for the Urban Migration Agent.

Two LLM-driven roles live here:

  1. Synthesis role (`self_consistency_synthesis`) - takes the user
     query + retrieved context and produces a structured answer using
     few-shot Chain-of-Thought in the format:
         EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE
     This is run k=3 times independently (Self-Consistency) and the
     three candidates are aggregated by majority vote on CONCLUSION,
     with CONFIDENCE used as a tie-breaker.

  2. Critic role (`critic_review`) - a SEPARATE agent role (satisfies
     the brief's "second agent" requirement) that checks the winning
     synthesis against the retrieved evidence before it's returned to
     the user, and produces a visible APPROVED / REJECTED verdict.

Both roles log token usage into the shared TokenBudget (guardrails.py)
and emit Langfuse spans.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

try:
    from langfuse.decorators import observe
except ImportError:  # pragma: no cover
    def observe(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator

from guardrails import TokenBudget

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# Same provider config switch as agent.py (LLM_PROVIDER in .env): both
# supported providers speak the OpenAI-compatible chat.completions API, so
# only the base_url/api_key differ.
_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_HAS_REAL_OPENAI_KEY = bool(_OPENAI_API_KEY) and not _OPENAI_API_KEY.startswith("sk-...")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
if not LLM_PROVIDER:
    LLM_PROVIDER = "openai" if _HAS_REAL_OPENAI_KEY else "ollama"

if LLM_PROVIDER == "openai":
    MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _client = OpenAI(api_key=_OPENAI_API_KEY)
elif LLM_PROVIDER == "ollama":
    MODEL_NAME = os.getenv("LLM_MODEL", "llama3.2:latest")
    _client = OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL)
else:
    raise ValueError(f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Use 'openai' or 'ollama'.")

SELF_CONSISTENCY_K = 3


# --------------------------------------------------------------------------
# Few-shot synthesis prompt
# --------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """You are the synthesis component of an urban migration research agent.
You are given a user question and a set of retrieved evidence passages
(from research documents and/or structured city data). You must reason
about the evidence and produce a structured answer in EXACTLY this
format, with all four sections present and in this order:

EVIDENCE: <list the specific facts from the provided context that are
relevant to the question, citing which source each comes from>
ANALYSIS: <reason step by step about what the evidence implies, how
different pieces of evidence interact, and any tension or gaps between
them>
CONCLUSION: <a direct, concrete answer to the user's question>
CONFIDENCE: <a single number between 0.0 and 1.0 representing your
confidence in the CONCLUSION, followed by a one-line justification for
that confidence level>

Never state a fact in EVIDENCE that does not appear in the provided
context. If the context is insufficient to answer confidently, say so
explicitly and lower your CONFIDENCE accordingly rather than guessing.

--- EXAMPLE 1 ---
Question: Is Toulouse a good candidate for absorbing 5,000 climate
migrants from a drought-affected region in the next 12 months?

Context:
[city_profile:Toulouse] housing_vacancy_rate=0.039, job_growth_rate=0.034,
school_capacity_utilization=0.90
[doc:receiving_city_capacity.md] Vacancy rates below 3% are considered a
tight housing market... school enrollment capacity is a commonly used
proxy for public service absorption capacity.

EVIDENCE: Toulouse has a housing vacancy rate of 3.9% (city_profile),
which sits just above the "tight market" threshold of 3% described in
receiving_city_capacity.md. Job growth is comparatively strong at 3.4%
year-over-year (city_profile). School capacity utilization is already
high at 90% (city_profile).
ANALYSIS: The housing market is not critically tight but has little
slack - a 5,000-person influx would likely push vacancy below the 3%
threshold within months, especially given the strong job growth already
attracting other new residents. The 90% school utilization leaves very
little room for new enrollment, which receiving_city_capacity.md flags
as a public-service capacity constraint independent of housing.
CONCLUSION: Toulouse can likely absorb the migrants economically (strong
job growth) but is a weak candidate on housing and school capacity
without prior infrastructure investment; a phased arrival over 18-24
months would materially reduce strain compared to a 12-month timeline.
CONFIDENCE: 0.72 - city-level data is specific and directly relevant,
but the analysis lacks information on planned school capacity expansion
or migrant skill profiles, which would sharpen the estimate.

--- EXAMPLE 2 ---
Question: What role do social networks play in migration destination
choice?

Context:
[doc:push_pull_factors.md] Social networks matter enormously - migrants
disproportionately move to cities where family members or people from
their home region have already settled, because these networks lower
the cost of arrival.

EVIDENCE: push_pull_factors.md states that migrants disproportionately
choose destinations where existing social networks (family, people from
the same home region) are already present.
ANALYSIS: The document frames this as a cost-reduction mechanism -
networks lower the cost of arrival by providing housing leads, job
referrals, and language support, rather than purely being an emotional
or cultural preference.
CONCLUSION: Social networks function as a pull factor by reducing the
practical costs of arrival (housing, jobs, language), which makes
existing diaspora concentration a strong predictor of destination choice
independent of a city's raw economic indicators.
CONFIDENCE: 0.65 - the mechanism is clearly described, but only one
source is available in the provided context, so this should not be
treated as a comprehensive account of destination choice.
--- END EXAMPLES ---

Now answer the actual question using only the context provided below.
"""


@dataclass
class SynthesisCandidate:
    raw_text: str
    evidence: str
    analysis: str
    conclusion: str
    confidence: float


@dataclass
class SynthesisResult:
    winning_candidate: SynthesisCandidate
    all_candidates: list[SynthesisCandidate]
    agreement_ratio: float  # fraction of the k candidates that agreed with the winner


def _parse_structured_response(text: str) -> SynthesisCandidate:
    """Extracts the four required sections from a model response."""

    def _extract(section: str, next_sections: list[str]) -> str:
        """Regex-slices the text between `section:` and whichever of
        next_sections appears first (or end-of-string if none given)."""
        if next_sections:
            pattern = rf"{section}:\s*(.*?)(?=" + "|".join(f"{s}:" for s in next_sections) + r")"
        else:
            pattern = rf"{section}:\s*(.*)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else ""

    evidence = _extract("EVIDENCE", ["ANALYSIS", "CONCLUSION", "CONFIDENCE"])
    analysis = _extract("ANALYSIS", ["CONCLUSION", "CONFIDENCE"])
    conclusion = _extract("CONCLUSION", ["CONFIDENCE"])
    confidence_block = _extract("CONFIDENCE", [])

    confidence_match = re.search(r"(\d\.\d+|\d)", confidence_block)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    confidence = max(0.0, min(1.0, confidence))

    return SynthesisCandidate(
        raw_text=text,
        evidence=evidence,
        analysis=analysis,
        conclusion=conclusion,
        confidence=confidence,
    )


@observe(name="reasoning.llm_call")
def _call_llm(system_prompt: str, user_prompt: str, token_budget: TokenBudget, label: str) -> str:
    """
    Single non-tool-calling completion against whichever provider
    LLM_PROVIDER resolved to (OpenAI or Ollama, both via the OpenAI SDK).
    Every call's usage feeds the shared TokenBudget.
    """
    response = _client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    usage = response.usage
    token_budget.add(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        label=label,
    )
    return response.choices[0].message.content or ""


def _build_context_block(context_chunks: list[str], sources: list[str]) -> str:
    """Formats retrieved chunks as `[doc:source] text` lines, so the LLM's
    EVIDENCE section can cite which source each fact came from."""
    lines = []
    for text, source in zip(context_chunks, sources):
        lines.append(f"[doc:{source}] {text}")
    return "\n\n".join(lines)


def _cluster_by_conclusion_similarity(candidates: list[SynthesisCandidate]) -> list[list[int]]:
    """
    Very lightweight agreement clustering: groups candidate indices whose
    CONCLUSION text shares a high proportion of overlapping significant
    words. This avoids requiring an extra embedding call just to compare
    3 short strings, while still catching "these two candidates actually
    agree" beyond exact string match.
    """
    def _keywords(text: str) -> set[str]:
        """Words longer than 4 chars, lowercased - a cheap proxy for the
        "significant" (non-stopword-ish) terms in a CONCLUSION."""
        return set(w for w in re.findall(r"\w+", text.lower()) if len(w) > 4)

    keyword_sets = [_keywords(c.conclusion) for c in candidates]
    clusters: list[list[int]] = []
    assigned = [False] * len(candidates)

    for i in range(len(candidates)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(candidates)):
            if assigned[j]:
                continue
            union = keyword_sets[i] | keyword_sets[j]
            overlap = keyword_sets[i] & keyword_sets[j]
            similarity = len(overlap) / len(union) if union else 0.0
            if similarity >= 0.25:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    return clusters


@observe(name="reasoning.self_consistency_synthesis")
def self_consistency_synthesis(
    question: str,
    context_chunks: list[str],
    sources: list[str],
    token_budget: TokenBudget,
    k: int = SELF_CONSISTENCY_K,
) -> SynthesisResult:
    """
    Runs the synthesis prompt k independent times (temperature-driven
    variation comes from the API default; each call is an independent
    sample) and aggregates by majority vote over conclusion clusters.
    Within the winning cluster, the candidate with the highest
    self-reported CONFIDENCE is returned as the representative answer.
    """
    context_block = _build_context_block(context_chunks, sources)
    user_prompt = f"Question: {question}\n\nContext:\n{context_block}"

    candidates: list[SynthesisCandidate] = []
    for i in range(k):
        raw = _call_llm(
            SYNTHESIS_SYSTEM_PROMPT,
            user_prompt,
            token_budget,
            label=f"synthesis_sample_{i+1}",
        )
        candidates.append(_parse_structured_response(raw))

    clusters = _cluster_by_conclusion_similarity(candidates)
    winning_cluster = max(clusters, key=len)
    agreement_ratio = len(winning_cluster) / len(candidates)

    winner_idx = max(winning_cluster, key=lambda i: candidates[i].confidence)
    winning_candidate = candidates[winner_idx]

    return SynthesisResult(
        winning_candidate=winning_candidate,
        all_candidates=candidates,
        agreement_ratio=agreement_ratio,
    )


# --------------------------------------------------------------------------
# Critic agent - the required "second agent role"
# --------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """You are the critic component of an urban migration research agent.
You are given the original question, the evidence context that was
retrieved, and a candidate answer produced by a separate synthesis
agent. Your job is NOT to re-answer the question. Your job is to check
the candidate answer for two failure modes:

1. Hallucination - does the CONCLUSION or EVIDENCE section state a
   specific fact (a number, a causal claim, a named statistic) that
   does NOT appear anywhere in the provided context? Minor phrasing
   differences or reasonable numerical inferences (e.g. "approaching
   90% threshold" for a value of 88%) are NOT hallucinations.
2. Overconfidence - is the stated CONFIDENCE above 0.9 when the
   retrieved context is thin or indirect? A confidence of 0.7–0.9
   backed by city-profile data and relevant corpus passages is
   acceptable; do NOT flag it unless the evidence is clearly
   insufficient for the claim.

Default to APPROVED. Only use REJECTED if you can cite a specific
sentence in the CONCLUSION that directly contradicts or is completely
absent from the provided context.

Respond in EXACTLY this format:

VERDICT: <APPROVED or REJECTED>
JUSTIFICATION: <one to three sentences explaining the verdict, citing
the specific claim that is unsupported if REJECTED>
"""


@dataclass
class CriticVerdict:
    verdict: str  # "APPROVED" or "REJECTED"
    justification: str


@observe(name="reasoning.critic_review")
def critic_review(
    question: str,
    context_chunks: list[str],
    sources: list[str],
    candidate: SynthesisCandidate,
    token_budget: TokenBudget,
) -> CriticVerdict:
    """Second agent role (rubric requirement): re-checks the winning synthesis
    candidate against the same evidence for hallucination/overconfidence and
    returns an APPROVED/REJECTED verdict - never re-answers the question itself."""
    context_block = _build_context_block(context_chunks, sources)
    user_prompt = (
        f"Question: {question}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Candidate answer:\n{candidate.raw_text}"
    )

    raw = _call_llm(
        CRITIC_SYSTEM_PROMPT,
        user_prompt,
        token_budget,
        label="critic_review",
    )

    verdict_match = re.search(r"VERDICT:\s*(APPROVED|REJECTED)", raw, re.IGNORECASE)
    justification_match = re.search(r"JUSTIFICATION:\s*(.*)", raw, re.DOTALL)

    return CriticVerdict(
        verdict=verdict_match.group(1).upper() if verdict_match else "REJECTED",
        justification=justification_match.group(1).strip() if justification_match else "Critic response could not be parsed.",
    )


# --------------------------------------------------------------------------
# Manual smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    budget = TokenBudget(max_tokens=20_000)

    demo_chunks = [
        "Vacancy rates below 3% are generally considered a tight housing market; "
        "cities in this range absorb new arrivals slowly.",
        "Toulouse housing_vacancy_rate=0.039, job_growth_rate_yoy=0.034, "
        "school_capacity_utilization=0.90",
    ]
    demo_sources = ["receiving_city_capacity.md", "cities.json"]
    question = "Is Toulouse a good candidate for absorbing 5,000 climate migrants in 12 months?"

    result = self_consistency_synthesis(question, demo_chunks, demo_sources, budget)
    print("=== Winning candidate ===")
    print("CONCLUSION:", result.winning_candidate.conclusion)
    print("CONFIDENCE:", result.winning_candidate.confidence)
    print("Agreement ratio:", result.agreement_ratio)

    verdict = critic_review(question, demo_chunks, demo_sources, result.winning_candidate, budget)
    print("\n=== Critic verdict ===")
    print(verdict.verdict, "-", verdict.justification)

    print("\n=== Token budget summary ===")
    print(budget.summary())