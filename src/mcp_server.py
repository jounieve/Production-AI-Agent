"""
mcp_server.py — MCP server exposing the Urban Migration Agent's tools.

Three tools:
  1. search_migration_evidence   — hybrid RAG search over the research corpus
  2. get_city_capacity_profile   — structured lookup of a receiving city's data
  3. compute_push_pull_index     — computed push/pull score for an origin/destination pair

Run standalone for local testing / MCP inspector:
    python src/mcp_server.py

Every tool catches its own exceptions and returns a structured error
dict instead of letting exceptions propagate uncaught, per rubric B.
"""

from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from retrieval import HybridRetriever

mcp = FastMCP("urban-migration-agent")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CORPUS_DIR = _DATA_DIR / "corpus"
_CITIES_PATH = _DATA_DIR / "cities" / "cities.json"

# Retriever is expensive to build (loads embedding + cross-encoder models),
# so it's built once, lazily, on first use rather than at import time.
_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(_CORPUS_DIR)
    return _retriever


def _load_cities() -> dict:
    with open(_CITIES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {city["name"].lower(): city for city in data["cities"]}


# --------------------------------------------------------------------------
# Tool 1: search_migration_evidence
# --------------------------------------------------------------------------

@mcp.tool()
def search_migration_evidence(query: str) -> dict:
    """
    Searches the research corpus (academic reports, articles on
    migration push/pull factors, climate displacement, and receiving
    city capacity) using hybrid search (BM25 + dense embeddings + RRF
    fusion) followed by cross-encoder reranking, and returns the most
    relevant parent-level text passages.

    Use when: you need qualitative evidence, definitions, mechanisms,
    or research findings about migration drivers, corridors, or
    capacity concepts (e.g. "why do social networks affect destination
    choice", "what counts as a tight housing market").

    Do NOT use when: you need a specific numeric value for a named city
    (use get_city_capacity_profile instead) or a computed comparison
    score between an origin and a destination (use
    compute_push_pull_index instead). This tool returns text passages,
    not numbers.

    Returns: a dict with:
      - "results": list of {"text": str, "source": str, "score": float},
        ordered by relevance (highest score first)
      - "query": the original query echoed back
      On failure: {"error": <message>, "results": []}

    Example:
      search_migration_evidence("what makes a city's housing market absorb migrants well")
      -> {
           "query": "...",
           "results": [
             {"text": "Vacancy rates below 3% are generally considered...",
              "source": "receiving_city_capacity.md", "score": 4.82},
             ...
           ]
         }
    """
    try:
        if not query or not query.strip():
            return {"error": "query must be a non-empty string", "results": []}

        retriever = _get_retriever()
        hits = retriever.search(query.strip())
        return {
            "query": query,
            "results": [
                {"text": h.text, "source": h.source, "score": round(h.score, 4)}
                for h in hits
            ],
        }
    except Exception as exc:  # noqa: BLE001 — intentionally broad: tools must never crash the server
        return {"error": f"search_migration_evidence failed: {exc}", "results": []}


# --------------------------------------------------------------------------
# Tool 2: get_city_capacity_profile
# --------------------------------------------------------------------------

@mcp.tool()
def get_city_capacity_profile(city_name: str) -> dict:
    """
    Looks up structured capacity data for a named receiving city:
    housing vacancy rate, job growth rate, industry diversity index,
    school capacity utilization, healthcare beds per 1000 residents,
    and public transit coverage index.

    Use when: you need concrete, current numeric indicators for a
    SPECIFIC named city (e.g. "what is Lyon's housing vacancy rate").

    Do NOT use when: the question is about migration concepts in
    general (use search_migration_evidence) or requires combining a
    city's capacity with an origin region's push factors into a single
    score (use compute_push_pull_index).

    Returns: a dict with:
      - "city": the matched city's full data dict, if found
      - "found": bool
      On failure or city not found: {"error": <message>, "found": False}

    Example:
      get_city_capacity_profile("Toulouse")
      -> {
           "found": true,
           "city": {
             "name": "Toulouse", "country": "France", "population": 493000,
             "housing_vacancy_rate": 0.039, "job_growth_rate_yoy": 0.034,
             ...
           }
         }
    """
    try:
        if not city_name or not city_name.strip():
            return {"error": "city_name must be a non-empty string", "found": False}

        cities = _load_cities()
        match = cities.get(city_name.strip().lower())

        if match is None:
            available = ", ".join(c["name"] for c in cities.values())
            return {
                "error": f"city '{city_name}' not found. Available cities: {available}",
                "found": False,
            }

        return {"found": True, "city": match}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"get_city_capacity_profile failed: {exc}", "found": False}


# --------------------------------------------------------------------------
# Tool 3: compute_push_pull_index
# --------------------------------------------------------------------------

# Simplified push-factor severity scores by category, on a 0-1 scale.
# In a production system these would come from a live data feed (drought
# indices, conflict trackers, unemployment stats); here they are
# illustrative constants the agent can reason over, clearly documented
# as such in REPORT.md's limitations section.
_PUSH_FACTOR_SEVERITY = {
    "climate": 0.75,
    "economic": 0.55,
    "conflict": 0.85,
}

# Weights for the composite capacity score, matching the weighting logic
# described in data/corpus/receiving_city_capacity.md: housing and labor
# market capacity weighted more heavily than public services.
_CAPACITY_WEIGHTS = {
    "housing_vacancy_rate": 0.35,
    "job_growth_rate_yoy": 0.30,
    "school_capacity_utilization": -0.15,  # higher utilization = less spare capacity
    "healthcare_beds_per_1000": 0.10,
    "public_transit_coverage_index": 0.10,
}


def _normalize(value: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.0


@mcp.tool()
def compute_push_pull_index(origin_region: str, destination_city: str, push_factor_type: str = "climate") -> dict:
    """
    Computes a combined push/pull index for a specific origin-region ->
    destination-city pair: a push severity score for the origin (based
    on the declared push_factor_type) and a composite pull/capacity
    score for the destination (based on its structured city data),
    returned together with an overall corridor risk assessment.

    Use when: the user asks to compare or evaluate a SPECIFIC migration
    corridor (e.g. "how strained would Lyon become if 3,000 people left
    a drought-affected region for it") and wants a single computed
    score rather than narrative evidence or raw city stats alone.

    Do NOT use when: no destination city is named (use
    search_migration_evidence for general corridor concepts instead) or
    when only raw city stats are needed (use
    get_city_capacity_profile).

    Args:
      origin_region: free-text name of the origin region (used for
        labeling only; push severity currently comes from
        push_factor_type, not from the region name itself — see
        limitations in REPORT.md)
      destination_city: must match a city in get_city_capacity_profile
      push_factor_type: one of "climate", "economic", "conflict"

    Returns: a dict with:
      - "origin_region", "destination_city", "push_factor_type"
      - "push_severity": float 0-1
      - "destination_capacity_score": float 0-1 (higher = more absorption capacity)
      - "corridor_risk": "low" | "medium" | "high"
      On failure: {"error": <message>}

    Example:
      compute_push_pull_index("Sahel region", "Toulouse", "climate")
      -> {
           "origin_region": "Sahel region", "destination_city": "Toulouse",
           "push_factor_type": "climate", "push_severity": 0.75,
           "destination_capacity_score": 0.61, "corridor_risk": "medium"
         }
    """
    try:
        if not origin_region or not destination_city:
            return {"error": "origin_region and destination_city are both required"}

        if push_factor_type not in _PUSH_FACTOR_SEVERITY:
            return {
                "error": (
                    f"push_factor_type must be one of {list(_PUSH_FACTOR_SEVERITY)}, "
                    f"got '{push_factor_type}'"
                )
            }

        cities = _load_cities()
        city = cities.get(destination_city.strip().lower())
        if city is None:
            return {"error": f"destination_city '{destination_city}' not found in city data"}

        push_severity = _PUSH_FACTOR_SEVERITY[push_factor_type]

        capacity_score = 0.5  # baseline
        capacity_score += _CAPACITY_WEIGHTS["housing_vacancy_rate"] * _normalize(city["housing_vacancy_rate"], 0.0, 0.10)
        capacity_score += _CAPACITY_WEIGHTS["job_growth_rate_yoy"] * _normalize(city["job_growth_rate_yoy"], -0.02, 0.05)
        capacity_score += _CAPACITY_WEIGHTS["school_capacity_utilization"] * _normalize(city["school_capacity_utilization"], 0.7, 1.0)
        capacity_score += _CAPACITY_WEIGHTS["healthcare_beds_per_1000"] * _normalize(city["healthcare_beds_per_1000"], 3.0, 8.0)
        capacity_score += _CAPACITY_WEIGHTS["public_transit_coverage_index"] * _normalize(city["public_transit_coverage_index"], 0.5, 1.0)
        capacity_score = max(0.0, min(1.0, capacity_score))

        # Risk = high push severity meeting low capacity.
        strain = push_severity * (1 - capacity_score)
        if strain >= 0.5:
            corridor_risk = "high"
        elif strain >= 0.25:
            corridor_risk = "medium"
        else:
            corridor_risk = "low"

        return {
            "origin_region": origin_region,
            "destination_city": city["name"],
            "push_factor_type": push_factor_type,
            "push_severity": round(push_severity, 3),
            "destination_capacity_score": round(capacity_score, 3),
            "corridor_risk": corridor_risk,
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": f"compute_push_pull_index failed: {exc}"}


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")