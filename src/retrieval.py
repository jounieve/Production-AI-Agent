"""
retrieval.py — Hybrid retrieval pipeline for the Urban Migration Agent.

Pipeline:
    1. Parent-child chunking of the corpus (children = small, searchable
       chunks; parents = larger context blocks returned to the LLM).
    2. Hybrid search over the CHILD chunks: BM25 (lexical) + dense
       embeddings (semantic), fused with Reciprocal Rank Fusion (RRF).
    3. Cross-encoder reranking of the fused candidates.
    4. Expansion from reranked child chunks -> their parent chunks before
       returning final context.

A `basic_retrieval()` function (plain top-k cosine similarity over
un-chunked documents) is also provided so RAGAS can measure a
"baseline" score to compare against the full hybrid+rerank pipeline
(see eval/ragas_eval.py). This baseline/final split is required by
rubric F.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# Langfuse span decorator. If langfuse isn't configured (e.g. running
# retrieval.py standalone / in CI), fall back to a no-op decorator so the
# module still imports cleanly.
try:
    from langfuse.decorators import observe
except ImportError:  # pragma: no cover
    def observe(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PARENT_CHUNK_SIZE = 1200      # chars, the block returned to the LLM
CHILD_CHUNK_SIZE = 300        # chars, the block that gets embedded/indexed
CHILD_CHUNK_OVERLAP = 50

RRF_K = 60                    # standard RRF damping constant
TOP_K_LEXICAL = 20
TOP_K_DENSE = 20
TOP_K_FUSED = 15              # candidates handed to the cross-encoder
TOP_K_FINAL = 5               # parent chunks returned after reranking


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class ParentChunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    id: str
    text: str
    parent_id: str
    source: str


@dataclass
class RetrievedContext:
    parent_id: str
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Simple sliding-window character splitter on whitespace boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # try not to cut mid-word
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c]


def build_parent_child_index(corpus_dir: str | Path) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """
    Reads every .txt/.md file in corpus_dir and produces:
      - parent chunks (larger context blocks, ~PARENT_CHUNK_SIZE chars)
      - child chunks (small blocks, ~CHILD_CHUNK_SIZE chars) each linked
        to the parent it was carved from via parent_id.

    Child chunks are what gets embedded and indexed for search;
    parent chunks are what gets returned to the LLM as context.
    """
    corpus_dir = Path(corpus_dir)
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for path in sorted(corpus_dir.glob("**/*")):
        if path.suffix.lower() not in (".txt", ".md") or not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")

        for parent_text in _split_text(raw, PARENT_CHUNK_SIZE, overlap=0):
            parent_id = str(uuid.uuid4())
            parents.append(
                ParentChunk(id=parent_id, text=parent_text, source=path.name)
            )
            for child_text in _split_text(parent_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP):
                children.append(
                    ChildChunk(
                        id=str(uuid.uuid4()),
                        text=child_text,
                        parent_id=parent_id,
                        source=path.name,
                    )
                )

    return parents, children


# --------------------------------------------------------------------------
# Hybrid retriever
# --------------------------------------------------------------------------

class HybridRetriever:
    """
    Owns the BM25 index, dense embeddings, and cross-encoder for the
    child-chunk corpus, plus the parent-chunk lookup table used to expand
    results before they're handed to the reasoning module.
    """

    def __init__(self, corpus_dir: str | Path):
        self.parents, self.children = build_parent_child_index(corpus_dir)
        self._parent_by_id = {p.id: p for p in self.parents}

        if not self.children:
            raise ValueError(
                f"No .txt/.md documents found in {corpus_dir}. "
                "Populate data/corpus/ before running retrieval (see data/README.md)."
            )

        # --- lexical index (BM25) ---
        self._tokenized_children = [self._tokenize(c.text) for c in self.children]
        self.bm25 = BM25Okapi(self._tokenized_children)

        # --- dense index ---
        self.dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        self._child_embeddings = self.dense_model.encode(
            [c.text for c in self.children],
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # --- cross-encoder reranker ---
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    # ---------------------------------------------------------------
    # Stage 1: lexical (BM25)
    # ---------------------------------------------------------------
    @observe(name="retrieval.bm25_search")
    def _bm25_search(self, query: str, top_k: int = TOP_K_LEXICAL) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked]

    # ---------------------------------------------------------------
    # Stage 1: dense (semantic)
    # ---------------------------------------------------------------
    @observe(name="retrieval.dense_search")
    def _dense_search(self, query: str, top_k: int = TOP_K_DENSE) -> list[tuple[int, float]]:
        q_emb = self.dense_model.encode([query], normalize_embeddings=True)[0]
        sims = self._child_embeddings @ q_emb
        ranked = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in ranked]

    # ---------------------------------------------------------------
    # Stage 2: Reciprocal Rank Fusion
    # ---------------------------------------------------------------
    @staticmethod
    def _rrf_fuse(
        rank_lists: list[list[tuple[int, float]]], k: int = RRF_K
    ) -> list[tuple[int, float]]:
        """
        Combines several ranked lists of (index, score) into a single
        ranking using Reciprocal Rank Fusion: rrf_score(d) = sum(1 / (k + rank(d)))
        over every list the document appears in. Fusion depends only on
        rank position, not on raw score scale, which is what makes it
        safe to combine BM25 scores with cosine similarities.
        """
        fused: dict[int, float] = {}
        for rank_list in rank_lists:
            for rank, (idx, _score) in enumerate(rank_list):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # ---------------------------------------------------------------
    # Stage 3: cross-encoder reranking
    # ---------------------------------------------------------------
    @observe(name="retrieval.rerank")
    def _rerank(
        self, query: str, candidate_indices: list[int], top_k: int = TOP_K_FINAL
    ) -> list[tuple[int, float]]:
        pairs = [[query, self.children[i].text] for i in candidate_indices]
        scores = self.cross_encoder.predict(pairs)
        order = np.argsort(scores)[::-1][:top_k]
        return [(candidate_indices[i], float(scores[i])) for i in order]

    # ---------------------------------------------------------------
    # Public entry point: full hybrid + rerank + parent expansion
    # ---------------------------------------------------------------
    @observe(name="retrieval.hybrid_search")
    def search(self, query: str, top_k: int = TOP_K_FINAL) -> list[RetrievedContext]:
        bm25_hits = self._bm25_search(query)
        dense_hits = self._dense_search(query)

        fused = self._rrf_fuse([bm25_hits, dense_hits])
        fused_indices = [idx for idx, _ in fused[:TOP_K_FUSED]]

        reranked = self._rerank(query, fused_indices, top_k=top_k)

        # Expand child -> parent, de-duplicating parents that were hit by
        # more than one of their children.
        seen_parents: set[str] = set()
        results: list[RetrievedContext] = []
        for child_idx, score in reranked:
            child = self.children[child_idx]
            if child.parent_id in seen_parents:
                continue
            seen_parents.add(child.parent_id)
            parent = self._parent_by_id[child.parent_id]
            results.append(
                RetrievedContext(
                    parent_id=parent.id,
                    text=parent.text,
                    source=parent.source,
                    score=score,
                )
            )
        return results

    # ---------------------------------------------------------------
    # Baseline for RAGAS comparison: plain top-k cosine similarity,
    # no BM25, no RRF, no reranking, no parent-child expansion.
    # Required by rubric F to establish a "before Block 1 improvements"
    # score.
    # ---------------------------------------------------------------
    @observe(name="retrieval.basic_retrieval")
    def basic_retrieval(self, query: str, top_k: int = TOP_K_FINAL) -> list[RetrievedContext]:
        q_emb = self.dense_model.encode([query], normalize_embeddings=True)[0]
        sims = self._child_embeddings @ q_emb
        ranked = np.argsort(sims)[::-1][:top_k]
        return [
            RetrievedContext(
                parent_id=self.children[i].parent_id,
                text=self.children[i].text,  # raw child text, no parent expansion
                source=self.children[i].source,
                score=float(sims[i]),
            )
            for i in ranked
        ]


# --------------------------------------------------------------------------
# Manual smoke test
# --------------------------------------------------------------------------

if __name__ == "__main__":
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "corpus"
    retriever = HybridRetriever(corpus_path)

    test_query = "housing capacity for climate migrants in receiving cities"

    print("=== Hybrid + rerank + parent-child ===")
    for r in retriever.search(test_query):
        print(f"[{r.score:.3f}] ({r.source}) {r.text[:120]}...")

    print("\n=== Basic retrieval (baseline) ===")
    for r in retriever.basic_retrieval(test_query):
        print(f"[{r.score:.3f}] ({r.source}) {r.text[:120]}...")