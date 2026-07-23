"""
eval/ragas_eval.py - RAGAS evaluation: baseline vs final pipeline.

"Baseline" = retrieval.basic_retrieval() (plain top-k cosine similarity,
no BM25, no RRF, no reranking, no parent-child expansion) + a direct
zero-shot answer (no few-shot CoT, no Self-Consistency).

"Final" = retrieval.HybridRetriever.search() (BM25 + dense + RRF fusion
+ cross-encoder reranking + parent-child expansion) + the full
reasoning.self_consistency_synthesis() pipeline (few-shot CoT,
EVIDENCE/ANALYSIS/CONCLUSION/CONFIDENCE format, k=3).

This isolates which techniques caused which RAGAS metric changes:
  - context_recall / context_precision changes -> attributable to the
    RETRIEVAL technique (hybrid+rerank+parent-child vs basic cosine)
  - faithfulness / answer_relevancy changes -> attributable to BOTH
    retrieval quality AND the reasoning technique (few-shot CoT +
    self-consistency vs direct zero-shot answer)

Run with:
    python eval/ragas_eval.py

Requires OPENAI_API_KEY in .env (used both as the agent's answer
generator and, via ChatOpenAI, as RAGAS's judge LLM).

Outputs eval/ragas_results.json with the full baseline vs final table,
plus a printed summary matching the REPORT.md section 3 table format.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from guardrails import TokenBudget
from reasoning import self_consistency_synthesis
from retrieval import HybridRetriever

load_dotenv()

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_QUESTIONS_PATH = _DATA_DIR / "eval_questions.json"
_CORPUS_DIR = _DATA_DIR / "corpus"
_OUTPUT_PATH = Path(__file__).resolve().parent / "ragas_results.json"

_client = OpenAI()


def _load_questions() -> list[dict]:
    """Loads the question/ground_truth pairs used for both baseline and final runs."""
    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def _baseline_answer(question: str, contexts: list[str]) -> str:
    """
    Direct zero-shot answer, no few-shot examples, no CoT structure, no
    Self-Consistency. Represents the "before Block 1 improvements"
    generation strategy, paired with basic_retrieval() context.
    """
    context_block = "\n\n".join(contexts)
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Answer the question using only this context.\n\n"
                    f"Context:\n{context_block}\n\nQuestion: {question}"
                ),
            }
        ],
    )
    return response.choices[0].message.content or ""


def _run_pipeline(questions: list[dict], retriever: HybridRetriever, mode: str) -> Dataset:
    """mode is 'baseline' or 'final'."""
    records = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    budget = TokenBudget(max_tokens=200_000)  # generous ceiling for a full eval run

    for item in questions:
        question = item["question"]
        print(f"  [{mode}] {item['id']}: {question[:60]}...")

        if mode == "baseline":
            hits = retriever.basic_retrieval(question, top_k=5)
            contexts = [h.text for h in hits]
            answer = _baseline_answer(question, contexts)
        else:
            hits = retriever.search(question, top_k=5)
            contexts = [h.text for h in hits]
            result = self_consistency_synthesis(question, contexts, [h.source for h in hits], budget)
            answer = result.winning_candidate.conclusion

        records["question"].append(question)
        records["answer"].append(answer)
        records["contexts"].append(contexts)
        records["ground_truth"].append(item["ground_truth"])

        time.sleep(0.5)  # light rate-limit courtesy

    return Dataset.from_dict(records)


def main():
    """Runs baseline then final pipelines over the eval questions, scores both
    with RAGAS, and writes the before/after comparison table to ragas_results.json."""
    print("Building retriever (loads embedding + cross-encoder models)...")
    retriever = HybridRetriever(_CORPUS_DIR)
    questions = _load_questions()
    print(f"Loaded {len(questions)} evaluation questions.\n")

    judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    metrics = [context_recall, context_precision, faithfulness, answer_relevancy]

    print("=== Running BASELINE pipeline (basic retrieval + zero-shot answer) ===")
    baseline_dataset = _run_pipeline(questions, retriever, mode="baseline")
    print("\nScoring baseline with RAGAS...")
    baseline_scores = evaluate(baseline_dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)

    print("\n=== Running FINAL pipeline (hybrid+rerank+parent-child + few-shot CoT + self-consistency) ===")
    final_dataset = _run_pipeline(questions, retriever, mode="final")
    print("\nScoring final with RAGAS...")
    final_scores = evaluate(final_dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings)

    comparison = {
        "num_questions": len(questions),
        "metrics": {},
    }
    metric_names = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]
    for name in metric_names:
        comparison["metrics"][name] = {
            "baseline": round(float(baseline_scores[name]), 4),
            "final": round(float(final_scores[name]), 4),
            "delta": round(float(final_scores[name]) - float(baseline_scores[name]), 4),
        }

    with open(_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'Metric':<20}{'Baseline':<12}{'Final':<12}{'Delta':<10}")
    print("=" * 70)
    for name, vals in comparison["metrics"].items():
        print(f"{name:<20}{vals['baseline']:<12}{vals['final']:<12}{vals['delta']:<10}")
    print("=" * 70)
    print(f"\nFull results saved to {_OUTPUT_PATH}")
    print("Copy this table directly into REPORT.md section 3 (Evaluation).")


if __name__ == "__main__":
    main()