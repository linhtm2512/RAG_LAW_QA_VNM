"""
evaluator.py
────────────
Evaluate RAG vs no-RAG answers using:
  • Exact Match (EM)
  • Token-level F1
  • ROUGE-L
  • Simple hallucination heuristic (answer contains keywords from reference)

Reads QA pairs from an Excel file with columns:
    question | reference_answer  (or: câu_hỏi | câu_trả_lời)
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List

import pandas as pd
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


# ─── Text normalisation ───────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics for EM / token-F1 comparison."""
    text = text.lower().strip()
    # Normalize unicode (NFC) – keep Vietnamese chars intact for ROUGE
    text = unicodedata.normalize("NFC", text)
    return text


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _normalize(text))


# ─── Metrics ──────────────────────────────────────────────────────────────────


def exact_match(prediction: str, reference: str) -> float:
    return float(_normalize(prediction) == _normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    score = scorer.score(reference, prediction)
    return round(score["rougeL"].fmeasure, 4)


def hallucination_score(prediction: str, reference: str) -> float:
    """
    Simple heuristic: what fraction of reference keywords appear in the
    prediction? Higher = less hallucination (better grounding).
    Returns a value in [0, 1].
    """
    ref_tokens = set(_tokenize(reference))
    pred_tokens = set(_tokenize(prediction))
    if not ref_tokens:
        return 1.0
    covered = ref_tokens & pred_tokens
    return round(len(covered) / len(ref_tokens), 4)


# ─── Batch evaluation ─────────────────────────────────────────────────────────


def evaluate_single(prediction: str, reference: str) -> Dict[str, float]:
    return {
        "exact_match": exact_match(prediction, reference),
        "token_f1": round(token_f1(prediction, reference), 4),
        "rouge_l": rouge_l(prediction, reference),
        "grounding": hallucination_score(prediction, reference),
    }


def load_qa_test_file(filepath: str) -> pd.DataFrame:
    """
    Load QA test pairs from Excel.
    Accepted column names (case-insensitive):
        question / câu_hỏi / câu hỏi
        answer / reference_answer / câu_trả_lời / câu trả lời
    """
    df = pd.read_excel(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    # Map to standard names
    q_aliases = ["question", "câu_hỏi", "câu hỏi", "cau_hoi", "cau hoi"]
    a_aliases = [
        "answer",
        "reference_answer",
        "câu_trả_lời",
        "câu trả lời",
        "cau_tra_loi",
        "cau tra loi",
        "expected_answer",
    ]

    q_col = next((c for c in df.columns if c in q_aliases), None)
    a_col = next((c for c in df.columns if c in a_aliases), None)

    if q_col is None or a_col is None:
        raise ValueError(
            f"Cannot find question/answer columns. Found: {list(df.columns)}"
        )

    return df[[q_col, a_col]].rename(columns={q_col: "question", a_col: "reference"})


def evaluate_batch(
    qa_pairs: List[
        Dict[str, str]
    ],  # [{"question": ..., "reference": ..., "rag_answer": ..., "no_rag_answer": ...}]
) -> Dict[str, Any]:
    """
    Compute aggregate metrics for a batch of QA results.

    Returns:
        {
            "per_question": List[Dict],
            "aggregate": {
                "rag":    {exact_match, token_f1, rouge_l, grounding},
                "no_rag": {exact_match, token_f1, rouge_l, grounding},
            }
        }
    """
    per_question = []
    rag_metrics: List[Dict] = []
    no_rag_metrics: List[Dict] = []

    for item in qa_pairs:
        q = item["question"]
        ref = item["reference"]
        rag_ans = item.get("rag_answer", "")
        no_rag_ans = item.get("no_rag_answer", "")

        rag_m = evaluate_single(rag_ans, ref)
        no_rag_m = evaluate_single(no_rag_ans, ref)

        per_question.append(
            {
                "question": q,
                "reference": ref,
                "rag_answer": rag_ans,
                "no_rag_answer": no_rag_ans,
                "rag_metrics": rag_m,
                "no_rag_metrics": no_rag_m,
            }
        )
        rag_metrics.append(rag_m)
        no_rag_metrics.append(no_rag_m)

    def _avg(metrics_list: List[Dict]) -> Dict:
        keys = metrics_list[0].keys() if metrics_list else []
        return {
            k: round(sum(m[k] for m in metrics_list) / len(metrics_list), 4)
            for k in keys
        }

    return {
        "per_question": per_question,
        "aggregate": {
            "rag": _avg(rag_metrics) if rag_metrics else {},
            "no_rag": _avg(no_rag_metrics) if no_rag_metrics else {},
        },
        "total": len(per_question),
    }
