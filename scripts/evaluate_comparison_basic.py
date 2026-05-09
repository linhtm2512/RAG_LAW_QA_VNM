import re
import string
from collections import Counter

import pandas as pd


INPUT_FILE = "comparison_outputs.csv"
OUTPUT_FILE = "comparison_metrics.csv"
SUMMARY_FILE = "summary_result.csv"


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_text(prediction) == normalize_text(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    rows = []

    for _, row in df.iterrows():
        reference = str(row["reference_answer"])
        rag_answer = str(row.get("rag_answer", ""))
        no_rag_answer = str(row.get("no_rag_answer", ""))

        rows.append(
            {
                "question": row["question"],
                "reference_answer": reference,
                "rag_answer": rag_answer,
                "no_rag_answer": no_rag_answer,
                "rag_exact_match": exact_match(rag_answer, reference),
                "rag_f1": f1_score(rag_answer, reference),
                "no_rag_exact_match": exact_match(no_rag_answer, reference),
                "no_rag_f1": f1_score(no_rag_answer, reference),
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "Cấu hình": "Không sử dụng RAG",
                "Exact Match": result_df["no_rag_exact_match"].mean(),
                "F1": result_df["no_rag_f1"].mean(),
            },
            {
                "Cấu hình": "Có sử dụng RAG",
                "Exact Match": result_df["rag_exact_match"].mean(),
                "F1": result_df["rag_f1"].mean(),
            },
        ]
    )

    summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")

    print(summary)


if __name__ == "__main__":
    main()
