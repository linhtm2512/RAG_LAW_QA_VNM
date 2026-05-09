import json
import time

import pandas as pd
import requests


BASE_URL = "http://localhost:8000"
QA_FILE = "QA_TEST.xlsx"
OUTPUT_FILE = "comparison_outputs.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        normalized = str(col).strip().lower()

        if normalized in {"question", "câu hỏi", "cau hoi"}:
            rename_map[col] = "question"

        if normalized in {
            "reference_answer",
            "ground_truth",
            "câu trả lời",
            "cau tra loi",
            "answer",
        }:
            rename_map[col] = "reference_answer"

    df = df.rename(columns=rename_map)

    missing = {"question", "reference_answer"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Current columns: {list(df.columns)}")

    return df


def call_compare(question: str, top_k: int = 5) -> dict:
    response = requests.post(
        f"{BASE_URL}/compare",
        json={
            "question": question,
            "top_k": top_k,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()


def extract_contexts(result: dict) -> list[str]:
    retrieved = result.get("rag", {}).get("retrieved", [])

    contexts = []

    for item in retrieved:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("page_content")
            if text:
                contexts.append(str(text))
        elif isinstance(item, str):
            contexts.append(item)

    return contexts


def main() -> None:
    df = pd.read_excel(QA_FILE)
    df = normalize_columns(df)

    rows = []

    for index, row in df.iterrows():
        question = str(row["question"]).strip()
        reference_answer = str(row["reference_answer"]).strip()

        print(f"[{index + 1}/{len(df)}] {question}")

        try:
            result = call_compare(question, top_k=5)

            rag_answer = result.get("rag", {}).get("answer", "")
            no_rag_answer = result.get("no_rag", {}).get("answer", "")
            contexts = extract_contexts(result)

            rows.append(
                {
                    "question": question,
                    "reference_answer": reference_answer,
                    "rag_answer": rag_answer,
                    "no_rag_answer": no_rag_answer,
                    "contexts": json.dumps(contexts, ensure_ascii=False),
                    "raw_response": json.dumps(result, ensure_ascii=False),
                    "error": "",
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "question": question,
                    "reference_answer": reference_answer,
                    "rag_answer": "",
                    "no_rag_answer": "",
                    "contexts": "[]",
                    "raw_response": "",
                    "error": str(exc),
                }
            )

        time.sleep(0.5)

    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
