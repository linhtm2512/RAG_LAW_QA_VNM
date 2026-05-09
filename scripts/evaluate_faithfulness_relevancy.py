import ast
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "comparison_outputs.csv"

DETAIL_OUTPUT_FILE = BASE_DIR / "faithfulness_relevancy_detail.csv"
SUMMARY_OUTPUT_FILE = BASE_DIR / "faithfulness_relevancy_summary.csv"

GEMINI_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"


def parse_contexts(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]

            if isinstance(parsed, str):
                return [parsed.strip()] if parsed.strip() else []

        except Exception:
            return [value]

    return []


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    return str(value).strip()


def build_faithfulness_prompt(question: str, answer: str, contexts: list[str]) -> str:
    joined_contexts = "\n\n".join(
        f"[Context {i + 1}]\n{context}"
        for i, context in enumerate(contexts)
    )

    return f"""
Bạn là bộ đánh giá hệ thống hỏi đáp RAG tiếng Việt.

Nhiệm vụ:
Đánh giá mức độ FAITHFULNESS của câu trả lời so với ngữ cảnh được cung cấp.

Định nghĩa:
- Faithfulness = 1.0 nếu toàn bộ thông tin quan trọng trong câu trả lời được hỗ trợ bởi ngữ cảnh.
- Faithfulness = 0.5 nếu câu trả lời chỉ được hỗ trợ một phần.
- Faithfulness = 0.0 nếu câu trả lời không được hỗ trợ bởi ngữ cảnh hoặc chứa thông tin bịa đặt.

Quy tắc:
- Chỉ dựa vào CONTEXTS.
- Không dùng kiến thức bên ngoài.
- Trả về JSON hợp lệ, không markdown.
- JSON phải có đúng các field:
  - score: số từ 0 đến 1
  - reason: giải thích ngắn bằng tiếng Việt

QUESTION:
{question}

ANSWER:
{answer}

CONTEXTS:
{joined_contexts}

JSON:
""".strip()


def extract_json(text: str) -> dict:
    text = text.strip()

    # Remove markdown fences if model returns ```json
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: find first JSON object
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "score": None,
        "reason": f"Cannot parse Gemini response: {text[:300]}",
    }


def clamp_score(value) -> float | None:
    try:
        score = float(value)
    except Exception:
        return None

    if score < 0:
        return 0.0

    if score > 1:
        return 1.0

    return score


def compute_faithfulness(
    *,
    client: genai.Client,
    question: str,
    answer: str,
    contexts: list[str],
    max_retries: int = 3,
) -> tuple[float | None, str]:
    if not answer:
        return None, "Empty answer"

    if not contexts:
        return 0.0, "No retrieved contexts"

    prompt = build_faithfulness_prompt(question, answer, contexts)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            parsed = extract_json(response.text or "")
            score = clamp_score(parsed.get("score"))

            return score, str(parsed.get("reason", ""))

        except Exception as exc:
            if attempt == max_retries:
                return None, f"Gemini error after {max_retries} retries: {exc}"

            time.sleep(2 * attempt)

    return None, "Unknown error"


def compute_answer_relevancy(
    *,
    embedding_model: SentenceTransformer,
    question: str,
    answer: str,
) -> float | None:
    if not question or not answer:
        return None

    embeddings = embedding_model.encode(
        [question, answer],
        normalize_embeddings=True,
    )

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]],
    )[0][0]

    # cosine can be [-1, 1]; map to [0, 1]
    normalized_score = (float(score) + 1.0) / 2.0

    if normalized_score < 0:
        return 0.0

    if normalized_score > 1:
        return 1.0

    return normalized_score


def evaluate_answer(
    *,
    client: genai.Client,
    embedding_model: SentenceTransformer,
    question: str,
    answer: str,
    contexts: list[str],
) -> dict:
    faithfulness_score, faithfulness_reason = compute_faithfulness(
        client=client,
        question=question,
        answer=answer,
        contexts=contexts,
    )

    answer_relevancy_score = compute_answer_relevancy(
        embedding_model=embedding_model,
        question=question,
        answer=answer,
    )

    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": answer_relevancy_score,
        "faithfulness_reason": faithfulness_reason,
    }


def require_columns(df: pd.DataFrame) -> None:
    required_columns = {
        "question",
        "rag_answer",
        "no_rag_answer",
        "contexts",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {missing}. "
            f"Current columns: {list(df.columns)}"
        )


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    if not INPUT_FILE.exists():
        raise RuntimeError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    require_columns(df)

    client = genai.Client(api_key=api_key)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    rows = []

    for index, row in df.iterrows():
        question = clean_text(row.get("question", ""))
        reference_answer = clean_text(row.get("reference_answer", ""))
        rag_answer = clean_text(row.get("rag_answer", ""))
        no_rag_answer = clean_text(row.get("no_rag_answer", ""))
        contexts = parse_contexts(row.get("contexts", "[]"))

        print(f"[{index + 1}/{len(df)}] {question}")

        rag_metrics = evaluate_answer(
            client=client,
            embedding_model=embedding_model,
            question=question,
            answer=rag_answer,
            contexts=contexts,
        )

        no_rag_metrics = evaluate_answer(
            client=client,
            embedding_model=embedding_model,
            question=question,
            answer=no_rag_answer,
            contexts=contexts,
        )

        rows.append(
            {
                "question": question,
                "reference_answer": reference_answer,
                "rag_answer": rag_answer,
                "no_rag_answer": no_rag_answer,
                "rag_faithfulness": rag_metrics["faithfulness"],
                "rag_answer_relevancy": rag_metrics["answer_relevancy"],
                "rag_faithfulness_reason": rag_metrics["faithfulness_reason"],
                "no_rag_faithfulness": no_rag_metrics["faithfulness"],
                "no_rag_answer_relevancy": no_rag_metrics["answer_relevancy"],
                "no_rag_faithfulness_reason": no_rag_metrics["faithfulness_reason"],
            }
        )

        # Avoid Gemini rate limit
        time.sleep(0.5)

    detail_df = pd.DataFrame(rows)
    detail_df.to_csv(DETAIL_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "Cấu hình": "Không sử dụng RAG",
                "Faithfulness": detail_df["no_rag_faithfulness"].mean(),
                "Answer Relevancy": detail_df["no_rag_answer_relevancy"].mean(),
            },
            {
                "Cấu hình": "Có sử dụng RAG",
                "Faithfulness": detail_df["rag_faithfulness"].mean(),
                "Answer Relevancy": detail_df["rag_answer_relevancy"].mean(),
            },
        ]
    )

    summary.to_csv(SUMMARY_OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nSaved files:")
    print(f"- {DETAIL_OUTPUT_FILE}")
    print(f"- {SUMMARY_OUTPUT_FILE}")

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
