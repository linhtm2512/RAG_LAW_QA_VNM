"""
rag_pipeline.py
───────────────
Full RAG pipeline:
  1. Encode user question  →  dense vector
  2. Retrieve top-k similar chunks from Qdrant
  3. Build prompt with retrieved context
  4. Call LLM (Gemini / OpenAI) to generate an answer

Also exposes a "no-RAG" path (pure LLM, no retrieval) for comparison.
"""

import logging
import textwrap
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from openai import OpenAI

from .config import Settings
from .embeddings import EmbeddingModel
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


# ─── Prompt Templates ─────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """Bạn là một trợ lý thông minh chuyên trả lời câu hỏi dựa trên các tài liệu pháp lý tiếng Việt.

Quy tắc:
1. Chỉ trả lời dựa trên CONTEXT được cung cấp.
2. Nếu thông tin không có trong CONTEXT, hãy trả lời: "Tôi không tìm thấy thông tin này trong tài liệu."
3. Trích dẫn nguồn tài liệu khi có thể (tên file).
4. Trả lời bằng tiếng Việt, rõ ràng và đầy đủ.
5. KHÔNG bịa đặt thông tin ngoài CONTEXT.
6. KHÔNG dùng ký hiệu LaTeX (\\[...\\], \\(...\\), \\text{...}). Viết công thức và phép tính bằng chữ thuần túy, ví dụ: "Thu nhập tính thuế = Lương gross - Giảm trừ gia cảnh = 30 triệu - 15,5 triệu = 14,5 triệu đồng/tháng"."""

NO_RAG_SYSTEM_PROMPT = """Bạn là một trợ lý thông minh trả lời câu hỏi về pháp luật Việt Nam.
Trả lời dựa trên kiến thức của bạn. Trả lời bằng tiếng Việt.
KHÔNG dùng ký hiệu LaTeX (\\[...\\], \\(...\\), \\text{...}). Viết công thức và phép tính bằng chữ thuần túy."""


def _build_rag_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Đoạn {i} – Nguồn: {source}]\n{text}")

    context_str = "\n\n".join(context_parts)
    return textwrap.dedent(f"""
    CONTEXT:
    {context_str}

    CÂU HỎI: {question}

    TRẢ LỜI:""").strip()


# ─── LLM callers ──────────────────────────────────────────────────────────────


def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model_name: str = "gemini-2.0-flash-lite",
) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt,
    )
    # Retry up to 3 times on rate-limit (429)
    for attempt in range(3):
        try:
            response = model.generate_content(user_prompt)
            return response.text.strip()
        except Exception as e:
            if (
                "429" in str(e)
                or "quota" in str(e).lower()
                or "RESOURCE_EXHAUSTED" in str(e)
            ):
                wait = 30 * (attempt + 1)
                logger.warning(
                    "Gemini rate limit hit (attempt %d/3). Waiting %ds…",
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        "Gemini API rate limit exceeded after 3 retries. Vui lòng thử lại sau 1 phút."
    )


def _call_openai(
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    model_name: str = "gpt-4o-mini",
    base_url: str = None,
) -> str:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


def _call_llm(system_prompt: str, user_prompt: str, settings: Settings) -> str:
    if settings.llm_provider == "openai":
        return _call_openai(
            system_prompt,
            user_prompt,
            api_key=settings.openai_api_key,
            model_name=settings.llm_model_openai,
        )
    if settings.llm_provider == "glm":
        return _call_openai(
            system_prompt,
            user_prompt,
            api_key=settings.glm_api_key,
            model_name=settings.llm_model_glm,
            base_url=settings.glm_base_url,
        )
    # default: gemini
    return _call_gemini(
        system_prompt,
        user_prompt,
        api_key=settings.gemini_api_key,
        model_name=settings.llm_model_gemini,
    )


# ─── RAG Pipeline class ───────────────────────────────────────────────────────


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
    ):
        self.settings = settings
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    # ── Main: with RAG ────────────────────────────────────────────────────────

    def answer_with_rag(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline.

        Returns:
            {
                "question":   str,
                "answer":     str,
                "retrieved":  List[{text, score, source, chunk_id}],
                "mode":       "rag",
            }
        """
        k = top_k or self.settings.top_k

        # 1. Embed query
        query_vector = self.embedding_model.encode_query(question)

        # 2. Retrieve similar chunks
        retrieved_chunks = self.vector_store.search(
            query_vector=query_vector,
            top_k=k,
            filter_source=filter_source,
        )

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "Không tìm thấy tài liệu liên quan trong cơ sở dữ liệu. "
                "Vui lòng kiểm tra lại hoặc tải tài liệu vào hệ thống.",
                "retrieved": [],
                "mode": "rag",
            }

        # 3. Build prompt + call LLM
        user_prompt = _build_rag_prompt(question, retrieved_chunks)
        answer = _call_llm(RAG_SYSTEM_PROMPT, user_prompt, self.settings)

        return {
            "question": question,
            "answer": answer,
            "retrieved": retrieved_chunks,
            "mode": "rag",
        }

    # ── Comparison: without RAG (pure LLM) ───────────────────────────────────

    def answer_without_rag(self, question: str) -> Dict[str, Any]:
        """
        Pure LLM answer – no retrieval step (for hallucination comparison).
        """
        answer = _call_llm(NO_RAG_SYSTEM_PROMPT, question, self.settings)
        return {
            "question": question,
            "answer": answer,
            "retrieved": [],
            "mode": "no_rag",
        }

    # ── Compare both modes ────────────────────────────────────────────────────

    def compare(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run both RAG and no-RAG and return side-by-side results."""
        rag_result = self.answer_with_rag(question, top_k=top_k)
        no_rag_result = self.answer_without_rag(question)
        return {
            "question": question,
            "rag": rag_result,
            "no_rag": no_rag_result,
        }
