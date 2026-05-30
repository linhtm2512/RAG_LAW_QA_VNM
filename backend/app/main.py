"""
main.py  –  FastAPI application
────────────────────────────────
Endpoints:
  GET  /health                   – liveness probe
  GET  /info                     – system info (model, collection stats)
  GET  /documents                – list PDF files in documents folder
  POST /ingest                   – ingest all PDFs into Qdrant
  DELETE /collection             – delete the vector collection
  POST /query                    – RAG answer
  POST /query/no-rag             – pure-LLM answer (no retrieval)
  POST /compare                  – RAG vs no-RAG side-by-side
  POST /evaluate                 – batch evaluation from QA_TEST.xlsx
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tqdm import tqdm

from .config import get_settings
from .document_loader import load_documents_from_folder
from .embeddings import get_embedding_model
from .evaluator import evaluate_batch, load_qa_test_file
from .rag_pipeline import RAGPipeline
from .vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Globals (initialised at startup) ────────────────────────────────────────

_store: VectorStore = None
_pipeline: RAGPipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy resources once on startup."""
    global _store, _pipeline
    settings = get_settings()

    logger.info("Loading embedding model: %s", settings.embedding_model)
    emb_model = get_embedding_model(settings.embedding_model)

    logger.info(
        "Connecting to Qdrant at %s:%d", settings.qdrant_host, settings.qdrant_port
    )
    _store = VectorStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.collection_name,
        vector_dim=emb_model.dim,
    )

    _pipeline = RAGPipeline(
        settings=settings,
        embedding_model=emb_model,
        vector_store=_store,
    )
    logger.info("RAG system ready.")
    yield
    logger.info("Shutting down.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vietnamese RAG QA API",
    description="Hệ thống hỏi–đáp tiếng Việt dựa trên RAG (PhoBERT / Sentence-BERT + Qdrant)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic schemas ─────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    filter_source: Optional[str] = None


class EvaluateRequest(BaseModel):
    qa_test_path: Optional[str] = None  # defaults to QA_TEST.xlsx in documents parent


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}


@app.get("/info", tags=["System"])
def info():
    settings = get_settings()
    emb = get_embedding_model(settings.embedding_model)
    collection = _store.collection_info() if _store else {}
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dim": emb.dim,
        "llm_provider": settings.llm_provider,
        "llm_model": (
            settings.llm_model_glm
            if settings.llm_provider == "glm"
            else settings.llm_model_gemini
            if settings.llm_provider == "gemini"
            else settings.llm_model_openai
        ),
        "collection": collection,
        "top_k": settings.top_k,
    }


@app.get("/documents", tags=["Documents"])
def list_documents():
    """List all PDF files currently in the documents folder."""
    settings = get_settings()
    folder = Path(settings.documents_path)
    if not folder.exists():
        return {"files": [], "count": 0}
    files = [
        {
            "name": f.name,
            "size_kb": round(f.stat().st_size / 1024, 1),
        }
        for f in sorted(list(folder.glob("*.pdf")) + list(folder.glob("*.docx")))
    ]
    return {"files": files, "count": len(files), "path": str(folder)}


@app.post("/ingest", tags=["Documents"])
def ingest_documents(recreate: bool = False):
    """
    Load all PDFs from the documents folder, encode them with the embedding
    model and upsert into Qdrant.

    Set `recreate=true` to wipe and rebuild the collection from scratch.
    """
    settings = get_settings()
    emb_model = get_embedding_model(settings.embedding_model)

    logger.info("Starting ingestion from %s", settings.documents_path)
    chunks, file_names = load_documents_from_folder(
        folder_path=settings.documents_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(
            status_code=404, detail="No PDF files found in documents folder."
        )

    logger.info(
        "Loaded %d chunks from %d files. Encoding…", len(chunks), len(file_names)
    )

    # Ensure collection exists
    _store.create_collection(recreate=recreate)

    # Batch encode
    texts = [c["text"] for c in chunks]
    vectors = emb_model.encode(texts, batch_size=32, show_progress=True)

    # Upsert
    total = _store.upsert_chunks(chunks, vectors)

    return {
        "status": "success",
        "files_processed": file_names,
        "chunks_ingested": total,
        "collection": _store.collection_info(),
    }


@app.delete("/collection", tags=["Documents"])
def delete_collection():
    """Delete the entire Qdrant collection (all vectors)."""
    _store.delete_collection()
    return {"status": "deleted", "collection": get_settings().collection_name}


@app.post("/query", tags=["QA"])
def query_rag(req: QueryRequest):
    """Answer a question using RAG (retrieval + LLM)."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return _pipeline.answer_with_rag(
        question=req.question,
        top_k=req.top_k,
        filter_source=req.filter_source,
    )


@app.post("/query/no-rag", tags=["QA"])
def query_no_rag(req: QueryRequest):
    """Answer a question using LLM only – no retrieval (baseline for comparison)."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return _pipeline.answer_without_rag(question=req.question)


@app.post("/compare", tags=["QA"])
def compare_rag_vs_no_rag(req: QueryRequest):
    """Run both RAG and no-RAG for the same question and return both answers."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return _pipeline.compare(question=req.question, top_k=req.top_k)


@app.post("/evaluate", tags=["Evaluation"])
def evaluate(req: EvaluateRequest):
    """
    Run batch evaluation against QA_TEST.xlsx.
    Returns per-question metrics and aggregate scores (EM, F1, ROUGE-L, grounding).
    """
    settings = get_settings()

    # Resolve path
    if req.qa_test_path:
        qa_path = req.qa_test_path
    else:
        # Search in documents folder first, then parent
        docs_folder = Path(settings.documents_path)
        docs_parent = docs_folder.parent
        candidates = (
            list(docs_folder.glob("QA_TEST*.xlsx"))
            + list(docs_folder.glob("qa_test*.xlsx"))
            + list(docs_parent.glob("QA_TEST*.xlsx"))
            + list(docs_parent.glob("qa_test*.xlsx"))
        )
        if not candidates:
            raise HTTPException(
                status_code=404, detail="QA_TEST.xlsx not found. Provide qa_test_path."
            )
        qa_path = str(candidates[0])

    logger.info("Loading QA test file: %s", qa_path)
    df = load_qa_test_file(qa_path)

    import time as _time

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        q = str(row["question"]).strip()
        ref = str(row["reference"]).strip()

        if i > 0:
            _time.sleep(7)  # Stay under Gemini free-tier rate limit (10 RPM)

        rag_res = _pipeline.answer_with_rag(q)
        _time.sleep(7)
        no_rag_res = _pipeline.answer_without_rag(q)

        results.append(
            {
                "question": q,
                "reference": ref,
                "rag_answer": rag_res["answer"],
                "no_rag_answer": no_rag_res["answer"],
            }
        )

    return evaluate_batch(results)
