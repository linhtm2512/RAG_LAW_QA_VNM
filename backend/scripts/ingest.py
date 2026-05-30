#!/usr/bin/env python3
"""
scripts/ingest.py
─────────────────
Standalone CLI script to ingest PDF documents from a folder into Qdrant.
Can run locally (outside Docker) or inside the backend container.

Usage:
    # Inside Docker container:
    docker exec rag-backend python -m scripts.ingest

    # Locally (with .env):
    python scripts/ingest.py --docs ./documents --recreate

Options:
    --docs PATH      Path to documents folder  (default: ./documents)
    --recreate       Drop and recreate the collection before ingestion
    --model NAME     Embedding model name (overrides .env)
    --host HOST      Qdrant host  (default: localhost)
    --port PORT      Qdrant port  (default: 6333)
"""

import argparse
import logging
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from app.config import get_settings
from app.document_loader import load_documents_from_folder
from app.embeddings import get_embedding_model
from app.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
    parser.add_argument("--docs", default=None, help="Documents folder")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    parser.add_argument("--model", default=None, help="Embedding model name")
    parser.add_argument("--host", default=None, help="Qdrant host")
    parser.add_argument("--port", type=int, default=None, help="Qdrant port")
    args = parser.parse_args()

    settings = get_settings()

    # CLI overrides
    docs_path = args.docs or settings.documents_path
    qdrant_host = args.host or settings.qdrant_host
    qdrant_port = args.port or settings.qdrant_port
    model_name = args.model or settings.embedding_model

    logger.info("═" * 55)
    logger.info("Vietnamese RAG – Document Ingestion Script")
    logger.info("═" * 55)
    logger.info("Documents folder : %s", docs_path)
    logger.info("Embedding model  : %s", model_name)
    logger.info("Qdrant           : %s:%d", qdrant_host, qdrant_port)
    logger.info("Collection       : %s", settings.collection_name)
    logger.info("Recreate         : %s", args.recreate)
    logger.info("─" * 55)

    # 1. Load + chunk documents
    logger.info("Step 1/3: Loading and chunking documents…")
    chunks, file_names = load_documents_from_folder(
        folder_path=docs_path,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        logger.error("No PDF files found in '%s'. Exiting.", docs_path)
        sys.exit(1)

    logger.info(
        "Loaded %d chunks from %d files: %s", len(chunks), len(file_names), file_names
    )

    # 2. Load embedding model
    logger.info("Step 2/3: Loading embedding model '%s'…", model_name)
    emb_model = get_embedding_model(model_name)
    logger.info("Model loaded. Vector dimension: %d", emb_model.dim)

    # 3. Encode all chunks
    logger.info("Encoding %d chunks (this may take a few minutes)…", len(chunks))
    texts = [c["text"] for c in chunks]
    vectors = emb_model.encode(texts, batch_size=32, show_progress=True)
    logger.info("Encoding complete. Shape: %s", vectors.shape)

    # 4. Upsert to Qdrant
    logger.info("Step 3/3: Upserting to Qdrant…")
    store = VectorStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=settings.collection_name,
        vector_dim=emb_model.dim,
    )
    store.create_collection(recreate=args.recreate)
    total = store.upsert_chunks(chunks, vectors)

    info = store.collection_info()
    logger.info("═" * 55)
    logger.info("✅ Ingestion complete!")
    logger.info("   Chunks upserted : %d", total)
    logger.info("   Total in DB      : %d", info["count"])
    logger.info("   Sources          : %s", store.list_sources())
    logger.info("═" * 55)


if __name__ == "__main__":
    main()
