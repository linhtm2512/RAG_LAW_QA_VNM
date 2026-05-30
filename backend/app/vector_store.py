"""
vector_store.py
───────────────
CRUD operations against Qdrant for storing and retrieving Vietnamese document
chunks as dense vectors.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper around QdrantClient providing:
        - Collection management (create / delete / info)
        - Batch upsert of document chunks
        - Similarity search (returns top-k results with scores + metadata)
    """

    def __init__(self, host: str, port: int, collection_name: str, vector_dim: int):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = QdrantClient(host=host, port=port, timeout=30)
        logger.info("Connected to Qdrant  %s:%d", host, port)

    # ── Collection management ─────────────────────────────────────────────────

    def collection_exists(self) -> bool:
        collections = [c.name for c in self.client.get_collections().collections]
        return self.collection_name in collections

    def create_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection (cosine similarity, fp32 vectors)."""
        if self.collection_exists():
            if recreate:
                self.client.delete_collection(self.collection_name)
                logger.info("Deleted existing collection '%s'", self.collection_name)
            else:
                logger.info(
                    "Collection '%s' already exists – skipping creation.",
                    self.collection_name,
                )
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '%s' (dim=%d, cosine)",
            self.collection_name,
            self.vector_dim,
        )

    def delete_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted collection '%s'.", self.collection_name)

    def collection_info(self) -> Dict[str, Any]:
        if not self.collection_exists():
            return {"exists": False, "count": 0}
        info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "count": info.points_count,
            "status": str(info.status),
            "vector_dim": self.vector_dim,
            "collection_name": self.collection_name,
        }

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: np.ndarray,
        batch_size: int = 64,
    ) -> int:
        """
        Insert / update document chunks with their embedding vectors.

        Args:
            chunks:     List of {"id": str, "text": str, "metadata": dict}
            vectors:    numpy array (n, dim) aligned with chunks
            batch_size: how many points to push per request

        Returns:
            Total number of upserted points.
        """
        if not self.collection_exists():
            self.create_collection()

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]

            points = [
                qmodels.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["id"])),
                    vector=vector.tolist(),
                    payload={
                        "doc_id": chunk["id"],
                        "text": chunk["text"],
                        **chunk.get("metadata", {}),
                    },
                )
                for chunk, vector in zip(batch_chunks, batch_vectors)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total += len(points)
            logger.debug("Upserted %d points (total so far: %d)", len(points), total)

        return total

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Nearest-neighbour search.

        Returns list of dicts:
            {"text": str, "score": float, "source": str, "chunk_id": int, ...}
        """
        query_filter = None
        if filter_source:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source",
                        match=qmodels.MatchValue(value=filter_source),
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload.get("text", ""),
                "score": round(hit.score, 4),
                "source": hit.payload.get("source", ""),
                "chunk_id": hit.payload.get("chunk_id", -1),
                "doc_id": hit.payload.get("doc_id", ""),
            }
            for hit in results
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def list_sources(self) -> List[str]:
        """Return distinct source file names stored in the collection."""
        if not self.collection_exists():
            return []
        # scroll through all points to collect unique sources
        sources = set()
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                with_payload=["source"],
                limit=256,
                offset=offset,
            )
            for r in records:
                src = r.payload.get("source", "")
                if src:
                    sources.add(src)
            if next_offset is None:
                break
            offset = next_offset
        return sorted(sources)
