"""
embeddings.py
─────────────
Singleton wrapper around SentenceTransformer that encodes Vietnamese text.

Supported models (set via EMBEDDING_MODEL env var):
  • keepitreal/vietnamese-sbert                    ← default, ~270 MB
  • VoVanPhuc/sup-SimCSE-VietNamese-phobert-base   ← PhoBERT-based, ~540 MB
  • intfloat/multilingual-e5-large                 ← multilingual, ~1.1 GB

The class is a singleton – the model is loaded once and reused across requests.
"""

import logging
from functools import lru_cache
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer with batch encoding and
    normalisation helpers.
    """

    def __init__(self, model_name: str = "keepitreal/vietnamese-sbert"):
        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded. Dimension: %d", self.dim)

    # ------------------------------------------------------------------
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more strings into dense vectors.

        Returns:
            numpy array of shape (n, dim)  for list input  OR
            numpy array of shape (dim,)    for single-string input.
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        # multilingual-e5 models require the "query:" / "passage:" prefix
        if "multilingual-e5" in self.model_name.lower():
            texts = [f"passage: {t}" for t in texts]

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return vectors[0] if single else vectors

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a *query* string (uses 'query:' prefix for E5 models).
        """
        if "multilingual-e5" in self.model_name.lower():
            query = f"query: {query}"
        return self._model.encode(
            [query],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )[0]


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Return the singleton EmbeddingModel instance (loaded once per process)."""
    return EmbeddingModel(model_name)
