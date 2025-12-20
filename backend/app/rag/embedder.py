"""
Embedder module fully compatible with LlamaIndex + Pydantic v2,
using SentenceTransformers locally.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding

from ..config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)


class JinaEmbedder(BaseEmbedding):
    """
    Correct implementation compatible with Pydantic v2 and LlamaIndex.
    """

    # Declare cache properly so Pydantic does NOT wrap it incorrectly
    _model_cache: SentenceTransformer | None = PrivateAttr(default=None)

    # ---------------------------------------------------------
    # INTERNAL LOADER
    # ---------------------------------------------------------
    def _load_model(self) -> SentenceTransformer:
        """Load the local HF embedding model only once."""
        if self._model_cache is None:
            logger.info(f"Loading local SentenceTransformer: {EMBEDDING_MODEL}")

            self._model_cache = SentenceTransformer(
                EMBEDDING_MODEL,
                trust_remote_code=True
            )

            logger.info("SentenceTransformer loaded successfully")

        return self._model_cache

    # ---------------------------------------------------------
    # REQUIRED BY LLAMAINDEX
    # ---------------------------------------------------------
    def _get_text_embedding(self, text: str) -> List[float]:
        model = self._load_model()

        if not text:
            return np.zeros(EMBEDDING_DIMENSION, dtype=np.float32).tolist()

        emb = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return emb.astype("float32").tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    # ---------------------------------------------------------
    # OPTIONAL PUBLIC API (compat with your code)
    # ---------------------------------------------------------
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()

        if not texts:
            return np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)

        emb = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True
        )

        return emb.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]

    def get_dimension(self) -> int:
        return EMBEDDING_DIMENSION


# ===== SINGLETON GLOBAL POUR L'EMBEDDER =====

_embedder_instance: JinaEmbedder | None = None


def get_embedder(force_reload: bool = False) -> JinaEmbedder:
    """
    Retourne TOUJOURS la MÊME instance JinaEmbedder (Singleton).
    
    TRÈS IMPORTANT : Cela garantit qu'on ne charge le modèle qu'UNE FOIS
    et qu'on le réinitialise après ingestion si nécessaire.
    
    Args:
        force_reload: Si True, recrée l'instance (après ingestion)
    
    Returns:
        JinaEmbedder singleton instance
    """
    global _embedder_instance
    
    if _embedder_instance is None or force_reload:
        logger.info("[EMBEDDER]Création de l'instance JinaEmbedder singleton...")
        _embedder_instance = JinaEmbedder()
    
    return _embedder_instance

