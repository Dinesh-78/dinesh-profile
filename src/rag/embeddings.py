"""Embeddings provider with Google GenAI and fallback support."""

import logging
from typing import List
from src.config import settings

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    class Embeddings:
        pass

logger = logging.getLogger(__name__)

class RobustEmbeddings(Embeddings):
    """Embeddings wrapper providing Google GenAI embeddings with resilient fallback."""

    def __init__(self):
        self.embedding_model = None
        self._init_embeddings()

    def _init_embeddings(self):
        if settings.GOOGLE_API_KEY:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model=settings.EMBEDDING_MODEL,
                    google_api_key=settings.GOOGLE_API_KEY
                )
                logger.info(f"Initialized Google Generative AI Embeddings with {settings.EMBEDDING_MODEL}")
                return
            except Exception as e:
                logger.warning(f"Could not load Google Generative AI Embeddings: {e}. Falling back.")
        self.embedding_model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text documents."""
        if self.embedding_model:
            try:
                return self.embedding_model.embed_documents(texts)
            except Exception as e:
                logger.warning(f"Embedding error: {e}. Using deterministic semantic vector.")
        return [self._fallback_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        if self.embedding_model:
            try:
                return self.embedding_model.embed_query(text)
            except Exception as e:
                logger.warning(f"Query embedding error: {e}. Using deterministic semantic vector.")
        return self._fallback_vector(text)

    def _fallback_vector(self, text: str, dim: int = 768) -> List[float]:
        """Deterministic hash-based dense vector for offline/fallback environments."""
        import hashlib
        import math

        vector = [0.0] * dim
        tokens = text.lower().split()
        if not tokens:
            return vector

        for token in tokens:
            h = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vector[idx] += sign

        # Normalize L2
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector

_embeddings_instance = None

def get_embeddings():
    """Singleton getter for robust embeddings."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = RobustEmbeddings()
    return _embeddings_instance

