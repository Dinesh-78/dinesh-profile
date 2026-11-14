"""Hybrid Retriever using LangChain EnsembleRetriever (BM25 + Chroma Dense)."""

import sys
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from src.config import settings
from src.rag.embeddings import get_embeddings
from src.rag.indexer import ProfileIndexer

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Production hybrid retriever leveraging LangChain BM25Retriever and EnsembleRetriever."""

    def __init__(self, top_k: Optional[int] = None, similarity_threshold: Optional[float] = None):
        self.top_k = top_k or settings.TOP_K_RESULTS
        self.similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        self.embeddings = get_embeddings()
        self.indexer = ProfileIndexer()
        self.documents: List[Document] = []
        self.ensemble_retriever = None
        self.chroma_retriever = None
        self.bm25_retriever = None
        self._initialize()

    def _initialize(self):
        """Initializes LangChain documents, BM25 retriever, and Chroma retriever."""
        self.documents = self.indexer.load_and_chunk()

        # 1. Initialize BM25Retriever
        try:
            from langchain_community.retrievers import BM25Retriever
            self.bm25_retriever = BM25Retriever.from_documents(self.documents)
            self.bm25_retriever.k = self.top_k
        except Exception as e:
            logger.warning(f"Could not initialize LangChain BM25Retriever: {e}")
            self.bm25_retriever = None

        # 2. Initialize Chroma VectorStore Retriever
        try:
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                from langchain_chroma import Chroma

            vector_store = Chroma(
                collection_name="dinesh_profile",
                embedding_function=self.embeddings,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
            # Ensure it has documents
            if vector_store._collection.count() == 0:
                vector_store = self.indexer.build_index()

            self.chroma_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k}
            )
        except Exception as e:
            logger.warning(f"Could not initialize LangChain Chroma retriever: {e}")
            self.chroma_retriever = None

        # 3. Combine with LangChain EnsembleRetriever (Reciprocal Rank Fusion)
        if self.chroma_retriever and self.bm25_retriever:
            try:
                from langchain.retrievers import EnsembleRetriever
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[self.chroma_retriever, self.bm25_retriever],
                    weights=[0.6, 0.4]
                )
                logger.info("Initialized LangChain EnsembleRetriever with BM25 + Chroma.")
            except Exception as e:
                logger.warning(f"Could not build EnsembleRetriever: {e}. Will use available retriever.")
                self.ensemble_retriever = self.chroma_retriever or self.bm25_retriever

    def retrieve_documents(self, query: str) -> List[Document]:
        """LangChain native retrieval returning List[Document]."""
        if self.ensemble_retriever:
            try:
                if hasattr(self.ensemble_retriever, "invoke"):
                    return self.ensemble_retriever.invoke(query)
                return self.ensemble_retriever.get_relevant_documents(query)
            except Exception as e:
                logger.warning(f"Ensemble invoke failed: {e}")

        if self.chroma_retriever:
            try:
                return self.chroma_retriever.invoke(query)
            except Exception as e:
                logger.warning(f"Chroma invoke failed: {e}")

        # Fallback to initial docs
        return self.documents[:self.top_k]

    def retrieve(self, query: str, section_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compatible retrieval method returning formatted hit dictionaries for the pipeline."""
        docs = self.retrieve_documents(query)

        hits = []
        for i, doc in enumerate(docs):
            sec = doc.metadata.get("section")
            if section_filter and sec != section_filter:
                continue

            hits.append({
                "id": doc.metadata.get("doc_id", f"doc_{i}"),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": round(1.0 - (i * 0.1), 3),
                "type": "langchain_ensemble"
            })
            if len(hits) >= self.top_k:
                break

        if not hits and self.documents:
            for i, doc in enumerate(self.documents[:2]):
                hits.append({
                    "id": doc.metadata.get("doc_id", f"fallback_{i}"),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 0.5,
                    "type": "fallback"
                })

        return hits
