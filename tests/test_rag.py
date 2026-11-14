"""Unit tests for Chunking, Indexing, and Retrieval."""

import pytest
from src.rag.indexer import ProfileIndexer
from src.rag.retriever import HybridRetriever

def test_directory_loader_and_splitting():
    from src.rag.loader import ProfileDirectoryLoader
    loader = ProfileDirectoryLoader()
    raw_docs = loader.load_documents()
    assert len(raw_docs) >= 1
    assert any("Dinesh" in d.page_content for d in raw_docs)

    split_docs = loader.load_and_split()
    assert len(split_docs) > 1
    assert all(hasattr(d, "page_content") for d in split_docs)
    assert all("source" in d.metadata for d in split_docs)

def test_profile_indexer_chunking():
    indexer = ProfileIndexer()
    docs = indexer.load_and_chunk()
    assert len(docs) > 0

    # Ensure sections exist in LangChain Documents
    sections = {d.metadata["section"] for d in docs}
    assert "personal_info" in sections
    assert "skills" in sections
    assert "experience" in sections
    assert "projects" in sections
    assert "faq" in sections

def test_hybrid_retriever_initialization_and_search():
    retriever = HybridRetriever(top_k=3)
    results = retriever.retrieve("What is Dinesh's experience with RAG and FastAPI?")
    assert len(results) > 0
    assert any("RAG" in r["content"] or "FastAPI" in r["content"] for r in results)

def test_retriever_recruiter_faq_hit():
    retriever = HybridRetriever(top_k=2)
    results = retriever.retrieve("Why should we hire Dinesh for an LLM engineer role?")
    assert len(results) > 0
    top_doc = results[0]
    assert "Dinesh" in top_doc["content"]

def test_efficient_agent_routing_and_caching():
    from src.rag.agent import EfficientProfileAgent
    agent = EfficientProfileAgent()

    # Fast intent route
    res1 = agent.run("hello")
    assert res1["cached"] is True
    assert res1["strategy"] == "fast_intent_router"
    assert res1["latency_ms"] < 20

    # RAG lookup with caching
    res2 = agent.run("What is Dinesh's experience with RAG?")
    assert "answer" in res2
    assert res2["cached"] is False

    # Second lookup should hit cache
    res3 = agent.run("What is Dinesh's experience with RAG?")
    assert res3["cached"] is True
    assert res3["latency_ms"] < 20

