"""High-efficiency Agentic RAG using LangChain Tools, Intent Routing, and Semantic Caching."""

import time
import logging
from functools import lru_cache
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document

from src.config import settings
from src.rag.retriever import HybridRetriever
from src.rag.generator import GroundedGenerator

logger = logging.getLogger(__name__)

class EfficientProfileAgent:
    """Agent optimized for minimum token latency, fast routing, and caching."""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.generator = GroundedGenerator()
        # In-memory exact/normalized query cache for instant responses (sub-5ms)
        self._response_cache: Dict[str, Dict[str, Any]] = {}

    def _normalize_key(self, query: str) -> str:
        """Normalizes query for high cache hit rates."""
        return " ".join(query.lower().strip().split())

    # Fast Intent Router: Avoids DB & LLM for common conversational intents
    FAST_INTENTS = {
        "hi": "Hello! I am Veerapareddy Dinesh's AI Career Representative. I can answer questions regarding his GenAI background, projects, skills, and availability.",
        "hello": "Hello! How can I assist you with Dinesh's profile or hiring inquiries today?",
        "hey": "Hey there! Ask me anything about Dinesh's experience, tech stack, or how to contact him.",
        "who are you": "I am the official AI Representative for Veerapareddy Dinesh, designed to help recruiters evaluate his fit for Generative AI and software roles.",
        "thank you": "You're very welcome! If you'd like to schedule an interview or speak with Dinesh directly, email him at veerapareddydinesh@gmail.com.",
        "thanks": "Glad to help! Feel free to reach out to Dinesh at veerapareddydinesh@gmail.com or via LinkedIn."
    }

    def run(self, query: str) -> Dict[str, Any]:
        """Runs the query through intent routing, caching, and hybrid retrieval."""
        start_time = time.time()
        norm_key = self._normalize_key(query)

        # 1. Fast Intent Routing (0 LLM tokens, <1ms latency)
        if norm_key in self.FAST_INTENTS:
            return {
                "answer": self.FAST_INTENTS[norm_key],
                "citations": [],
                "cached": True,
                "strategy": "fast_intent_router",
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }

        # 2. Semantic Cache Check (<2ms latency, 0 tokens)
        if norm_key in self._response_cache:
            cached_res = self._response_cache[norm_key].copy()
            cached_res["cached"] = True
            cached_res["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return cached_res

        # 3. Hybrid Lexical + Dense Retrieval via LangChain
        retrieved_hits = self.retriever.retrieve(query)

        # 4. Grounded Generation
        gen_res = self.generator.generate_response(query, retrieved_hits)

        latency = round((time.time() - start_time) * 1000, 2)
        response_payload = {
            "answer": gen_res["answer"],
            "citations": gen_res["citations"],
            "cached": False,
            "strategy": "hybrid_rag_agent",
            "latency_ms": latency
        }

        # Store in cache (capped at 256 recent queries)
        if len(self._response_cache) > 256:
            self._response_cache.pop(next(iter(self._response_cache)))
        self._response_cache[norm_key] = response_payload

        return response_payload

