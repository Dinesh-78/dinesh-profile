"""Production FastAPI Backend for Dinesh's Career Profile RAG Pipeline."""

import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.guardrails.pipeline import get_pipeline, PipelineResponse
from src.evals.run_evals import run_evaluation_suite
from src.rag.indexer import ProfileIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Veerapareddy Dinesh - AI Career Representative API",
    description="Production-grade RAG pipeline with dual-stage guardrails and AI evals.",
    version="1.0.0"
)

# CORS middleware for modern frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str = Field(..., example="What is Dinesh's experience with RAG and LLM guardrails?")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=None)
    section_filter: Optional[str] = Field(default=None, example="experience")

class ChatResponse(BaseModel):
    query: str
    response: str
    guardrail_status: str
    category: str
    citations: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]
    latency_ms: float
    model_used: str

@app.on_event("startup")
def startup_event():
    """Ensure ChromaDB index is warmed up at startup."""
    try:
        indexer = ProfileIndexer()
        indexer.build_index()
        logger.info("Knowledge base vector store ready.")
    except Exception as e:
        logger.warning(f"Startup indexing note: {e}")

@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "Veerapareddy Dinesh Portfolio RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "guardrails_enabled": settings.GUARDRAILS_ENABLED,
        "model": settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.GROQ_MODEL
    }

@app.post("/api/chat", response_model=ChatResponse, tags=["RAG Chat"])
def chat(request: ChatRequest):
    """Processes a user/recruiter query through the guarded RAG pipeline."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    pipeline = get_pipeline()
    pipeline_res: PipelineResponse = pipeline.process(
        query=request.query,
        chat_history=request.chat_history,
        section_filter=request.section_filter
    )

    return ChatResponse(
        query=pipeline_res.query,
        response=pipeline_res.response,
        guardrail_status=pipeline_res.guardrail_status,
        category=pipeline_res.category,
        citations=pipeline_res.citations,
        diagnostics={
            "input_guardrail": pipeline_res.input_guardrail,
            "output_guardrail": pipeline_res.output_guardrail
        },
        latency_ms=pipeline_res.latency_ms,
        model_used=pipeline_res.model_used
    )

@app.get("/api/profile/summary", tags=["Profile"])
def profile_summary():
    """Returns structured summary of Dinesh's profile."""
    import json
    from pathlib import Path
    data_path = Path(settings.DATA_PATH)
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Profile data not found.")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "name": data.get("personal_info", {}).get("full_name"),
        "title": data.get("personal_info", {}).get("title"),
        "email": data.get("personal_info", {}).get("email"),
        "github": data.get("personal_info", {}).get("github"),
        "linkedin": data.get("personal_info", {}).get("linkedin"),
        "summary": data.get("personal_info", {}).get("summary"),
        "target_roles": data.get("career_preferences", {}).get("target_roles"),
        "featured_projects_count": len(data.get("featured_projects", []))
    }

@app.post("/api/evals/run", tags=["Evaluations"])
def trigger_evaluations():
    """Triggers the automated AI evaluation suite and returns composite benchmark results."""
    try:
        report = run_evaluation_suite()
        if not report:
            raise HTTPException(status_code=500, detail="Evaluation failed to produce report.")
        return {
            "status": "completed",
            "benchmark_score": report.overall_benchmark_score,
            "guardrail_accuracy": report.guardrail_accuracy,
            "faithfulness": report.avg_faithfulness,
            "relevance": report.avg_answer_relevance,
            "context_recall": report.avg_context_recall,
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests
        }
    except Exception as e:
        logger.error(f"Error during eval execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

