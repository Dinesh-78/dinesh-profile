"""Complete guarded RAG pipeline orchestrator."""

import time
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.config import settings
from src.rag.retriever import HybridRetriever
from src.rag.generator import GroundedGenerator
from src.guardrails.input_guardrail import InputGuardrail, GuardrailResult
from src.guardrails.output_guardrail import OutputGuardrail, OutputGuardrailResult

logger = logging.getLogger(__name__)

class PipelineResponse(BaseModel):
    """Structured response from the complete Guarded RAG pipeline."""
    query: str
    response: str
    guardrail_status: str = Field(description="'PASSED', 'REDIRECTED', or 'BLOCKED'")
    category: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    input_guardrail: Dict[str, Any]
    output_guardrail: Optional[Dict[str, Any]] = None
    latency_ms: float
    model_used: str = ""

class GuardedProfilePipeline:
    """Production orchestrator combining Input Guardrails, Hybrid Retrieval, Grounded Generation, and Output Guardrails."""

    def __init__(self):
        self.input_guardrail = InputGuardrail(strict_mode=settings.GUARDRAIL_STRICT_MODE)
        self.output_guardrail = OutputGuardrail()
        self.retriever = HybridRetriever()
        self.generator = GroundedGenerator()

    def process(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        section_filter: Optional[str] = None
    ) -> PipelineResponse:
        """Executes the complete guarded RAG pipeline."""
        start_time = time.time()

        # Step 1: Input Guardrail Evaluation
        if settings.GUARDRAILS_ENABLED:
            input_res: GuardrailResult = self.input_guardrail.validate(query)
            if not input_res.passed:
                latency = round((time.time() - start_time) * 1000, 2)
                return PipelineResponse(
                    query=query,
                    response=input_res.suggested_response,
                    guardrail_status=input_res.action,
                    category=input_res.category,
                    citations=[],
                    input_guardrail=input_res.model_dump(),
                    output_guardrail=None,
                    latency_ms=latency,
                    model_used="guardrail-interceptor"
                )
        else:
            input_res = GuardrailResult(passed=True, action="ALLOW", reason="Guardrails disabled", category="OK")

        # Step 2: Hybrid Retrieval
        retrieved_docs = self.retriever.retrieve(query, section_filter=section_filter)

        # Step 3: Generation
        gen_result = self.generator.generate_response(query, retrieved_docs, chat_history)
        raw_answer = gen_result["answer"]
        citations = gen_result["citations"]
        model_used = gen_result.get("model_used", "unknown")

        # Step 4: Output Guardrail Evaluation
        if settings.GUARDRAILS_ENABLED:
            output_res: OutputGuardrailResult = self.output_guardrail.validate(
                output_text=raw_answer,
                retrieved_docs=retrieved_docs,
                user_query=query
            )
            final_answer = output_res.sanitized_output
            output_dump = output_res.model_dump()
        else:
            final_answer = raw_answer
            output_dump = {"passed": True, "notes": ["Output guardrails disabled"]}

        latency = round((time.time() - start_time) * 1000, 2)

        return PipelineResponse(
            query=query,
            response=final_answer,
            guardrail_status="PASSED",
            category="OK",
            citations=citations,
            input_guardrail=input_res.model_dump(),
            output_guardrail=output_dump,
            latency_ms=latency,
            model_used=model_used
        )

_pipeline_instance = None

def get_pipeline() -> GuardedProfilePipeline:
    """Singleton getter for the RAG pipeline."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = GuardedProfilePipeline()
    return _pipeline_instance

