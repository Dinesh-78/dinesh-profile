"""Input guardrails for prompt injection defense, topic enforcement, and PII protection."""

import re
import logging
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class GuardrailResult(BaseModel):
    """Result of an input guardrail check."""
    passed: bool
    action: str = Field(description="'ALLOW', 'REDIRECT', or 'BLOCK'")
    reason: str
    category: str = Field(description="'OK', 'PROMPT_INJECTION', 'OFF_TOPIC', 'UNSAFE_PII'")
    suggested_response: str = ""

class InputGuardrail:
    """Multi-tiered input guardrail ensuring queries stay safe, relevant, and in-scope."""

    # Prompt injection and jailbreak signatures
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous\s+|above\s+|prior\s+)?(instructions|prompts|rules)",
        r"(?i)disregard\s+(the\s+)?(previous\s+|above\s+|prior\s+)?(instructions|system\s+message)",
        r"(?i)system\s+override",
        r"(?i)you\s+are\s+now\s+(in\s+)?(developer\s+mode|dan\s+mode|unrestricted\s+mode|god\s+mode)",
        r"(?i)jailbreak",
        r"(?i)reveal\s+(your|the)\s+(system\s+prompt|initial\s+instructions|hidden\s+prompt)",
        r"(?i)what\s+(is|are)\s+your\s+(exact\s+)?(system\s+prompt|system\s+instructions)",
        r"(?i)print\s+(the\s+)?system\s+prompt",
        r"(?i)bypass\s+(all\s+)?(safety|filters|guardrails)",
        r"(?i)do\s+anything\s+now",
        r"(?i)repeat\s+everything\s+above",
        r"(?i)<\s*system\s*>",
        r"(?i)\[\s*INST\s*\]",
        r"(?i)###\s*(instruction|system):",
    ]

    # Sensitive PII keywords that should not be queried or leaked
    SENSITIVE_PII_PATTERNS = [
        r"(?i)(home\s+address|street\s+address|bank\s+account|credit\s+card|social\s+security|aadhaar|ssn|passport\s+number|pin\s+code)",
        r"(?i)(password|secret\s+key|private\s+key|api\s+key|token)\s*(for|of|to)?",
    ]

    # Relevant keywords to assess topical relevance
    ON_TOPIC_KEYWORDS = {
        "dinesh", "veerapareddy", "experience", "skill", "skills", "project", "projects",
        "work", "job", "career", "role", "roles", "resume", "cv", "hire", "hiring", "recruiter",
        "interview", "tech", "technology", "stack", "python", "ai", "ml", "genai", "llm",
        "rag", "langchain", "llamaindex", "fastapi", "docker", "education", "college", "degree",
        "contact", "email", "linkedin", "github", "notice", "notice period", "availability",
        "salary", "relocation", "remote", "hybrid", "qualification", "portfolio", "strengths",
        "weakness", "certifications", "engineering", "architect", "vector", "chromadb", "groq",
        "gemini", "agent", "agents", "langgraph", "eval", "evals", "ragas"
    }

    # Explicit off-topic indicators (unrelated questions)
    OFF_TOPIC_PATTERNS = [
        r"(?i)\b(recipe|bake\s+a\s+cake|how\s+to\s+cook)\b",
        r"(?i)\b(write\s+a\s+love\s+poem|dating\s+advice|flirt)\b",
        r"(?i)\b(horoscope|astrology|zodiac)\b",
        r"(?i)\b(hack\s+a\s+wifi|sql\s+injection\s+tutorial|bypass\s+paywall)\b",
        r"(?i)\b(sports\s+scores|who\s+won\s+the\s+match|weather\s+in)\b",
        r"(?i)\b(crypto\s+price|bitcoin\s+prediction|stock\s+tips)\b",
    ]

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._compiled_injections = [re.compile(p) for p in self.INJECTION_PATTERNS]
        self._compiled_pii = [re.compile(p) for p in self.SENSITIVE_PII_PATTERNS]
        self._compiled_off_topic = [re.compile(p) for p in self.OFF_TOPIC_PATTERNS]

    def validate(self, query: str) -> GuardrailResult:
        """Evaluates an incoming user query against injection, PII, and off-topic checks."""
        clean_query = query.strip()
        if not clean_query:
            return GuardrailResult(
                passed=False,
                action="REDIRECT",
                reason="Query is empty.",
                category="OFF_TOPIC",
                suggested_response="Hello! I am Dinesh's AI Career Representative. Feel free to ask about his experience, technical skills, projects, or how to get in touch!"
            )

        # 1. Prompt Injection & Jailbreak check
        for pattern in self._compiled_injections:
            if pattern.search(clean_query):
                logger.warning(f"Prompt injection pattern detected: '{clean_query}'")
                return GuardrailResult(
                    passed=False,
                    action="BLOCK",
                    reason="Prompt injection / system prompt extraction attempt detected.",
                    category="PROMPT_INJECTION",
                    suggested_response=(
                        "I am configured to represent Dinesh's professional background and skills. "
                        "System instructions and administrative overrides are protected. "
                        "How can I help you evaluate Dinesh for your technical roles?"
                    )
                )

        # 2. Sensitive PII extraction check
        for pattern in self._compiled_pii:
            if pattern.search(clean_query):
                logger.warning(f"Sensitive PII access attempt detected: '{clean_query}'")
                return GuardrailResult(
                    passed=False,
                    action="BLOCK",
                    reason="Request seeks private PII or credentials.",
                    category="UNSAFE_PII",
                    suggested_response=(
                        "For privacy and security reasons, personal identity credentials, private addresses, "
                        "and internal secrets are not disclosed. You can reach Dinesh directly for authorized "
                        "recruiter inquiries at veerapareddydinesh@gmail.com."
                    )
                )

        # 3. Off-Topic Check
        for pattern in self._compiled_off_topic:
            if pattern.search(clean_query):
                return GuardrailResult(
                    passed=False,
                    action="REDIRECT",
                    reason="Query is explicitly off-topic from Dinesh's profile and hiring context.",
                    category="OFF_TOPIC",
                    suggested_response=(
                        "I specialize specifically in answering questions regarding Veerapareddy Dinesh's "
                        "career, technical expertise in Generative AI / LLM engineering, featured projects, and hiring availability. "
                        "Feel free to ask questions like: 'What is Dinesh's experience with RAG?' or 'How can I schedule an interview?'"
                    )
                )

        # 4. Strict Mode Topical Keyword Verification
        if self.strict_mode:
            words = set(re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", clean_query.lower()))
            matches = words.intersection(self.ON_TOPIC_KEYWORDS)
            # If the query is relatively long (> 5 words) and has zero overlap with any career/technical/profile terms
            if len(words) >= 6 and len(matches) == 0:
                logger.info(f"Query flagged as low-relevance to profile: '{clean_query}'")
                return GuardrailResult(
                    passed=False,
                    action="REDIRECT",
                    reason="Query has low semantic overlap with professional profile topics.",
                    category="OFF_TOPIC",
                    suggested_response=(
                        "I'm here to provide information about Dinesh's software engineering and Generative AI career. "
                        "Could you please relate your question to Dinesh's background, projects, or technical qualifications?"
                    )
                )

        return GuardrailResult(
            passed=True,
            action="ALLOW",
            reason="Input passed all safety, injection, and relevance guardrail checks.",
            category="OK"
        )

