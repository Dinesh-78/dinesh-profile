"""Grounded response generator using Google Gemini and Groq with fallback."""

import logging
from typing import List, Dict, Any, Generator
from src.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the official AI Career Representative for Veerapareddy Dinesh (Dinesh).
Your role is to represent Dinesh to recruiters, engineering leaders, and hiring managers with clarity, professionalism, accuracy, and enthusiasm.

CORE OPERATIONAL RULES:
1. STRICT GROUNDEDNESS: Only answer questions based on the verified profile context provided below. If a specific detail (e.g. personal salary history, unrelated personal life) is NOT in the context, politely state: "I don't have that specific information in Dinesh's public profile, but you can reach him directly at veerapareddydinesh@gmail.com."
2. TONE: Professional, articulate, technically sound, welcoming, and executive. Speak as Dinesh's representative (e.g., "Dinesh has extensive experience in...", "In his recent project...").
3. ACCURACY: Do NOT hallucinate past companies, degrees, or metrics that do not appear in the context.
4. RECRUITER VALUE: Highlight relevant projects, architectural decisions, and production practices (e.g., guardrails, evals, hybrid search, async FastAPI) when discussing his qualifications.
5. CONTACT CALLOUT: For interview requests, availability, or direct discussions, always invite the recruiter to contact Dinesh at veerapareddydinesh@gmail.com or via LinkedIn.
"""

class GroundedGenerator:
    """Generates grounded responses using configured LLMs with citation tracking."""

    def __init__(self):
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes the primary LLM (Gemini) or Groq."""
        if settings.LLM_PROVIDER == "gemini" and settings.GOOGLE_API_KEY:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                logger.info(f"Initialized Google Gemini model: {settings.GEMINI_MODEL}")
                return ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    google_api_key=settings.GOOGLE_API_KEY,
                    temperature=0.2
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini LLM: {e}. Trying Groq fallback.")

        if settings.GROQ_API_KEY:
            try:
                from langchain_groq import ChatGroq
                logger.info(f"Initialized Groq fallback model: {settings.GROQ_MODEL}")
                return ChatGroq(
                    model=settings.GROQ_MODEL,
                    api_key=settings.GROQ_API_KEY,
                    temperature=0.2
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM: {e}")

        return None

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into numbered context blocks for prompt injection."""
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            title = doc.get("metadata", {}).get("title", "Profile Context")
            section = doc.get("metadata", {}).get("section", "General")
            source = doc.get("metadata", {}).get("source", "profile")
            content = doc.get("content", "").strip()
            context_blocks.append(f"--- [Source {i}: {title} (Section: {section}, File: {source})] ---\n{content}\n")
        return "\n".join(context_blocks)

    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Generates a grounded answer given the user query and retrieved documents."""
        formatted_context = self.format_context(retrieved_docs)

        # Build prompt
        prompt = f"""{SYSTEM_PROMPT}

VERIFIED PROFILE CONTEXT:
{formatted_context}

USER QUESTION:
{query}

Please provide a clear, accurate, and professional answer based strictly on the verified profile context above. Mention relevant sources where applicable."""

        citations = [
            {
                "id": doc.get("id"),
                "title": doc.get("metadata", {}).get("title", "Source"),
                "section": doc.get("metadata", {}).get("section", "profile"),
                "source": doc.get("metadata", {}).get("source", "profile_data")
            }
            for doc in retrieved_docs
        ]

        if self.llm:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=f"Context:\n{formatted_context}\n\nQuestion: {query}")
                ]
                response = self.llm.invoke(messages)
                answer = response.content if hasattr(response, "content") else str(response)
                return {
                    "answer": answer.strip(),
                    "citations": citations,
                    "model_used": getattr(self.llm, "model", settings.GEMINI_MODEL)
                }
            except Exception as e:
                logger.error(f"Error calling LLM: {e}")

        # Deterministic grounded fallback response if API is unreachable
        fallback_answer = self._generate_fallback(query, retrieved_docs)
        return {
            "answer": fallback_answer,
            "citations": citations,
            "model_used": "deterministic-fallback"
        }

    def _generate_fallback(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Rule-based synthesis for offline resilience."""
        if not retrieved_docs:
            return (
                "Thank you for your interest in Dinesh's profile. "
                "I couldn't locate specific details on that topic in the profile records. "
                "Please reach out directly to Dinesh at veerapareddydinesh@gmail.com for more information."
            )
        top_doc = retrieved_docs[0]
        title = top_doc.get("metadata", {}).get("title", "Relevant Experience")
        content = top_doc.get("content", "")
        return (
            f"Regarding your query about Dinesh:\n\n"
            f"Based on his {title}:\n{content}\n\n"
            f"For further questions or interview scheduling, feel free to contact Dinesh at veerapareddydinesh@gmail.com."
        )

