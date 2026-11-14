"""Grounded response generator using Google Gemini and Groq with fallback."""

import logging
from typing import List, Dict, Any, Generator
from src.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the official AI Career Representative for Veerapareddy Dinesh (Dinesh).
Your role is to represent Dinesh to recruiters, engineering leaders, and hiring managers with complete factual accuracy, professionalism, and clarity.

GROUND TRUTH CAREER FACTS FOR DINESH:
- Full Name: Veerapareddy Dinesh
- Contact: dineshveerapareddy@gmail.com | Phone: +91-8374330906 | LinkedIn: https://www.linkedin.com/in/dinesh-veerapareddy/ | Portfolio: https://www.veerapareddydinesh.cyou/
- Title: AI Engineer | GenAI Full Stack | Forward Deployed Engineer
- Current Role & Company: Associate Software Engineer I at Carelon Global Solutions India (Bengaluru, Feb 2026 – Present), owning the healthcare Benefits and Insurance module (Java, Spring Boot, Angular, AWS).
- Previous Experience:
  1. Prajwal Fintech Private Limited (Hyderabad, Mar 2025 – Feb 2026): Software Engineer leading MERN stack development, Kafka, Redis caching, microservices, CI/CD with Docker on GCP/AWS.
  2. MOVE37 Productions (Remote, May 2024 – Mar 2025): Full Stack Developer building AI audio-transcription feature, Next.js, TypeScript.
- Total Experience: ~3 years (2.5+ years production experience).
- Notice Period / Availability: 15 days (Available to join in 15 days).
- Current Salary: 33,000 INR per month | Current CTC: 4,50,000 INR.
- Education: B.Tech in Computer Science and Engineering from KL University (2020 – 2024), CGPA: 8.9 / 10.0.
- Certifications: AWS Certified Cloud Practitioner, Wipro Java Full Stack Certification.

CORE OPERATIONAL RULES:
1. STRICT ACCURACY: Answer questions based strictly on the verified profile context provided below and the ground truth facts above. Do NOT invent past companies, degrees, or experiences.
2. TONE: Professional, executive, technically precise, and welcoming. Speak in the third person as Dinesh's representative (e.g. "Dinesh is currently working at Carelon Global Solutions...", "Dinesh has 2.5+ years of production experience...").
3. RECRUITER VALUE: Clearly articulate his architectural skills (Kafka, Redis, Spring Boot, FastAPI, RAG, tool-calling agents) and his quantifiable impacts (20% API latency cut at Carelon, 40% performance gain at Prajwal Fintech).
4. CONTACT: For interview requests, availability, or salary questions, always provide his direct contact email (dineshveerapareddy@gmail.com) and phone (+91-8374330906).
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
        """Generates a grounded answer given the user query, retrieved documents, and Neo4j graph facts."""
        formatted_context = self.format_context(retrieved_docs)

        # Retrieve Neo4j Knowledge Graph context (GraphRAG)
        from src.rag.graph_store import get_graph_store
        graph_store = get_graph_store()
        graph_context = graph_store.query_graph_context(query)

        # Build prompt combining Vector Context + Neo4j Graph Facts
        combined_context = f"{formatted_context}\n\n{graph_context}"

        citations = [
            {
                "id": doc.get("id"),
                "title": doc.get("metadata", {}).get("title", "Source"),
                "section": doc.get("metadata", {}).get("section", "profile"),
                "source": doc.get("metadata", {}).get("source", "profile_data")
            }
            for doc in retrieved_docs
        ]

        # 1. Fast, high-accuracy Google Gemini REST invocation (<1s latency)
        if settings.GOOGLE_API_KEY:
            try:
                import httpx
                model_name = settings.GEMINI_MODEL
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={settings.GOOGLE_API_KEY}"
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"{SYSTEM_PROMPT}\n\nVERIFIED PROFILE & GRAPH CONTEXT:\n{combined_context}\n\nUSER QUESTION:\n{query}"
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.2,
                        "maxOutputTokens": 1024
                    }
                }
                resp = httpx.post(url, json=payload, timeout=12.0)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        if text and text.strip():
                            return {
                                "answer": text.strip(),
                                "citations": citations,
                                "model_used": model_name
                            }
                else:
                    logger.warning(f"Gemini API returned {resp.status_code}: {resp.text}")
            except Exception as ex:
                logger.error(f"Direct REST API error: {ex}")

        # 2. LangChain fallback
        if self.llm:
            try:
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=f"Context:\n{formatted_context}\n\nQuestion: {query}")
                ]
                response = self.llm.invoke(messages)
                if hasattr(response, "content"):
                    if isinstance(response.content, list):
                        answer = "".join([p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in response.content])
                    else:
                        answer = str(response.content)
                else:
                    answer = str(response)

                if answer and answer.strip():
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
                "Please reach out directly to Dinesh at dineshveerapareddy@gmail.com for more information."
            )
        top_doc = retrieved_docs[0]
        title = top_doc.get("metadata", {}).get("title", "Relevant Experience")
        content = top_doc.get("content", "")
        return (
            f"Regarding your query about Dinesh:\n\n"
            f"Based on his {title}:\n{content}\n\n"
            f"For further questions or interview scheduling, feel free to contact Dinesh at dineshveerapareddy@gmail.com."
        )

