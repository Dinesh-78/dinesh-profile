"""Output guardrail for faithfulness, persona enforcement, and toxicity prevention."""

import re
import logging
from typing import List, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class OutputGuardrailResult(BaseModel):
    """Result of an output guardrail check."""
    passed: bool
    sanitized_output: str
    hallucination_flag: bool = False
    toxicity_flag: bool = False
    notes: List[str] = Field(default_factory=list)

class OutputGuardrail:
    """Evaluates and sanitizes generated answers before serving them to the user."""

    # Disallowed toxic or offensive patterns
    TOXIC_PATTERNS = [
        r"(?i)\b(hate|stupid|idiot|incompetent|kill|die|f\*\*\*|bitch|bastard)\b"
    ]

    # Specific ungrounded buzzwords that should never be claimed unless explicitly in context
    KNOWN_FABRICATIONS = [
        "worked at Google as Director",
        "PhD from Stanford",
        "invented Bitcoin",
        "won Nobel Prize",
        "worked at NASA"
    ]

    def __init__(self):
        self._compiled_toxic = [re.compile(p) for p in self.TOXIC_PATTERNS]

    def validate(
        self,
        output_text: str,
        retrieved_docs: List[Dict[str, Any]],
        user_query: str
    ) -> OutputGuardrailResult:
        """Validates generated text against context and safety policies."""
        sanitized = output_text.strip()
        notes = []
        hallucination_flag = False
        toxicity_flag = False

        # 1. Toxicity check
        for pattern in self._compiled_toxic:
            if pattern.search(sanitized):
                toxicity_flag = True
                notes.append("Potentially toxic or inappropriate phrasing detected.")
                sanitized = re.sub(pattern, "[redacted]", sanitized)

        # 2. Known high-impact fabrication check
        for fab in self.KNOWN_FABRICATIONS:
            if fab.lower() in sanitized.lower():
                hallucination_flag = True
                notes.append(f"Detected ungrounded high-impact hallucination: '{fab}'")
                sanitized = (
                    "Dinesh has a strong engineering background in Generative AI and software development. "
                    "For accurate, verified career credentials, please consult his official resume or reach out at veerapareddydinesh@gmail.com."
                )

        # 3. Groundedness validation heuristic
        # If retrieved_docs is empty and output presents specific ungrounded facts
        if not retrieved_docs and len(sanitized) > 150:
            notes.append("Output generated without retrieved context; appending verification disclaimer.")
            if "veerapareddydinesh@gmail.com" not in sanitized:
                sanitized += "\n\n*Note: To verify detailed questions beyond the public profile, please contact Dinesh at veerapareddydinesh@gmail.com.*"

        # 4. Professional Contact check
        # If user asks about contact, interview, or hire, make sure email is present
        hiring_intents = ["interview", "hire", "contact", "email", "phone", "reach", "schedule"]
        if any(intent in user_query.lower() for intent in hiring_intents):
            if "veerapareddydinesh@gmail.com" not in sanitized:
                sanitized += "\n\nYou can reach Dinesh directly at **veerapareddydinesh@gmail.com** or via [LinkedIn](https://linkedin.com/in/veerapareddydinesh)."
                notes.append("Appended direct contact information for hiring query.")

        passed = not toxicity_flag and not hallucination_flag

        return OutputGuardrailResult(
            passed=passed,
            sanitized_output=sanitized,
            hallucination_flag=hallucination_flag,
            toxicity_flag=toxicity_flag,
            notes=notes
        )

