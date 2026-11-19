import re
import json
import logging
from typing import List, Optional
from backend.llm_provider import get_llm_provider

logger = logging.getLogger("intellirag.decomposer")

QUESTION_STARTERS = (
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "list", "explain", "describe", "identify", "state", "compare"
)

DECOMPOSITION_SYSTEM_PROMPT = """You are an intelligent query decomposition engine for a general-purpose Question-Answering system.
Your job is to analyze the user question and determine whether and how to break it down.

Rules:
1. 1-Hop (Single Concept or Question):
   If the question is about a single topic, definition, mechanism, or fact (e.g., "What is evaporation?", "Explain how gradient descent works", "What is the difference between TCP and UDP?", "What is her CGPA at VIT-AP University?"), DO NOT over-split it! Keep it as a SINGLE focused sub-question.
2. Multi-Part / Independent Questions:
   If the user asks multiple distinct questions combined with "and", "as well as", or punctuation (e.g., "What is evaporation, what conditions make it faster, and how do clouds form?"), decompose it into distinct sub-questions.
3. Dependent / Chained Multi-Hop Questions:
   If answering requires sequential discovery (e.g., "Find the director of Inception and what awards they won"), break it into the logical steps.
4. Entity Preservation:
   Ensure each sub-question retains the necessary nouns/entities so it can be understood in isolation.
5. Output Format:
   Output ONLY a valid JSON list of strings, for example:
   ["Sub-question 1?", "Sub-question 2?"]
   Do NOT wrap in extra markdown or commentary."""


def _clean_sub_question(text: str) -> str:
    cleaned = re.sub(r"^[\s,\-–—•*\d.)]+", "", text).strip()
    cleaned = re.sub(
        r"^(?:and|also|furthermore|moreover|then|secondly|thirdly|finally)\s+",
        "",
        cleaned,
        flags=re.I,
    ).strip()
    if not cleaned:
        return ""
    cleaned = cleaned[0].upper() + cleaned[1:]
    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".") + "?"
    return cleaned


def _fallback_rule_decomposition(question: str) -> List[str]:
    """Fallback linguistic splitter for offline mode."""
    q = question.strip()
    if q.count("?") >= 2:
        parts = [_clean_sub_question(p) for p in q.split("?") if len(p.strip()) >= 8]
        if len(parts) >= 2:
            return parts[:4]

    starter_pattern = "|".join(QUESTION_STARTERS)
    split_regex = rf"(?:,\s*(?:and\s+|or\s+)?|\s+(?:and|while|whereas|also)\s+|;\s*)(?=(?:{starter_pattern})\b)"
    raw_parts = re.split(split_regex, q, flags=re.I)
    cleaned_parts = [_clean_sub_question(p) for p in raw_parts if len(p.strip()) >= 10]
    if len(cleaned_parts) >= 2:
        return cleaned_parts[:4]

    single = _clean_sub_question(q)
    return [single] if single else [q]


def decompose_question(question: str, is_multihop_hint: Optional[bool] = None) -> List[str]:
    """
    Decomposes a user query into 1 to 4 sequential sub-questions using Gemini.
    Leaves 1-hop and single-concept questions as a single sub-question.
    """
    q = question.strip()
    if not q:
        return []

    # If already known to be a simple 1-hop question and contains no conjunctions/multiple question marks
    if is_multihop_hint is False and q.count("?") <= 1 and not re.search(r"\b(and|while|as well as|furthermore)\b", q, re.I):
        return [_clean_sub_question(q)]

    llm = get_llm_provider()
    if llm.is_available() and "Gemini" in llm.name:
        try:
            prompt = f"User Question: {q}\nOutput JSON list of sub-questions:"
            raw_response = llm.generate(
                prompt=prompt,
                system_prompt=DECOMPOSITION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=256,
            )
            cleaned_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip(), flags=re.MULTILINE).strip()
            match = re.search(r"\[.*\]", cleaned_json, re.DOTALL)
            if match:
                items = json.loads(match.group(0))
                if isinstance(items, list) and items:
                    parsed = [_clean_sub_question(str(item)) for item in items if str(item).strip()]
                    if parsed:
                        return parsed[:4]
        except Exception as exc:
            logger.warning(f"LLM question decomposition failed, using fallback: {exc}")

    return _fallback_rule_decomposition(q)
