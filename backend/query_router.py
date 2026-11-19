import re
import json
import logging
from typing import Dict, Any, List, Optional
from backend.llm_provider import get_llm_provider
from backend.vector_store import SessionVectorStore

logger = logging.getLogger("intellirag.router")

ROUTER_SYSTEM_PROMPT = """You are the Intent Classification and Query Understanding engine for IntelliRAG.
Analyze the user's question, accounting for informal language, spelling mistakes, and grammatical errors.

Inputs provided:
1. User Question
2. Document Name (if any)
3. Document Sample Topics/Summary (if any)

Your tasks:
1. "intent": Choose exactly one:
   - "DOCUMENT_SEARCH": The question asks specifically about the contents, people, data, or policies in the uploaded document.
   - "GENERAL_KNOWLEDGE": The question asks about general computer science, AI/ML, science, history, geography, mathematics, or world facts NOT specific to the document.
   - "HYBRID": The question requires comparing or combining information from the document with external world knowledge.
   - "AMBIGUOUS": The question is too vague or underspecified (e.g., "what about it?", "is it right?") to answer definitively without explaining possible interpretations.
2. "cleaned_query": Rephrase the question into clear, standard, grammatically correct English while preserving 100% of the user's intended meaning.
3. "is_multihop": True if answering requires 2 or more distinct logical steps or separate factual inquiries; False if it is a single-step question.
4. "ambiguity_note": If AMBIGUOUS, provide a brief note on what is unclear and how you are interpreting it; otherwise null.

Respond ONLY with valid JSON in this exact structure:
{
  "intent": "DOCUMENT_SEARCH" | "GENERAL_KNOWLEDGE" | "HYBRID" | "AMBIGUOUS",
  "cleaned_query": "Cleaned question text",
  "is_multihop": false,
  "ambiguity_note": null
}"""


def route_and_understand_query(
    question: str,
    vector_store: Optional[SessionVectorStore] = None,
) -> Dict[str, Any]:
    """
    Classifies intent, normalizes informal language, and decides knowledge source strategy.
    """
    raw_q = question.strip()
    if not raw_q:
        return {
            "intent": "AMBIGUOUS",
            "cleaned_query": "",
            "is_multihop": False,
            "ambiguity_note": "The question was empty.",
        }

    llm = get_llm_provider()
    doc_name = vector_store.filename if vector_store else "None"
    doc_preview = ""
    if vector_store and vector_store.chunks:
        # Give a 200-character preview of top 2 chunks so the router knows document domain
        sample_texts = " | ".join(c["text"][:120].replace("\n", " ") for c in vector_store.chunks[:2])
        doc_preview = f"Sample Excerpt: {sample_texts}"

    if llm.is_available() and "Gemini" in llm.name:
        prompt = (
            f"User Question: {raw_q}\n"
            f"Uploaded Document: {doc_name}\n"
            f"{doc_preview}\n\n"
            f"JSON Output:"
        )
        try:
            resp = llm.generate(
                prompt=prompt,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=300,
            )
            clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.strip(), flags=re.MULTILINE).strip()
            match = re.search(r"\{.*\}", clean_json, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                intent = data.get("intent", "DOCUMENT_SEARCH")
                cleaned = data.get("cleaned_query") or raw_q
                is_multihop = bool(data.get("is_multihop", False))
                ambiguity_note = data.get("ambiguity_note")
                return {
                    "intent": intent,
                    "cleaned_query": cleaned,
                    "is_multihop": is_multihop,
                    "ambiguity_note": ambiguity_note,
                }
        except Exception as exc:
            logger.warning(f"Router LLM classification failed, using heuristic: {exc}")

    # Fallback heuristic if LLM router is unavailable
    is_multihop = bool(
        raw_q.count("?") >= 2
        or re.search(r"\b(and what|along with|furthermore|as well as)\b", raw_q, re.I)
    )

    intent = "DOCUMENT_SEARCH" if vector_store else "GENERAL_KNOWLEDGE"
    if vector_store:
        # Check similarity of question against vector store
        probe = vector_store.search(raw_q, top_k=1)
        sim = probe[0]["similarity_score"] if probe else 0.0
        if sim < 0.25:
            # Low similarity to document -> general knowledge
            intent = "GENERAL_KNOWLEDGE"

    return {
        "intent": intent,
        "cleaned_query": raw_q,
        "is_multihop": is_multihop,
        "ambiguity_note": None,
    }

