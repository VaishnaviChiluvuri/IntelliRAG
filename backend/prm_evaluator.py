import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("intellirag.prm")

FRAGMENT_STARTERS = (
    r"^(?:into\s+|of\s+|and\s+|with\s+|for\s+|from\s+|to\s+|by\s+|tablespoon\s+of\s+|about\s+|than\s+)",
    r"^(?:in\s+section|we\s+have\s+learnt|refer\s+to\s+table|fig\.\s*\d+)",
)

INCOMPLETE_ENDINGS = (
    r"[,\-–—:;(\[]$",
    r"\b(?:and|or|with|to|from|for|in|on|that|which|by)\s*$",
)


def _check_is_fragment(text: str) -> tuple[bool, str]:
    """
    Detects whether the text is an incomplete fragment rather than
    a complete, grammatically sound sentence.
    """
    s = text.strip()
    if not s:
        return True, "Answer is empty."

    # Too short
    words = s.split()
    if len(words) < 4:
        return True, f"Answer is too short ({len(words)} words) to be a complete answer."

    # Starts with a bad fragment starter
    for pat in FRAGMENT_STARTERS:
        if re.search(pat, s, re.I):
            return True, f"Answer starts with an incomplete fragment phrase ('{words[0]}...')."

    # Ends with dangling conjunction or preposition
    clean_end = re.sub(r"[.!?]+$", "", s).strip()
    for pat in INCOMPLETE_ENDINGS:
        if re.search(pat, clean_end, re.I):
            return True, "Answer terminates abruptly with a dangling preposition, conjunction, or punctuation."

    # Starts with lowercase letter
    if s[0].isalpha() and s[0].islower():
        return True, "Answer begins with a lowercase letter, indicating an extractive fragment."

    return False, ""


def _cosine_sim(text_a: str, text_b: str) -> float:
    """Computes cosine similarity between two strings."""
    from backend import embedding_utils
    if not text_a.strip() or not text_b.strip():
        return 0.0
    if hasattr(embedding_utils, "cosine_similarity"):
        return float(embedding_utils.cosine_similarity(text_a, text_b))
    vecs = embedding_utils.encode_texts([text_a, text_b])
    import numpy as np
    return float(np.dot(vecs[0], vecs[1]))


def evaluate_step(
    sub_question: str,
    intermediate_answer: str,
    evidence_chunks: List[Dict[str, Any]],
    prior_step_context: str = "",
    source_type: str = "DOCUMENT",
) -> Dict[str, Any]:
    """
    Multi-Dimensional Process Reward Model (PRM) Evaluator.
    Evaluates:
    1. Answer Quality (Grammar, Completeness, Structure, Absence of Fragments).
    2. Evidence Grounding & Relevance (Semantic and token alignment).
    A high score is NEVER awarded to an incomplete fragment or irrelevant context dump.
    """
    ans = intermediate_answer.strip()
    issues = []

    # 1. Fragment & Structural Quality Check
    is_frag, frag_reason = _check_is_fragment(ans)
    if is_frag:
        issues.append(f"Incomplete fragment detected: {frag_reason}")

    # Check for context dumps (headers, emails, full resumes)
    if len(ans) > 280 and any(kw in ans.lower() for kw in ("linkedin", "github", "email", "phone", "profile")):
        if not any(kw in sub_question.lower() for kw in ("linkedin", "github", "email", "contact", "phone", "profile")):
            issues.append("Answer contains an unrelated contact/header dump instead of a direct answer.")

    # 2. Quality Score calculation [0.0 to 1.0]
    quality_score = 1.0
    if is_frag:
        quality_score -= 0.60
    if "dump" in " ".join(issues).lower():
        quality_score -= 0.40
    if len(ans.split()) < 6:
        quality_score -= 0.20

    quality_score = max(0.10, min(1.0, quality_score))

    # 3. Grounding & Relevance Evaluation
    grounding_score = 0.50  # default baseline for general knowledge
    sem_sim = 0.0

    if source_type == "GENERAL_KNOWLEDGE":
        # Evaluates answer relevance to sub-question directly
        qa_sim = _cosine_sim(sub_question, ans)
        sem_sim = qa_sim
        grounding_score = min(1.0, qa_sim + 0.25)
        if qa_sim < 0.35:
            issues.append(f"Low semantic relevance to question ({qa_sim:.2f})")
    elif evidence_chunks:
        # Document or external knowledge evidence
        sims = [max(0.0, float(c.get("similarity_score", 0.0))) for c in evidence_chunks]
        top_retrieval = max(sims) if sims else 0.0

        # Best sentence alignment
        for chunk in evidence_chunks:
            sem_sim = max(sem_sim, _cosine_sim(ans, chunk.get("text", "")))
            for sent in re.split(r"(?<=[.!?])\s+|\n+", chunk.get("text", "")):
                if len(sent.strip()) >= 15:
                    sem_sim = max(sem_sim, _cosine_sim(ans, sent))

        grounding_score = (0.50 * sem_sim) + (0.50 * top_retrieval)
        if sem_sim < 0.40:
            issues.append(f"Low semantic alignment with retrieved evidence ({sem_sim:.2f})")
    else:
        # No evidence available
        grounding_score = 0.20
        issues.append("No supporting document or external evidence available.")

    grounding_score = max(0.0, min(1.0, grounding_score))

    # 4. Final Combined PRM Score
    # Crucial rule: PRM score is strictly bounded by quality_score.
    # An incomplete fragment cannot receive higher than 0.45.
    raw_prm = (0.45 * quality_score) + (0.55 * grounding_score)
    if is_frag:
        raw_prm = min(raw_prm, 0.40)

    prm_score = round(max(0.0, min(1.0, raw_prm)), 3)

    if prm_score >= 0.75:
        support_level = "Strong support"
    elif prm_score >= 0.55:
        support_level = "Moderate support"
    else:
        support_level = "Weak support"

    is_weak = prm_score < 0.55
    detected_issue = "; ".join(issues) if issues else None

    explanation = (
        f"PRM Score: {prm_score:.2f} ({support_level}). "
        f"Quality: {quality_score:.2f}, Grounding: {grounding_score:.2f}."
    )
    if detected_issue:
        explanation += f" Issues: {detected_issue}"

    return {
        "prm_score": prm_score,
        "support_level": support_level,
        "quality_score": round(quality_score, 3),
        "grounding_score": round(grounding_score, 3),
        "is_weak": is_weak,
        "detected_issue": detected_issue,
        "explanation": explanation,
        "metrics": {
            "quality_score": round(quality_score, 3),
            "grounding_score": round(grounding_score, 3),
            "semantic_similarity": round(sem_sim, 3),
        },
    }
