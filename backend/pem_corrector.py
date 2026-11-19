import re
import logging
from typing import List, Dict, Any, Set, Optional
from backend.vector_store import SessionVectorStore
from backend.llm_provider import get_llm_provider
from backend.prm_evaluator import evaluate_step
from backend.external_knowledge import fetch_external_knowledge

logger = logging.getLogger("intellirag.pem")


def attempt_pem_correction(
    sub_question: str,
    original_answer: str,
    detected_issue: str,
    current_evidence: List[Dict[str, Any]],
    vector_store: Optional[SessionVectorStore] = None,
    prior_step_context: str = "",
    source_type: str = "DOCUMENT",
) -> Dict[str, Any]:
    """
    Process Error Model (PEM) diagnostic corrector:
    1. Identifies error (fragment, context dump, weak grounding, or missing facts).
    2. Executes targeted secondary retrieval in document index if relevant.
    3. If document search is insufficient, queries external knowledge (Wikipedia / DuckDuckGo).
    4. Regenerates a complete, grammatically sound answer using Gemini.
    5. Re-evaluates via PRM to verify that quality and grounding issues are resolved.
    """
    extra_chunks: List[Dict[str, Any]] = []
    external_hit: Optional[Dict[str, Any]] = None
    merged_evidence = list(current_evidence)

    # 1. Secondary Document Retrieval if vector store exists
    if vector_store and source_type != "GENERAL_KNOWLEDGE":
        existing_ids = {c.get("chunk_id") for c in current_evidence if "chunk_id" in c}
        # Reformulate query with keywords
        keywords = re.findall(r"[a-zA-Z0-9]{3,}", sub_question)
        stop = {"what", "which", "where", "when", "does", "explain", "describe", "from", "with", "that"}
        ref_q = " ".join([k for k in keywords if k.lower() not in stop]) or sub_question

        extra_chunks = vector_store.search(
            query=ref_q,
            top_k=2,
            exclude_chunk_ids=existing_ids,
        )
        merged_evidence.extend(extra_chunks)

    # 2. Check if evidence is still insufficient -> Query external knowledge
    max_sim = max((c.get("similarity_score", 0.0) for c in merged_evidence), default=0.0)
    if (not merged_evidence or max_sim < 0.28) and source_type != "GENERAL_KNOWLEDGE":
        logger.info(f"PEM fetching external knowledge for: {sub_question}")
        ext_res = fetch_external_knowledge(sub_question)
        if ext_res.get("used"):
            external_hit = ext_res
            merged_evidence.append(
                {
                    "chunk_id": f"ext_{sub_question[:10]}",
                    "filename": ext_res.get("title", "External Source"),
                    "page": None,
                    "section": ext_res.get("source", "External Knowledge"),
                    "citation": f"[{ext_res.get('source', 'External Knowledge')}: {ext_res.get('title')}]",
                    "text": ext_res.get("summary", ""),
                    "similarity_score": 0.85,
                }
            )

    ext_detail = ""
    if external_hit:
        ext_source = external_hit.get("source", "External")
        ext_title = external_hit.get("title", "")
        ext_detail = f" and {ext_source} for {ext_title}"

    error_explanation = (
        f"PEM Diagnosis: Initial step flagged ({detected_issue or 'quality or evidence gap'}). "
        f"Retrieved {len(extra_chunks)} extra document chunk(s){ext_detail}."
    )

    # 3. Targeted Regeneration
    llm = get_llm_provider()
    evidence_block = ""
    if merged_evidence:
        evidence_block = "\n\nAvailable Evidence:\n" + "\n\n".join(
            f"{c.get('citation', '[Document]')}: {c.get('text', '')}"
            for c in merged_evidence[:4]
        )

    system_prompt = (
        "You are an expert answer corrector for a QA system. "
        "The previous attempt failed quality or grounding checks (e.g., was a fragment or unsupported). "
        "Write a complete, clear, grammatically sound, and direct answer to the sub-question. "
        "Never output partial fragments such as 'into water...' or 'tablespoon of...'. "
        "If evidence is provided, base your answer on it. If evidence is absent, state what is known accurately or explain the concept clearly."
    )

    prompt = (
        f"Sub-Question: {sub_question}\n"
        f"Detected Issue in Previous Attempt: {detected_issue}\n"
        f"Prior Context: {prior_step_context}\n"
        f"{evidence_block}\n\n"
        f"Provide the corrected, complete, and direct answer:"
    )

    regenerated = ""
    try:
        regenerated = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=250,
        )
    except Exception as exc:
        logger.warning(f"PEM regeneration error: {exc}")

    if not regenerated or len(regenerated.strip()) < 10:
        regenerated = original_answer

    # 4. Re-evaluate
    re_eval = evaluate_step(
        sub_question=sub_question,
        intermediate_answer=regenerated,
        evidence_chunks=merged_evidence,
        prior_step_context=prior_step_context,
        source_type=source_type if not external_hit else "EXTERNAL",
    )

    new_score = re_eval["prm_score"]
    accepted = new_score >= 0.55

    return {
        "attempted": True,
        "error_explanation": error_explanation,
        "additional_evidence": extra_chunks,
        "corrected_answer": regenerated,
        "re_eval_score": new_score,
        "re_eval_explanation": re_eval["explanation"],
        "accepted": accepted,
        "status": "Accepted (PEM Corrected)" if accepted else "Unresolved",
        "external_hit": external_hit,
    }
