import re
import logging
from typing import List, Dict, Any, Callable, Optional
from backend.vector_store import SessionVectorStore
from backend.query_router import route_and_understand_query
from backend.question_decomposer import decompose_question
from backend.llm_provider import get_llm_provider
from backend.prm_evaluator import evaluate_step, _check_is_fragment
from backend.pem_corrector import attempt_pem_correction
from backend.citation_manager import collect_citations_and_excerpts
from backend.external_knowledge import fetch_external_knowledge

logger = logging.getLogger("intellirag.engine")


def _generate_step_answer(
    sub_question: str,
    evidence_chunks: List[Dict[str, Any]],
    prior_steps: List[Dict[str, Any]],
    source_type: str = "DOCUMENT",
    ambiguity_note: Optional[str] = None,
) -> str:
    """
    Generates a clear, grounded, complete answer for an individual sub-question.
    Works for document-grounded, external, and general-knowledge questions.
    """
    llm = get_llm_provider()

    prior_context = ""
    if prior_steps:
        prior_context = "Prior Steps Resolved:\n" + "\n".join(
            f"- Hop {s['step_number']} ({s['sub_question']}): {s['final_answer']}"
            for s in prior_steps
        ) + "\n\n"

    evidence_context = ""
    if evidence_chunks and source_type != "GENERAL_KNOWLEDGE":
        context_lines = [
            f"{c.get('citation', '[Document]')}: {c.get('text', '')}"
            for c in evidence_chunks[:4]
        ]
        evidence_context = "Retrieved Evidence:\n" + "\n\n".join(context_lines) + "\n\n"

    ambiguity_prompt = ""
    if ambiguity_note:
        ambiguity_prompt = f"Ambiguity Note: {ambiguity_note}\n\n"

    system_prompt = (
        "You are an expert, general-purpose Question-Answering engine. "
        "Your highest priority is to provide a clear, accurate, grammatically complete, and direct answer. "
        "Rules:\n"
        "1. NEVER return partial sentences or extractive fragments (e.g., 'into water...', 'tablespoon of...'). "
        "2. Directly answer the question in full, well-structured sentences.\n"
        "3. If evidence is provided, base your answer strictly on the facts in the evidence.\n"
        "4. If answering from general technical or scientific knowledge, explain the concepts accurately and authoritatively.\n"
        "5. If prior steps contain verified values, build upon them logically."
    )

    prompt = (
        f"{ambiguity_prompt}"
        f"{prior_context}"
        f"{evidence_context}"
        f"Sub-Question: {sub_question}\n"
        f"Complete Direct Answer:"
    )

    try:
        ans = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.15,
            max_tokens=300,
        )
        if ans and len(ans.strip()) >= 10:
            return ans.strip()
    except Exception as exc:
        logger.warning(f"Step answer generation error: {exc}")

    # Fallback if generation failed
    if evidence_chunks:
        top_text = evidence_chunks[0].get("text", "")
        # Find first complete sentence
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_text) if len(s.strip()) > 20]
        if sentences:
            return sentences[0]
        return top_text[:200]

    return "Information for this step could not be verified."


def _synthesize_final_answer(
    original_question: str,
    cleaned_question: str,
    steps: List[Dict[str, Any]],
    intent: str,
    wiki_result: Optional[Dict[str, Any]] = None,
    ambiguity_note: Optional[str] = None,
) -> str:
    """
    Synthesizes intermediate reasoning steps into a cohesive, grammatically unified,
    and logically connected final answer.
    """
    accepted_steps = [s for s in steps if s["status"] != "Unresolved"]
    llm = get_llm_provider()

    steps_text = "\n\n".join(
        f"Step {s['step_number']} ({s['sub_question']}):\n{s['final_answer']}"
        for s in (accepted_steps or steps)
    )

    all_citations = []
    for s in steps:
        for ev in s.get("retrieved_evidence", []):
            cite = ev.get("citation")
            if cite and cite not in all_citations:
                all_citations.append(cite)

    external_block = ""
    if wiki_result and wiki_result.get("used"):
        external_block = f"\n\nSupplementary External Knowledge ({wiki_result.get('source', 'External')}):\n{wiki_result.get('title')}: {wiki_result.get('summary')}"

    system_prompt = (
        "You are an intelligent general-purpose Question-Answering synthesizer. "
        "Synthesize a complete, natural-language, logically connected response that directly answers the user's question.\n"
        "Guidelines:\n"
        "1. Write in clear, grammatically flawless, natural prose.\n"
        "2. Answer every aspect of the user's question completely.\n"
        "3. If intermediate steps were resolved, synthesize them smoothly rather than just listing them.\n"
        "4. When referencing facts from an uploaded document, preserve source citations (e.g. [Document.pdf, Page X]).\n"
        "5. If information comes from external knowledge (like Wikipedia or DuckDuckGo), cite the source clearly.\n"
        "6. If answering from general technical knowledge, explain the topic clearly with helpful context.\n"
        "7. If the question was ambiguous, acknowledge the possible interpretations and address them."
    )

    prompt = (
        f"Original User Question: {original_question}\n"
        f"Interpreted Question: {cleaned_question}\n"
        f"Question Intent: {intent}\n"
        f"{f'Ambiguity Note: {ambiguity_note}' if ambiguity_note else ''}\n\n"
        f"Verified Reasoning Steps:\n{steps_text}"
        f"{external_block}\n\n"
        f"Write the comprehensive, unified, and directly answering final response:"
    )

    try:
        synthesized = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2,
            max_tokens=700,
        )
        if synthesized and len(synthesized.strip()) >= 20:
            return synthesized.strip()
    except Exception as exc:
        logger.warning(f"Final synthesis error: {exc}")

    # Fallback assembly
    assembled = " ".join(s["final_answer"].rstrip(".") + "." for s in (accepted_steps or steps))
    if all_citations:
        assembled += f" [{' '.join(all_citations[:3])}]"
    if wiki_result and wiki_result.get("used"):
        assembled += f"\n\n[{wiki_result.get('source')}: {wiki_result.get('title')}] {wiki_result.get('summary')}"
    return assembled


def _validate_and_repair_final_answer(
    final_answer: str,
    original_question: str,
    llm: Any,
) -> str:
    """
    Multi-Point Quality Validator:
    1. Grammatical completeness (starts with capital, ends with punctuation).
    2. Absence of fragments (e.g. 'tablespoon of water...').
    3. Relevance and completeness.
    4. Repairs the text if minor issues are detected.
    """
    cleaned = final_answer.strip()
    if not cleaned:
        return "I was unable to find verified information to answer this question."

    # Fragment check
    is_frag, reason = _check_is_fragment(cleaned)
    if is_frag and llm.is_available() and "Gemini" in llm.name:
        logger.info(f"Validator detected issue ({reason}), repairing with LLM...")
        repair_prompt = (
            f"The following answer text is an incomplete fragment or grammatically broken:\n"
            f"'{cleaned}'\n\n"
            f"Rewrite it into a complete, clear, grammatically correct and meaningful answer for the question: '{original_question}'."
        )
        try:
            repaired = llm.generate(
                prompt=repair_prompt,
                system_prompt="You are a grammar and answer completion validator. Convert broken fragments into complete, meaningful sentences.",
                temperature=0.1,
                max_tokens=400,
            )
            if repaired and len(repaired.strip()) > 15:
                cleaned = repaired.strip()
        except Exception:
            pass

    # Ensure capitalization and valid ending punctuation
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if not cleaned.endswith((".", "!", "?", "]", ")", '"', "*")):
        cleaned += "."

    return cleaned


def run_multihop_pipeline(
    question: str,
    vector_store: SessionVectorStore,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """
    General-Purpose Multi-Hop Reasoning Orchestrator:
    1. Question Understanding & Intent Classification (Document vs General vs Hybrid vs Ambiguous).
    2. Intelligent Decomposition (Preserves 1-hop simplicity, decomposes complex multi-hop).
    3. Dynamic Source Selection (FAISS+BM25 Document, External Knowledge, or Gemini Knowledge).
    4. Step-by-Step Reasoning & Multi-Dimensional PRM/PEM Evaluation.
    5. Final Synthesis, Multi-Point Validation, and Clean Source Attribution.
    """
    def notify(stage: str, detail: str):
        if progress_callback:
            progress_callback(stage, detail)

    llm = get_llm_provider()
    clean_raw_q = question.strip()

    # Stage 1: Question Understanding & Routing
    notify("document_retrieval", "Analyzing question intent and context...")
    routing_info = route_and_understand_query(clean_raw_q, vector_store)
    intent = routing_info["intent"]
    cleaned_query = routing_info["cleaned_query"]
    is_multihop_hint = routing_info.get("is_multihop")
    ambiguity_note = routing_info.get("ambiguity_note")

    logger.info(f"Question: '{clean_raw_q}' -> Intent: {intent}, Multihop: {is_multihop_hint}")

    # Stage 2: Question Decomposition
    notify("question_decomposition", f"Decomposing query (Intent: {intent})...")
    sub_questions = decompose_question(cleaned_query, is_multihop_hint=is_multihop_hint)
    if not sub_questions:
        sub_questions = [cleaned_query]

    reasoning_steps: List[Dict[str, Any]] = []
    external_hits_collected: List[Dict[str, Any]] = []
    any_pem_triggered = False

    # Stages 3, 4, 5, 6: Iterative Hop Processing
    for idx, sub_q in enumerate(sub_questions, start=1):
        notify("evidence_extraction", f"Step {idx}/{len(sub_questions)}: Sourcing evidence for '{sub_q[:45]}...'")

        ev_chunks: List[Dict[str, Any]] = []
        source_type = "DOCUMENT"

        if intent == "GENERAL_KNOWLEDGE":
            # For general knowledge, prioritize Gemini conceptual reasoning + external search
            source_type = "GENERAL_KNOWLEDGE"
            # Optional external verification
            ext = fetch_external_knowledge(sub_q)
            if ext.get("used"):
                external_hits_collected.append(ext)
                ev_chunks.append({
                    "chunk_id": f"ext_{idx}",
                    "filename": ext.get("title", "External"),
                    "page": None,
                    "section": ext.get("source", "External Knowledge"),
                    "citation": f"[{ext.get('source')}: {ext.get('title')}]",
                    "text": ext.get("summary", ""),
                    "similarity_score": 0.85,
                })
        else:
            # Document or Hybrid query -> Search hybrid FAISS + BM25 index
            ev_chunks = vector_store.search(sub_q, top_k=3)
            max_sim = max((c.get("similarity_score", 0.0) for c in ev_chunks), default=0.0)

            # If document lacks relevant evidence, seamlessly fall back to external knowledge
            if not ev_chunks or max_sim < 0.28:
                ext = fetch_external_knowledge(sub_q)
                if ext.get("used"):
                    source_type = "EXTERNAL"
                    external_hits_collected.append(ext)
                    ev_chunks.append({
                        "chunk_id": f"ext_{idx}",
                        "filename": ext.get("title", "External"),
                        "page": None,
                        "section": ext.get("source", "External Knowledge"),
                        "citation": f"[{ext.get('source')}: {ext.get('title')}]",
                        "text": ext.get("summary", ""),
                        "similarity_score": 0.80,
                    })
                elif intent == "HYBRID" or max_sim < 0.20:
                    source_type = "GENERAL_KNOWLEDGE"

        notify("intermediate_reasoning", f"Step {idx}/{len(sub_questions)}: Reasoning with verified evidence...")
        inter_answer = _generate_step_answer(
            sub_question=sub_q,
            evidence_chunks=ev_chunks,
            prior_steps=reasoning_steps,
            source_type=source_type,
            ambiguity_note=ambiguity_note,
        )

        notify("prm_evaluation", f"Step {idx}/{len(sub_questions)}: Evaluating quality & grounding with PRM...")
        prior_ctx = " ".join(s["final_answer"] for s in reasoning_steps)
        prm_res = evaluate_step(
            sub_question=sub_q,
            intermediate_answer=inter_answer,
            evidence_chunks=ev_chunks,
            prior_step_context=prior_ctx,
            source_type=source_type,
        )

        pem_res = None
        final_step_answer = inter_answer
        final_score = prm_res["prm_score"]
        status = "Accepted"

        if prm_res["is_weak"]:
            any_pem_triggered = True
            notify("pem_correction", f"Step {idx}/{len(sub_questions)}: PRM weak ({final_score:.2f}) — running PEM correction...")
            pem_res = attempt_pem_correction(
                sub_question=sub_q,
                original_answer=inter_answer,
                detected_issue=prm_res["detected_issue"] or "Weak quality or grounding",
                current_evidence=ev_chunks,
                vector_store=vector_store,
                prior_step_context=prior_ctx,
                source_type=source_type,
            )
            if pem_res["additional_evidence"]:
                ev_chunks = ev_chunks + pem_res["additional_evidence"]
            if pem_res.get("external_hit"):
                external_hits_collected.append(pem_res["external_hit"])

            final_step_answer = pem_res["corrected_answer"]
            final_score = pem_res["re_eval_score"]
            status = pem_res["status"]

        reasoning_steps.append(
            {
                "step_number": idx,
                "sub_question": sub_q,
                "source_type": source_type,
                "retrieved_evidence": ev_chunks,
                "intermediate_answer": inter_answer,
                "prm_score": prm_res["prm_score"],
                "quality_score": prm_res.get("quality_score", 1.0),
                "grounding_score": prm_res.get("grounding_score", 1.0),
                "support_level": prm_res["support_level"],
                "evaluation_explanation": prm_res["explanation"],
                "detected_issue": prm_res["detected_issue"],
                "pem_correction": pem_res,
                "re_evaluation_score": pem_res["re_eval_score"] if pem_res else None,
                "final_score": final_score,
                "final_answer": final_step_answer,
                "status": status,
            }
        )

    if not any_pem_triggered:
        notify("pem_correction", "All steps verified by PRM; PEM correction not required.")

    # Global external knowledge attribution
    primary_external = external_hits_collected[0] if external_hits_collected else None

    # Stage 7: Final Answer Synthesis & Multi-Point Validation
    notify("final_answer_generation", "Synthesizing comprehensive final answer...")
    raw_final_answer = _synthesize_final_answer(
        original_question=clean_raw_q,
        cleaned_question=cleaned_query,
        steps=reasoning_steps,
        intent=intent,
        wiki_result=primary_external,
        ambiguity_note=ambiguity_note,
    )

    validated_final_answer = _validate_and_repair_final_answer(
        final_answer=raw_final_answer,
        original_question=clean_raw_q,
        llm=llm,
    )

    avg_score = round(
        sum(s["final_score"] for s in reasoning_steps) / max(1, len(reasoning_steps)), 3
    )
    if avg_score >= 0.75:
        overall_support = "Strong support"
    elif avg_score >= 0.55:
        overall_support = "Moderate support"
    else:
        overall_support = "Moderate / General Knowledge"

    citation_bundle = collect_citations_and_excerpts(reasoning_steps, primary_external)

    notify("completed", "Reasoning pipeline completed successfully.")

    # Knowledge source summary label for UI clarity
    source_summary = "Uploaded Document"
    if intent == "GENERAL_KNOWLEDGE":
        source_summary = "General Technical Knowledge (Gemini)"
    elif primary_external and citation_bundle["document_citations"]:
        source_summary = "Hybrid (Document + External Knowledge)"
    elif primary_external:
        source_summary = f"External Knowledge ({primary_external.get('source', 'Wikipedia')})"

    return {
        "question": clean_raw_q,
        "cleaned_question": cleaned_query,
        "intent": intent,
        "knowledge_source": source_summary,
        "final_answer": validated_final_answer,
        "overall_prm_score": avg_score,
        "overall_support_level": overall_support,
        "score_disclaimer": f"IntelliRAG Multi-Source Reasoning ({source_summary}) with PRM validation.",
        "reasoning_steps": reasoning_steps,
        "document_citations": citation_bundle["document_citations"],
        "supporting_excerpts": citation_bundle["supporting_excerpts"],
        "external_citations": citation_bundle["external_citations"],
        "wikipedia_fallback_used": bool(primary_external),
        "wikipedia_notice": (
            f"Supplementary knowledge retrieved from {primary_external.get('source')}: {primary_external.get('title')}."
            if primary_external
            else None
        ),
    }
