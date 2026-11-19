from typing import List, Dict, Any


def collect_citations_and_excerpts(
    reasoning_steps: List[Dict[str, Any]],
    wiki_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Deduplicates document citations and supporting excerpts strictly from
    retrieved chunks, keeping Wikipedia external citations strictly separate.
    """
    seen_chunks = set()
    document_citations: List[str] = []
    supporting_excerpts: List[Dict[str, Any]] = []

    for step in reasoning_steps:
        for ev in step.get("retrieved_evidence", []):
            cid = ev["chunk_id"]
            if cid not in seen_chunks:
                seen_chunks.add(cid)
                if ev["citation"] not in document_citations:
                    document_citations.append(ev["citation"])
                supporting_excerpts.append(
                    {
                        "citation": ev["citation"],
                        "page": ev.get("page"),
                        "section": ev.get("section"),
                        "similarity_score": ev.get("similarity_score"),
                        "excerpt": ev["text"],
                    }
                )

    external_citations: List[Dict[str, Any]] = []
    if wiki_result and wiki_result.get("used"):
        external_citations.append(
            {
                "source": "Wikipedia (External Knowledge)",
                "title": wiki_result.get("title"),
                "url": wiki_result.get("url"),
                "summary": wiki_result.get("summary"),
            }
        )

    return {
        "document_citations": document_citations,
        "supporting_excerpts": supporting_excerpts,
        "external_citations": external_citations,
    }