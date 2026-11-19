import time
import json
from pathlib import Path
from typing import List, Dict, Any

from backend.vector_store import SessionVectorStore
from backend.reasoning_engine import run_multihop_pipeline

DOCUMENTS = {
    "resume": {
        "path": "sample_documents/CHILUVURI_SAI_SRI_VAISHNAVI(Resume).pdf",
        "queries": [
            # 1-Hop Queries
            {
                "hop": 1,
                "question": "What is her CGPA at VIT-AP University?",
                "expected_keywords": ["8.99", "vit-ap"],
            },
            {
                "hop": 1,
                "question": "How is user authentication implemented in the HealthHub project?",
                "expected_keywords": ["jwt"],
            },
            {
                "hop": 1,
                "question": "Which organization issued her MERN Full Stack Certification?",
                "expected_keywords": ["ethnus", "codemithra"],
            },
            # 2-Hop Queries
            {
                "hop": 2,
                "question": "What is her CGPA at VIT-AP University, and what rank did she secure in AP Eamcet?",
                "expected_keywords": ["8.99", "3741"],
            },
            {
                "hop": 2,
                "question": "What web technologies are listed under her skills, and which virtual internship certificate did she complete in Java Programming?",
                "expected_keywords": ["react", "node", "codsoft"],
            },
            # 3-Hop Queries
            {
                "hop": 3,
                "question": "What programming languages does she know, which databases are listed in her skills, and what certificate did she receive from TCS iON?",
                "expected_keywords": ["java", "python", "mongodb", "mysql", "young professional"],
            },
            {
                "hop": 3,
                "question": "What is her specialization in Computer Science, how is user authentication implemented in the HealthHub project, and what rank did she secure in AP Eamcet?",
                "expected_keywords": ["artificial intelligence", "machine learning", "jwt", "3741"],
            },
        ],
    },
    "ncert_science": {
        "path": "sample_documents/science txtbook.pdf",
        "queries": [
            # 1-Hop Queries
            {
                "hop": 1,
                "question": "What is evaporation?",
                "expected_keywords": ["conversion of water", "vapour", "evaporation"],
            },
            {
                "hop": 1,
                "question": "What is condensation?",
                "expected_keywords": ["water vapour", "liquid", "condensation"],
            },
            {
                "hop": 1,
                "question": "Why does water stored in an earthen pot stay cool?",
                "expected_keywords": ["seeps", "evaporates", "cooling"],
            },
            # 2-Hop Queries
            {
                "hop": 2,
                "question": "What is evaporation, and what is condensation?",
                "expected_keywords": ["vapour", "liquid", "evaporation", "condensation"],
            },
            {
                "hop": 2,
                "question": "What conditions make evaporation faster, and how do clouds form in the atmosphere?",
                "expected_keywords": ["exposed area", "humidity", "condenses", "droplets"],
            },
            # 3-Hop Queries
            {
                "hop": 3,
                "question": "What are the three states of water, what is the process of conversion of water into vapour called, and what is the continuous circulation of water between the Earth's surface and atmosphere called?",
                "expected_keywords": ["solid", "liquid", "evaporation", "water cycle"],
            },
        ],
    },
}


def load_pdf_into_store(pdf_path: str) -> SessionVectorStore:
    """Chunks a PDF using existing backend utilities or installed PDF libraries."""
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Test PDF not found: {pdf_path}")

    # 1. Try using your backend's own loader/chunker first
    for mod_name, fn_name in [
        ("backend.pdf_loader", "load_and_chunk_pdf"),
        ("backend.pdf_loader", "process_pdf"),
        ("backend.chunker", "chunk_pdf"),
        ("backend.ingestion", "process_pdf"),
    ]:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            chunks = fn(str(path_obj), path_obj.name)
            return SessionVectorStore(
                session_id=f"bench_{path_obj.stem}",
                filename=path_obj.name,
                chunks=chunks,
            )
        except Exception:
            continue

    # 2. Extract page texts using whichever PDF library is already in your venv
    page_texts = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path_obj))
        for idx, page in enumerate(reader.pages, start=1):
            page_texts.append((idx, page.extract_text() or ""))
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path_obj))
            for idx, page in enumerate(reader.pages, start=1):
                page_texts.append((idx, page.extract_text() or ""))
        except ImportError:
            import pdfplumber
            with pdfplumber.open(str(path_obj)) as pdf:
                for idx, page in enumerate(pdf.pages, start=1):
                    page_texts.append((idx, page.extract_text() or ""))

    chunks = []
    chunk_idx = 0
    for page_num, text in page_texts:
        text = text.strip()
        if not text:
            continue
        words = text.split()
        step = 80
        for i in range(0, max(1, len(words)), step):
            chunk_text = " ".join(words[i : i + 100])
            chunks.append({
                "chunk_id": f"{path_obj.stem}_p{page_num}_c{chunk_idx}",
                "text": chunk_text,
                "page": page_num,
                "source": path_obj.name,
                "citation": f"{path_obj.name}, Page {page_num}",
            })
            chunk_idx += 1

    return SessionVectorStore(
        session_id=f"bench_{path_obj.stem}",
        filename=path_obj.name,
        chunks=chunks,
    )


def evaluate_answer(answer: str, expected_keywords: List[str]) -> float:
    ans_low = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_low)
    return round(hits / max(1, len(expected_keywords)), 3)


def main():
    all_results: List[Dict[str, Any]] = []
    hop_summary: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: []}

    for doc_name, cfg in DOCUMENTS.items():
        pdf_file = cfg["path"]
        if not Path(pdf_file).exists():
            print(f"[SKIP] {pdf_file} not found.")
            continue

        print(f"\n=== Indexing Document: {pdf_file} ===")
        store = load_pdf_into_store(pdf_file)

        for item in cfg["queries"]:
            hop = item["hop"]
            q = item["question"]

            t0 = time.perf_counter()
            out = run_multihop_pipeline(q, store)
            latency_ms = round((time.perf_counter() - t0) * 1000, 1)

            final_ans = out.get("final_answer", "")
            prm_score = out.get("overall_prm_score", 0.0)
            steps = out.get("reasoning_steps", [])
            pem_triggered = sum(1 for s in steps if s.get("pem_correction") is not None)
            pem_recovered = sum(1 for s in steps if s.get("status") == "Accepted (PEM Corrected)")
            has_citation = bool(out.get("document_citations")) and not out.get("wikipedia_fallback_used", False)
            kw_recall = evaluate_answer(final_ans, item["expected_keywords"])

            record = {
                "document": doc_name,
                "hop": hop,
                "question": q,
                "final_answer": final_ans,
                "keyword_recall": kw_recall,
                "exact_pass": kw_recall >= 0.75,
                "overall_prm_score": prm_score,
                "support_level": out.get("overall_support_level"),
                "num_steps": len(steps),
                "pem_triggered": pem_triggered,
                "pem_recovered": pem_recovered,
                "grounded_citation": has_citation,
                "latency_ms": latency_ms,
            }
            all_results.append(record)
            hop_summary[hop].append(record)
            print(
                f"[{hop}-Hop] Pass={record['exact_pass']} | PRM={prm_score:.3f} | "
                f"PEM={pem_recovered}/{pem_triggered} | Latency={latency_ms}ms | Q: {q[:55]}..."
            )

    print("\n================ IEEE PAPER SUMMARY TABLE ================")
    print(f"{'Query Type':<12} | {'N':<3} | {'Accuracy (%)':<13} | {'Avg PRM':<8} | {'Grounded Cite (%)':<18} | {'Avg Latency (ms)':<16}")
    print("-" * 82)
    for hop in (1, 2, 3):
        rows = hop_summary[hop]
        if not rows:
            continue
        n = len(rows)
        acc = round(100.0 * sum(1 for r in rows if r["exact_pass"]) / n, 1)
        avg_prm = round(sum(r["overall_prm_score"] for r in rows) / n, 3)
        cite_rate = round(100.0 * sum(1 for r in rows if r["grounded_citation"]) / n, 1)
        avg_lat = round(sum(r["latency_ms"] for r in rows) / n, 1)
        print(f"{f'{hop}-Hop':<12} | {n:<3} | {acc:<13} | {avg_prm:<8} | {cite_rate:<18} | {avg_lat:<16}")

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved detailed per-query traces to benchmark_results.json")


if __name__ == "__main__":
    main()