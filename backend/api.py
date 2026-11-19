import uuid
import time
import asyncio
from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from backend.pdf_processor import validate_file, extract_segments, chunk_segments
from backend.vector_store import SessionVectorStore
from backend.reasoning_engine import run_multihop_pipeline
from backend.llm_provider import get_llm_provider

router = APIRouter()

# In-memory session stores with timestamps
SESSIONS: Dict[str, SessionVectorStore] = {}
SESSION_META: Dict[str, Dict[str, Any]] = {}
PIPELINE_STATUS: Dict[str, Dict[str, Any]] = {}
STATUS_TIMESTAMPS: Dict[str, float] = {}

MAX_STORED_STATUSES = 100
STATUS_TTL_SECONDS = 900  # 15 minutes


def _cleanup_old_statuses():
    """Removes stale pipeline statuses to prevent memory growth."""
    now = time.time()
    expired = [
        req_id for req_id, ts in STATUS_TIMESTAMPS.items()
        if now - ts > STATUS_TTL_SECONDS
    ]
    for req_id in expired:
        PIPELINE_STATUS.pop(req_id, None)
        STATUS_TIMESTAMPS.pop(req_id, None)


class AskRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    request_id: str | None = None


@router.get("/health")
def health_check():
    llm = get_llm_provider()
    return {
        "status": "ok",
        "service": "Document-Based Multi-Hop RAG (IntelliRAG 2.0)",
        "active_sessions": len(SESSIONS),
        "llm_provider": llm.name,
        "llm_available": llm.is_available(),
    }


@router.get("/status/{request_id}")
def get_pipeline_status(request_id: str):
    _cleanup_old_statuses()
    status = PIPELINE_STATUS.get(request_id)
    if not status:
        return {"request_id": request_id, "stage": "idle", "detail": "Waiting..."}
    return status


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename or "uploaded_document"
        validate_file(filename, content)

        segments = extract_segments(filename, content)
        chunks = chunk_segments(filename, segments, chunk_size=550, chunk_overlap=100)

        session_id = str(uuid.uuid4())
        # Run synchronous embedding in thread pool to prevent blocking event loop
        store = await asyncio.to_thread(
            SessionVectorStore,
            session_id=session_id,
            filename=filename,
            chunks=chunks,
        )

        pages = sorted({c["page"] for c in chunks if c["page"] is not None})
        meta = {
            "session_id": session_id,
            "filename": filename,
            "file_size_bytes": len(content),
            "page_count": len(pages) if pages else None,
            "chunk_count": len(chunks),
            "uploaded_at": time.time(),
        }

        SESSIONS[session_id] = store
        SESSION_META[session_id] = meta

        return {
            "message": "Document processed, embedded with MiniLM, and indexed with FAISS + BM25 successfully.",
            **meta,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}") from exc


@router.post("/ask")
async def ask_question(payload: AskRequest):
    session_id = payload.session_id.strip()
    question = payload.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if session_id not in SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please upload a document first.",
        )

    req_id = payload.request_id or str(uuid.uuid4())
    PIPELINE_STATUS[req_id] = {
        "request_id": req_id,
        "stage": "document_retrieval",
        "detail": "Starting multi-hop pipeline...",
        "completed_stages": [],
    }
    STATUS_TIMESTAMPS[req_id] = time.time()

    def on_progress(stage: str, detail: str):
        prev = PIPELINE_STATUS.get(req_id, {})
        completed = list(prev.get("completed_stages", []))
        prev_stage = prev.get("stage")
        if prev_stage and prev_stage != stage and prev_stage not in completed:
            completed.append(prev_stage)
        PIPELINE_STATUS[req_id] = {
            "request_id": req_id,
            "stage": stage,
            "detail": detail,
            "completed_stages": completed,
        }
        STATUS_TIMESTAMPS[req_id] = time.time()

    try:
        store = SESSIONS[session_id]
        # Run pipeline asynchronously in a worker thread to ensure polling endpoints remain responsive
        result = await asyncio.to_thread(
            run_multihop_pipeline,
            question=question,
            vector_store=store,
            progress_callback=on_progress,
        )
        return {
            "request_id": req_id,
            "session_id": session_id,
            "document_name": store.filename,
            **result,
        }
    except Exception as exc:
        PIPELINE_STATUS[req_id] = {
            "request_id": req_id,
            "stage": "error",
            "detail": str(exc),
            "completed_stages": [],
        }
        STATUS_TIMESTAMPS[req_id] = time.time()
        raise HTTPException(status_code=500, detail=f"Reasoning pipeline failed: {exc}") from exc
