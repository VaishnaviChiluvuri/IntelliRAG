import io
import os
import re
from typing import List, Dict, Any
from pypdf import PdfReader
from docx import Document

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit for 8GB RAM stability


def validate_file(filename: str, content: bytes) -> None:
    if not filename:
        raise ValueError("Uploaded file must have a valid filename.")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. Please upload a PDF, DOCX, or TXT file."
        )
    if not content or len(content) == 0:
        raise ValueError("The uploaded file is empty (0 bytes).")
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def extract_segments(filename: str, content: bytes) -> List[Dict[str, Any]]:
    """
    Extracts structured text segments with page or section metadata.
    Returns a list of dicts: {"text": str, "page": int | None, "section": str}
    """
    ext = os.path.splitext(filename)[1].lower()
    segments: List[Dict[str, Any]] = []

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Failed to parse PDF file: {exc}") from exc

        if len(reader.pages) == 0:
            raise ValueError("The PDF file contains no pages.")

        for idx, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_text(raw)
            if cleaned:
                segments.append(
                    {
                        "text": cleaned,
                        "page": idx,
                        "section": f"Page {idx}",
                    }
                )

    elif ext == ".docx":
        try:
            doc = Document(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Failed to parse DOCX file: {exc}") from exc

        current_section = "Section 1"
        section_idx = 1
        buffer: List[str] = []

        for para in doc.paragraphs:
            txt = para.text.strip()
            if not txt:
                continue
            style_name = (para.style.name or "").lower() if para.style else ""
            if "heading" in style_name:
                if buffer:
                    segments.append(
                        {
                            "text": _clean_text("\n".join(buffer)),
                            "page": None,
                            "section": current_section,
                        }
                    )
                    buffer = []
                section_idx += 1
                current_section = f"Section: {txt[:60]}"
            buffer.append(txt)

        # Also extract tables from DOCX
        for t_idx, table in enumerate(doc.tables, start=1):
            rows_text = []
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                rows_text.append(" | ".join(cells))
            if rows_text:
                buffer.append(f"[Table {t_idx}]\n" + "\n".join(rows_text))

        if buffer:
            segments.append(
                {
                    "text": _clean_text("\n".join(buffer)),
                    "page": None,
                    "section": current_section if section_idx > 1 else "Section 1",
                }
            )

    elif ext == ".txt":
        decoded = None
        for encoding in ("utf-8", "utf-16", "latin-1"):
            try:
                decoded = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if decoded is None:
            raise ValueError("Could not decode TXT file with UTF-8 or Latin-1 encoding.")

        cleaned = _clean_text(decoded)
        if cleaned:
            paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
            for i in range(0, len(paragraphs), 3):
                sec_num = (i // 3) + 1
                segments.append(
                    {
                        "text": "\n\n".join(paragraphs[i : i + 3]),
                        "page": None,
                        "section": f"Section {sec_num}",
                    }
                )

    if not segments:
        raise ValueError(
            "No extractable text found in the document. If this is a scanned image PDF, OCR is not enabled."
        )

    return segments


def chunk_segments(
    filename: str,
    segments: List[Dict[str, Any]],
    chunk_size: int = 550,
    chunk_overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Splits extracted segments into overlapping sentence-aware chunks while preserving
    source filename, page number (if PDF), and section/chunk identifiers.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_counter = 1

    for seg in segments:
        text = seg["text"]
        page = seg["page"]
        section = seg["section"]

        if len(text) <= chunk_size:
            citation_label = (
                f"[{filename}, Page {page}]"
                if page is not None
                else f"[{filename}, {section}, Chunk {chunk_counter}]"
            )
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_counter}",
                    "chunk_index": chunk_counter,
                    "filename": filename,
                    "page": page,
                    "section": section,
                    "citation": citation_label,
                    "text": text,
                }
            )
            chunk_counter += 1
            continue

        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    citation_label = (
                        f"[{filename}, Page {page}]"
                        if page is not None
                        else f"[{filename}, {section}, Chunk {chunk_counter}]"
                    )
                    chunks.append(
                        {
                            "chunk_id": f"chunk_{chunk_counter}",
                            "chunk_index": chunk_counter,
                            "filename": filename,
                            "page": page,
                            "section": section,
                            "citation": citation_label,
                            "text": current_chunk,
                        }
                    )
                    chunk_counter += 1
                    overlap_text = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
                    current_chunk = f"{overlap_text} {sentence}".strip()
                else:
                    for start in range(0, len(sentence), chunk_size - chunk_overlap):
                        piece = sentence[start : start + chunk_size].strip()
                        if piece:
                            citation_label = (
                                f"[{filename}, Page {page}]"
                                if page is not None
                                else f"[{filename}, {section}, Chunk {chunk_counter}]"
                            )
                            chunks.append(
                                {
                                    "chunk_id": f"chunk_{chunk_counter}",
                                    "chunk_index": chunk_counter,
                                    "filename": filename,
                                    "page": page,
                                    "section": section,
                                    "citation": citation_label,
                                    "text": piece,
                                }
                            )
                            chunk_counter += 1
                    current_chunk = ""

        if current_chunk:
            citation_label = (
                f"[{filename}, Page {page}]"
                if page is not None
                else f"[{filename}, {section}, Chunk {chunk_counter}]"
            )
            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_counter}",
                    "chunk_index": chunk_counter,
                    "filename": filename,
                    "page": page,
                    "section": section,
                    "citation": citation_label,
                    "text": current_chunk,
                }
            )
            chunk_counter += 1

    return chunks