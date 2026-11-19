import threading
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

_MODEL_LOCK = threading.Lock()
_EMBEDDER: SentenceTransformer | None = None
_FLAN_TOKENIZER = None
_FLAN_MODEL = None

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_NAME = "google/flan-t5-small"


def get_embedding_model() -> SentenceTransformer:
    """Loads and caches all-MiniLM-L6-v2 on CPU once."""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _MODEL_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _EMBEDDER


def get_flan_model():
    """Loads and caches google/flan-t5-small on CPU once with low memory overhead."""
    global _FLAN_TOKENIZER, _FLAN_MODEL
    if _FLAN_MODEL is None or _FLAN_TOKENIZER is None:
        with _MODEL_LOCK:
            if _FLAN_MODEL is None or _FLAN_TOKENIZER is None:
                _FLAN_TOKENIZER = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
                _FLAN_MODEL = AutoModelForSeq2SeqLM.from_pretrained(
                    GEN_MODEL_NAME,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                _FLAN_MODEL.to("cpu")
                _FLAN_MODEL.eval()
    return _FLAN_TOKENIZER, _FLAN_MODEL


def encode_texts(texts: List[str]) -> np.ndarray:
    """Returns L2-normalized float32 embeddings suitable for FAISS IndexFlatIP."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)
    model = get_embedding_model()
    vecs = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def cosine_similarity(text_a: str, text_b: str) -> float:
    """Computes cosine similarity in [0, 1] between two strings."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vecs = encode_texts([text_a, text_b])
    sim = float(np.dot(vecs[0], vecs[1]))
    return max(0.0, min(1.0, sim))


def generate_with_flan(prompt: str, max_new_tokens: int = 128) -> str:
    """Runs CPU inference with FLAN-T5-small respecting 512-token input limits."""
    try:
        tokenizer, model = get_flan_model()
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as exc:
        raise RuntimeError(f"FLAN-T5-small generation failed: {exc}") from exc