import re
from typing import List, Dict, Any, Set, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from backend.embedding_utils import encode_texts


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25."""
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class SessionVectorStore:
    """
    Per-session hybrid retrieval store combining:
    1. FAISS IndexFlatIP (cosine similarity with L2-normalized dense embeddings)
    2. BM25Okapi (lexical / exact term keyword matching)
    3. Hybrid score fusion with reciprocal rank / convex combination
    """

    def __init__(self, session_id: str, filename: str, chunks: List[Dict[str, Any]]):
        if not chunks:
            raise ValueError("Cannot build index from empty chunks.")
        self.session_id = session_id
        self.filename = filename
        self.chunks = chunks

        # 1. Dense Index (FAISS)
        texts = [c["text"] for c in chunks]
        embeddings = encode_texts(texts)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # 2. Sparse Index (BM25)
        self.tokenized_corpus = [_tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 4,
        exclude_chunk_ids: Optional[Set[str]] = None,
        alpha: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining dense FAISS cosine similarity and sparse BM25 scores.
        alpha: weight for dense similarity (1.0 = pure dense, 0.0 = pure BM25).
        """
        clean_q = query.strip()
        if not clean_q or not self.chunks:
            return []

        exclude_chunk_ids = exclude_chunk_ids or set()
        num_chunks = len(self.chunks)

        # --- Dense Search ---
        q_vec = encode_texts([clean_q])
        # Search all chunks to get complete dense scores for fusion
        dense_scores_all, dense_indices_all = self.index.search(q_vec, num_chunks)

        dense_score_map: Dict[int, float] = {}
        for score, idx in zip(dense_scores_all[0], dense_indices_all[0]):
            if idx >= 0:
                # Cosine similarity is in [-1, 1], clamp to [0, 1]
                dense_score_map[idx] = max(0.0, min(1.0, float(score)))

        # --- BM25 Search ---
        q_tokens = _tokenize(clean_q)
        bm25_raw_scores = self.bm25.get_scores(q_tokens) if q_tokens else np.zeros(num_chunks)
        max_bm25 = float(np.max(bm25_raw_scores)) if len(bm25_raw_scores) > 0 and np.max(bm25_raw_scores) > 0 else 1.0

        # --- Score Fusion & Reranking ---
        candidates = []
        for idx, chunk in enumerate(self.chunks):
            if chunk["chunk_id"] in exclude_chunk_ids:
                continue

            dense_score = dense_score_map.get(idx, 0.0)
            norm_bm25 = float(bm25_raw_scores[idx]) / max_bm25 if max_bm25 > 0 else 0.0

            # Hybrid linear combination
            hybrid_score = (alpha * dense_score) + ((1.0 - alpha) * norm_bm25)

            item = dict(chunk)
            item["dense_score"] = round(dense_score, 4)
            item["bm25_score"] = round(norm_bm25, 4)
            item["similarity_score"] = round(hybrid_score, 4)
            candidates.append(item)

        # Sort descending by hybrid similarity score
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

        return candidates[:top_k]
