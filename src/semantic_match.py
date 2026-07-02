"""
semantic_match.py  —  Embedding-Based Semantic Retrieval
=========================================================
Two-stage semantic search pipeline:

  Stage 1 — Dense retrieval (sentence-transformers + FAISS)
    Encode the JD and all candidates with all-MiniLM-L6-v2 (90MB,
    runs on CPU in ~2-3 min for 100K candidates, cached to disk so
    subsequent runs are instant). Retrieve top-K by cosine similarity
    via a flat FAISS index.

  Stage 2 — BM25 fallback
    If sentence-transformers or FAISS is unavailable (CI, sandbox, or
    model weights not yet downloaded), transparently falls back to
    BM25Okapi. The rest of the pipeline is identical in both cases.

Embedding cache:
  Candidate embeddings are saved to models/cache/candidate_embeddings.npy
  after the first run. On subsequent runs they load in ~1s instead of
  ~2-3 min. Delete the cache file to force re-encoding.

Usage:
  matcher = SemanticMatcher(candidates, cache_dir="models/cache")
  scores  = matcher.score_all(jd_text)  # → {candidate_id: float 0-1}
"""

from __future__ import annotations
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z][a-z0-9+\-_.#]*")


# ---------------------------------------------------------------------------
# Candidate text builder  (shared by both backends)
# ---------------------------------------------------------------------------
def _candidate_text(candidate: dict) -> str:
    """
    Builds a rich text representation of a candidate that captures
    what they've actually *done*, not just what they've listed.
    Career-history descriptions are weighted 3x by repetition so the
    embedding space naturally favours demonstrated over claimed skills.
    """
    p = candidate.get("profile", {})
    parts = [
        p.get("current_title", ""),
        p.get("headline", ""),
        p.get("summary", ""),
    ]
    for job in candidate.get("career_history", []):
        title = job.get("title", "")
        desc  = job.get("description", "")
        # Repeat description to upweight real work over bare skill tags
        parts += [title, desc, desc, desc]
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    for c in candidate.get("certifications", []):
        parts.append(c.get("name", ""))
    return " ".join(filter(None, parts))


# ---------------------------------------------------------------------------
# Dense backend  (sentence-transformers + FAISS)
# ---------------------------------------------------------------------------
class DenseRetriever:
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, candidates: List[dict], cache_dir: str = "models/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.candidate_ids = [c["candidate_id"] for c in candidates]
        self._model = None
        self._index = None
        self._candidate_vecs: Optional[np.ndarray] = None

        cache_file = self.cache_dir / "candidate_embeddings.npy"
        id_file    = self.cache_dir / "candidate_ids.txt"

        if cache_file.exists() and id_file.exists():
            cached_ids = id_file.read_text().splitlines()
            if cached_ids == self.candidate_ids:
                logger.info("Loading cached candidate embeddings …")
                self._candidate_vecs = np.load(str(cache_file))
                self._build_index()
                return

        logger.info("Encoding %d candidates with %s …", len(candidates), self.MODEL_NAME)
        self._model = self._load_model()
        texts = [_candidate_text(c) for c in candidates]
        t0 = time.time()
        self._candidate_vecs = self._encode(texts, batch_size=256, desc="Encoding candidates")
        logger.info("Encoded in %.1fs", time.time() - t0)

        np.save(str(cache_file), self._candidate_vecs)
        id_file.write_text("\n".join(self.candidate_ids))
        self._build_index()

    def _load_model(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self.MODEL_NAME)

    def _encode(self, texts: List[str], batch_size: int = 256, desc: str = "") -> np.ndarray:
        try:
            from tqdm import tqdm
            show_progress = True
        except ImportError:
            show_progress = False

        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → cosine = dot product
        )
        return vecs.astype(np.float32)

    def _build_index(self):
        import faiss
        dim = self._candidate_vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # Inner product on L2-normed vecs = cosine
        self._index.add(self._candidate_vecs)

    def score_all(self, jd_text: str) -> Dict[str, float]:
        if self._model is None:
            # Re-load model for JD encoding (first call after cache load)
            self._model = self._load_model()
        jd_vec = self._encode([jd_text])          # shape (1, dim)
        scores_raw, _ = self._index.search(jd_vec, len(self.candidate_ids))
        raw = scores_raw[0]                        # cosine scores in [-1, 1]
        # Map [-1, 1] → [0, 1]
        normalized = (raw + 1.0) / 2.0
        return dict(zip(self.candidate_ids, normalized.tolist()))


# ---------------------------------------------------------------------------
# BM25 backend  (fallback)
# ---------------------------------------------------------------------------
class BM25Retriever:
    def __init__(self, candidates: List[dict]):
        from rank_bm25 import BM25Okapi
        self.candidate_ids = [c["candidate_id"] for c in candidates]
        tokenized = [_TOKEN_RE.findall(_candidate_text(c).lower()) for c in candidates]
        self._bm25 = BM25Okapi(tokenized)

    def score_all(self, jd_text: str) -> Dict[str, float]:
        query = _TOKEN_RE.findall(jd_text.lower())
        raw = self._bm25.get_scores(query)
        lo, hi = float(raw.min()), float(raw.max())
        span = (hi - lo) or 1.0
        return {cid: (float(s) - lo) / span for cid, s in zip(self.candidate_ids, raw)}


# ---------------------------------------------------------------------------
# Public façade  (auto-selects backend)
# ---------------------------------------------------------------------------
class SemanticMatcher:
    """
    Drop-in replacement for the original SemanticMatcher.
    Prefers the dense embedding backend; silently falls back to BM25
    if sentence-transformers or FAISS is unavailable.
    """

    def __init__(
        self,
        candidates: List[dict],
        cache_dir: str = "models/cache",
        force_bm25: bool = False,
    ):
        self._backend_name = "bm25"
        if force_bm25:
            self._backend = BM25Retriever(candidates)
            return

        try:
            import sentence_transformers  # noqa: F401
            import faiss                  # noqa: F401
            self._backend = DenseRetriever(candidates, cache_dir=cache_dir)
            self._backend_name = "dense_embedding"
        except Exception as e:
            logger.warning("Dense backend unavailable (%s). Falling back to BM25.", e)
            self._backend = BM25Retriever(candidates)

    @property
    def backend(self) -> str:
        return self._backend_name

    def score_all(self, jd_text: str) -> Dict[str, float]:
        return self._backend.score_all(jd_text)
