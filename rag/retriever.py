"""Hybrid retrieval: BM25 keyword search + dense embeddings, fused with RRF.

BM25 is pure Python and always available. Dense search uses `fastembed`
(a small ONNX model, CPU only, no API cost). If it can't load, the retriever
falls back to BM25 and says so, so the app never breaks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import CHUNKS_PATH, EMBEDDINGS_PATH

log = logging.getLogger(__name__)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RRF_K = 60

STOPWORDS = set("""
a an and are as at be by can could do does for from has have how i if in into is it its
me my of on or our should so that the their them then there these they this to was we
what when where which who why will with would you your about also any may such than
define defined definition mean means meaning refer refers describe described say says
involve involves purpose explain tell give list according
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*")


def tokenize(text: str) -> list[str]:
    tokens = []
    for t in TOKEN_RE.findall(text.lower()):
        if t in STOPWORDS:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]  # crude plural folding: "risks" -> "risk"
        tokens.append(t)
    return tokens


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    location: str
    url: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused score used for ranking
    bm25: float = 0.0
    dense: float = 0.0


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def _corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


class Retriever:
    def __init__(self, chunks: list[Chunk], use_dense: bool = True,
                 embeddings_path: Path = EMBEDDINGS_PATH):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.dense_error: str | None = None
        self._model = None
        self._matrix = None
        if use_dense:
            try:
                self._init_dense(embeddings_path)
            except Exception as e:  # missing package, offline, etc.
                self.dense_error = f"{type(e).__name__}: {e}"
                log.warning("Dense retrieval disabled: %s", self.dense_error)

    @property
    def dense_available(self) -> bool:
        return self._matrix is not None

    def _init_dense(self, embeddings_path: Path) -> None:
        import numpy as np
        from fastembed import TextEmbedding

        self._model = TextEmbedding(EMBED_MODEL, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))
        meta_path = embeddings_path.with_suffix(".meta.json")
        corpus_hash = _corpus_hash(self.chunks)
        if embeddings_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("hash") == corpus_hash and meta.get("model") == EMBED_MODEL:
                self._matrix = np.load(embeddings_path)
                return
        # Not cached (or stale): embed the corpus now and try to cache it.
        vecs = np.array(list(self._model.passage_embed([c.text for c in self.chunks])),
                        dtype="float32")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        self._matrix = vecs
        try:
            np.save(embeddings_path, vecs)
            meta_path.write_text(json.dumps({"hash": corpus_hash, "model": EMBED_MODEL}))
        except OSError:
            pass

    def _bm25_scores(self, query: str):
        return self.bm25.get_scores(tokenize(query))

    def _dense_scores(self, query: str):
        import numpy as np
        q = np.array(next(iter(self._model.query_embed(query))), dtype="float32")
        q /= np.linalg.norm(q)
        return self._matrix @ q

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode in ("hybrid", "dense") and not self.dense_available:
            mode = "bm25"
        bm25 = self._bm25_scores(query)
        dense = self._dense_scores(query) if mode in ("hybrid", "dense") else None

        def ranking(scores):
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        if mode == "bm25":
            fused = {i: float(bm25[i]) for i in ranking(bm25)[:k]}
        elif mode == "dense":
            fused = {i: float(dense[i]) for i in ranking(dense)[:k]}
        else:
            # Reciprocal Rank Fusion: robust to the two scores being on different scales.
            fused: dict[int, float] = {}
            for scores in (bm25, dense):
                for rank, i in enumerate(ranking(scores)[:50]):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(chunk=self.chunks[i], score=fused[i], bm25=float(bm25[i]),
                    dense=float(dense[i]) if dense is not None else 0.0) for i in top]

    def max_dense_similarity(self, query: str) -> float | None:
        """Cosine similarity of the closest chunk, or None without embeddings."""
        if not self.dense_available:
            return None
        return float(max(self._dense_scores(query)))

    def max_keyword_overlap(self, query: str) -> float:
        """Best BM25 score for the query; 0 means no shared terms with the corpus."""
        scores = self._bm25_scores(query)
        return float(max(scores)) if len(scores) else 0.0
