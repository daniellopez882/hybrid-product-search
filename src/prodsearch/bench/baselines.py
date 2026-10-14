"""bm25s and faiss-cpu, used only as reference points in the benchmark (dev dependencies)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class Bm25sBaseline:
    """bm25s over the same analysed tokens as the in-house single-field BM25."""

    def __init__(self, corpus_tokens: Sequence[Sequence[str]], backend: str = "numpy") -> None:
        import bm25s

        self.backend = backend
        self.retriever = bm25s.BM25(k1=1.2, b=0.75, method="lucene", backend=backend)
        self.retriever.index([list(t) for t in corpus_tokens], show_progress=False)

    def search(self, tokens: Sequence[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        if not tokens:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        docs, scores = self.retriever.retrieve([list(tokens)], k=k, show_progress=False)
        # bm25s pads with zero-score documents when fewer than k match; those are not hits.
        keep = scores[0] > 0
        return docs[0][keep].astype(np.int32), scores[0][keep]

    def nbytes(self) -> int:
        return int(sum(v.nbytes for v in self.retriever.scores.values() if hasattr(v, "nbytes")))


class FaissHnswBaseline:
    """faiss IndexHNSWFlat with inner product, same M and efConstruction as the in-house graph."""

    def __init__(self, vectors: np.ndarray, M: int, ef_construction: int, threads: int) -> None:
        import faiss

        faiss.omp_set_num_threads(threads)
        self.faiss = faiss
        self.index = faiss.IndexHNSWFlat(vectors.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def search(self, query: np.ndarray, k: int, ef: int) -> np.ndarray:
        self.index.hnsw.efSearch = max(ef, k)
        _, ids = self.index.search(query.reshape(1, -1).astype(np.float32), k)
        return ids[0][ids[0] >= 0].astype(np.int32)

    def nbytes(self) -> int:
        return int(self.faiss.serialize_index(self.index).nbytes)
