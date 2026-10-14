"""Field-weighted BM25F over the compressed layout, with 16-bit quantised impacts.

Scoring follows the simple BM25F of Robertson, Zaragoza and Taylor (2004): per-field term
frequencies are length-normalised with a per-field b, weighted, summed into one pseudo
frequency, and saturated once with k1:

    tf~(t, d) = sum_f  w_f * tf_f(t, d) / (1 - b_f + b_f * len_f(d) / avglen_f)
    s(t, d)   = idf(t) * tf~ / (k1 + tf~),   idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

s(t, d) is computed at index time and quantised linearly against the largest s in the index,
so a query score is an integer sum. Changing weights, b or k1 only recomputes impacts and block
maxima; the doc-id layout is built once.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from prodsearch.lexical.layout import PostingLayout
from prodsearch.lexical.wand import exhaustive_topk, score_docs, wand_topk

QUANT_LEVELS = 65535
METHODS = ("bmw", "wand", "exhaustive")


@dataclass(frozen=True)
class BM25FConfig:
    weights: Mapping[str, float]
    b: Mapping[str, float]
    k1: float = 1.2

    def label(self) -> str:
        return ",".join(f"{f}={self.weights[f]:g}" for f in self.weights)

    def to_json(self) -> dict:
        return {"weights": dict(self.weights), "b": dict(self.b), "k1": self.k1}

    @classmethod
    def from_json(cls, data: Mapping) -> BM25FConfig:
        return cls(weights=dict(data["weights"]), b=dict(data["b"]), k1=float(data["k1"]))


# Chosen before any WANDS run: titles count double, attribute values half, b lower on short fields.
DEFAULT_BM25F = BM25FConfig(
    weights={"name": 2.0, "class": 1.0, "category": 1.0, "features": 0.5, "description": 1.0},
    b={"name": 0.5, "class": 0.5, "category": 0.5, "features": 0.75, "description": 0.75},
)
SINGLE_FIELD_BM25 = BM25FConfig(weights={"all": 1.0}, b={"all": 0.75})


@dataclass
class FieldedPostings:
    """Per-field term frequencies for every (term, doc) pair, in term-then-doc order."""

    fields: tuple[str, ...]
    vocab: dict[str, int]
    n_docs: int
    term_ptr: np.ndarray
    docs: np.ndarray
    tf: np.ndarray
    lengths: np.ndarray

    @classmethod
    def build(cls, field_tokens: Mapping[str, Sequence[Sequence[str]]]) -> FieldedPostings:
        fields = tuple(field_tokens)
        n_docs = len(next(iter(field_tokens.values()))) if fields else 0
        vocab: dict[str, int] = {}
        keyed: list[tuple[np.ndarray, np.ndarray]] = []
        lengths = np.zeros((len(fields), n_docs), dtype=np.float32)
        for fi, name in enumerate(fields):
            docs_tokens = field_tokens[name]
            if len(docs_tokens) != n_docs:
                raise ValueError(f"field {name!r} has {len(docs_tokens)} docs, expected {n_docs}")
            lens = np.fromiter((len(t) for t in docs_tokens), dtype=np.int64, count=n_docs)
            lengths[fi] = lens
            ids = np.fromiter(
                (vocab.setdefault(w, len(vocab)) for toks in docs_tokens for w in toks),
                dtype=np.int64,
                count=int(lens.sum()),
            )
            keys = ids * n_docs + np.repeat(np.arange(n_docs, dtype=np.int64), lens)
            keyed.append(np.unique(keys, return_counts=True))
        all_keys = (
            np.unique(np.concatenate([k for k, _ in keyed])) if keyed else np.zeros(0, np.int64)
        )
        tf = np.zeros((len(fields), all_keys.shape[0]), dtype=np.float32)
        for fi, (keys, counts) in enumerate(keyed):
            tf[fi, np.searchsorted(all_keys, keys)] = counts
        terms = all_keys // max(n_docs, 1)
        term_ptr = np.zeros(len(vocab) + 1, dtype=np.int64)
        np.cumsum(np.bincount(terms, minlength=len(vocab)), out=term_ptr[1:])
        return cls(
            fields=fields,
            vocab=vocab,
            n_docs=n_docs,
            term_ptr=term_ptr,
            docs=(all_keys % max(n_docs, 1)).astype(np.int32),
            tf=tf,
            lengths=lengths,
        )

    def pseudo_tf(self, config: BM25FConfig) -> np.ndarray:
        out = np.zeros(self.docs.shape[0], dtype=np.float64)
        for fi, name in enumerate(self.fields):
            weight = config.weights.get(name, 0.0)
            avg = float(self.lengths[fi].mean()) if self.n_docs else 0.0
            if weight == 0.0 or avg == 0.0:
                continue
            b = config.b.get(name, 0.75)
            norm = 1.0 - b + b * self.lengths[fi][self.docs] / avg
            tf = self.tf[fi]
            # With b = 1 an empty field has norm 0; its tf is 0 too and must stay 0, not NaN.
            out += weight * np.divide(tf, norm, out=np.zeros_like(out), where=tf > 0)
        return out


@dataclass
class LexicalIndex:
    vocab: dict[str, int]
    layout: PostingLayout
    impacts: np.ndarray
    blk_max: np.ndarray
    term_max: np.ndarray
    idf: np.ndarray
    scale: float
    config: BM25FConfig
    _ix: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        lay = self.layout
        self._ix = (
            lay.words,
            lay.blk_last,
            lay.blk_bits,
            lay.blk_word,
            lay.blk_post,
            lay.term_blk,
            self.impacts,
            self.blk_max,
            self.term_max,
        )

    @property
    def n_docs(self) -> int:
        return self.layout.n_docs

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        postings: FieldedPostings,
        config: BM25FConfig,
        block_size: int = 128,
        layout: PostingLayout | None = None,
    ) -> LexicalIndex:
        if layout is None:
            layout = PostingLayout.build(
                postings.docs, postings.term_ptr, postings.n_docs, block_size
            )
        pseudo = postings.pseudo_tf(config)
        counts = np.diff(postings.term_ptr)
        term_of = np.repeat(np.arange(counts.shape[0]), counts)
        # A posting whose only occurrences sit in zero-weight fields does not count towards df.
        df = np.bincount(term_of, weights=(pseudo > 0), minlength=counts.shape[0])
        n = postings.n_docs
        idf = np.log1p((n - df + 0.5) / (df + 0.5))
        raw = idf[term_of] * pseudo / (config.k1 + pseudo)
        top = float(raw.max()) if raw.size else 0.0
        scale = top / QUANT_LEVELS if top > 0 else 1.0
        impacts = np.where(raw > 0, np.clip(np.rint(raw / scale), 1, QUANT_LEVELS), 0)
        impacts = impacts.astype(np.uint16)
        return cls(
            vocab=postings.vocab,
            layout=layout,
            impacts=impacts,
            blk_max=_segment_max(impacts, layout.blk_post),
            term_max=_segment_max(impacts, postings.term_ptr),
            idf=idf.astype(np.float32),
            scale=scale,
            config=config,
        )

    # ------------------------------------------------------------------ query
    def query_terms(self, tokens: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        counts = Counter(t for t in tokens if t in self.vocab)
        terms = np.array([self.vocab[t] for t in counts], dtype=np.int32)
        weights = np.array(list(counts.values()), dtype=np.int64)
        return terms, weights

    def search(
        self,
        tokens: Sequence[str],
        k: int,
        mask: np.ndarray | None = None,
        method: str = "bmw",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """Top-k (doc ids, BM25F scores, kernel statistics) for an analysed query."""
        terms, weights = self.query_terms(tokens)
        m = _NO_MASK if mask is None else mask
        bs = self.layout.block_size
        if method == "bmw":
            docs, scores, stats = wand_topk(self._ix, terms, weights, k, m, bs, True)
        elif method == "wand":
            docs, scores, stats = wand_topk(self._ix, terms, weights, k, m, bs, False)
        elif method == "exhaustive":
            docs, scores, stats = exhaustive_topk(self._ix, terms, weights, k, m, self.n_docs, bs)
        else:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        return (
            docs,
            scores.astype(np.float64) * self.scale,
            {"scored": int(stats[0]), "blocks_decoded": int(stats[1]), "skips": int(stats[2])},
        )

    def score(self, tokens: Sequence[str], docs: np.ndarray) -> np.ndarray:
        terms, weights = self.query_terms(tokens)
        raw = score_docs(
            self._ix, terms, weights, np.asarray(docs, dtype=np.int32), self.layout.block_size
        )
        return raw.astype(np.float64) * self.scale

    # ------------------------------------------------------------------ persistence
    def nbytes(self) -> dict[str, int]:
        sizes = self.layout.nbytes()
        sizes["impacts"] = int(self.impacts.nbytes)
        sizes["block_max"] = int(self.blk_max.nbytes + self.term_max.nbytes)
        sizes["total"] = sum(sizes.values())
        return sizes

    def save(self, path: Path) -> None:
        terms = sorted(self.vocab, key=self.vocab.__getitem__)
        meta = {
            "block_size": self.layout.block_size,
            "n_docs": self.n_docs,
            "scale": self.scale,
            "config": self.config.to_json(),
        }
        np.savez(
            path,
            meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
            vocab=np.frombuffer("\n".join(terms).encode(), dtype=np.uint8),
            impacts=self.impacts,
            blk_max=self.blk_max,
            term_max=self.term_max,
            idf=self.idf,
            **self.layout.arrays(),
        )

    @classmethod
    def load(cls, path: Path) -> LexicalIndex:
        with np.load(path) as a:
            meta = json.loads(a["meta"].tobytes().decode())
            text = a["vocab"].tobytes().decode()
            vocab = {t: i for i, t in enumerate(text.split("\n"))} if text else {}
            layout = PostingLayout.from_arrays(meta["block_size"], meta["n_docs"], a)
            return cls(
                vocab=vocab,
                layout=layout,
                impacts=a["impacts"],
                blk_max=a["blk_max"],
                term_max=a["term_max"],
                idf=a["idf"],
                scale=float(meta["scale"]),
                config=BM25FConfig.from_json(meta["config"]),
            )


_NO_MASK = np.zeros(0, dtype=np.uint8)


def _segment_max(values: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """Max of values[ptr[i]:ptr[i+1]] for each non-empty segment (empty segments give 0)."""
    out = np.zeros(ptr.shape[0] - 1, dtype=np.int32)
    starts = ptr[:-1]
    nonempty = ptr[1:] > starts
    if values.size and nonempty.any():
        out[nonempty] = np.maximum.reduceat(values, starts[nonempty]).astype(np.int32)
    return out
