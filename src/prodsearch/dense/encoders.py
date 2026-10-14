"""ONNX bi-encoder (bge-small-en-v1.5) and cross-encoder (ms-marco MiniLM) on onnxruntime CPU."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from prodsearch.sources import BI_ENCODER, CROSS_ENCODER, HubModel

# bge-small-en-v1.5 is trained to embed short queries behind this instruction.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Encoder(Protocol):
    dim: int

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class PairScorer(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> np.ndarray: ...


def _session(path: Path, threads: int | None, spin: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    # The cascade alternates two ONNX sessions with numba kernels on the same cores. Spinning
    # workers busy-wait for the next parallel section, so after one session returns its threads
    # still hold the cores the next stage needs; the benchmark measures both settings.
    options.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _tokenizer(path: Path, max_length: int, pair: bool) -> Tokenizer:
    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length, strategy="only_second" if pair else "longest_first")
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    return tok


def _feeds(encodings) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
    }


class OnnxBiEncoder:
    """CLS pooling and L2 normalisation, as the bge model card prescribes."""

    dim = 384

    def __init__(
        self,
        model: HubModel = BI_ENCODER,
        max_length: int = 128,
        batch_size: int = 32,
        threads: int | None = None,
        spin: bool = False,
    ) -> None:
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = _tokenizer(model.path("tokenizer.json"), max_length, pair=False)
        self.session = _session(model.path(model.onnx_file), threads, spin)

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        out = self.session.run(["last_hidden_state"], _feeds(self.tokenizer.encode_batch(texts)))
        cls = out[0][:, 0, :]
        return cls / np.linalg.norm(cls, axis=1, keepdims=True).clip(min=1e-12)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        result = np.empty((len(texts), self.dim), dtype=np.float32)
        # Batching texts of similar length keeps padding, and so wasted compute, small.
        order = np.argsort([len(t) for t in texts], kind="stable")
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            result[idx] = self._embed([texts[i] for i in idx])
        return result

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([BGE_QUERY_PREFIX + text])[0].astype(np.float32)


class OnnxCrossEncoder:
    """Relevance logits for (query, passage) pairs; the passage side is truncated, not the query."""

    def __init__(
        self,
        model: HubModel = CROSS_ENCODER,
        max_length: int = 128,
        batch_size: int = 25,
        threads: int | None = None,
        spin: bool = False,
    ) -> None:
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = _tokenizer(model.path("tokenizer.json"), max_length, pair=True)
        self.session = _session(model.path(model.onnx_file), threads, spin)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        scores = np.empty(len(passages), dtype=np.float32)
        for start in range(0, len(passages), self.batch_size):
            chunk = passages[start : start + self.batch_size]
            enc = self.tokenizer.encode_batch([(query, p) for p in chunk])
            logits = self.session.run(["logits"], _feeds(enc))[0]
            scores[start : start + len(chunk)] = logits[:, 0]
        return scores
