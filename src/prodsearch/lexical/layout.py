"""Compressed postings: d-gaps bit-packed per fixed-size block, plus per-block skip metadata.

Each term's postings (sorted doc ids) are cut into blocks of `block_size` documents. A block
stores its d-gaps minus one with a single bit width chosen for that block (frame of reference
without exceptions), so a block of consecutive ids costs zero bits. Per block the layout keeps

* `blk_last`  the last doc id, which lets a cursor skip a block without decoding it,
* `blk_bits`  the bit width, `blk_word` the first packed 32-bit word, `blk_post` the first
  posting position (used to address the impact array, which is stored unpacked as uint16).

Block maxima of the impacts live with the impacts (see `bm25f.py`), because they change with the
field weights while this layout does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

END = np.int32(2**31 - 1)


@nb.njit(cache=True)
def _bits_for(value: int) -> int:
    bits = 0
    while (1 << bits) <= value:
        bits += 1
    return bits


@nb.njit(cache=True)
def _build_layout(docs: np.ndarray, term_ptr: np.ndarray, block_size: int):
    n_terms = term_ptr.shape[0] - 1
    term_blk = np.empty(n_terms + 1, dtype=np.int64)
    n_blocks = 0
    for t in range(n_terms):
        term_blk[t] = n_blocks
        n_blocks += (term_ptr[t + 1] - term_ptr[t] + block_size - 1) // block_size
    term_blk[n_terms] = n_blocks

    blk_last = np.empty(n_blocks, dtype=np.int32)
    blk_bits = np.empty(n_blocks, dtype=np.uint8)
    blk_word = np.empty(n_blocks + 1, dtype=np.int64)
    blk_post = np.empty(n_blocks + 1, dtype=np.int64)
    n_words = 0
    b = 0
    for t in range(n_terms):
        lo = term_ptr[t]
        hi = term_ptr[t + 1]
        prev = -1
        for start in range(lo, hi, block_size):
            stop = min(start + block_size, hi)
            widest = 0
            p = prev
            for j in range(start, stop):
                gap = docs[j] - p - 1
                widest = max(widest, gap)
                p = docs[j]
            bits = _bits_for(widest)
            blk_bits[b] = bits
            blk_last[b] = docs[stop - 1]
            blk_word[b] = n_words
            blk_post[b] = start
            n_words += ((stop - start) * bits + 31) // 32
            prev = docs[stop - 1]
            b += 1
    blk_word[n_blocks] = n_words
    blk_post[n_blocks] = docs.shape[0]

    words = np.zeros(n_words, dtype=np.uint32)
    for t in range(n_terms):
        prev = -1
        for b in range(term_blk[t], term_blk[t + 1]):
            bits = np.int64(blk_bits[b])
            base = blk_word[b]
            bitpos = np.int64(0)
            for j in range(blk_post[b], blk_post[b + 1]):
                value = np.int64(docs[j] - prev - 1)
                prev = docs[j]
                if bits == 0:
                    continue
                w = base + (bitpos >> 5)
                shift = bitpos & 31
                words[w] = np.uint32((np.int64(words[w]) | (value << shift)) & 0xFFFFFFFF)
                if shift + bits > 32:
                    words[w + 1] = np.uint32(np.int64(words[w + 1]) | (value >> (32 - shift)))
                bitpos += bits
    return words, blk_last, blk_bits, blk_word, blk_post, term_blk


@nb.njit(cache=True, nogil=True)
def decode_block(words, blk_bits, blk_word, blk_post, b, prev, out) -> int:
    """Write the doc ids of block `b` into `out`; `prev` is the previous block's last id or -1."""
    n = blk_post[b + 1] - blk_post[b]
    bits = np.int64(blk_bits[b])
    if bits == 0:
        for i in range(n):
            prev += 1
            out[i] = prev
        return n
    mask = (np.int64(1) << bits) - 1
    base = blk_word[b]
    bitpos = np.int64(0)
    for i in range(n):
        w = base + (bitpos >> 5)
        shift = bitpos & 31
        value = np.int64(words[w]) >> shift
        if shift + bits > 32:
            value |= np.int64(words[w + 1]) << (32 - shift)
        prev += (value & mask) + 1
        out[i] = prev
        bitpos += bits
    return n


@nb.njit(cache=True)
def _decode_all(words, blk_bits, blk_word, blk_post, blk_last, term_blk, block_size):
    n_terms = term_blk.shape[0] - 1
    out = np.empty(blk_post[-1], dtype=np.int32)
    buf = np.empty(block_size, dtype=np.int32)
    for t in range(n_terms):
        prev = -1
        for b in range(term_blk[t], term_blk[t + 1]):
            n = decode_block(words, blk_bits, blk_word, blk_post, b, prev, buf)
            start = blk_post[b]
            for i in range(n):
                out[start + i] = buf[i]
            prev = blk_last[b]
    return out


@dataclass
class PostingLayout:
    """Doc-id side of an inverted index. Independent of scoring parameters."""

    block_size: int
    n_docs: int
    words: np.ndarray
    blk_last: np.ndarray
    blk_bits: np.ndarray
    blk_word: np.ndarray
    blk_post: np.ndarray
    term_blk: np.ndarray

    @classmethod
    def build(
        cls, docs: np.ndarray, term_ptr: np.ndarray, n_docs: int, block_size: int = 128
    ) -> PostingLayout:
        """`docs` holds every term's ascending doc ids back to back; `term_ptr` delimits terms."""
        if block_size < 1:
            raise ValueError("block_size must be positive")
        arrays = _build_layout(
            np.ascontiguousarray(docs, dtype=np.int32),
            np.ascontiguousarray(term_ptr, dtype=np.int64),
            block_size,
        )
        return cls(block_size, n_docs, *arrays)

    @property
    def n_terms(self) -> int:
        return self.term_blk.shape[0] - 1

    @property
    def n_postings(self) -> int:
        return int(self.blk_post[-1])

    def decode(self) -> np.ndarray:
        """All doc ids, uncompressed, in posting order (for tests and diagnostics)."""
        return _decode_all(
            self.words,
            self.blk_bits,
            self.blk_word,
            self.blk_post,
            self.blk_last,
            self.term_blk,
            self.block_size,
        )

    def nbytes(self) -> dict[str, int]:
        return {
            "doc_id_words": int(self.words.nbytes),
            "block_metadata": int(
                self.blk_last.nbytes
                + self.blk_bits.nbytes
                + self.blk_word.nbytes
                + self.blk_post.nbytes
                + self.term_blk.nbytes
            ),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "words": self.words,
            "blk_last": self.blk_last,
            "blk_bits": self.blk_bits,
            "blk_word": self.blk_word,
            "blk_post": self.blk_post,
            "term_blk": self.term_blk,
        }

    @classmethod
    def from_arrays(cls, block_size: int, n_docs: int, a: dict[str, np.ndarray]) -> PostingLayout:
        return cls(
            block_size,
            n_docs,
            a["words"],
            a["blk_last"],
            a["blk_bits"],
            a["blk_word"],
            a["blk_post"],
            a["term_blk"],
        )
