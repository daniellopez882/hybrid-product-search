"""Top-k document-at-a-time kernels over the compressed layout: exhaustive, WAND and block-max WAND.

Scores are integer sums of quantised impacts, so every kernel computes bit-identical scores no
matter in which order it adds terms. Ties are broken towards the lower doc id. Because documents
are visited in ascending id order, "enter the heap only if strictly above the threshold" gives
exactly that tie-break, which is what lets the pruned kernels promise the exhaustive top-k
(`tests/test_wand.py` checks it on random corpora).

Index arrays travel as one tuple `ix`:
    (words, blk_last, blk_bits, blk_word, blk_post, term_blk, impacts, blk_max, term_max)
"""

from __future__ import annotations

import numba as nb
import numpy as np

from prodsearch.lexical.layout import END, decode_block

# Statistics returned next to the results.
STAT_SCORED = 0  # documents whose full score was computed
STAT_DECODED = 1  # blocks decoded
STAT_SKIPS = 2  # block-max skips (BMW only)
N_STATS = 3


@nb.njit(cache=True, nogil=True, inline="always")
def _worse(s1, d1, s2, d2) -> bool:
    return s1 < s2 or (s1 == s2 and d1 > d2)


@nb.njit(cache=True, nogil=True)
def _sift_down(hs, hd, size, i) -> None:
    while True:
        left = 2 * i + 1
        if left >= size:
            return
        worst = left
        right = left + 1
        if right < size and _worse(hs[right], hd[right], hs[left], hd[left]):
            worst = right
        if _worse(hs[worst], hd[worst], hs[i], hd[i]):
            hs[i], hs[worst] = hs[worst], hs[i]
            hd[i], hd[worst] = hd[worst], hd[i]
            i = worst
        else:
            return


@nb.njit(cache=True, nogil=True)
def heap_offer(hs, hd, size, k, score, doc) -> int:
    """Offer (score, doc) to a size-k heap whose root is the current worst. Returns the new size."""
    if size < k:
        i = size
        hs[i] = score
        hd[i] = doc
        while i > 0:
            parent = (i - 1) // 2
            if _worse(hs[i], hd[i], hs[parent], hd[parent]):
                hs[i], hs[parent] = hs[parent], hs[i]
                hd[i], hd[parent] = hd[parent], hd[i]
                i = parent
            else:
                break
        return size + 1
    if score > hs[0]:
        hs[0] = score
        hd[0] = doc
        _sift_down(hs, hd, size, 0)
    return size


@nb.njit(cache=True, nogil=True)
def heap_sorted(hs, hd, size):
    """Drain the heap into (doc ids, scores) ordered best first."""
    docs = np.empty(size, dtype=np.int32)
    scores = np.empty(size, dtype=np.int64)
    n = size
    while n > 0:
        docs[n - 1] = hd[0]
        scores[n - 1] = hs[0]
        n -= 1
        hs[0] = hs[n]
        hd[0] = hd[n]
        _sift_down(hs, hd, n, 0)
    return docs, scores


@nb.njit(cache=True, nogil=True)
def _passes(mask, doc) -> bool:
    return mask.shape[0] == 0 or mask[doc] != 0


# ----------------------------------------------------------------------------- cursors
# Cursor state for query term i lives in row i of these arrays:
#   cur[i, 0] first block of the term, cur[i, 1] last block, cur[i, 2] current block,
#   cur[i, 3] position inside the decoded block, cur[i, 4] current doc id (END when exhausted)
# and buf[i] holds the decoded doc ids of the current block.
C_FIRST, C_LAST, C_BLK, C_POS, C_DOC = 0, 1, 2, 3, 4


@nb.njit(cache=True, nogil=True)
def _load(ix, cur, buf, i, b, stats) -> None:
    words, blk_last, blk_bits, blk_word, blk_post = ix[0], ix[1], ix[2], ix[3], ix[4]
    prev = -1 if b == cur[i, C_FIRST] else np.int64(blk_last[b - 1])
    decode_block(words, blk_bits, blk_word, blk_post, b, prev, buf[i])
    stats[STAT_DECODED] += 1
    cur[i, C_BLK] = b
    cur[i, C_POS] = 0
    cur[i, C_DOC] = buf[i, 0]


@nb.njit(cache=True, nogil=True)
def _open(ix, cur, buf, i, term, stats) -> None:
    term_blk = ix[5]
    cur[i, C_FIRST] = term_blk[term]
    cur[i, C_LAST] = term_blk[term + 1] - 1
    if cur[i, C_LAST] < cur[i, C_FIRST]:
        cur[i, C_DOC] = END
        return
    _load(ix, cur, buf, i, cur[i, C_FIRST], stats)


@nb.njit(cache=True, nogil=True)
def _next(ix, cur, buf, i, stats) -> None:
    blk_post = ix[4]
    b = cur[i, C_BLK]
    pos = cur[i, C_POS] + 1
    if pos < blk_post[b + 1] - blk_post[b]:
        cur[i, C_POS] = pos
        cur[i, C_DOC] = buf[i, pos]
    elif b < cur[i, C_LAST]:
        _load(ix, cur, buf, i, b + 1, stats)
    else:
        cur[i, C_DOC] = END


@nb.njit(cache=True, nogil=True)
def _seek(ix, cur, buf, i, target, stats) -> None:
    """Move cursor i to its first doc >= target, skipping whole blocks via `blk_last`."""
    if cur[i, C_DOC] >= target:
        return
    blk_last = ix[1]
    b = cur[i, C_BLK]
    last = cur[i, C_LAST]
    while b <= last and blk_last[b] < target:
        b += 1
    if b > last:
        cur[i, C_DOC] = END
        return
    if b != cur[i, C_BLK]:
        _load(ix, cur, buf, i, b, stats)
    pos = cur[i, C_POS]
    while buf[i, pos] < target:
        pos += 1
    cur[i, C_POS] = pos
    cur[i, C_DOC] = buf[i, pos]


@nb.njit(cache=True, nogil=True)
def _impact(ix, cur, i) -> int:
    blk_post, impacts = ix[4], ix[6]
    return np.int64(impacts[blk_post[cur[i, C_BLK]] + cur[i, C_POS]])


@nb.njit(cache=True, nogil=True)
def _shallow(ix, cur, i, target):
    """Block max and last doc of the block of cursor i that would hold `target`, without decoding.

    A term with no posting at or after `target` contributes nothing there and bounds nothing.
    """
    blk_last, blk_max = ix[1], ix[7]
    b = cur[i, C_BLK]
    last = cur[i, C_LAST]
    while b <= last and blk_last[b] < target:
        b += 1
    if b > last:
        return np.int64(0), np.int64(END) - 1
    return np.int64(blk_max[b]), np.int64(blk_last[b])


@nb.njit(cache=True, nogil=True)
def _sort_by_doc(order, cur, m) -> None:
    for a in range(1, m):
        x = order[a]
        j = a - 1
        while j >= 0 and cur[order[j], C_DOC] > cur[x, C_DOC]:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = x


# ----------------------------------------------------------------------------- kernels
@nb.njit(cache=True, nogil=True)
def wand_topk(ix, terms, weights, k, mask, block_size, use_block_max):
    """WAND (Broder et al., 2003) or, with `use_block_max`, block-max WAND (Ding and Suel, 2011)."""
    term_max = ix[8]
    m = terms.shape[0]
    stats = np.zeros(N_STATS, dtype=np.int64)
    hs = np.empty(max(k, 1), dtype=np.int64)
    hd = np.empty(max(k, 1), dtype=np.int32)
    size = 0
    if m == 0 or k <= 0:
        docs, scores = heap_sorted(hs, hd, 0)
        return docs, scores, stats

    cur = np.zeros((m, 5), dtype=np.int64)
    buf = np.empty((m, block_size), dtype=np.int32)
    ub = np.empty(m, dtype=np.int64)
    order = np.arange(m)
    for i in range(m):
        _open(ix, cur, buf, i, terms[i], stats)
        ub[i] = np.int64(term_max[terms[i]]) * weights[i]

    while True:
        theta = hs[0] if size == k else np.int64(0)
        _sort_by_doc(order, cur, m)

        # Pivot: first cursor at which the sum of upper bounds can beat the threshold.
        acc = np.int64(0)
        p = -1
        for j in range(m):
            if cur[order[j], C_DOC] == END:
                break
            acc += ub[order[j]]
            if acc > theta:
                p = j
                break
        if p < 0:
            break
        pivot = cur[order[p], C_DOC]
        # Cursors already on the pivot contribute to it, so they join the pivot set.
        while p + 1 < m and cur[order[p + 1], C_DOC] == pivot:
            p += 1

        if use_block_max:
            bound = np.int64(0)
            nearest_end = np.int64(END) - 1
            for j in range(p + 1):
                bmax, bend = _shallow(ix, cur, order[j], pivot)
                bound += bmax * weights[order[j]]
                nearest_end = min(nearest_end, bend)
            if bound <= theta:
                # No doc in [pivot, nearest_end] can beat theta from these blocks, and the
                # cursors after p only start at cur[order[p + 1]].
                target = nearest_end + 1
                if p + 1 < m:
                    target = min(target, cur[order[p + 1], C_DOC])
                best = order[0]
                for j in range(1, p + 1):
                    if ub[order[j]] > ub[best]:
                        best = order[j]
                stats[STAT_SKIPS] += 1
                _seek(ix, cur, buf, best, target, stats)
                continue

        if cur[order[0], C_DOC] == pivot:
            if _passes(mask, pivot):
                score = np.int64(0)
                for j in range(p + 1):
                    score += _impact(ix, cur, order[j]) * weights[order[j]]
                stats[STAT_SCORED] += 1
                # Zero-impact postings (terms seen only in zero-weight fields) never match.
                if score > 0:
                    size = heap_offer(hs, hd, size, k, score, pivot)
            for j in range(p + 1):
                _next(ix, cur, buf, order[j], stats)
        else:
            # Move the strongest cursor that is still before the pivot up to it.
            best = order[0]
            for j in range(1, p + 1):
                if cur[order[j], C_DOC] < pivot and ub[order[j]] > ub[best]:
                    best = order[j]
            _seek(ix, cur, buf, best, pivot, stats)

    docs, scores = heap_sorted(hs, hd, size)
    return docs, scores, stats


@nb.njit(cache=True, nogil=True)
def exhaustive_topk(ix, terms, weights, k, mask, n_docs, block_size):
    """Term-at-a-time accumulation over every posting; the reference the pruned kernels match."""
    words, blk_last, blk_bits, blk_word, blk_post, term_blk, impacts = (
        ix[0],
        ix[1],
        ix[2],
        ix[3],
        ix[4],
        ix[5],
        ix[6],
    )
    stats = np.zeros(N_STATS, dtype=np.int64)
    acc = np.zeros(n_docs, dtype=np.int64)
    buf = np.empty(block_size, dtype=np.int32)
    for i in range(terms.shape[0]):
        t = terms[i]
        prev = -1
        for b in range(term_blk[t], term_blk[t + 1]):
            n = decode_block(words, blk_bits, blk_word, blk_post, b, prev, buf)
            stats[STAT_DECODED] += 1
            base = blk_post[b]
            for j in range(n):
                acc[buf[j]] += np.int64(impacts[base + j]) * weights[i]
            prev = np.int64(blk_last[b])
    hs = np.empty(max(k, 1), dtype=np.int64)
    hd = np.empty(max(k, 1), dtype=np.int32)
    size = 0
    if k > 0:
        for d in range(n_docs):
            if acc[d] > 0 and _passes(mask, d):
                stats[STAT_SCORED] += 1
                size = heap_offer(hs, hd, size, k, acc[d], d)
    docs, scores = heap_sorted(hs, hd, size)
    return docs, scores, stats


@nb.njit(cache=True, nogil=True)
def score_docs(ix, terms, weights, docs, block_size):
    """Exact scores of the given documents (any order), e.g. dense-only candidates in fusion."""
    order = np.argsort(docs)
    out = np.zeros(docs.shape[0], dtype=np.int64)
    m = terms.shape[0]
    stats = np.zeros(N_STATS, dtype=np.int64)
    cur = np.zeros((max(m, 1), 5), dtype=np.int64)
    buf = np.empty((max(m, 1), block_size), dtype=np.int32)
    for i in range(m):
        _open(ix, cur, buf, i, terms[i], stats)
        for j in range(order.shape[0]):
            d = docs[order[j]]
            _seek(ix, cur, buf, i, d, stats)
            if cur[i, C_DOC] == END:
                break
            if cur[i, C_DOC] == d:
                out[order[j]] += _impact(ix, cur, i) * weights[i]
    return out
