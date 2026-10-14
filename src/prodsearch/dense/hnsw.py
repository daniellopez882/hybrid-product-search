"""HNSW (Malkov and Yashunin, 2018) as numba kernels, for unit-norm vectors under inner product.

Insertion follows Algorithm 1 of the paper: greedy descent through the layers above the new
node's level, then an ef_construction beam search per layer, neighbour selection with the
diversity heuristic (Algorithm 4, without candidate extension or pruned-connection keeping), and
bidirectional links whose overflow is shrunk with the same heuristic. Layer 0 allows 2*M links,
upper layers M.

Storage: layer 0 links are an (N, 2M) int32 matrix; the links of node i on layer l >= 1 are row
`up_off[i] + l - 1` of an (R, M) matrix, where R is the total number of upper-layer slots.

Query-time filters are applied inside the layer-0 beam search: nodes outside the filter are still
expanded (so the graph stays navigable) but never enter the result set.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numba as nb
import numpy as np


@nb.njit(cache=True, nogil=True, fastmath=True, inline="always")
def _dist(X: np.ndarray, i: int, q: np.ndarray) -> float:
    acc = np.float32(0.0)
    for j in range(q.shape[0]):
        acc += X[i, j] * q[j]
    return np.float32(1.0) - acc


# Binary min-heap on (key, id) pairs stored in two arrays.
@nb.njit(cache=True, nogil=True, inline="always")
def _push(keys: np.ndarray, ids: np.ndarray, size: int, key: float, ident: int) -> int:
    i = size
    keys[i] = key
    ids[i] = ident
    while i > 0:
        parent = (i - 1) >> 1
        if keys[parent] <= keys[i]:
            break
        keys[i], keys[parent] = keys[parent], keys[i]
        ids[i], ids[parent] = ids[parent], ids[i]
        i = parent
    return size + 1


@nb.njit(cache=True, nogil=True, inline="always")
def _pop(keys: np.ndarray, ids: np.ndarray, size: int) -> int:
    """Remove the root; the caller reads keys[0]/ids[0] first."""
    size -= 1
    keys[0] = keys[size]
    ids[0] = ids[size]
    i = 0
    while True:
        left = 2 * i + 1
        if left >= size:
            break
        child = left
        if left + 1 < size and keys[left + 1] < keys[left]:
            child = left + 1
        if keys[i] <= keys[child]:
            break
        keys[i], keys[child] = keys[child], keys[i]
        ids[i], ids[child] = ids[child], ids[i]
        i = child
    return size


@nb.njit(cache=True, nogil=True)
def _search_layer(
    X, q, eps, n_eps, ef, layer, nb0, cnt0, nbu, cntu, up_off, ws_visit, ws_epoch,
    cand_k, cand_i, res_k, res_i, mask, out_d, out_i,
) -> int:  # fmt: skip
    """Beam search on one layer; writes up to ef results to out_d/out_i in ascending distance."""
    ws_epoch[0] += 1
    epoch = ws_epoch[0]
    use_mask = mask.shape[0] > 0
    n_cand = 0
    n_res = 0  # max-heap through negated keys: res_k[0] is minus the farthest kept distance
    for j in range(n_eps):
        e = eps[j]
        if ws_visit[e] == epoch:
            continue
        ws_visit[e] = epoch
        d = _dist(X, e, q)
        n_cand = _push(cand_k, cand_i, n_cand, d, e)
        if not use_mask or mask[e] != 0:
            n_res = _push(res_k, res_i, n_res, -d, e)
            if n_res > ef:
                n_res = _pop(res_k, res_i, n_res)
    while n_cand > 0:
        dc = cand_k[0]
        c = cand_i[0]
        n_cand = _pop(cand_k, cand_i, n_cand)
        if n_res >= ef and dc > -res_k[0]:
            break
        row = up_off[c] + layer - 1
        count = cnt0[c] if layer == 0 else cntu[row]
        for j in range(count):
            e = nb0[c, j] if layer == 0 else nbu[row, j]
            if ws_visit[e] == epoch:
                continue
            ws_visit[e] = epoch
            d = _dist(X, e, q)
            if n_res < ef or d < -res_k[0]:
                n_cand = _push(cand_k, cand_i, n_cand, d, e)
                if not use_mask or mask[e] != 0:
                    n_res = _push(res_k, res_i, n_res, -d, e)
                    if n_res > ef:
                        n_res = _pop(res_k, res_i, n_res)
    n_out = n_res
    while n_res > 0:
        out_d[n_res - 1] = -res_k[0]
        out_i[n_res - 1] = res_i[0]
        n_res = _pop(res_k, res_i, n_res)
    return n_out


@nb.njit(cache=True, nogil=True)
def _greedy(
    X: np.ndarray,
    q: np.ndarray,
    start: int,
    layer: int,
    nb0: np.ndarray,
    cnt0: np.ndarray,
    nbu: np.ndarray,
    cntu: np.ndarray,
    up_off: np.ndarray,
) -> int:
    cur = start
    cur_d = _dist(X, cur, q)
    changed = True
    while changed:
        changed = False
        row = up_off[cur] + layer - 1
        count = cnt0[cur] if layer == 0 else cntu[row]
        for j in range(count):
            e = nb0[cur, j] if layer == 0 else nbu[row, j]
            d = _dist(X, e, q)
            if d < cur_d:
                cur_d = d
                cur = e
                changed = True
    return cur


@nb.njit(cache=True, nogil=True)
def _select(
    X: np.ndarray, cand_d: np.ndarray, cand_i: np.ndarray, n: int, m: int, out: np.ndarray
) -> int:
    """Diversity heuristic: keep e only if it is closer to the base than to every kept node."""
    kept = 0
    for j in range(n):
        if kept >= m:
            break
        e = cand_i[j]
        good = True
        for s in range(kept):
            if _dist(X, e, X[out[s]]) < cand_d[j]:
                good = False
                break
        if good:
            out[kept] = e
            kept += 1
    return kept


@nb.njit(cache=True, nogil=True)
def _link(
    X: np.ndarray,
    node: int,
    nbr: int,
    layer: int,
    cap: int,
    nb0: np.ndarray,
    cnt0: np.ndarray,
    nbu: np.ndarray,
    cntu: np.ndarray,
    up_off: np.ndarray,
    tmp_d: np.ndarray,
    tmp_i: np.ndarray,
    sel: np.ndarray,
) -> None:
    """Add `node` to `nbr`'s links on `layer`, shrinking with the heuristic when full."""
    row = up_off[nbr] + layer - 1
    count = cnt0[nbr] if layer == 0 else cntu[row]
    if count < cap:
        if layer == 0:
            nb0[nbr, count] = node
            cnt0[nbr] = count + 1
        else:
            nbu[row, count] = node
            cntu[row] = count + 1
        return
    base = X[nbr]
    for j in range(count):
        e = nb0[nbr, j] if layer == 0 else nbu[row, j]
        tmp_i[j] = e
        tmp_d[j] = _dist(X, e, base)
    tmp_i[count] = node
    tmp_d[count] = _dist(X, node, base)
    order = np.argsort(tmp_d[: count + 1])
    sd = tmp_d[: count + 1][order]
    si = tmp_i[: count + 1][order]
    kept = _select(X, sd, si, count + 1, cap, sel)
    for j in range(kept):
        if layer == 0:
            nb0[nbr, j] = sel[j]
        else:
            nbu[row, j] = sel[j]
    if layer == 0:
        cnt0[nbr] = kept
    else:
        cntu[row] = kept


@nb.njit(cache=True, nogil=True)
def _insert_range(
    X, levels, lo, hi, M, ef_c, nb0, cnt0, nbu, cntu, up_off, state, ws_visit, ws_epoch,
    cand_k, cand_i, res_k, res_i, w_d, w_i, eps, links, tmp_d, tmp_i, sel, no_mask,
) -> None:  # fmt: skip
    for i in range(lo, hi):
        level = levels[i]
        if state[0] < 0:
            state[0] = i
            state[1] = level
            continue
        q = X[i]
        top = state[1]
        cur = state[0]
        for layer in range(top, level, -1):
            cur = _greedy(X, q, cur, layer, nb0, cnt0, nbu, cntu, up_off)
        eps[0] = cur
        n_eps = 1
        for layer in range(min(level, top), -1, -1):
            n = _search_layer(
                X, q, eps, n_eps, ef_c, layer, nb0, cnt0, nbu, cntu, up_off, ws_visit,
                ws_epoch, cand_k, cand_i, res_k, res_i, no_mask, w_d, w_i,
            )  # fmt: skip
            kept = _select(X, w_d, w_i, n, M, links)
            cap = 2 * M if layer == 0 else M
            row = up_off[i] + layer - 1
            for j in range(kept):
                if layer == 0:
                    nb0[i, j] = links[j]
                else:
                    nbu[row, j] = links[j]
            if layer == 0:
                cnt0[i] = kept
            else:
                cntu[row] = kept
            for j in range(kept):
                _link(X, i, links[j], layer, cap, nb0, cnt0, nbu, cntu, up_off, tmp_d, tmp_i, sel)
            for j in range(n):
                eps[j] = w_i[j]
            n_eps = n
        if level > top:
            state[0] = i
            state[1] = level


@nb.njit(cache=True, nogil=True)
def _knn(
    X, q, k, ef, entry, top, nb0, cnt0, nbu, cntu, up_off, ws_visit, ws_epoch,
    cand_k, cand_i, res_k, res_i, eps, mask, out_d, out_i,
) -> int:  # fmt: skip
    cur = entry
    for layer in range(top, 0, -1):
        cur = _greedy(X, q, cur, layer, nb0, cnt0, nbu, cntu, up_off)
    eps[0] = cur
    return _search_layer(
        X, q, eps, 1, max(ef, k), 0, nb0, cnt0, nbu, cntu, up_off, ws_visit, ws_epoch,
        cand_k, cand_i, res_k, res_i, mask, out_d, out_i,
    )  # fmt: skip


_NO_MASK = np.zeros(0, dtype=np.uint8)


@dataclass
class _Workspace:
    visit: np.ndarray
    epoch: np.ndarray
    cand_k: np.ndarray
    cand_i: np.ndarray
    res_k: np.ndarray
    res_i: np.ndarray

    @classmethod
    def make(cls, n: int, ef: int) -> _Workspace:
        return cls(
            visit=np.zeros(n, dtype=np.int32),
            epoch=np.zeros(1, dtype=np.int64),
            cand_k=np.empty(n + 1, dtype=np.float32),
            cand_i=np.empty(n + 1, dtype=np.int32),
            res_k=np.empty(ef + 2, dtype=np.float32),
            res_i=np.empty(ef + 2, dtype=np.int32),
        )


@dataclass
class HNSWIndex:
    vectors: np.ndarray
    M: int
    ef_construction: int
    levels: np.ndarray
    nb0: np.ndarray
    cnt0: np.ndarray
    nbu: np.ndarray
    cntu: np.ndarray
    up_off: np.ndarray
    entry: int
    top: int
    _local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        M: int = 16,
        ef_construction: int = 200,
        seed: int = 7,
        progress: Callable[[int, int], None] | None = None,
        chunk: int = 2048,
    ) -> HNSWIndex:
        X = np.ascontiguousarray(vectors, dtype=np.float32)
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        # Level ~ floor(-ln(U) * mL) with mL = 1 / ln(M), as in the paper.
        uniform = 1.0 - rng.random(n)
        levels = np.floor(-np.log(uniform) / math.log(M)).astype(np.int32)
        up_off = np.zeros(n, dtype=np.int64)
        np.cumsum(levels[:-1], out=up_off[1:])
        n_up = int(levels.sum())
        nb0 = np.full((n, 2 * M), -1, dtype=np.int32)
        cnt0 = np.zeros(n, dtype=np.int32)
        nbu = np.full((max(n_up, 1), M), -1, dtype=np.int32)
        cntu = np.zeros(max(n_up, 1), dtype=np.int32)
        state = np.array([-1, 0], dtype=np.int64)
        ws = _Workspace.make(n, ef_construction)
        w_d = np.empty(ef_construction + 1, dtype=np.float32)
        w_i = np.empty(ef_construction + 1, dtype=np.int32)
        eps = np.empty(ef_construction + 1, dtype=np.int32)
        links = np.empty(2 * M + 2, dtype=np.int32)
        tmp_d = np.empty(2 * M + 2, dtype=np.float32)
        tmp_i = np.empty(2 * M + 2, dtype=np.int32)
        sel = np.empty(2 * M + 2, dtype=np.int32)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            _insert_range(
                X, levels, lo, hi, M, ef_construction, nb0, cnt0, nbu, cntu, up_off, state,
                ws.visit, ws.epoch, ws.cand_k, ws.cand_i, ws.res_k, ws.res_i, w_d, w_i, eps,
                links, tmp_d, tmp_i, sel, _NO_MASK,
            )  # fmt: skip
            if progress is not None:
                progress(hi, n)
        return cls(
            vectors=X,
            M=M,
            ef_construction=ef_construction,
            levels=levels,
            nb0=nb0,
            cnt0=cnt0,
            nbu=nbu,
            cntu=cntu,
            up_off=up_off,
            entry=int(state[0]),
            top=int(state[1]),
        )

    def _workspace(self, ef: int) -> _Workspace:
        ws = getattr(self._local, "ws", None)
        if ws is None or ws.res_k.shape[0] < ef + 2:
            ws = _Workspace.make(self.n, ef)
            self._local.ws = ws
            self._local.eps = np.empty(1, dtype=np.int32)
        return ws

    def search(
        self, query: np.ndarray, k: int, ef: int = 64, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate top-k (ids, inner products), best first."""
        if self.n == 0 or k <= 0:
            return np.zeros(0, np.int32), np.zeros(0, np.float32)
        width = max(ef, k)
        ws = self._workspace(width)
        out_d = np.empty(width + 1, dtype=np.float32)
        out_i = np.empty(width + 1, dtype=np.int32)
        q = np.ascontiguousarray(query, dtype=np.float32)
        n = _knn(
            self.vectors, q, k, ef, self.entry, self.top, self.nb0, self.cnt0, self.nbu,
            self.cntu, self.up_off, ws.visit, ws.epoch, ws.cand_k, ws.cand_i, ws.res_k,
            ws.res_i, self._local.eps, _NO_MASK if mask is None else mask, out_d, out_i,
        )  # fmt: skip
        n = min(n, k)
        return out_i[:n].copy(), (1.0 - out_d[:n]).astype(np.float32)

    def nbytes(self) -> int:
        """Graph bytes (links, counts, offsets, levels); the vectors are counted separately."""
        arrays = (self.nb0, self.cnt0, self.nbu, self.cntu, self.up_off, self.levels)
        return int(sum(a.nbytes for a in arrays))

    def save(self, path: Path) -> None:
        np.savez(
            path,
            params=np.array([self.M, self.ef_construction, self.entry, self.top], dtype=np.int64),
            levels=self.levels,
            nb0=self.nb0,
            cnt0=self.cnt0,
            nbu=self.nbu,
            cntu=self.cntu,
            up_off=self.up_off,
        )

    @classmethod
    def load(cls, path: Path, vectors: np.ndarray) -> HNSWIndex:
        with np.load(path) as a:
            M, ef_c, entry, top = (int(v) for v in a["params"])
            return cls(
                vectors=np.ascontiguousarray(vectors, dtype=np.float32),
                M=M,
                ef_construction=ef_c,
                levels=a["levels"],
                nb0=a["nb0"],
                cnt0=a["cnt0"],
                nbu=a["nbu"],
                cntu=a["cntu"],
                up_off=a["up_off"],
                entry=entry,
                top=top,
            )


def exact_search(
    vectors: np.ndarray, query: np.ndarray, k: int, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force top-k by inner product, ties to the lower id."""
    sims = vectors @ query.astype(np.float32)
    candidates = np.arange(vectors.shape[0]) if mask is None else np.flatnonzero(mask)
    sims_c = sims[candidates]
    k = min(k, candidates.shape[0])
    if k == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.float32)
    part = np.argpartition(-sims_c, k - 1)[:k] if k < sims_c.shape[0] else np.arange(k)
    order = np.lexsort((candidates[part], -sims_c[part]))
    top = part[order]
    return candidates[top].astype(np.int32), sims_c[top].astype(np.float32)
