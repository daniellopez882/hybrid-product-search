"""Query-time attribute filters compiled to a per-document byte mask.

The mask is passed into the retrieval kernels rather than applied afterwards: block-max WAND
skips non-matching documents before scoring them, and HNSW keeps expanding through non-matching
nodes but only admits matching ones to its result set. A post-filter would instead return fewer
than k hits whenever the filter is selective.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from prodsearch.analysis import fold
from prodsearch.catalog import Catalog, parse_features, split_classes


@dataclass(frozen=True)
class Filter:
    classes: tuple[str, ...] = ()
    category: str | None = None
    min_rating: float | None = None
    min_reviews: float | None = None
    attributes: tuple[tuple[str, str], ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.classes
            or self.category
            or self.min_rating is not None
            or self.min_reviews is not None
            or self.attributes
        )

    def describe(self) -> dict[str, object]:
        out: dict[str, object] = {}
        if self.classes:
            out["class"] = list(self.classes)
        if self.category:
            out["category"] = self.category
        if self.min_rating is not None:
            out["min_rating"] = self.min_rating
        if self.min_reviews is not None:
            out["min_reviews"] = self.min_reviews
        if self.attributes:
            out["attr"] = [f"{k}={v}" for k, v in self.attributes]
        return out


def _norm_path(path: str) -> tuple[str, ...]:
    return tuple(fold(p).strip() for p in path.split("/") if p.strip())


def _csr(groups: Mapping[str, list[int]]) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    keys = {key: i for i, key in enumerate(groups)}
    ptr = np.zeros(len(keys) + 1, dtype=np.int64)
    np.cumsum([len(v) for v in groups.values()], out=ptr[1:])
    docs = np.fromiter((d for v in groups.values() for d in v), dtype=np.int32, count=int(ptr[-1]))
    return keys, ptr, docs


@dataclass
class AttributeIndex:
    catalog: Catalog
    _classes: dict[str, list[int]] = field(init=False, repr=False)
    _paths: dict[tuple[str, ...], list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._classes = {}
        self._paths = {}
        for doc, (raw_class, path) in enumerate(
            zip(self.catalog.classes, self.catalog.categories, strict=True)
        ):
            for c in split_classes(raw_class):
                self._classes.setdefault(fold(c), []).append(doc)
            parts = _norm_path(path)
            for depth in range(1, len(parts) + 1):
                self._paths.setdefault(parts[:depth], []).append(doc)

    @cached_property
    def _features(self) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
        # Built on first use: 2.4 M (key, value) occurrences on WANDS, rarely needed.
        groups: dict[str, list[int]] = {}
        for doc, raw in enumerate(self.catalog.features):
            for key, value in set(parse_features(raw)):
                groups.setdefault(f"{key}={value}", []).append(doc)
        return _csr(groups)

    def mask(self, flt: Filter) -> np.ndarray | None:
        """uint8 mask over documents, or None when the filter constrains nothing."""
        if flt.is_empty():
            return None
        n = len(self.catalog)
        keep = np.ones(n, dtype=bool)
        if flt.classes:
            any_class = np.zeros(n, dtype=bool)
            for c in flt.classes:
                any_class[self._classes.get(fold(c).strip(), [])] = True
            keep &= any_class
        if flt.category:
            in_path = np.zeros(n, dtype=bool)
            in_path[self._paths.get(_norm_path(flt.category), [])] = True
            keep &= in_path
        if flt.min_rating is not None:
            keep &= np.nan_to_num(self.catalog.average_rating, nan=-1.0) >= flt.min_rating
        if flt.min_reviews is not None:
            keep &= np.nan_to_num(self.catalog.review_count, nan=0.0) >= flt.min_reviews
        if flt.attributes:
            keys, ptr, docs = self._features
            for key, value in flt.attributes:
                has = np.zeros(n, dtype=bool)
                slot = keys.get(f"{key.strip().lower()}={value.strip().lower()}")
                if slot is not None:
                    has[docs[ptr[slot] : ptr[slot + 1]]] = True
                keep &= has
        return keep.astype(np.uint8)
