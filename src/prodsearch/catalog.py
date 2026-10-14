"""The WANDS catalogue, queries and graded judgements, in the shapes the engine needs.

Documents are addressed by their row position (0..N-1) everywhere inside the engine; the WANDS
`product_id` is kept alongside for output.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FIELDS: tuple[str, ...] = ("name", "class", "category", "features", "description")
GRADES: dict[str, int] = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

csv.field_size_limit(1 << 30)


def _rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)


def _float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def parse_features(raw: str) -> list[tuple[str, str]]:
    """`key:value|key : value|...` into normalised (key, value) pairs; empty values dropped."""
    pairs = []
    for item in raw.split("|"):
        key, _, value = item.partition(":")
        key, value = key.strip().lower(), value.strip().lower()
        if key and value:
            pairs.append((key, value))
    return pairs


def split_classes(raw: str) -> list[str]:
    return [c.strip() for c in raw.split("|") if c.strip()]


@dataclass
class Catalog:
    product_ids: np.ndarray
    names: list[str]
    classes: list[str]
    categories: list[str]
    descriptions: list[str]
    features: list[str]
    average_rating: np.ndarray
    review_count: np.ndarray

    def __len__(self) -> int:
        return len(self.names)

    def field_text(self, doc: int, name: str) -> str:
        if name == "name":
            return self.names[doc]
        if name == "class":
            return self.classes[doc].replace("|", " ")
        if name == "category":
            return self.categories[doc].replace("/", " ")
        if name == "features":
            # Values only: keys such as "overallwidth-sidetoside" are schema names, not words
            # a shopper types.
            return " ".join(v for _, v in parse_features(self.features[doc]))
        if name == "description":
            return self.descriptions[doc]
        raise KeyError(name)

    def passage(self, doc: int) -> str:
        """The text the dense encoder and the cross-encoder read for one product."""
        parts = [
            self.names[doc],
            self.classes[doc].replace("|", ", "),
            self.categories[doc],
            self.descriptions[doc],
        ]
        return ". ".join(p for p in parts if p)

    def subset(self, docs: Sequence[int]) -> Catalog:
        idx = np.asarray(docs, dtype=np.int64)
        return Catalog(
            product_ids=self.product_ids[idx],
            names=[self.names[i] for i in idx],
            classes=[self.classes[i] for i in idx],
            categories=[self.categories[i] for i in idx],
            descriptions=[self.descriptions[i] for i in idx],
            features=[self.features[i] for i in idx],
            average_rating=self.average_rating[idx],
            review_count=self.review_count[idx],
        )


@dataclass(frozen=True)
class Query:
    qid: int
    text: str
    query_class: str


@dataclass
class Judgements:
    """Graded labels per query, keyed by internal document position."""

    grades: dict[int, dict[int, int]] = field(default_factory=dict)
    rows: int = 0
    conflicts: int = 0

    def for_query(self, qid: int) -> dict[int, int]:
        return self.grades.get(qid, {})

    def __len__(self) -> int:
        return sum(len(g) for g in self.grades.values())


def load_catalog(path: Path) -> Catalog:
    rows = list(_rows(path))
    return Catalog(
        product_ids=np.array([int(r["product_id"]) for r in rows], dtype=np.int64),
        names=[r["product_name"].strip() for r in rows],
        classes=[r["product_class"].strip() for r in rows],
        categories=[r["category hierarchy"].strip() for r in rows],
        descriptions=[r["product_description"].strip() for r in rows],
        features=[r["product_features"] for r in rows],
        average_rating=np.array([_float(r["average_rating"]) for r in rows], dtype=np.float32),
        review_count=np.array([_float(r["review_count"]) for r in rows], dtype=np.float32),
    )


def load_queries(path: Path) -> list[Query]:
    return [
        Query(qid=int(r["query_id"]), text=r["query"].strip(), query_class=r["query_class"].strip())
        for r in _rows(path)
    ]


def load_judgements(path: Path, catalog: Catalog) -> Judgements:
    """WANDS repeats some (query, product) rows; a repeated pair keeps its highest grade."""
    position = {int(pid): i for i, pid in enumerate(catalog.product_ids)}
    out: dict[int, dict[int, int]] = {}
    rows = 0
    conflicted: set[tuple[int, int]] = set()
    for r in _rows(path):
        rows += 1
        doc = position.get(int(r["product_id"]))
        if doc is None:
            continue
        qid = int(r["query_id"])
        grades = out.setdefault(qid, {})
        grade = GRADES[r["label"]]
        if doc in grades and grades[doc] != grade:
            conflicted.add((qid, doc))
        grades[doc] = max(grade, grades.get(doc, 0))
    return Judgements(out, rows=rows, conflicts=len(conflicted))


@dataclass
class Wands:
    catalog: Catalog
    queries: list[Query]
    judgements: Judgements

    def restrict(self, n_queries: int, docs_per_query: int | None = None) -> Wands:
        """The first `n_queries` queries and a catalogue of only products judged for them.

        With `docs_per_query`, each query contributes at most that many judged products (the
        lowest ids), which keeps smoke runs small enough to embed in seconds.
        """
        queries = self.queries[:n_queries]
        docs = sorted(
            {d for q in queries for d in sorted(self.judgements.for_query(q.qid))[:docs_per_query]}
        )
        remap = {old: new for new, old in enumerate(docs)}
        grades = {
            q.qid: {remap[d]: g for d, g in self.judgements.for_query(q.qid).items() if d in remap}
            for q in queries
        }
        rows = sum(len(g) for g in grades.values())
        return Wands(self.catalog.subset(docs), queries, Judgements(grades, rows=rows))


def load_wands(directory: Path) -> Wands:
    catalog = load_catalog(directory / "product.csv")
    return Wands(
        catalog=catalog,
        queries=load_queries(directory / "query.csv"),
        judgements=load_judgements(directory / "label.csv", catalog),
    )
