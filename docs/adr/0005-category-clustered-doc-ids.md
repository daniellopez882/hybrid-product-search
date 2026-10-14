# 5. Assign internal doc ids in category order

Status: accepted

## Context

WANDS product ids carry no structure: neighbouring ids are unrelated products. Two parts of the
lexical index depend on which documents sit next to each other in id space:

- **Compression.** A block's bit width is set by its widest d-gap (ADR 2). When the products that
  share a term are scattered, gaps are wide everywhere.
- **Block maxima.** Block-max WAND can only skip a block whose maximum impact is low. With
  scattered ids, every block of a common term such as "table" mixes strong and weak matches, so
  its maximum is close to the term maximum and the block-max check rarely prunes.

A first measurement on the full catalogue in catalogue order, made while the build was still
contended by other processes, showed block-max WAND decoding nearly every block it visited and
running no faster than exhaustive scoring. That measurement is not a benchmark result; it is what
prompted this decision, and the benchmark now reports both orders.

## Decision

Postings use internal doc ids assigned by sorting products on (category path, class, name). The
index keeps the permutation: kernels run on internal ids, and `LexicalIndex.search` and
`LexicalIndex.score` translate to and from catalogue positions, including filter masks. Dense
vectors, the catalogue and every API keep catalogue positions; only the inverted index sees the
internal order.

## Consequences

- Products of one category share terms, so their postings cluster. That should shrink gaps and
  make blocks more homogeneous in score, which is what block-max skipping needs; the benchmark
  reports whether it does on WANDS.
- Ties between equal BM25F scores are broken by internal id, so equal-scoring products of one
  category come out in name order rather than in WANDS id order. The exactness guarantee is
  unchanged: the pruned kernels match the exhaustive one in the same order (tested with random
  permutations in `tests/test_wand.py`).
- The benchmark builds the same BM25F index in catalogue order too and reports bits per posting
  and kernel latency for both orders, so the effect is measured rather than assumed.
- The order is computed from catalogue fields, so adding a product means rebuilding the index;
  there is no incremental update path (there is none for the layout anyway).

## Alternatives rejected

- **Catalogue order.** Kept as the measured baseline.
- **Graph-based reordering (BP / recursive graph bisection).** Better compression in the
  literature, but an optimisation problem of its own; category order gets most of the locality
  for free because the catalogue already has a taxonomy.
- **Ordering by popularity (review count).** Groups strong documents for impact-ordered
  processing, but does not cluster the term vocabulary, which is what both the gaps and the block
  maxima need.
