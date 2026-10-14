# 1. Quantise BM25F term scores to 16-bit integers at index time

Status: accepted

## Context

Block-max WAND skips documents whose upper bound cannot beat the current top-k threshold. The
claim worth making about it is that it returns *exactly* the exhaustive top-k, and the test
suite has to be able to check that claim on random corpora, including the order of tied
documents.

With floating-point scores that check is fragile. The exhaustive scorer adds term contributions
term-at-a-time, WAND adds them document-at-a-time in cursor order, and the two orders round
differently in the last bit. A document can then sit exactly on the threshold in one kernel and
just under it in the other, and ties between documents come out in different orders. Tests end
up comparing with tolerances, which hides real off-by-one bugs in the skipping logic.

BM25F also has a cost at query time: if the per-field frequencies and lengths are stored, every
scored posting needs five divisions and a saturation.

## Decision

The full BM25F contribution `s(t, d) = idf(t) * tf~ / (k1 + tf~)` is computed once at index time
and quantised linearly to `uint16` against the largest `s` in the index (any positive score maps
to at least 1). A query score is an integer sum `sum_t qtf(t) * q(t, d)`, accumulated in int64.
Block maxima and term maxima are maxima of those integers.

All kernels break ties towards the lower document id. Because documents are visited in ascending
id order, "enter the heap only if strictly above the threshold" implements that tie-break, and
the pruning tests use strict comparisons as well.

## Consequences

- WAND, BMW and the exhaustive kernel return bit-identical (doc, score) lists; the hypothesis
  tests in `tests/test_wand.py` compare with `array_equal`, against a NumPy brute force that
  never touches the compressed layout.
- The quantisation step is 1/65535 of the largest term score in the index, well below the
  differences that decide rankings; `tests/test_bm25f.py` bounds the error against the float
  formula by one step.
- Changing field weights, b or k1 means recomputing impacts. That is one vectorised NumPy pass
  over the postings and leaves the doc-id layout untouched, which is what makes the 81-point
  field-weight grid in the benchmark affordable; the results file records how long it takes.
- Impacts take 2 bytes per posting, uncompressed; the benchmark reports their share of the
  index bytes next to the packed doc ids.

## Alternatives rejected

- **Float32 impacts with a tolerance in the tests.** Simpler, but the tests could no longer tell
  a correct tie from a skipping bug.
- **Lucene-style query-time scoring from stored frequencies and norms.** Lets weights change per
  query, but costs a division per field per scored posting and still has the rounding problem.
- **8-bit impacts.** Halves impact storage, but with 256 levels distinct BM25F scores collide
  often, and those collisions are then ordered by document id instead of by score.
