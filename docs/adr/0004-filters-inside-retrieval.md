# 4. Apply attribute filters inside both retrievers, with an exact path for selective filters

Status: accepted

## Context

Product search is usually faceted: a shopper picks a class ("Bar Stools"), a category branch, a
minimum rating or an attribute value ("color = cream"). Filtering the final top-k afterwards
returns fewer than k results whenever the filter is selective, and the missing results are not
recoverable without searching deeper.

## Decision

A filter compiles to a byte mask over documents (`AttributeIndex.mask`), built from class and
category-prefix maps and a lazily built (key, value) index over product features. The mask goes
into the kernels:

- **BM25F (WAND / BMW):** a pivot document that fails the mask is not scored and its cursors
  move on. Pruning stays exact, since the heap only ever holds documents that pass.
- **HNSW:** the layer-0 beam keeps expanding through nodes that fail the mask, so the graph stays
  connected, but only passing nodes enter the result set.
- **Selective filters:** when fewer than 2,000 products pass, the dense side switches to an exact
  dot product over just those products. With few passing nodes the beam has to expand through
  many failing neighbours before it fills its result set, while the exact scan is one small
  matrix-vector product and has perfect recall.

## Consequences

- Every retrieval stage returns a full k whenever k products pass the filter.
- The pruned lexical kernels match the exhaustive filtered top-k exactly; the benchmark counts
  any query where they differ.
- The benchmark measures filtered HNSW recall against exact filtered search with the query's own
  WANDS class as the facet, so the cost of in-graph filtering is reported, not assumed.
- The feature index holds one entry per (key, value) occurrence in the catalogue, so it is built
  only when an attribute filter is first requested.

## Alternatives rejected

- **Post-filtering the fused list.** Short result pages for selective filters.
- **Over-fetching then post-filtering.** Needs a depth that depends on selectivity, which is not
  known in advance.
- **One HNSW graph per class.** Fast for class filters only, multiplies memory and does not
  handle ratings or attributes.
