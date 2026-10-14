# 3. Tune fusion and field weights on held-out folds, fuse over full scores of the union

Status: accepted

## Context

WANDS has 480 queries. Two parts of the pipeline have free parameters that move nDCG: the BM25F
field weights and the alpha of the convex combination. Choosing them on all 480 queries and
reporting on the same 480 overstates the gain of tuning, and the gain is what the comparison
against plain BM25 and RRF is about.

Convex fusion also needs a score from both retrievers for every candidate. A document that only
the dense retriever found has no BM25F score in the lexical top-100, and vice versa. The common
fix is to impute the minimum of the list, which treats "ranked 101st" and "matches no query term"
alike.

## Decision

- Queries are dealt into 5 folds by a seeded permutation. For each fold, the BM25F weights are
  chosen from an 81-point grid by mean nDCG@10 on the other four folds, then alpha is chosen from
  21 values in [0, 1] the same way, given that fold's weights. Each query is scored once, by the
  settings chosen without it. Candidate order breaks ties, with the a-priori default weights
  first.
- Convex fusion scores the union of both candidate lists with both retrievers: the lexical index
  scores dense-only hits exactly (`score_docs`, a seek per term), and the dense side is a dot
  product with the stored product vector. Each score vector is z-normalised over the union and
  combined as `alpha * z_lex + (1 - alpha) * z_dense`.
- RRF uses k = 60 and is not tuned, so it is an honest untuned reference.
- The configuration the latency benchmark runs is tuned on all queries, and it is only used for
  timing.

## Consequences

- The relevance table reports what a fresh query set would see, within the noise the bootstrap
  intervals show. The per-fold choices are written to the results file, so instability (folds
  picking different weights) is visible.
- Scoring the union exactly costs one `score_docs` call per query, a few hundred seeks.
- 81 x 21 settings are evaluated, which is cheap because impacts are recomputed without
  rebuilding the doc-id layout (ADR 1) and fusion is NumPy over at most 200 candidates.

## Alternatives rejected

- **Tune on all queries.** Overstates tuned methods against untuned ones.
- **Min-score imputation for missing candidates.** Cheaper, but conflates near-misses with
  non-matches, and the exact score is available.
- **Min-max normalisation.** Driven by the single best and worst candidate of each list; z-scores
  use the whole union and are less sensitive to one outlier score.
- **Learning-to-rank over many features.** Not identifiable with 480 queries without a separate
  training set.
