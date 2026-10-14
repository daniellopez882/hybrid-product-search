# 2. Bit-pack d-gaps per 128-document block, with skip data outside the payload

Status: accepted

## Context

The postings layout serves three access patterns:

1. sequential decoding (exhaustive scoring, and `next()` in WAND),
2. skipping forward to the first document >= a target (`seek()` in WAND),
3. asking "what is the largest impact in the block that would hold document d?" without
   decoding anything (the shallow move of block-max WAND).

Catalogue text gives a mix of very long lists (colours, "table", category words that appear in
thousands of products) and many short ones. The kernels are numba functions, so whatever the
format is has to decode in a tight loop without Python objects.

## Decision

Each term's sorted document ids are cut into blocks of 128 postings. A block stores
`gap - 1` for every posting, all with one bit width chosen for that block (frame of reference
without exceptions); runs of consecutive ids cost zero bits. Per block, separate arrays keep the
last document id, the bit width, the offset of the first packed 32-bit word and the offset of the
first posting. Block maxima of the impacts sit in another array, next to the impacts.

`seek()` walks the `blk_last` array and decodes only the block that holds the target.
`_shallow()` does the same walk and reads `blk_max` without decoding.

## Consequences

- Shallow moves are a scan over a small int32 array, which is what makes block-max checks cheap
  enough to run at every pivot.
- Decoding a block is one loop with a shift and a mask per posting; no branches on the data.
- A single large gap in a block widens every posting in it. PForDelta-style exception lists would
  avoid that; on WANDS the benchmark reports the resulting bits per posting next to the 32 bits
  of a plain int32 array, so the cost of the simplification is visible.
- The block size is a parameter of the layout. Tests run with blocks of 1 to 128 postings so
  block boundaries are crossed constantly; production uses 128.

## Alternatives rejected

- **Variable-byte (VByte) gaps.** Byte-aligned and simple, but decoding is data-dependent
  branching per posting, and block maxima still need separate skip structures.
- **PForDelta with exceptions.** Better compression on skewed gap distributions, at the price of
  a second decode pass for exceptions; not worth it for a 43k-product catalogue.
- **Elias-Fano.** Excellent random access, but its `nextGEQ` is harder to keep tight in numba,
  and it does not by itself give per-block impact maxima.
- **Uncompressed int32 ids.** Fastest to read, four bytes per posting.
