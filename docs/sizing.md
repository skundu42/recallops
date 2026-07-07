# Storage sizing (FR-1.4)

RecallOps stores embeddings, not just their hashes, so that evals and
counterfactual arms can replay retrieval with **zero re-embedding** (FR-1.5,
FR-4.2, FR-7.2). Embeddings dominate the footprint; chunk text and SQLite
metadata are small next to them.

## Embedding footprint

Embeddings are stored in **fp16** columnar Parquet by default, keyed by
`embedding_key`. The size of one embedding-model version is:

```
bytes = chunks x dims x 2        # 2 bytes per fp16 component
```

| Chunks | dims = 256 | dims = 1536 |
|-------:|-----------:|------------:|
| 10k    | 5.12 MB    | 30.72 MB    |
| 50k    | 25.60 MB   | 153.60 MB   |
| 150k   | 76.80 MB   | 460.80 MB   |
| 1M     | 512.00 MB  | 3.07 GB     |
| 10M    | 5.12 GB    | 30.72 GB    |

(1 MB = 10^6 bytes, 1 GB = 10^9 bytes.)

The PRD reference point (FR-1.4) is **150k chunks x 1536 dims**: fp32 is
~0.92 GB ("~1 GB per model version"); fp16 halves that to **0.46 GB**. fp16 is
the default because retrieval rankings are insensitive to the last few bits of
mantissa, and it halves both disk and cache pressure. Store `n` model versions
(e.g. before/after a migration) and multiply the column by `n`.

## Corpus-overhead target (<= 2.5x)

The footprint target (NFR §12) is that the **local store overhead stays within
2.5x the corpus text size** at one embedding-model version (fp16). Worked
example at 1536 dims:

- Assume ~500 tokens/chunk and ~4 chars/token => ~2000 chars (~2 KB UTF-8) of
  source text per chunk.
- One fp16 embedding per chunk is `1536 x 2 = 3072` bytes ≈ 1.5x the chunk text.
- Chunk text is stored once per chunkset (deduplicated by `chunkset_key`), plus
  small SQLite manifest/index rows and the fp16 embedding column.

So at 1536 dims a single model version lands near ~1.5x corpus text, comfortably
under 2.5x; smaller models (256 dims => 512 bytes/chunk) are a fraction of the
text size. The budget is spent primarily on **embeddings**: the knobs that
move it are `dims`, fp16-vs-fp32, and how many model versions you retain.
Retention is managed with `recall gc --keep N` (keep the last N snapshots plus
pinned ones; FR-1.8), which prunes unreferenced chunk and embedding artifacts.

## Scale tiers

Attribution fidelity, not storage, is what the scale tiers (§14) govern.
Storage grows linearly with `chunks x dims x versions`; the tiers describe how
counterfactual attribution degrades (always labeled, never silent) as corpora
cross 500k and 5M chunks. See the README "Honest at scale" section.
