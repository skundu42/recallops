#!/usr/bin/env bash
#
# RecallOps quickstart: Journey J1 (PRD §11), end to end, no API keys.
#
# The story: ingest a corpus, generate a golden set, confirm retrieval is green,
# ship a chunker change that regresses it, then let RecallOps *prove* the cause
# with a counterfactual arm, and revert back to green. Everything runs on the
# built-in local embedding provider, so it costs $0.00 and needs no network.
#
# Run it:
#     RECALL="uv run recall" ./examples/quickstart.sh      # from a checkout
#     ./examples/quickstart.sh                             # if `recall` is installed
#
set -euo pipefail

# `recall` command (override with e.g. RECALL="uv run recall").
RECALL="${RECALL:-recall}"

# The example corpus ships next to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="$SCRIPT_DIR/corpus"

# Work in a throwaway directory so we never touch your repo.
WORK="$(mktemp -d)"
cd "$WORK"
echo "Working in $WORK"
echo

# 1. Initialise the project (writes recall.yaml + the .recall provenance store).
$RECALL init --source "$CORPUS"

# Use dense-only retrieval for this demo so a single chunker change is easy to
# read. (The default is hybrid dense+BM25.) We rewrite recall.yaml explicitly to
# show the full pipeline the engine tracks.
cat > recall.yaml <<YAML
project: quickstart
source: $CORPUS
adapter:
  type: local
pipeline:
  parser:
    tool: text-v1
  chunker:
    tool: recall.chunkers.markdown_heading
    params:
      max_tokens: 800
      overlap: 120
  embedding:
    provider: local
    model: hash-v1
    dims: 256
    params:
      seed: 0
      ngram: [1, 2]
  index:
    adapter: local
  retrieve:
    top_k: 10
    hybrid: null
gate:
  mode: statistical
  primary_metric: recall@5
  q: 0.05
costs:
  max_cost: 0.0
YAML

# 2. Ingest the corpus -> immutable snapshot A. Grab its id from the first token
#    of the ingest line (`snap_...  docs=12 chunks=...`).
echo
echo "== Ingest snapshot A (markdown_heading) =="
INGEST_A="$($RECALL ingest)"
echo "$INGEST_A"
SNAP_A="$(printf '%s\n' "$INGEST_A" | awk 'NR==1{print $1}')"

# 3. Generate a deterministic golden dataset from snapshot A's chunks.
echo
echo "== Generate golden dataset =="
$RECALL dataset generate --n 20 --seed 0 --name gold

# 4. Eval snapshot A: this is our GREEN baseline (high recall@5).
echo
echo "== Eval snapshot A (expect GREEN) =="
$RECALL eval gold-v1 --snapshot "$SNAP_A"

# 5. Ship a chunker change -> snapshot B. fixed_token(30,0) shatters the heading
#    sections into tiny windows, evicting some targets from the top-5.
echo
echo "== Ingest snapshot B (fixed_token 30/0) =="
INGEST_B="$($RECALL ingest --chunker recall.chunkers.fixed_token \
                           --chunk-params '{"max_tokens": 30, "overlap": 0}')"
echo "$INGEST_B"
SNAP_B="$(printf '%s\n' "$INGEST_B" | awk 'NR==1{print $1}')"

# 6. Eval snapshot B, now RED: recall@5 has dropped versus A.
echo
echo "== Eval snapshot B (expect RED) =="
$RECALL eval gold-v1 --snapshot "$SNAP_B"

# 7. Diff A -> B with deep attribution. The funnel localises the failing stage
#    and the counterfactual arm returns a VERIFIED cause: `chunk`.
echo
echo "== Diff A -> B with deep (verified) attribution =="
$RECALL diff "$SNAP_A" "$SNAP_B" --dataset gold-v1 --attribute deep

# 8. Revert the chunker (recall.yaml still holds markdown_heading, so this simply
#    re-ingests). It is content-addressed: the revert reproduces snapshot A and
#    the eval is GREEN again.
echo
echo "== Revert chunker -> reproduces snapshot A, GREEN again =="
INGEST_C="$($RECALL ingest)"
echo "$INGEST_C"
SNAP_C="$(printf '%s\n' "$INGEST_C" | awk 'NR==1{print $1}')"
if [ "$SNAP_C" = "$SNAP_A" ]; then
  echo "revert reproduced snapshot A ($SNAP_A) exactly; provenance is content-addressed."
fi
$RECALL eval gold-v1 --snapshot "$SNAP_A"

echo
echo "Done. Explore the store under $WORK/.recall"
