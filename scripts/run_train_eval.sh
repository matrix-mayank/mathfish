#!/usr/bin/env bash
# Run train -> inference -> evaluate.
# From mathfish_new: bash scripts/run_train_eval.sh
#   QUICK=--quick         : fast 50-step run
#   IMPROVED=1            : better accuracy (500 steps, 8 negatives, lower temp)
#   TRAIN_N=2000          : sample 2000 train examples from HF, then train (default: 500)

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_N=${TRAIN_N:-500}
TRAIN_STEPS=${TRAIN_STEPS:-200}
QUICK=${QUICK:-}
OUT_DIR="${OUT_DIR:-outputs/biencoder_${TRAIN_N}}"
[ -n "${IMPROVED}" ] && IMPROVED_ARG=--improved || IMPROVED_ARG=

TRAIN_FILE="data/processed/problems_train${TRAIN_N}.jsonl"
if [ ! -f "$TRAIN_FILE" ] || [ -n "${FORCE_SAMPLE}" ]; then
  echo "=== 0. Sample train (n=$TRAIN_N) ==="
  python scripts/sample_train.py --n "$TRAIN_N"
fi

echo "=== 1. Train ($TRAIN_FILE) ==="
python scripts/train_biencoder.py \
  --problems_file "$TRAIN_FILE" \
  --from_hf --max_steps "$TRAIN_STEPS" --batch_size 8 \
  --output_dir "$OUT_DIR" \
  $QUICK $IMPROVED_ARG

echo "=== 2. Inference ==="
python scripts/run_inference.py \
  --checkpoint_dir "$OUT_DIR/checkpoint" \
  --problems_file data/processed/problems_dev100.jsonl \
  --from_hf --output_file data/processed/predictions_dev100.jsonl --top_k 5

echo "=== 3. Evaluate ==="
python scripts/evaluate_alignment.py \
  --predictions_file data/processed/predictions_dev100.jsonl \
  --gold_file data/processed/problems_dev100.jsonl \
  --standards_path data/cache/achieve-the-core/standards.jsonl \
  --domain_groups_path data/cache/achieve-the-core/domain_groups.json

echo "Done."
