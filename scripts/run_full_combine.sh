#!/usr/bin/env bash
set -euo pipefail

# AutoDL one-click script: convert BIO results to span format and run combiner

# ---------- Config ----------
ROOT_DIR="/root/autodl-tmp/HybridNER"
DATANAME="conll03"
DATA_DIR="$ROOT_DIR/combination/results/$DATANAME"

# BIO result files to convert (relative to DATA_DIR)
BIO_FILES=(
  "conll03_CflairWnon_lstmCrf_1_test_9241.txt"
  "conll03_CbertWglove_lstmCrf_1_test_9201.txt"
)

# Your local spanNER result (auto-detect the first match if not set)
LOCAL_SPAN_FILE=""

# ---------- Prepare ----------
cd "$ROOT_DIR"
mkdir -p "$DATA_DIR"

echo "[1/3] Converting BIO results to span format..."
for f in "${BIO_FILES[@]}"; do
  in_path="$DATA_DIR/$f"
  if [[ -f "$in_path" ]]; then
    python scripts/bio_result_to_span_txt.py \
      --input "$in_path" \
      --output "$DATA_DIR"
  else
    echo "  - WARN: missing $in_path, skip."
  fi
done

echo "[2/3] Locating your local spanNER result..."
if [[ -z "$LOCAL_SPAN_FILE" ]]; then
  # pick the first conll03_spanNER_local_test_*.txt in DATA_DIR
  LOCAL_SPAN_FILE=$(ls -1 "$DATA_DIR"/conll03_spanNER_local_test_*.txt 2>/dev/null | head -n 1 || true)
fi
if [[ -z "$LOCAL_SPAN_FILE" ]]; then
  echo "ERROR: Not found: $DATA_DIR/conll03_spanNER_local_test_*.txt"
  echo "Please copy your spanNER result into $DATA_DIR and rerun."
  exit 1
fi
echo "  - Using: $LOCAL_SPAN_FILE"

# Construct converted file names
MODEL_A="conll03_CflairWnon_lstmCrf_1_test_9241_span.txt"
MODEL_B="conll03_CbertWglove_lstmCrf_1_test_9201_span.txt"

echo "[3/3] Running combiner (weighted_f1 + weighted_cat)..."
python main.py \
  --dataname "$DATANAME" \
  --data_dir "$DATA_DIR" \
  --use_combiner \
  --comb_model_results \
    "$MODEL_A" \
    "$MODEL_B" \
    "$(basename "$LOCAL_SPAN_FILE")" \
  --comb_strategy all \
  --comb_result_dir comb_result \
  --comb_wscore 0.5 --comb_wf1 1.5

echo "Done. Results saved to: $ROOT_DIR/comb_result"


