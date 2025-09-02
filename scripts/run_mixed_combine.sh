#!/usr/bin/env bash
set -euo pipefail

# Mixed combiner script: BIO format for first 3 strategies, span format for 4th strategy

# ---------- Config ----------
ROOT_DIR="/root/autodl-tmp/HybridNER"
DATANAME="conll03"
DATA_DIR="$ROOT_DIR/combination/results/$DATANAME"

# Original BIO result files (for first 3 strategies)
BIO_FILES=(
  "conll03_CflairWnon_lstmCrf_1_test_9241.txt"
  "conll03_CbertWglove_lstmCrf_1_test_9201.txt"
)

# Your local spanNER result (auto-detect the first match if not set)
LOCAL_SPAN_FILE=""

# ---------- Prepare ----------
cd "$ROOT_DIR"
mkdir -p "$DATA_DIR"

echo "[1/3] Converting BIO results to span format (for 4th strategy only)..."
for f in "${BIO_FILES[@]}"; do
  in_path="$DATA_DIR/$f"
  if [[ -f "$in_path" ]]; then
    python scripts/bio_result_to_span_txt.py \
      --input "$in_path" \
      --output "$DATA_DIR"
    echo "  - Converted: ${f%.*}_span.txt"
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

# Also convert local spanNER result to span format if needed
LOCAL_SPAN_FILE_SPAN="${LOCAL_SPAN_FILE%.*}_span.txt"
if [[ ! -f "$LOCAL_SPAN_FILE_SPAN" ]]; then
  echo "  - Converting local spanNER to span format..."
  python scripts/bio_result_to_span_txt.py \
    --input "$LOCAL_SPAN_FILE" \
    --output "$DATA_DIR"
fi

echo "[3/3] Running mixed combiner..."
echo "  - First 3 strategies (VM, VOF1, VCF1): using BIO format files"
echo "  - 4th strategy (SpanNER): using span format files + probability"

# Construct file lists
BIO_MODEL_A="${BIO_FILES[0]}"
BIO_MODEL_B="${BIO_FILES[1]}"
LOCAL_BIO="$(basename "$LOCAL_SPAN_FILE")"

SPAN_MODEL_A="${BIO_FILES[0]%.*}_span.txt"
SPAN_MODEL_B="${BIO_FILES[1]%.*}_span.txt"
LOCAL_SPAN="$(basename "$LOCAL_SPAN_FILE_SPAN")"

python main.py \
  --dataname "$DATANAME" \
  --data_dir "$DATA_DIR" \
  --use_combiner \
  --comb_model_results \
    "$BIO_MODEL_A" \
    "$BIO_MODEL_B" \
    "$LOCAL_BIO" \
    "$SPAN_MODEL_A" \
    "$SPAN_MODEL_B" \
    "$LOCAL_SPAN" \
  --comb_strategy "all" \
  --comb_result_dir comb_result \
  --comb_wscore 0.5 --comb_wf1 1.5

echo "===== Done ====="
echo "Strategy explanation:"
echo "  - VM/VOF1/VCF1: Used original BIO format files for better voting accuracy"
echo "  - SpanNER: Used span format files + probability for confidence-based combination"
echo "Results saved to: $ROOT_DIR/comb_result"

