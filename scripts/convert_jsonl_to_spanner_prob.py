#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert a JSONL prediction file (produced by inference) to SpanNER combiner
probability file format (pickle).

Input JSONL schema per line (produced by framework.metric → get_predict_prune):
{
  "sentence": "...",
  "entity": [
    {
      "entity": "Japan",
      "span": [start_idx, end_idx],
      "pred": "LOC",
      "answer": "LOC",
      "confidence": 0.98,
      "uncertainty": 0.1,
      "res": 1
    },
    ...
  ]
}

Output pickle schema expected by combiner.models.dataread.read_span_score:
{
  (start_idx, end_idx, sent_id): {
    "ORG": prob, "PER": prob, "LOC": prob, "MISC": prob, "O": prob
  },
  ...
}

Usage:
  python scripts/convert_jsonl_to_spanner_prob.py \
    --jsonl my_results/conll03_confidence_local_model.jsonl \
    --out combination/results/conll03/conll03_spanner_prob.pkl \
    --classes ORG PER LOC MISC \
    --default 0.5

Notes:
- Only the predicted label's probability is available in the JSONL ("confidence").
  Other labels (including 'O') will be filled with a default score (0.5 by default).
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSONL predictions to SpanNER probability pickle"
    )
    parser.add_argument("--jsonl", required=True, help="Input prediction JSONL path")
    parser.add_argument("--out", required=True, help="Output pickle path")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=["ORG", "PER", "LOC", "MISC"],
        help="Entity classes to include (excluding 'O')",
    )
    parser.add_argument(
        "--default",
        type=float,
        default=0.5,
        help="Default probability for labels without explicit confidence",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def build_prob_dict(
    jsonl_path: str, classes: List[str], default_prob: float
) -> Dict[Tuple[int, int, int], Dict[str, float]]:
    prob_dict: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        for sent_id, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            ents = obj.get("entity", []) or []
            for ent in ents:
                span = ent.get("span", [0, 0])
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    continue
                try:
                    start_idx, end_idx = int(span[0]), int(span[1])
                except Exception:
                    continue

                key = (start_idx, end_idx, int(sent_id))
                # initialize with default scores
                label2prob = {label: float(default_prob) for label in classes}
                label2prob["O"] = float(default_prob)

                pred_label = ent.get("pred", "O")
                try:
                    conf = float(ent.get("confidence", default_prob))
                except Exception:
                    conf = float(default_prob)

                # clamp prob to [0,1]
                conf = max(0.0, min(1.0, conf))
                if pred_label not in label2prob:
                    # include unseen label if present in prediction
                    label2prob[pred_label] = conf
                else:
                    label2prob[pred_label] = conf

                prob_dict[key] = label2prob

    return prob_dict


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)
    prob_dict = build_prob_dict(args.jsonl, args.classes, args.default)
    with open(args.out, "wb") as fout:
        pickle.dump(prob_dict, fout)
    print(f"wrote: {args.out}; entries: {len(prob_dict)}; classes: {args.classes}; default={args.default}")


if __name__ == "__main__":
    main()

