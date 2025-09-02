#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert prediction JSONL to files required by the SpanNER combiner.

Features
- Generate span-format TXT for combiner (sentence<TAB>entity:: s,e:: gold:: pred ...)
- Generate span probability PKL for combiner (dict[(sid,eid,sentid)] -> {label: prob, 'O': prob})

Usage (example on AutoDL):
  python scripts/convert_predictions.py \
    --jsonl my_results/conll03_confidence_local_model.jsonl \
    --out_dir combination/results/conll03 \
    --dataname conll03 \
    --classes ORG PER LOC MISC \
    --f1tag 8762

This will create
  - combination/results/conll03/conll03_spanNER_local_test_8762.txt
  - combination/results/conll03/conll03_spanner_prob.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_jsonl(jsonl_path: str) -> List[dict]:
    with open(jsonl_path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin]


def write_span_txt(examples: List[dict], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as fout:
        for obj in examples:
            sentence: str = obj.get("sentence", "")
            parts: List[str] = [sentence]
            entities = obj.get("entity", []) or []
            for ent in entities:
                # span: [start, end]
                span = ent.get("span", [0, 0])
                try:
                    sidx, eidx = int(span[0]), int(span[1])
                except Exception:
                    sidx, eidx = 0, 0
                gold = ent.get("answer", "O") or "O"
                pred = ent.get("pred", "O") or "O"
                token_text = ent.get("entity", "")
                parts.append(f"{token_text}:: {sidx},{eidx}:: {gold}:: {pred}")
            fout.write("\t".join(parts) + "\n")


def write_prob_pkl(
    examples: List[dict], out_path: str, classes: List[str]
) -> Tuple[int, str]:
    ensure_dir(os.path.dirname(out_path))
    # Key: (sid, eid, sentid) -> {label: prob, 'O': prob}
    prob_map: Dict[Tuple[int, int, int], Dict[str, float]] = {}

    for sentid, obj in enumerate(examples):
        entities = obj.get("entity", []) or []
        for ent in entities:
            span = ent.get("span", [0, 0])
            try:
                sidx, eidx = int(span[0]), int(span[1])
            except Exception:
                sidx, eidx = 0, 0

            pred = ent.get("pred", "O") or "O"
            try:
                conf = float(ent.get("confidence", 0.5))
            except Exception:
                conf = 0.5

            key = (sidx, eidx, int(sentid))
            label2prob: Dict[str, float] = {c: 0.5 for c in classes}
            label2prob["O"] = 0.5
            # Overwrite with the predicted label confidence when available
            label2prob[pred] = conf
            prob_map[key] = label2prob

    with open(out_path, "wb") as fout:
        pickle.dump(prob_map, fout)
    return len(prob_map), out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert JSONL predictions for combiner")
    parser.add_argument("--jsonl", required=True, help="Path to prediction JSONL produced by inference")
    parser.add_argument("--out_dir", required=True, help="Directory to write outputs (will be created)")
    parser.add_argument("--dataname", default="conll03", help="Dataset name used in output filenames")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=["ORG", "PER", "LOC", "MISC"],
        help="Entity classes; default for conll03",
    )
    parser.add_argument(
        "--f1tag",
        default="9000",
        help="4-digit F1 tag used in span TXT filename, e.g., 8762 means F1=0.8762",
    )
    parser.add_argument(
        "--span_txt_only",
        action="store_true",
        help="If set, only generate span TXT and skip probability PKL",
    )
    parser.add_argument(
        "--prob_pkl_only",
        action="store_true",
        help="If set, only generate probability PKL and skip span TXT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = load_jsonl(args.jsonl)

    # Output paths
    span_txt_path = os.path.join(
        args.out_dir, f"{args.dataname}_spanNER_local_test_{args.f1tag}.txt"
    )
    prob_pkl_path = os.path.join(args.out_dir, f"{args.dataname}_spanner_prob.pkl")

    if not args.prob_pkl_only:
        write_span_txt(examples, span_txt_path)
        print(f"Wrote span TXT: {span_txt_path}")

    if not args.span_txt_only:
        n_items, prob_path = write_prob_pkl(examples, prob_pkl_path, args.classes)
        print(f"Wrote prob PKL: {prob_path} (items: {n_items})")


if __name__ == "__main__":
    main()