#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert BIO-format NER result file (word gold pred, blank line per sentence)
into the SpanNER-style span file:

  sentence_content\t token_text:: start,end:: gold:: pred [\t ...]

This does NOT require probabilities. It unifies BIO and span-based outputs
into a single internal representation that the combiner can consume.

Usage:
  python scripts/bio_result_to_span_txt.py \
    --input combination/results/conll03/conll03_CbertWglove_lstmCrf_72102467_test_9088.txt \
    --output combination/results/conll03/conll03_CbertWglove_lstmCrf_72102467_test_9088_span.txt

If --output is a directory, the converted filename will reuse the input base
name and add suffix "_span.txt".
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BIO result -> SpanNER span txt")
    p.add_argument("--input", required=True, help="BIO result file path")
    p.add_argument("--output", required=True, help="output file or directory")
    return p.parse_args()


def get_chunks_onesent(tags: List[str]) -> List[Tuple[str, int, int]]:
    """Extract chunks from BIO tags for a single sentence.

    Returns a list of (label, start_index, end_index_inclusive).
    """
    chunks: List[Tuple[str, int, int]] = []
    start, lab = -1, None
    for i, t in enumerate(tags + ["O"]):  # sentinel O at end
        if t.startswith("B-"):
            if lab is not None:
                chunks.append((lab, start, i - 1))
            lab = t[2:]
            start = i
        elif t.startswith("I-"):
            # continuation of current chunk
            if lab is None:
                lab = t[2:]
                start = i
        else:  # t == 'O' or outside
            if lab is not None:
                chunks.append((lab, start, i - 1))
                lab = None
                start = -1
    return chunks


def read_bio_result(path: str) -> List[Tuple[List[str], List[str], List[str]]]:
    """Read BIO result file into list of (words, gold_tags, pred_tags)."""
    sents: List[Tuple[List[str], List[str], List[str]]] = []
    words: List[str] = []
    gold: List[str] = []
    pred: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("-DOCSTART-"):
                if words:
                    sents.append((words, gold, pred))
                    words, gold, pred = [], [], []
                continue
            parts = line.split()
            if len(parts) < 3:
                # tolerate malformed lines
                continue
            w, g, p = parts[0], parts[1], parts[2]
            words.append(w)
            gold.append(g)
            pred.append(p)
    if words:
        sents.append((words, gold, pred))
    return sents


def convert_one(input_path: str, output_path: str) -> None:
    sents = read_bio_result(input_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for sent_id, (words, gold, pred) in enumerate(sents):
            # build gold/pred chunks and union by (sid,eid)
            gold_chunks = get_chunks_onesent(gold)
            pred_chunks = get_chunks_onesent(pred)

            gold_map: Dict[Tuple[int, int], str] = {(s, e): lab for lab, s, e in gold_chunks}
            pred_map: Dict[Tuple[int, int], str] = {(s, e): lab for lab, s, e in pred_chunks}
            keys = sorted(set(gold_map.keys()) | set(pred_map.keys()))

            fields: List[str] = [" ".join(words)]
            for (s, e) in keys:
                g = gold_map.get((s, e), "O")
                p = pred_map.get((s, e), "O")
                token_text = " ".join(words[s : e + 1])
                fields.append(f"{token_text}:: {s},{e}:: {g}:: {p}")
            out.write("\t".join(fields) + "\n")


def main():
    args = parse_args()
    in_path = args.input
    out_path = args.output
    if os.path.isdir(out_path):
        base = os.path.basename(in_path)
        name, ext = os.path.splitext(base)
        out_path = os.path.join(out_path, f"{name}_span.txt")
    convert_one(in_path, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


