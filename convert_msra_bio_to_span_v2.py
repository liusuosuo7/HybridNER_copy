#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable
from collections import Counter, defaultdict

DATASET_ROOT = "/root/autodl-tmp/HybridNER/dataset"
MSRA_DIR = None  # 若为空会自动在 dataset 下寻找包含 "msra" 的目录

# 明确你的文件名
FILES = {
    "train": "msra_train_bio.txt",
    "dev":   "msra_dev_bio.txt",
    "test":  "msra_test_bio.txt",
}

# 将 NR/NS/NT 映射为 PER/LOC/ORG（不区分大小写）
LABEL_MAP = {
    "NR": "PER", "NS": "LOC", "NT": "ORG",
    "PER": "PER", "LOC": "LOC", "ORG": "ORG"
}

def find_msra_dir(root: str) -> Path:
    root_p = Path(root)
    # 优先 dataset/msra
    for name in ["msra", "MSRA", "Msra"]:
        p = root_p / name
        if p.exists():
            return p
    # 其次递归找
    for p in root_p.rglob("*"):
        if p.is_dir() and "msra" in p.name.lower():
            return p
    return root_p

def normalize_tag(raw: str) -> Optional[Tuple[str, str]]:
    """
    将原始标签规整为 (prefix, label) 或 None (代表 O)。
    兼容：O / 0 / o / B-NS / B_NS / S-NR / M-NT / I-ORG 等。
    """
    t = raw.strip()
    if not t:
        return None
    # 0 / o 视为 O
    if t in {"O", "o", "0"}:
        return None
    # 统一分隔符
    t = t.replace("_", "-")
    # 如果没有连字符，可能是裸标签（极少见），直接当 O
    if "-" not in t:
        return None
    prefix, lab = t.split("-", 1)
    prefix = prefix.upper()
    lab_up = lab.upper()
    # NR/NS/NT -> PER/LOC/ORG；其余保持原样（但全部大写）
    lab_std = LABEL_MAP.get(lab_up, lab_up)
    return prefix, lab_std

def read_bio_file(fp: Path) -> List[Tuple[List[str], List[str]]]:
    """
    读取 BIO/BMES 文件，行内可能是：
      字 [其它列……] 标签
    句与句之间空行分隔。
    返回 [(tokens, tags), ...]
    """
    sents = []
    toks, tags = [], []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks:
                    sents.append((toks, tags))
                    toks, tags = [], []
                continue
            parts = line.split()
            if len(parts) == 1:
                token, tag = parts[0], "O"
            else:
                token, tag = parts[0], parts[-1]
            toks.append(token)
            tags.append(tag)
    if toks:
        sents.append((toks, tags))
    return sents

def token_starts(tokens: List[str]) -> List[int]:
    starts = []
    pos = 0
    for t in tokens:
        starts.append(pos)
        pos += len(t)
    return starts

def to_spans(tokens: List[str], tags: List[str]) -> Dict[str, str]:
    """
    单句 BIO/BMES -> { "start;end": "LABEL", ... }  (闭区间)
    """
    starts = token_starts(tokens)
    n = len(tokens)
    spans = {}

    i = 0
    while i < n:
        norm = normalize_tag(tags[i])
        if norm is None:
            i += 1
            continue
        prefix, lab = norm

        # 单字实体（S-）
        if prefix == "S":
            s = starts[i]
            e = s + len(tokens[i]) - 1
            spans[f"{s};{e}"] = lab
            i += 1
            continue

        # 以 B/I/M/E 任一开头，都向后吞并到 E 或遇到断开
        if prefix in {"B", "I", "M", "E"}:
            s = starts[i]
            e = s + len(tokens[i]) - 1
            j = i + 1
            while j < n:
                nxt = normalize_tag(tags[j])
                if nxt is None:
                    break
                p2, lab2 = nxt
                if lab2 != lab:
                    break
                if p2 in {"I", "M"}:
                    e = starts[j] + len(tokens[j]) - 1
                    j += 1
                    continue
                if p2 == "E":
                    e = starts[j] + len(tokens[j]) - 1
                    j += 1
                    break
                if p2 in {"B", "S"}:
                    break
            spans[f"{s};{e}"] = lab
            i = j
            continue

        i += 1

    return spans

def convert_split(msra_dir: Path, split_key: str, filename: str, out_dir: Path) -> Optional[Path]:
    src = msra_dir / filename
    if not src.exists():
        print(f"[WARN] {split_key}: 未找到文件 {src}")
        return None

    print(f"[INFO] 读取 {split_key}: {src}")
    sents = read_bio_file(src)
    items = []
    ent_counter = Counter()
    total_entities = 0

    for toks, tg in sents:
        ctx = "".join(toks)
        spans = to_spans(toks, tg)
        for lab in spans.values():
            ent_counter[lab] += 1
            total_entities += 1
        items.append({"context": ctx, "span_posLabel": spans})

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"msra_{split_key}_span.json"
    with out_fp.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[OK] {split_key}: 写出 {out_fp}  |  样本数={len(items)}  实体数={total_entities}  按类型={dict(ent_counter)}")
    if total_entities == 0:
        print(f"[WARNING] {split_key}: 没有解析到任何实体，请确认标签是否为 BIO/BMES，或是否只有 O/0 标签。")
    else:
        # 打印一个带实体的样例
        for eg in items:
            if eg["span_posLabel"]:
                print("[SAMPLE]", json.dumps(eg, ensure_ascii=False))
                break
    return out_fp

def main():
    base = find_msra_dir(DATASET_ROOT) if MSRA_DIR is None else Path(MSRA_DIR)
    print(f"[INFO] MSRA 目录：{base}")

    out_dir = base / "msra_span"
    results = []
    for k, fn in FILES.items():
        fp = convert_split(base, k, fn, out_dir)
        if fp:
            results.append(str(fp))

    if not results:
        print("[ERROR] 未生成任何文件。请检查文件是否存在/编码/命名。")
    else:
        print("[DONE] 生成：")
        for p in results:
            print("  -", p)

if __name__ == "__main__":
    main()
