#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

DATASET_ROOT = "/root/autodl-tmp/HybridNER/dataset"

# 可能的 MSRA 目录名（大小写/变体都兜底）
MSRA_DIR_CANDIDATES = ["msra", "MSRA", "Msra"]

# 各 split 可能的文件名（尽量鲁棒，BIO/BMES 都可）
CANDIDATES = {
    "train": [
        "train.txt", "train.bio", "train.char", "train.char.bmes",
        "msra_train_bio.txt", "msra_train.txt", "train.bmes", "train"
    ],
    "dev": [
        "dev.txt", "valid.txt", "dev.bio", "msra_dev_bio.txt",
        "msra_dev.txt", "validation.txt", "dev.bmes", "valid.bmes", "dev"
    ],
    "test": [
        "test.txt", "test.bio", "test.char.bmes",
        "msra_test_bio.txt", "msra_test.txt", "test.bmes", "test"
    ],
}

def find_msra_dir(root: str) -> Path:
    root_p = Path(root)
    # 先直接命中 /dataset/msra
    for d in MSRA_DIR_CANDIDATES:
        p = root_p / d
        if p.exists() and p.is_dir():
            return p
    # 其次在 dataset 下递归搜索包含“msra”的目录
    for p in root_p.rglob("*"):
        if p.is_dir() and "msra" in p.name.lower():
            return p
    # 否则就返回根，让后续报友好提示
    return root_p

def first_existing(base_dir: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = base_dir / n
        if p.exists() and p.is_file():
            return p
    return None

def read_bio_like(fp: Path) -> List[Tuple[List[str], List[str]]]:
    """
    读取 BIO/BMES（逐行：token [其它列…] tag；句间空行分隔）
    返回：[(tokens, tags), ...]
    """
    sents = []
    tokens, tags = [], []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    sents.append((tokens, tags))
                    tokens, tags = [], []
                continue
            # 兼容多列，最后一列当作标签，第一列当作 token
            parts = line.split()
            if len(parts) == 1:
                # 只有 token（当作 O）
                tok, tag = parts[0], "O"
            else:
                tok, tag = parts[0], parts[-1]
            tokens.append(tok)
            tags.append(tag)
    if tokens:
        sents.append((tokens, tags))
    return sents

def label_from_tag(tag: str) -> Optional[Tuple[str, str]]:
    """
    将 tag 标准化为 (scheme, label)
    支持 O / B-XXX / I-XXX / S-XXX / M-XXX / E-XXX
    """
    t = tag.strip()
    if t == "O":
        return None
    if "-" in t:
        prefix, lab = t.split("-", 1)
        return (prefix.upper(), lab)
    # 不规范标签，直接当 O
    return None

def tokens_to_char_offsets(tokens: List[str]) -> List[int]:
    """每个 token 在拼接后的字符串里的起始字符下标（不插空格）"""
    starts = []
    cur = 0
    for tok in tokens:
        starts.append(cur)
        cur += len(tok)
    return starts

def bio_bmes_to_spans(tokens: List[str], tags: List[str]) -> Dict[str, str]:
    """
    将一句的 BIO/BMES 标签转为“start;end”闭区间跨度字典
    """
    starts = tokens_to_char_offsets(tokens)
    n = len(tokens)
    spans = {}

    i = 0
    while i < n:
        info = label_from_tag(tags[i])
        if info is None:
            i += 1
            continue
        prefix, lab = info
        lab = lab.upper()

        # 单字实体（S-）
        if prefix == "S":
            start = starts[i]
            end = start + len(tokens[i]) - 1
            spans[f"{start};{end}"] = lab
            i += 1
            continue

        # 以 B- 开头，向后吞 I-/M-，直到遇到非同类或 E-
        if prefix in ("B", "I", "M", "E"):  # 兼容不规范开头
            start = starts[i]
            end = start + len(tokens[i]) - 1
            j = i + 1
            while j < n:
                nxt = label_from_tag(tags[j])
                if nxt is None:
                    break
                p2, lab2 = nxt
                if lab2.upper() != lab:
                    break
                if p2 in ("I", "M"):
                    end = starts[j] + len(tokens[j]) - 1
                    j += 1
                    continue
                if p2 == "E":
                    end = starts[j] + len(tokens[j]) - 1
                    j += 1
                    break
                if p2 == "B" or p2 == "S":
                    break
            spans[f"{start};{end}"] = lab
            i = j
            continue

        # 其它前缀兜底跳过
        i += 1

    return spans

def convert_split(msra_dir: Path, split_key: str, out_dir: Path) -> Optional[Path]:
    src = first_existing(msra_dir, CANDIDATES[split_key])
    if not src:
        print(f"[WARN] 未找到 {split_key} 文件（在 {msra_dir} 下尝试 {CANDIDATES[split_key]}）")
        return None

    print(f"[INFO] 读取：{src}")
    data = read_bio_like(src)

    items = []
    for tokens, tags in data:
        text = "".join(tokens)
        span_map = bio_bmes_to_spans(tokens, tags)
        items.append({
            "context": text,
            "span_posLabel": span_map
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"msra_{split_key}_span.json"
    with out_fp.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"[OK] 已写出：{out_fp}（句子数：{len(items)}）")
    # 打个样本
    if items:
        eg = items[0]
        print("[SAMPLE]", json.dumps(eg, ensure_ascii=False))
    return out_fp

def main():
    msra_dir = find_msra_dir(DATASET_ROOT)
    print(f"[INFO] MSRA 目录：{msra_dir}")
    if not msra_dir.exists():
        raise FileNotFoundError(f"未找到数据目录：{msra_dir}")

    out_dir = msra_dir / "msra_span"
    results = []
    for k in ["train", "dev", "test"]:
        fp = convert_split(msra_dir, k, out_dir)
        if fp:
            results.append(str(fp))

    if not results:
        print("[ERROR] 未生成任何文件，请检查数据文件是否存在/命名是否匹配。")
    else:
        print("[DONE] 生成文件：")
        for p in results:
            print("  -", p)

if __name__ == "__main__":
    main()
