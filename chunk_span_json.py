#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path("/root/autodl-tmp/HybridNER/dataset/msra")
IN_FILES = ["msra_train_span.json", "msra_dev_span.json", "msra_test_span.json"]
OUT_SUFFIX = "_chunked.json"

MAX_CHARS = 510     # 留出 [CLS]/[SEP]
STRIDE = 128        # 滑窗步长，避免信息丢失

def split_sample(ctx: str, spans: Dict[str, str], max_chars=510, stride=128):
    """
    将一个样本切分为若干片段（≤max_chars），并且把在窗口内完整落入的实体
    映射到新片段上（起止坐标做平移）。跨越窗口边界的实体会被丢弃（常规做法）。
    """
    n = len(ctx)
    start = 0
    out = []
    while start < n:
        end = min(start + max_chars, n)
        sub_ctx = ctx[start:end]
        sub_spans = {}
        for k, v in spans.items():
            s, e = map(int, k.split(";"))
            # 仅保留完全落在窗口内的实体
            if s >= start and e < end:
                sub_spans[f"{s - start};{e - start}"] = v
        out.append({"context": sub_ctx, "span_posLabel": sub_spans})
        if end == n:
            break
        # 滑窗前进，保留重叠
        start = max(0, end - stride)
    return out

def process_file(in_path: Path):
    data = json.loads(in_path.read_text(encoding="utf-8"))
    new_data: List[Dict] = []
    for eg in data:
        ctx = eg["context"]
        spans = eg.get("span_posLabel", {})
        chunks = split_sample(ctx, spans, MAX_CHARS, STRIDE)
        new_data.extend(chunks)
    out_path = in_path.with_name(in_path.stem + OUT_SUFFIX)
    out_path.write_text(json.dumps(new_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {in_path.name}  ->  {out_path.name} | 原样本 {len(data)} -> 新样本 {len(new_data)}")
    return out_path

def main():
    print(f"[INFO] 数据目录: {DATA_DIR}")
    for name in IN_FILES:
        p = DATA_DIR / name
        if not p.exists():
            print(f"[WARN] 跳过，未找到 {p}")
            continue
        process_file(p)

if __name__ == "__main__":
    main()
