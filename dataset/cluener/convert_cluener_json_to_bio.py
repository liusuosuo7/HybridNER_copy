# -*- coding: utf-8 -*-
"""
Convert CLUENER JSON (folder: ./cluener) to BIO format.
- 支持两种输入：JSON 数组 / JSON Lines（逐行 JSON）
- 输入:  ./cluener/{train.json, dev.json, test.json}
- 输出:  ./cluener_bio/{train.bio, dev.bio, test.bio}
- BIO: 每行 "字符 标签"，句间空行
"""

import os, sys, json

IN_DIR = "./cluener"
OUT_DIR = "./cluener_bio"
os.makedirs(OUT_DIR, exist_ok=True)

def load_samples(path):
    """返回 list[dict]，兼容 JSON 数组 和 JSON Lines 两种格式"""
    # 先尝试按整个 JSON 解析
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass  # 回退到 JSON Lines

    # JSON Lines：逐行解析
    samples = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} 第{idx}行不是合法 JSON：{e}")
    return samples

def spans_from_label(label_dict, text):
    """
    label 例子:
    {"address":{"北京":[[0,1]]}, "organization":{"清华大学":[[10,13],[20,23]]}}
    返回 [(start, end_inclusive, TYPE), ...]
    """
    spans = []
    if not label_dict:
        return spans
    for etype, ent_map in label_dict.items():
        if not ent_map:
            continue
        for _ent_text, pos_list in ent_map.items():
            if not pos_list:
                continue
            for st, ed in pos_list:
                # CLUENER 的索引是闭区间 [st, ed]
                if not isinstance(st, int) or not isinstance(ed, int):
                    continue
                if st < 0 or ed < st or st >= len(text):
                    continue
                ed = min(ed, len(text) - 1)
                spans.append((st, ed, etype.upper()))
    # 去重并排序
    spans = sorted(set(spans), key=lambda x: (x[0], x[1], x[2]))
    return spans

def to_bio(text, spans):
    tags = ["O"] * len(text)
    for st, ed, etype in spans:
        if st >= len(text):
            continue
        ed = min(ed, len(text) - 1)
        tags[st] = f"B-{etype}"
        for i in range(st + 1, ed + 1):
            tags[i] = f"I-{etype}"
    lines = []
    for ch, tg in zip(text, tags):
        lines.append(f"{ch} {tg}")
    lines.append("")  # 样本间空行
    return "\n".join(lines)

def convert(split):
    in_path = os.path.join(IN_DIR, f"{split}.json")
    out_path = os.path.join(OUT_DIR, f"{split}.bio")
    if not os.path.exists(in_path):
        sys.exit(f"未找到 {in_path}")
    data = load_samples(in_path)
    with open(out_path, "w", encoding="utf-8") as w:
        for item in data:
            text = item.get("text", "")
            label = item.get("label", {})  # test 集无标签也可
            spans = spans_from_label(label, text)
            w.write(to_bio(text, spans))
    print(f"[OK] {split}: {len(data)} 样本 -> {out_path}")

def main():
    for sp in ["train", "dev", "test"]:
        convert(sp)
    print("全部完成，输出目录：", os.path.abspath(OUT_DIR))

if __name__ == "__main__":
    main()
