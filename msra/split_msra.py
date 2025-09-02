import random

# 读取训练集文件
with open("msra_train_bio.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 按句子切分（空行表示句子结束）
sentences = []
sentence = []
for line in lines:
    if line.strip():
        sentence.append(line)
    else:
        if sentence:
            sentences.append(sentence)
            sentence = []
if sentence:
    sentences.append(sentence)

print(f"总句子数: {len(sentences)}")

# 固定随机种子保证可复现
random.seed(42)
random.shuffle(sentences)

# 按 9:1 划分训练集和验证集
split_idx = int(len(sentences) * 0.9)
train_sents = sentences[:split_idx]
dev_sents = sentences[split_idx:]

# 保存新的训练集
with open("msra_train_bio.txt", "w", encoding="utf-8") as f:
    for s in train_sents:
        f.writelines(s)
        f.write("\n")

# 保存验证集
with open("msra_dev_bio.txt", "w", encoding="utf-8") as f:
    for s in dev_sents:
        f.writelines(s)
        f.write("\n")

print(f"训练集: {len(train_sents)}  验证集: {len(dev_sents)}")
