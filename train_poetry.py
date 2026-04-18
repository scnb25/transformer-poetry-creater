import os
import json
import re
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 设备
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ======================
# 数据加载
# ======================
def clean_text(s):
    return re.sub(r"[^\u4e00-\u9fa5]", "", s)

def load_poetry(root_dir):
    data = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        poems = json.load(f)
                        if not isinstance(poems, list):
                            continue
                        for poem in poems:
                            if "paragraphs" not in poem:
                                continue
                            text = "".join(poem["paragraphs"])
                            text = clean_text(text)
                            if len(text) > 10:
                                data.append(text)
                except:
                    continue
    return data

data_path = r"D:\transformer\dataset\chinese-poetry-master"
sentences = load_poetry(data_path)

print("数据量:", len(sentences))

# 🔥 控制数据量（防过拟合）
sentences = sentences[:4000]

# ======================
# 构建词表
# ======================
chars = "".join(sentences)
counter = Counter(chars)

vocab = {c:i+2 for i,(c,_) in enumerate(counter.items())}
vocab["<pad>"] = 0
vocab["<unk>"] = 1

inv_vocab = {i:c for c,i in vocab.items()}
vocab_size = len(vocab)

def encode(s):
    return [vocab.get(c,1) for c in s]

data = [encode(s) for s in sentences]

# ======================
# Dataset
# ======================
MAX_LEN = 50

def pad(seq):
    seq = seq[:MAX_LEN]
    return seq + [0]*(MAX_LEN - len(seq))

inputs = []
targets = []

for seq in data:
    if len(seq) < 5:
        continue
    seq = pad(seq)
    inputs.append(seq[:-1])
    targets.append(seq[1:])

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

dataset = torch.utils.data.TensorDataset(inputs, targets)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# ======================
# 位置编码
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** (i/d_model)))

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ======================
# Transformer模型（优化版）
# ======================
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 128)
        self.pos = PositionalEncoding(128)
        self.dropout = nn.Dropout(0.3)

        self.transformer = nn.Transformer(
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos(x)
        x = self.dropout(x)

        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)

        out = self.transformer(x, x, tgt_mask=mask)

        return self.fc(out)

model = TransformerModel().to(device)

# ======================
# 训练
# ======================
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 18

for epoch in range(EPOCHS):
    total_loss = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        out = out.reshape(-1, vocab_size)
        y = y.reshape(-1)

        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    ppl = math.exp(avg_loss)

    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")

    # 🔥 早停（关键）
    if avg_loss < 2.5:
        print("提前停止，防止过拟合")
        break

# ======================
# 文本生成（优化版）
# ======================
def generate(start_char, max_len=20):
    model.eval()
    result = [vocab.get(start_char,1)]

    while len(result) < max_len:
        x = torch.tensor(result).unsqueeze(0).to(device)

        out = model(x)
        logits = out[0, -1]

        logits = logits / 1.1

        k = 30
        topk = torch.topk(logits, k)

        indices = topk.indices
        probs = F.softmax(topk.values, dim=-1)

        next_char = indices[torch.multinomial(probs, 1)].item()

        if next_char in result[-3:]:
            continue

        result.append(next_char)

    text = "".join([inv_vocab.get(i,"") for i in result])

    # 👉 20字 → 4句5言诗
    formatted = ""
    for i, c in enumerate(text):
        formatted += c
        if (i+1) % 5 == 0:
            if (i+1) % 10 == 0:
                formatted += "。\n"
            else:
                formatted += "，"

    return formatted

print("\n生成示例：")
print(generate("春"))
print(generate("山"))
print(generate("夜"))

# ======================
# 简单可视化
# ======================
sample = inputs[0][:20].unsqueeze(0).to(device)

plt.figure(figsize=(6,5))
sns.heatmap(sample.cpu(), cmap="viridis")
plt.title("Input Visualization")
plt.show()