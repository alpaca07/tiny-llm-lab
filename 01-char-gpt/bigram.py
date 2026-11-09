"""Step 1-1: 문자 단위 바이그램 언어모델.

흐름: 데이터 다운로드 -> 문자 토크나이징 -> 배치 만들기 -> 모델 -> 학습 -> 생성
바이그램 = "직전 문자 하나만 보고 다음 문자를 예측". 가장 단순한 언어모델이고,
이후 단계(어텐션 추가)의 출발점이 된다.
"""
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 하이퍼파라미터 ----------------
batch_size = 32      # 한 번에 병렬로 처리할 시퀀스 수
block_size = 8       # 컨텍스트 길이 (바이그램은 사실상 1글자만 사용)
max_iters = 3000
eval_interval = 300
eval_iters = 200
lr = 1e-2
seed = 1337
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(seed)

# ---------------- 데이터 ----------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "input.txt")

if not os.path.exists(DATA_PATH):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()

# ---------------- 문자 토크나이저 ----------------
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s: str) -> list[int]:
    return [stoi[c] for c in s]


def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split: str):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])  # 한 칸 밀린 정답
    return x.to(device), y.to(device)


# ---------------- 모델 ----------------
class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # 각 문자 -> "다음 문자 분포(logits)"를 바로 꺼내는 룩업 테이블
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # (B, T, V)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # 마지막 위치만 사용
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    model = BigramLM(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"device={device} vocab_size={vocab_size} chars={len(text):,}")
    print(f"학습 전 기대 loss ≈ ln({vocab_size}) = {torch.log(torch.tensor(float(vocab_size))):.3f}")

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            l = estimate_loss(model)
            print(f"step {it:5d} | train {l['train']:.4f} | val {l['val']:.4f}")
        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)  # 줄바꿈 문자로 시작
    print("\n--- 생성 결과 ---")
    print(decode(model.generate(ctx, 300)[0].tolist()))
