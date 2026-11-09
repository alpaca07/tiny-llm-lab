"""Step 1-2: 문자 단위 미니 GPT (decoder-only Transformer).

bigram.py에서 달라진 점:
  1) 위치 임베딩 추가 (어텐션은 순서를 모르기 때문)
  2) 셀프 어텐션 (Head -> MultiHeadAttention)
  3) FeedForward + 잔차 연결 + LayerNorm => Block
  4) Block을 n_layer번 쌓기
이론 설명은 THEORY.md 참고.

실행 예: python gpt.py --max_iters 3000
"""
import argparse
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 하이퍼파라미터 ----------------
p = argparse.ArgumentParser()
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--block_size", type=int, default=64)   # 컨텍스트 길이
p.add_argument("--n_embd", type=int, default=64)       # 임베딩 차원 (d_model)
p.add_argument("--n_head", type=int, default=4)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--max_iters", type=int, default=3000)
p.add_argument("--eval_interval", type=int, default=300)
p.add_argument("--eval_iters", type=int, default=100)
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)

# ---------------- 데이터 (bigram.py와 동일) ----------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "input.txt")
if not os.path.exists(DATA_PATH):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
with open(DATA_PATH, encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
    x = torch.stack([d[i : i + args.block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + args.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------- 모델 ----------------
class Head(nn.Module):
    """셀프 어텐션 헤드 1개: softmax(Q K^T / sqrt(d)) V, 미래는 마스킹."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(args.n_embd, head_size, bias=False)
        self.query = nn.Linear(args.n_embd, head_size, bias=False)
        self.value = nn.Linear(args.n_embd, head_size, bias=False)
        # 하삼각 행렬: 자기 자신과 과거만 볼 수 있게 하는 마스크 (학습 대상 아님)
        self.register_buffer("tril", torch.tril(torch.ones(args.block_size, args.block_size)))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, hs)  "내가 가진 정보의 라벨"
        q = self.query(x)  # (B, T, hs)  "내가 찾는 정보"
        # 유사도 점수: (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # 미래 차단
        wei = F.softmax(wei, dim=-1)  # 각 행의 합 = 1 (과거 토큰에 대한 가중치)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs)  "내가 실제로 전달할 내용"
        return wei @ v     # (B, T, hs)


class MultiHeadAttention(nn.Module):
    """헤드 여러 개를 병렬로 돌리고 이어붙인 뒤 다시 섞어준다 (projection)."""

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, args.n_embd)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """토큰마다 독립적으로 적용되는 2층 MLP (4배로 넓혔다 줄이기)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(args.n_embd, 4 * args.n_embd),
            nn.ReLU(),
            nn.Linear(4 * args.n_embd, args.n_embd),
            nn.Dropout(args.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """통신(어텐션) -> 계산(FFN). 각각 pre-LayerNorm + 잔차 연결."""

    def __init__(self):
        super().__init__()
        head_size = args.n_embd // args.n_head
        self.sa = MultiHeadAttention(args.n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # 잔차 연결
        x = x + self.ffwd(self.ln2(x))  # 잔차 연결
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, args.n_embd)      # 토큰 의미
        self.pos_emb = nn.Embedding(args.block_size, args.n_embd)  # 위치 정보
        self.blocks = nn.Sequential(*[Block() for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)
        self.lm_head = nn.Linear(args.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))  # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))  # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -args.block_size :]  # 컨텍스트 길이를 넘기면 최근 것만
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split)
            losses[k] = model(x, y)[1].item()
        out[split] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    model = GPT().to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0:
            l = estimate_loss(model)
            print(f"step {it:5d} | train {l['train']:.4f} | val {l['val']:.4f}")
        x, y = get_batch("train")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- 생성 결과 ---")
    print(decode(model.generate(ctx, 500)[0].tolist()))
