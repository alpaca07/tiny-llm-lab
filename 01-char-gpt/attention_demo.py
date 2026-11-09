"""어텐션의 핵심 아이디어를 작은 텐서로 확인하는 데모.

"각 위치가 자기 자신과 과거 토큰들의 (가중) 평균을 구한다"를 세 가지 방법으로 계산해서
결과가 같다는 것을 확인하고, 마지막에 셀프 어텐션이 그 가중치를 '데이터에서' 만든다는 걸 본다.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(1337)
B, T, C = 1, 5, 4
x = torch.randn(B, T, C)

# 버전 1: for 루프 (가장 직관적, 느림)
out1 = torch.zeros(B, T, C)
for t in range(T):
    out1[:, t] = x[:, : t + 1].mean(dim=1)

# 버전 2: 하삼각 행렬곱 (행 합 = 1로 정규화)
tril = torch.tril(torch.ones(T, T))
w = tril / tril.sum(dim=1, keepdim=True)
out2 = w @ x

# 버전 3: -inf 마스킹 + softmax (어텐션이 실제로 쓰는 방식)
mask = torch.zeros(T, T).masked_fill(tril == 0, float("-inf"))
w3 = F.softmax(mask, dim=-1)  # 점수가 전부 0이면 균등 가중치
out3 = w3 @ x

print("v1 == v2:", torch.allclose(out1, out2, atol=1e-6))
print("v2 == v3:", torch.allclose(out2, out3, atol=1e-6))
print("\n균등 가중치 행렬 (행 = 현재 위치, 열 = 참조 위치):")
print(w3)

# 셀프 어텐션: 점수(0)를 상수 대신 q·k 로 바꾸면 가중치가 '내용에 따라' 달라진다
hs = 4
q = torch.nn.Linear(C, hs, bias=False)(x)
k = torch.nn.Linear(C, hs, bias=False)(x)
scores = q @ k.transpose(-2, -1) * hs**-0.5
att = F.softmax(scores.masked_fill(tril == 0, float("-inf")), dim=-1)
print("\n셀프 어텐션 가중치 (내용 기반, 여전히 미래는 0 / 행 합 = 1):")
print(att[0].detach())
