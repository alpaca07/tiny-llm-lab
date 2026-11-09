# Step 1 실습 가이드 — 문자 단위 GPT

이론은 [THEORY.md](THEORY.md). 순서대로 따라 하면 된다.

## 0. 환경 세팅 (Google Colab)

### 0-1. 노트북 만들기 + GPU 켜기
1. [colab.research.google.com](https://colab.research.google.com) 접속 → **새 노트북**
2. 상단 메뉴 **런타임 → 런타임 유형 변경 → 하드웨어 가속기: T4 GPU** → 저장
3. 첫 셀에서 GPU가 잡혔는지 확인 (`!`로 시작하면 셸 명령):

```python
!nvidia-smi
import torch
print(torch.__version__, torch.cuda.is_available())   # True 가 나와야 함
```

`False`면 런타임 유형이 GPU로 안 바뀐 것이다. 2번을 다시 확인.

### 0-2. 레포 가져오기
레포가 public이라 로그인 없이 clone된다.

```python
!git clone https://github.com/alpaca07/tiny-llm-lab.git
%cd tiny-llm-lab/01-char-gpt
```

- Colab에는 PyTorch, numpy, matplotlib이 **이미 설치**되어 있다. `pip install -r requirements.txt`는 하지 않는다 (torch를 다시 깔아 오래 걸리고 버전이 꼬일 수 있음).
- 데이터(tiny shakespeare, 약 1MB)는 첫 실행 때 `data/input.txt`로 자동 다운로드된다.

### 0-3. 실행

```python
!python bigram.py
!python attention_demo.py
!python gpt.py --max_iters 3000
```

- `gpt.py`는 `cuda`가 있으면 자동으로 GPU를 쓴다. 출력 첫 줄이 `device=cuda`인지 확인.

### 0-4. 코드 수정하고 다시 돌리기 (실습 4용)
- 왼쪽 **폴더 아이콘(파일 패널)** → `tiny-llm-lab/01-char-gpt/gpt.py` 더블클릭 → 에디터에서 수정 → 다시 `!python gpt.py ...`
- 실험 후 원래대로 되돌리기: `!git checkout gpt.py`
- 레포에 새 커밋이 올라왔을 때 최신으로 받기: `!git pull`

### 0-5. 알아둘 점
- **세션이 끊기면 Colab 안의 파일이 전부 사라진다.** 다시 0-2부터 하면 된다. (데이터는 자동으로 다시 받아진다)
- **오래 걸리는 학습**(Step 3 이후)의 체크포인트는 구글 드라이브에 저장해야 한다:

```python
from google.colab import drive
drive.mount('/content/drive')
# 저장 경로 예: /content/drive/MyDrive/tiny-llm-lab/ckpt.pt
```

- 코드 수정과 커밋은 **로컬(또는 Claude와 작업하는 쪽)에서 하고 Colab은 실행용**으로 쓰는 걸 추천한다. Colab에서 push하려면 GitHub 토큰이 필요해 번거롭다.
- 무료 GPU는 사용 시간 제한이 있고, 한동안 쓰지 않으면 연결이 끊긴다. 학습 셀을 돌려놓고 창을 오래 방치하지 말 것.
- GPU 사용량이 많아 연결이 거부되면 시간을 두고 다시 시도하거나 CPU 런타임으로도 600 step 정도는 돌릴 수 있다.

---

## 실습 1. 바이그램 돌려보기

```bash
python bigram.py
```

**확인할 것**
- 시작 loss ≈ 4.7 → 끝 loss ≈ 2.46~2.49 (3000 step)
- 생성 글이 철자 비슷한 조각(`Thr toud en:`)일 뿐 단어가 아닌 것

**스스로 답해보기**
1. 기준선 `ln 65 ≈ 4.17`보다 시작 loss가 높은 이유는? (THEORY 5장)
2. 3000 step을 더 늘려도 2.4 아래로 못 내려가는 이유는? (THEORY 6장)

---

## 실습 2. 어텐션 데모

```bash
python attention_demo.py
```

**확인할 것**
- `v1 == v2`, `v2 == v3` 둘 다 `True`
- 마지막 가중치 행렬: **각 행의 합이 1**, **오른쪽 위(미래)가 전부 0**
- 균등 평균 행렬과 달리 셀프 어텐션 행렬은 값이 제각각 (내용 기반)

**해보기**: 스크립트의 `torch.manual_seed(1337)` 값을 바꿔서 다시 돌려보고, 행 합 1·미래 0은 그대로인데 값만 바뀌는 걸 확인.

---

## 실습 3. 미니 GPT 학습

```bash
python gpt.py --max_iters 3000
```

- 파라미터 약 0.21M. Colab GPU면 CPU보다 훨씬 빠르다. (CPU로는 600 step이 2분 안팎이었다.)
- 먼저 빨리 확인하고 싶으면 `--max_iters 600`.

**확인할 것**
- val loss가 바이그램(≈2.49)보다 낮아진다 (600 step에서 ≈2.0)
- 생성 글에 `AUSLF LENDBENCE:` 같은 **대사 형식(이름: 대사)** 이 나타난다 → 문자 몇 개를 넘는 구조를 배웠다는 신호

---

## 실습 4. 변수 하나씩 바꿔보기 (ablation)

기준: `--max_iters 600 --eval_interval 600`, seed 1337 기준 val loss.
(아래 숫자는 참고용 — 기기·라이브러리 버전에 따라 조금 다를 수 있다.)

| 실험 | 방법 | 참고 val loss | 해석 |
|------|------|--------------|------|
| 기준 | `python gpt.py ...` | 2.01 | |
| 레이어 줄이기 | `--n_layer 1` | 2.12 | 깊이가 줄면 나빠진다 |
| 헤드 1개 | `--n_head 1` | 2.03 | 이 작은 모델에선 차이가 작다 |
| 컨텍스트 짧게 | `--block_size 8` | 2.21 | 볼 수 있는 과거가 줄면 나빠진다 |
| **마스크 제거** | `Head.forward`의 `masked_fill` 줄을 주석 처리 | **0.04** | 미래를 보고 정답을 베낌 (커닝) |
| √d 스케일 제거 | `* k.shape[-1] ** -0.5` 삭제 | 2.00 | 이 크기에선 차이 거의 없음 (아래 참고) |

**마스크 제거 결과 읽는 법**
- loss가 0.04로 비정상적으로 낮은 건 모델이 똑똑해진 게 아니라 **정답을 훔쳐본 것**.
- 이 상태로 글을 생성시켜 보면(생성 시엔 미래가 없다) 엉망이다. 직접 확인해볼 것.

**√d 스케일이 차이가 없었던 이유**
- 헤드 크기가 16이라 내적 분산이 ~16으로 작아서 softmax가 아직 포화되지 않는다.
- 헤드 크기를 키우면(`--n_embd 256 --n_head 1` → head_size 256) 스케일 유무 차이가 커진다. 직접 시도해보기 (CPU는 느리니 `--max_iters 300` 정도로).

---

## 실습 5. 샘플링 온도

`gpt.py` 맨 아래 `generate(ctx, 500)`를 바꿔서 비교한다.

```python
for t in (0.5, 1.0, 1.5):
    print(f"\n===== temperature {t} =====")
    print(decode(model.generate(ctx, 300, temperature=t)[0].tolist()))
```

- 0.5: 안전한 글자만 골라서 반복적·보수적
- 1.5: 다양하지만 철자가 무너짐

**왜?** logits를 온도로 나눈 뒤 softmax하니까, 온도가 낮으면 분포가 뾰족해지고 높으면 평평해진다.

---

## 셀프 체크 (설명할 수 있으면 통과)

1. 학습용 정답 `y`는 입력 `x`와 어떤 관계인가? 조각 하나에서 예측 문제가 몇 개 생기나?
2. Q, K, V는 각각 무슨 역할인가?
3. 마스크가 없으면 왜 안 되나?
4. 위치 임베딩이 없으면 어떤 문제가 생기나?
5. 어텐션과 FFN은 각각 "무엇을" 하나? (토큰 간 교환 vs 토큰별 계산)
6. 잔차 연결은 왜 필요한가?

## 완료 기준

- [x] 실습 1~3을 직접 실행했다
- [x] 실습 4 중 마스크 제거 실험을 해봤다
- [x] 셀프 체크 6개를 내 말로 설명할 수 있다

다음: [Step 2 — BPE 토크나이저](../02-bpe-tokenizer/PRACTICE.md)
