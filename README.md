# tiny-llm-lab

Building an LLM from scratch, step by step — hobby project (tokenizer → GPT → Llama-style upgrades → Korean)

처음부터 직접 만들면서 LLM이 어떻게 동작하는지 이해하는 개인 학습 레포. 천천히, 단계별로 쌓아간다.

## Roadmap

| # | 폴더 | 내용 | 상태 |
|---|------|------|------|
| 1 | `01-char-gpt/` | 문자 단위 미니 GPT (셰익스피어) — 이론은 [THEORY.md](01-char-gpt/THEORY.md) | ✅ |
| 2 | `02-bpe-tokenizer/` | BPE 토크나이저 직접 구현 | ⬜ |
| 3 | `03-gpt-pipeline/` | 사전학습 → 파인튜닝 → 명령어 튜닝 한 사이클 | ⬜ |
| 4 | `04-llama-style/` | RoPE, RMSNorm, SwiGLU, GQA, KV 캐시 하나씩 적용 + 비교 | ⬜ |
| 5 | `05-korean-lm/` | 한국어 코퍼스로 학습, 한국어 토크나이징 비교 | ⬜ |
| 6 | `06-advanced/` | mixed precision, FlashAttention, DPO, 양자화, 서빙 | ⬜ |

각 폴더의 `PRACTICE.md`에 해당 단계의 실습 가이드(할 일, 확인할 것, 셀프 체크, 완료 기준)가 있다.
단계를 끝내면 `SUMMARY.md`에 그날 배운 핵심 이론을 짧게 정리한다. ([Step 1 요약](01-char-gpt/SUMMARY.md))

## References

- Andrej Karpathy — [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY), [nanoGPT](https://github.com/karpathy/nanoGPT)
- Sebastian Raschka — *Build a Large Language Model (From Scratch)*

## Setup

기본 실습 환경은 **Google Colab (T4 GPU)**. 단계별 세팅 방법은 [01-char-gpt/PRACTICE.md](01-char-gpt/PRACTICE.md)의 "0. 환경 세팅" 참고.

```python
# Colab 셀
!git clone https://github.com/alpaca07/tiny-llm-lab.git
%cd tiny-llm-lab/01-char-gpt
```

로컬에서 돌릴 때:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Notes

- 모델 체크포인트(`*.pt`)와 `data/`는 용량 문제로 git에 올리지 않는다.
