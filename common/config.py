"""전 노트북 공통 설정.

**벤더·모델명을 노트북 본문에 하드코딩하지 않는다.** 개강일에 모델이 바뀌면
이 파일 한 곳만 고친다(전 과정 공통 원칙: 벤더 종속 회피).

강사 확인 사항
  - MODEL_ID: 개강 2주 전에 접근 가능 여부와 라이선스를 재확인할 것
  - TOKENIZER_ZOO: 5종 사전 다운로드하여 오프라인 캐시 구성(§7-1)
"""
from __future__ import annotations

# ------------------------------------------------------------------ 학습 대상
# T4(16GB) 기준. 0.5B가 기본, 1.5B는 max_seq_len을 줄이면 가능, 3B는 선택 확장.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

MODEL_CANDIDATES = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    # 3B 이상은 T4에서 max_seq_len 512 + batch 1 + grad checkpointing 필요.
    # 강사가 §7-2 실측 후 아래 주석을 해제할 것.
    # "3b": "Qwen/Qwen2.5-3B-Instruct",
}

# ------------------------------------------------- 토크나이저 비교용 (010·020)
# 5종의 한국어 fertility를 비교한다. 접근이 막힌 모델은 제거해도 실습이 성립한다.
TOKENIZER_ZOO = {
    "tiktoken:cl100k_base": "cl100k_base",          # tiktoken (BPE)
    "Qwen2.5": "Qwen/Qwen2.5-0.5B-Instruct",        # BPE
    "Llama-3.2": "meta-llama/Llama-3.2-1B-Instruct",  # BPE (gated — 토큰 필요)
    "Gemma-2": "google/gemma-2-2b-it",              # SentencePiece Unigram (gated)
    "KLUE-RoBERTa": "klue/roberta-base",            # WordPiece (한국어 특화)
}

# ------------------------------------------------------------------ 데이터
DATASET_ID = "KorQuAD/squad_kor_v1"
N_TRAIN = 2000          # T4 15~25분 기준. 시간이 부족하면 1000으로 줄인다.
N_EVAL = 200            # §3.12-5: 200건이므로 신뢰구간이 필요하다
N_DOMAIN = 30           # 수강생이 직접 작성 (§3.9 실습)

# ------------------------------------------------- SFT 태스크 정의 (구조화 출력)
# 왜 JSON 출력인가: 0.5B급 소형 모델에서 EM/F1은 잘 안 오르지만
# **형식 준수율은 뚜렷하게 오른다**(§3.5-4, §3.12-4). 파인튜닝의 효과를
# 정직하게 관측할 수 있는 지표를 태스크 설계에 미리 심어 둔 것이다.
SYSTEM_PROMPT = (
    "당신은 주어진 지문에서만 근거를 찾아 답하는 한국어 질의응답 도우미입니다. "
    'answer와 evidence 두 개의 키를 가진 JSON 객체만 출력하십시오. '
    "설명이나 인사말을 덧붙이지 마십시오."
)
REQUIRED_FIELDS = ["answer", "evidence"]
MAX_OUTPUT_CHARS = 600   # clean_stop 판정 기준

# ------------------------------------------------------------------ 학습 설정
MAX_SEQ_LEN = 768        # 020의 p95 결과로 덮어쓴다
LORA = dict(r=16, alpha=32, dropout=0.05, target_modules="all-linear")
TRAIN = dict(
    epochs=2,
    per_device_batch_size=1,
    grad_accum=8,          # 유효 배치 = 1 x 8 = 8
    lr=2e-4,               # LoRA는 전체 파인튜닝보다 큰 LR을 쓴다 (§3.10-3)
    warmup_ratio=0.03,
    lr_scheduler="cosine",
    max_grad_norm=1.0,     # fp16 발산 방지 (§3.11-3)
    logging_steps=10,
    eval_steps=50,
    save_steps=100,        # Colab 세션 끊김 대비
    gradient_checkpointing=True,
)

# ------------------------------------------------------------------ 생성 설정
# 평가는 항상 이 설정으로 한다. 조건마다 다르게 하면 비교가 아니다 (§3.12-1).
GEN_EVAL = dict(max_new_tokens=160, do_sample=False, temperature=None, top_p=None)
GEN_DEMO = dict(max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9)

SEED = 42


def summary() -> str:
    return (
        f"모델        : {MODEL_ID}\n"
        f"데이터      : {DATASET_ID}  (train {N_TRAIN} / eval {N_EVAL} / domain {N_DOMAIN})\n"
        f"max_seq_len : {MAX_SEQ_LEN}\n"
        f"LoRA        : r={LORA['r']} alpha={LORA['alpha']} target={LORA['target_modules']}\n"
        f"유효 배치   : {TRAIN['per_device_batch_size']} x {TRAIN['grad_accum']} "
        f"= {TRAIN['per_device_batch_size'] * TRAIN['grad_accum']}\n"
        f"LR / epochs : {TRAIN['lr']} / {TRAIN['epochs']}\n"
        f"seed        : {SEED}"
    )
