"""chat template 적용과 손실 마스킹.

2일차 7교시(§3.9-3, §3.9-4) 슬라이드의 실물이다. 이 파일이 이 저장소에서
가장 중요하다 — 모델과 라이브러리는 바뀌지만 여기서 배우는 것은 남는다.

핵심 두 가지
  1) 모델마다 특수 토큰이 다르다. 문자열을 직접 조립하지 말고
     tokenizer.apply_chat_template()을 쓴다.
  2) 프롬프트 토큰까지 손실에 넣으면 모델이 '질문을 따라 쓰는 법'을 배운다.
     프롬프트 구간의 라벨을 -100으로 덮는다(= 손실에서 제외).

`render_mask()`로 마스킹 결과를 색으로 출력하는 것이 학습 활동의 핵심이다.
숫자 배열로는 아무도 확인하지 않는다.
"""
from __future__ import annotations

IGNORE_INDEX = -100

# ANSI 색상 (Colab·VS Code 터미널 출력에서 동작)
_DIM = "\033[2;37m"       # 회색  = 마스킹됨(손실 제외)
_HL = "\033[1;97;44m"     # 파랑  = 학습됨(손실 포함)
_END = "\033[0m"


def build_messages(instruction: str, system: str | None = None,
                   context: str | None = None) -> list[dict]:
    """지시-응답 한 건을 messages 형식으로. context는 지시문 앞에 붙인다."""
    msgs: list[dict] = []
    if system:
        msgs.append({"role": "system", "content": system})
    user = f"{context.strip()}\n\n{instruction.strip()}" if context else instruction.strip()
    msgs.append({"role": "user", "content": user})
    return msgs


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """추론용 프롬프트 문자열. 생성 시작 토큰까지 붙는다.

    학습(아래 build_example)과 추론(여기)에서 **같은 템플릿**을 써야 한다.
    다르면 학습 효과가 사라진다 — 가장 흔한 실패 원인이다.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_example(tokenizer, messages: list[dict], response: str,
                  max_len: int = 1024, mask_prompt: bool = True) -> dict:
    """학습 1건을 input_ids / labels / attention_mask로 만든다.

    mask_prompt=False로 두면 프롬프트까지 학습된다. 이 옵션은 실습에서
    '마스킹 유/무 비교'(§3.9-4)를 위해 일부러 남겨 둔 것이다. 실무에서는 True.
    """
    prompt = render_prompt(tokenizer, messages)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    eos = tokenizer.eos_token or ""
    answer_ids = tokenizer(response.strip() + eos, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + answer_ids
    if mask_prompt:
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)
    else:
        labels = list(input_ids)

    truncated = len(input_ids) > max_len
    if truncated:
        # 뒤를 자르면 정답과 EOS가 사라진다. 잘린 건은 통계로 남기고
        # 실제 학습에서는 제외하는 것이 안전하다(410에서 필터링).
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "n_prompt": len(prompt_ids),
        "n_answer": len(answer_ids),
        "n_total": len(prompt_ids) + len(answer_ids),
        "truncated": truncated,
    }


def render_mask(tokenizer, example: dict, max_tokens: int = 400) -> str:
    """마스킹 결과를 색으로 렌더링한다.

    회색 = 손실에서 제외된 구간(프롬프트) / 파랑 = 학습되는 구간(정답+EOS)

    확인할 것 세 가지
      1) 특수 토큰(<|im_start|> 등)이 회색 구간에 정상적으로 들어갔는가
      2) 정답만 파랑인가
      3) 파랑 구간 **끝에 EOS가 있는가** — 없으면 모델이 멈추는 법을 못 배운다
    """
    ids, labels = example["input_ids"], example["labels"]
    out, cur, cur_masked = [], [], None
    for i, (tid, lab) in enumerate(zip(ids, labels)):
        if i >= max_tokens:
            out.append(f"{_END}\n... (이하 {len(ids) - max_tokens} 토큰 생략)")
            break
        masked = lab == IGNORE_INDEX
        if masked != cur_masked and cur:
            out.append((_DIM if cur_masked else _HL) + tokenizer.decode(cur) + _END)
            cur = []
        cur_masked = masked
        cur.append(tid)
    if cur:
        out.append((_DIM if cur_masked else _HL) + tokenizer.decode(cur) + _END)
    header = (f"{_DIM}■{_END} 회색 = 손실 제외(프롬프트, {example['n_prompt']}토큰)   "
              f"{_HL}■{_END} 파랑 = 학습됨(정답+EOS, {example['n_answer']}토큰)\n")
    return header + "".join(out)


def check_example(tokenizer, example: dict) -> dict:
    """마스킹이 제대로 되었는지 자동 점검. 실습에서 assert로 쓴다."""
    labels = example["labels"]
    trainable = [l for l in labels if l != IGNORE_INDEX]
    eos_id = tokenizer.eos_token_id
    problems = []
    if not trainable:
        problems.append("학습되는 토큰이 0개입니다 — 전부 마스킹되었습니다(loss가 NaN이 됩니다).")
    if eos_id is not None and (not trainable or trainable[-1] != eos_id):
        problems.append("학습 구간이 EOS로 끝나지 않습니다 — 모델이 응답을 멈추지 못합니다.")
    if example["truncated"]:
        problems.append(f"max_len을 초과해 잘렸습니다(총 {example['n_total']}토큰) — 정답이 손실되었을 수 있습니다.")
    if len(trainable) / max(len(labels), 1) > 0.95:
        problems.append("거의 전부가 학습 구간입니다 — 프롬프트 마스킹이 적용되지 않은 것 같습니다.")
    return {"ok": not problems, "problems": problems,
            "trainable_tokens": len(trainable), "total_tokens": len(labels)}


def pad_batch(batch: list[dict], pad_token_id: int) -> dict:
    """가변 길이 배치를 오른쪽 패딩. labels의 패딩은 -100이어야 한다."""
    import torch

    n = max(len(b["input_ids"]) for b in batch)
    ids, labs, att = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_token_id] * k)
        labs.append(b["labels"] + [IGNORE_INDEX] * k)
        att.append(b["attention_mask"] + [0] * k)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labs),
        "attention_mask": torch.tensor(att),
    }
