"""평가 지표.

3일차 4교시(§3.12-1~5) 슬라이드와 1:1로 대응한다.

  - normalize / exact_match / char_f1 : KorQuAD 방식(한국어는 문자 단위 F1)
  - format_compliance                 : 형식 준수율 — 소형 모델에서 가장 잘 오르는 지표
  - bootstrap_ci                      : 표본 200건의 한계를 신뢰구간으로 표현
  - confusion / macro_f1              : HWP 1일차 7번 미충족 항목 해소

주의: 지표 함수를 노트북마다 다시 쓰지 말 것. 조건별로 다른 코드로 채점하면
비교가 성립하지 않는다(§3.12-1 '평가의 최소 요건').
"""
from __future__ import annotations

import json
import random
import re
import string
import unicodedata
from collections import Counter

_PUNCT = set(string.punctuation) | set("·…“”‘’「」『』〈〉《》―–—")


# ------------------------------------------------------------------ 정규화

def normalize(text: str) -> str:
    """KorQuAD 채점 관례에 맞춘 정규화: 유니코드 정규화 → 공백/구두점 제거."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = re.sub(r"\s+", "", text)
    return text


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def char_f1(pred: str, gold: str) -> float:
    """한국어는 어절 단위가 불안정하므로 문자 단위 F1을 쓴다(KorQuAD 공식 방식)."""
    p, g = normalize(pred), normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def score_qa(preds: list[str], golds: list[str]) -> dict:
    """정답이 여러 개일 수 있으면 golds의 원소를 '||'로 구분해 넣는다."""
    assert len(preds) == len(golds), "예측 수와 정답 수가 다릅니다"
    ems, f1s = [], []
    for p, g in zip(preds, golds):
        cands = [c for c in str(g).split("||") if c.strip()] or [""]
        ems.append(max(exact_match(p, c) for c in cands))
        f1s.append(max(char_f1(p, c) for c in cands))
    return {
        "n": len(preds),
        "em": round(100 * sum(ems) / len(ems), 2),
        "f1": round(100 * sum(f1s) / len(f1s), 2),
        "em_raw": ems,
        "f1_raw": f1s,
    }


# --------------------------------------------------------- 형식 준수율 (§3.12-4)

def parse_json_lenient(text: str) -> dict | None:
    """모델 출력에서 첫 JSON 객체를 꺼낸다. 실패하면 None.

    엄격 파싱만 하면 '```json' 펜스나 앞말 때문에 전부 실패로 잡혀
    형식 준수율이 0에 붙는다. 관대 파싱과 엄격 파싱을 따로 재는 것이
    이 지표의 핵심이다.
    """
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def format_compliance(outputs: list[str], required_fields: list[str],
                      max_chars: int | None = None) -> dict:
    """형식 준수율 4항목.

    strict_json : 앞뒤 군더더기 없이 그대로 json.loads 되는 비율
    lenient_json: 펜스/앞말을 걷어내면 파싱되는 비율
    fields      : 필수 필드를 모두 가진 비율
    clean_stop  : 지정 길이를 넘기지 않고 끝난 비율 (EOS 학습 여부의 대리 지표)
    """
    n = max(len(outputs), 1)
    strict = lenient = fields = clean = 0
    for o in outputs:
        o = o or ""
        try:
            obj = json.loads(o.strip())
            if isinstance(obj, dict):
                strict += 1
        except Exception:
            pass
        obj = parse_json_lenient(o)
        if obj is not None:
            lenient += 1
            if all(k in obj for k in required_fields):
                fields += 1
        if max_chars is None or len(o) <= max_chars:
            clean += 1
    return {
        "n": len(outputs),
        "strict_json_pct": round(100 * strict / n, 2),
        "lenient_json_pct": round(100 * lenient / n, 2),
        "required_fields_pct": round(100 * fields / n, 2),
        "clean_stop_pct": round(100 * clean / n, 2),
    }


# --------------------------------------------------------- 신뢰구간 (§3.12-5)

def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42, scale: float = 100.0) -> dict:
    """표본 평균의 부트스트랩 신뢰구간.

    '78% vs 81%'가 유의한 차이가 아닐 수 있다는 것을 보이는 데 쓴다.
    """
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "half_width": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    mean = sum(values) / n
    return {
        "mean": round(scale * mean, 2),
        "lo": round(scale * lo, 2),
        "hi": round(scale * hi, 2),
        "half_width": round(scale * (hi - lo) / 2, 2),
        "n": n,
    }


def ci_text(ci: dict, unit: str = "") -> str:
    return f"{ci['mean']}{unit} (95% CI {ci['lo']}~{ci['hi']}, n={ci['n']})"


def diff_ci(a: list[float], b: list[float], n_boot: int = 2000,
            alpha: float = 0.05, seed: int = 42, scale: float = 100.0) -> dict:
    """짝지어진 두 조건의 차이(b-a)에 대한 신뢰구간.

    같은 문항을 두 조건으로 채점했으므로 짝지어 비교해야 검정력이 높다.
    구간이 0을 포함하면 '개선을 확인하지 못했다'가 정확한 결론이다.
    """
    assert len(a) == len(b), "두 조건의 문항 수가 다릅니다 — 동일 표본이 아닙니다"
    d = [y - x for x, y in zip(a, b)]
    ci = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed, scale=scale)
    ci["significant"] = bool(ci["lo"] > 0 or ci["hi"] < 0)
    return ci


# ----------------------------------------------------- 분류 지표 (HWP 미충족 해소)

def confusion(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred), key=str)
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return {"labels": [str(l) for l in labels], "matrix": m}


def macro_f1(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    c = confusion(y_true, y_pred, labels)
    m, labs = c["matrix"], c["labels"]
    per = {}
    for i, l in enumerate(labs):
        tp = m[i][i]
        fp = sum(m[r][i] for r in range(len(labs))) - tp
        fn = sum(m[i]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[l] = {"precision": round(100 * prec, 2), "recall": round(100 * rec, 2),
                  "f1": round(100 * f1, 2), "support": sum(m[i])}
    total = sum(sum(r) for r in m)
    acc = sum(m[i][i] for i in range(len(labs))) / total if total else 0.0
    return {
        "accuracy": round(100 * acc, 2),
        "macro_f1": round(sum(v["f1"] for v in per.values()) / max(len(per), 1), 2),
        "per_class": per,
        "confusion": c,
    }


def print_confusion(c: dict) -> None:
    labs, m = c["labels"], c["matrix"]
    w = max(6, max(len(l) for l in labs) + 1)
    print("실제\\예측".ljust(w), "".join(l.rjust(w) for l in labs))
    for i, l in enumerate(labs):
        print(l.ljust(w), "".join(str(v).rjust(w) for v in m[i]))
