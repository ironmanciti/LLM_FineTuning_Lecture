"""노트북 사이의 데이터 계약.

이 저장소의 노트북은 서로를 직접 호출하지 않는다. 대신 `artifacts/`
아래 정해진 경로에 파일을 쓰고, 뒤 노트북이 그 파일을 읽는다.
경로를 바꾸면 뒤 노트북이 전부 깨지므로 여기만 고쳐서 쓴다.

    artifacts/
      env.json                          <- 300 (실행 환경 스냅샷)
      tokens/token_cost.json            <- 020 (fertility, p95, 권장 max_seq_len)
      data/train.jsonl                  <- 410
      data/eval.jsonl                   <- 410
      data/domain30.jsonl               <- 410 (수강생 직접 작성)
      data/data_report.json             <- 410 (품질 5항목 점검 결과)
      runs/<run_id>/config.json         <- 420 (재현용 설정 전량)
      runs/<run_id>/history.json        <- 420 (손실 곡선 원본)
      runs/<run_id>/loss_curve.png      <- 420
      runs/<run_id>/adapter/            <- 420 (LoRA 어댑터)
      eval/<run_id>/<cond>/metrics.json <- 430  cond = base | tuned | merged
      eval/<run_id>/<cond>/preds.jsonl  <- 430
      eval/<run_id>/rubric.json         <- 430 (자기 도메인 정성 채점)
      eval/<run_id>/regression.json     <- 430 (회귀 세트 20문항)
      merged/<run_id>/model_card.md     <- 440
      report/<run_id>/final_report.md   <- 450
      report/<run_id>/comparison.csv    <- 450
      reference/                        <- 강사 사전 실행 결과 (GPU 실패 시 대체 입력)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

# 강사 사전 실행 결과. GPU를 못 쓰는 수강생은 USE_REFERENCE=True로 두고
# 430·450만 돌려도 분석 활동에 참여할 수 있다.
REFERENCE = ART / "reference"


def _p(path: Path, mkdir: bool = True) -> Path:
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------ 경로 규약

def env_path() -> Path:
    return _p(ART / "env.json")


def token_cost_path() -> Path:
    return _p(ART / "tokens" / "token_cost.json")


def data_path(name: str) -> Path:
    """name: train.jsonl | eval.jsonl | domain30.jsonl | data_report.json"""
    return _p(ART / "data" / name)


def run_dir(run_id: str) -> Path:
    d = ART / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def eval_dir(run_id: str, cond: str | None = None) -> Path:
    d = ART / "eval" / run_id / cond if cond else ART / "eval" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def merged_dir(run_id: str) -> Path:
    d = ART / "merged" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir(run_id: str) -> Path:
    d = ART / "report" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_runs() -> list[str]:
    base = ART / "runs"
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def latest_run() -> str | None:
    runs = list_runs()
    return runs[-1] if runs else None


# --------------------------------------------------------------- 입출력 유틸

def save_json(path: Path, obj: Any) -> Path:
    _p(Path(path)).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(path)


def load_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    path = _p(Path(path))
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def resolve(path: Path, use_reference: bool = False) -> Path:
    """use_reference=True면 artifacts/reference/ 아래의 같은 상대경로를 먼저 본다.

    GPU 실패로 자기 결과가 없는 수강생이 강사 결과로 분석만 수행할 때 사용.
    """
    path = Path(path)
    if use_reference:
        try:
            rel = path.relative_to(ART)
        except ValueError:
            rel = Path(path.name)
        cand = REFERENCE / rel
        if cand.exists():
            print(f"[reference] {cand} 를 사용합니다 (강사 사전 실행 결과)")
            return cand
    return path
