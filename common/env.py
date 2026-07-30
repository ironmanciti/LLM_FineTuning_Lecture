"""실행 환경 고정과 측정.

전 노트북 공통. 이 모듈이 하는 일은 세 가지뿐이다.
  1) 재현성 고정 (seed)
  2) 환경 진단 (GPU / dtype / bf16 지원 여부)
  3) VRAM 측정

교육 포인트: T4(Turing)는 bf16을 지원하지 않는다. `pick_dtype()`이
자동으로 fp16으로 내려가는 것을 학습자에게 반드시 보여줄 것.
(3일차 §3.11-3 'NaN 대응' 슬라이드의 근거가 된다)
"""
from __future__ import annotations

import os
import random
import platform
from dataclasses import dataclass, asdict

SEED = 42


def set_seed(seed: int = SEED) -> int:
    """파이썬·numpy·torch의 난수를 한 번에 고정한다."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed


@dataclass
class EnvInfo:
    python: str
    platform: str
    torch: str | None
    cuda_available: bool
    gpu_name: str | None
    gpu_total_gb: float | None
    supports_bf16: bool
    recommended_dtype: str
    attn_implementation: str
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def describe_env() -> EnvInfo:
    """지금 이 런타임이 무엇인지 한 번에 진단한다."""
    notes: list[str] = []
    torch_ver = None
    cuda = False
    name = None
    total = None
    bf16 = False

    try:
        import torch

        torch_ver = torch.__version__
        cuda = torch.cuda.is_available()
        if cuda:
            name = torch.cuda.get_device_name(0)
            total = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            try:
                bf16 = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16 = False
    except ImportError:
        notes.append("torch가 설치되지 않았습니다. CPU 전용 노트북(010·020·410·450)만 실행 가능합니다.")

    if cuda and not bf16:
        notes.append(
            "이 GPU는 bf16을 지원하지 않습니다(T4/Turing 계열). fp16을 사용하며, "
            "학습 중 loss가 NaN이 되면 학습률을 낮추거나 grad clipping을 확인하십시오."
        )
    if cuda and name and ("T4" in name):
        notes.append(
            "T4는 FlashAttention을 지원하지 않습니다. attn_implementation='sdpa'(또는 'eager')를 사용합니다."
        )
    if not cuda:
        notes.append("GPU가 없습니다. 300·420·430·440은 GPU 런타임(Colab: 런타임 > 런타임 유형 변경 > T4)이 필요합니다.")

    dtype = "bfloat16" if bf16 else ("float16" if cuda else "float32")
    attn = "sdpa" if cuda else "eager"

    return EnvInfo(
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch_ver,
        cuda_available=cuda,
        gpu_name=name,
        gpu_total_gb=total,
        supports_bf16=bf16,
        recommended_dtype=dtype,
        attn_implementation=attn,
        notes=notes,
    )


def print_env() -> EnvInfo:
    info = describe_env()
    print("=" * 62)
    print(f"Python {info.python}  |  torch {info.torch}")
    print(f"GPU            : {info.gpu_name or '없음'}"
          + (f"  ({info.gpu_total_gb} GB)" if info.gpu_total_gb else ""))
    print(f"bf16 지원      : {info.supports_bf16}")
    print(f"권장 dtype     : {info.recommended_dtype}")
    print(f"attention 구현 : {info.attn_implementation}")
    print("=" * 62)
    for n in info.notes:
        print(f"  [주의] {n}")
    return info


def pick_dtype():
    """이 런타임에서 쓸 torch dtype을 돌려준다."""
    import torch

    info = describe_env()
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        info.recommended_dtype
    ]


# ---------------------------------------------------------------- VRAM 측정

def vram_reset() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def vram_gb() -> dict:
    """현재/최대 할당량을 GB로. GPU가 없으면 0."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "peak_gb": 0.0, "reserved_gb": 0.0}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        }
    except ImportError:
        return {"allocated_gb": 0.0, "peak_gb": 0.0, "reserved_gb": 0.0}


class vram_probe:
    """with 블록의 VRAM 피크를 잰다.

        with vram_probe("모델 로드") as p:
            model = ...
        print(p.result)
    """

    def __init__(self, label: str = ""):
        self.label = label
        self.result: dict = {}

    def __enter__(self):
        vram_reset()
        self.start = vram_gb()
        return self

    def __exit__(self, *exc):
        end = vram_gb()
        self.result = {
            "label": self.label,
            "start_gb": self.start["allocated_gb"],
            "end_gb": end["allocated_gb"],
            "peak_gb": end["peak_gb"],
            "delta_gb": round(end["allocated_gb"] - self.start["allocated_gb"], 3),
        }
        print(f"[VRAM] {self.label}: +{self.result['delta_gb']} GB (피크 {self.result['peak_gb']} GB)")
        return False


def kv_cache_gb(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int,
                batch: int = 1, bytes_per_elem: int = 2) -> float:
    """KV 캐시 크기 계산.

    2(K와 V) x layers x kv_heads x head_dim x seq_len x batch x dtype바이트

    §3.2-4 슬라이드의 계산식과 동일하다. 300 노트북에서 이 값과
    실측 VRAM 증가분을 대조하는 것이 학습 활동이다.
    """
    total = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per_elem
    return round(total / 1024**3, 4)
