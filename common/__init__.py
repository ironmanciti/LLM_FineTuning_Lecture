"""LLM 작동 원리와 오픈웨이트 모델 파인튜닝 — 공용 모듈.

노트북에서는 다음처럼 쓴다.

    import sys; sys.path.append("..")   # Colab에서 저장소 루트를 잡는 경우
    from common import config, env, artifacts as art, metrics, chat, regression
"""
from . import config, env, artifacts, metrics, chat, regression  # noqa: F401

__all__ = ["config", "env", "artifacts", "metrics", "chat", "regression"]
