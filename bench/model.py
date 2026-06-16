"""
임베딩 모델 래퍼.

모델 교체 방법:
  1. 이 파일의 EmbeddingModel 클래스를 그대로 두고
     model_id 인자만 바꾸면 SentenceTransformers가 지원하는 모든 모델로 전환 가능.
  2. 완전히 다른 프레임워크(e.g. vLLM, OpenAI API)를 쓰려면:
     - encode_docs() / encode_queries() 시그니처를 유지하는 새 클래스를 만들어
     - runner.py 의 EmbeddingModel(...) 호출 부분만 교체하면 됨.

지원 모델별 자동 처리:
  - Qwen3-Embedding: query 인코딩에 prompt_name="query" 자동 적용
    (Source: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
  - Qwen3-Embedding: flash_attention_2 자동 활성화 (Qwen3 decoder 계열 지원)
  - BGE-M3 등 XLM-RoBERTa 계열: flash_attention_2 비활성 (미지원)
    (Source: https://huggingface.co/BAAI/bge-m3)
"""
from __future__ import annotations

import gc

import numpy as np

# Qwen3-Embedding: query 인코딩 시 prompt_name="query" 필요.
# passages/docs 는 프롬프트 없이 인코딩.
# Source: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
_QWEN3_MODELS = frozenset({
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
})

# flash_attention_2는 decoder-only Qwen3 구조에서만 지원.
# XLM-RoBERTa(BGE-M3 백본) 등 encoder 계열은 미지원.
_FLASH_ATTN_MODELS = _QWEN3_MODELS


class EmbeddingModel:
    """SentenceTransformer 기반 임베딩 모델 래퍼."""

    def __init__(self, model_id: str, dtype: str = "auto") -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_id = model_id

        _dtype_map = {"auto": "auto", "fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
        dtype_str = _dtype_map.get(dtype, "auto")

        model_kwargs: dict = {}
        if dtype_str == "auto":
            model_kwargs["torch_dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = getattr(torch, dtype_str)

        if model_id in _FLASH_ATTN_MODELS:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self._model = SentenceTransformer(model_id, model_kwargs=model_kwargs)

        # 임베딩 차원 사전 확인
        self._dim: int = self._model.encode(["dim probe"], show_progress_bar=False).shape[1]

    @property
    def dim(self) -> int:
        return self._dim

    def encode_docs(self, texts: list[str], batch_size: int) -> np.ndarray:
        """passages / documents 인코딩 — 프롬프트 없음."""
        raw = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return _to_numpy(raw)

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        """쿼리 인코딩 — 모델에 따라 instruction prompt 자동 적용."""
        kwargs: dict = {
            "batch_size":          batch_size,
            "show_progress_bar":   False,
            "normalize_embeddings": True,
        }
        if self.model_id in _QWEN3_MODELS:
            kwargs["prompt_name"] = "query"

        raw = self._model.encode(texts, **kwargs)
        return _to_numpy(raw)

    def close(self) -> None:
        """모델 언로드 및 GPU 캐시 해제."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass
        del self._model
        gc.collect()


def _to_numpy(raw) -> np.ndarray:
    if hasattr(raw, "cpu"):
        raw = raw.detach().cpu()
    return np.asarray(raw)
