"""
임베딩 모델 래퍼.

모델 교체 방법:
  1. 이 파일의 EmbeddingModel 클래스를 그대로 두고
     model_id 인자만 바꾸면 SentenceTransformers가 지원하는 모든 모델로 전환 가능.
  2. 완전히 다른 프레임워크(e.g. vLLM, OpenAI API)를 쓰려면:
     - encode_docs() / encode_queries() 시그니처를 유지하는 새 클래스를 만들어
     - runner.py 의 build_model() 호출 부분만 교체하면 됨.

BGE-M3 vector_mode:
  - "dense"  : 1024-dim dense vector (SentenceTransformer 기본)
  - "sparse" : lexical sparse vector {token_id: weight} (FlagEmbedding)
  - "colbert": per-token multi-vector [n_tokens × 1024] MaxSim (FlagEmbedding)

Qwen3-Embedding:
  - query 인코딩에 prompt_name="query" 자동 적용
  - flash_attention_2 자동 활성화
"""
from __future__ import annotations

import gc

import numpy as np

_BGE_M3_ID      = "BAAI/bge-m3"
_BGE_M3_MODES   = frozenset({"dense", "sparse", "colbert"})

_QWEN3_MODELS = frozenset({
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
})
_FLASH_ATTN_MODELS = _QWEN3_MODELS


# ── Dense 모델 (SentenceTransformer) ─────────────────────────────────────────

class EmbeddingModel:
    """SentenceTransformer 기반 dense 임베딩 모델."""

    vector_mode = "dense"

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

        processor_kwargs: dict = {}
        if model_id in _FLASH_ATTN_MODELS:
            # last_token_pool 방식: 마지막 위치가 실제 토큰이어야 함 (flash_attn 유무 관계없이 필수)
            processor_kwargs["padding_side"] = "left"
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("  [모델] flash_attn 감지 → flash_attention_2 활성화", flush=True)
            except Exception:
                print("  [모델] flash_attn 없음/오류 → 표준 attention 사용", flush=True)

        self._model = SentenceTransformer(model_id, model_kwargs=model_kwargs,
                                          processor_kwargs=processor_kwargs)
        if model_id in _FLASH_ATTN_MODELS and hasattr(self._model, "tokenizer"):
            self._model.tokenizer.padding_side = "left"

        self._actual_dtype: str = "unknown"
        try:
            _actual_dtype = next(self._model.parameters()).dtype
            self._actual_dtype = str(_actual_dtype).replace("torch.", "")
            print(f"  [모델 dtype] 요청={dtype!r} → 실제={_actual_dtype}", flush=True)
        except StopIteration:
            pass

        self._dim: int = self._model.encode(["dim probe"], show_progress_bar=False).shape[1]

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def actual_dtype(self) -> str:
        return self._actual_dtype

    def _encode_multi_gpu(self, texts: list[str], batch_size: int,
                          normalize: bool = True, prompt_name: str | None = None) -> np.ndarray:
        import torch
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        # encode_multi_process는 prompt_name 미지원 → Qwen3 쿼리는 단일 GPU
        if n_gpu > 1 and prompt_name is None:
            pool = self._model.start_multi_process_pool(
                target_devices=[f"cuda:{i}" for i in range(n_gpu)]
            )
            try:
                raw = self._model.encode_multi_process(
                    texts, pool, batch_size=batch_size, normalize_embeddings=normalize
                )
            finally:
                self._model.stop_multi_process_pool(pool)
        else:
            kwargs: dict = {"batch_size": batch_size, "show_progress_bar": False,
                            "normalize_embeddings": normalize}
            if prompt_name:
                kwargs["prompt_name"] = prompt_name
            raw = self._model.encode(texts, **kwargs)
        return _to_numpy(raw)

    def encode_docs(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode_multi_gpu(texts, batch_size, normalize=True)

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        prompt = "query" if self.model_id in _QWEN3_MODELS else None
        return self._encode_multi_gpu(texts, batch_size, normalize=True, prompt_name=prompt)

    def close(self) -> None:
        _release_gpu()
        del self._model
        gc.collect()


# ── BGE-M3 다중 표현 모델 (FlagEmbedding) ────────────────────────────────────

class BGEM3Model:
    """
    BGE-M3 dense / sparse / colbert 3가지 vector 모드.

    dense  → np.ndarray [n, 1024]
    sparse → list[dict[int, float]]  (lexical_weights)
    colbert→ list[np.ndarray]        (각 문서별 [n_tokens, 1024])
    """

    def __init__(self, vector_mode: str = "dense", dtype: str = "auto") -> None:
        from FlagEmbedding import BGEM3FlagModel

        assert vector_mode in _BGE_M3_MODES, f"지원하지 않는 BGE-M3 모드: {vector_mode}"

        import torch
        self.model_id    = _BGE_M3_ID
        self.vector_mode = vector_mode
        use_fp16 = dtype in ("auto", "fp16")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            n_gpu = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(n_gpu)]
        else:
            devices = None
        print(f"[BGEM3] devices: {devices}", flush=True)

        self._model = BGEM3FlagModel(_BGE_M3_ID, use_fp16=use_fp16, devices=devices)
        self._actual_dtype = "float16" if use_fp16 else "float32"
        print(f"  [모델 dtype] 요청={dtype!r} → use_fp16={use_fp16}", flush=True)

        self._dim = 1024  # dense 및 colbert 토큰 차원

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def actual_dtype(self) -> str:
        return self._actual_dtype

    def _encode(self, texts: list[str], batch_size: int):
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):
            out = self._model.encode(
                texts,
                batch_size=batch_size,
                return_dense=        self.vector_mode == "dense",
                return_sparse=       self.vector_mode == "sparse",
                return_colbert_vecs= self.vector_mode == "colbert",
            )
        return out

    def encode_docs(self, texts: list[str], batch_size: int):
        out = self._encode(texts, batch_size)
        if self.vector_mode == "dense":
            return _to_numpy(out["dense_vecs"])
        elif self.vector_mode == "sparse":
            return out["lexical_weights"]
        else:
            return out["colbert_vecs"]

    def encode_queries(self, texts: list[str], batch_size: int):
        return self.encode_docs(texts, batch_size)

    def close(self) -> None:
        _release_gpu()
        del self._model
        gc.collect()


# ── 팩토리 ───────────────────────────────────────────────────────────────────

def build_model(model_id: str, vector_mode: str = "dense", dtype: str = "auto"):
    """model_id와 vector_mode에 따라 적절한 모델 인스턴스를 반환."""
    if model_id == _BGE_M3_ID and vector_mode != "dense":
        return BGEM3Model(vector_mode=vector_mode, dtype=dtype)
    return EmbeddingModel(model_id, dtype=dtype)


# ── 유틸 ─────────────────────────────────────────────────────────────────────

def _to_numpy(raw) -> np.ndarray:
    if hasattr(raw, "cpu"):
        raw = raw.detach().cpu()
    return np.asarray(raw)


def _release_gpu() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except ImportError:
        pass
