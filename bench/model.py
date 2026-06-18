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
            model_kwargs["attn_implementation"] = "flash_attention_2"
            # last_token_pool 방식: 마지막 위치가 실제 토큰이어야 함
            processor_kwargs["padding_side"] = "left"

        self._model = SentenceTransformer(model_id, model_kwargs=model_kwargs,
                                          processor_kwargs=processor_kwargs)
        # processor_kwargs만으로 padding_side가 반영 안 될 수 있으므로 명시적으로 설정
        if model_id in _FLASH_ATTN_MODELS and hasattr(self._model, "tokenizer"):
            self._model.tokenizer.padding_side = "left"
            print(f"  [모델] padding_side={self._model.tokenizer.padding_side}", flush=True)

        self._dim: int = self._model.encode(["dim probe"], show_progress_bar=False).shape[1]

        # padding_side 검증: 길이 차이가 큰 두 텍스트를 배치로 인코딩 vs 단건으로 비교
        # cosine ≈ 1.0이면 left padding 정상, << 1.0이면 right padding으로 마지막 토큰이 PAD
        if model_id in _FLASH_ATTN_MODELS:
            _t_short = "안녕"
            _t_long  = "안녕하세요 " * 60  # 단어 60개 → 배치 내 길이 차이 극대화
            _batch   = self._model.encode([_t_short, _t_long], normalize_embeddings=True,
                                          show_progress_bar=False)
            _single  = self._model.encode([_t_short], normalize_embeddings=True,
                                          show_progress_bar=False)
            _cos = float(np.dot(_batch[0], _single[0]))
            print(
                f"  [padding 검증] 배치 vs 단건 cosine={_cos:.4f}"
                f"  (1.0에 가까워야 left padding 정상; << 1.0이면 PAD 토큰 풀링 버그)",
                flush=True,
            )
            del _t_short, _t_long, _batch, _single

    @property
    def dim(self) -> int:
        return self._dim

    def encode_docs(self, texts: list[str], batch_size: int) -> np.ndarray:
        raw = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return _to_numpy(raw)

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        kwargs: dict = {
            "batch_size":           batch_size,
            "show_progress_bar":    False,
            "normalize_embeddings": True,
        }
        if self.model_id in _QWEN3_MODELS:
            kwargs["prompt_name"] = "query"

        raw = self._model.encode(texts, **kwargs)
        return _to_numpy(raw)

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
        devices = ["cuda:0"] if cuda_available else None
        print(f"[BGEM3] device: {devices}", flush=True)

        self._model = BGEM3FlagModel(_BGE_M3_ID, use_fp16=use_fp16, devices=devices)

        self._dim = 1024  # dense 및 colbert 토큰 차원

    @property
    def dim(self) -> int:
        return self._dim

    def _encode(self, texts: list[str], batch_size: int):
        import contextlib, io, psutil, os, torch
        proc = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / 1e9
        gpu_before = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        with contextlib.redirect_stderr(io.StringIO()):
            out = self._model.encode(
                texts,
                batch_size=batch_size,
                return_dense=        self.vector_mode == "dense",
                return_sparse=       self.vector_mode == "sparse",
                return_colbert_vecs= self.vector_mode == "colbert",
            )

        mem_after = proc.memory_info().rss / 1e9
        gpu_after = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"  [encode] RAM {mem_before:.1f}→{mem_after:.1f}GB  GPU {gpu_before:.1f}→{gpu_after:.1f}GB  n={len(texts)}", flush=True)
        return out

    def encode_docs(self, texts: list[str], batch_size: int):
        out = self._encode(texts, batch_size)
        if self.vector_mode == "dense":
            return _to_numpy(out["dense_vecs"])
        elif self.vector_mode == "sparse":
            return out["lexical_weights"]   # list[dict[str, float]]
        else:
            vecs = out["colbert_vecs"]      # list[np.ndarray [n_tokens, 1024]]
            total_mb = sum(v.nbytes for v in vecs) / 1e6
            print(f"  [colbert] {len(vecs)}docs  avg_tokens={sum(v.shape[0] for v in vecs)/len(vecs):.0f}  total={total_mb:.0f}MB", flush=True)
            return vecs

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
