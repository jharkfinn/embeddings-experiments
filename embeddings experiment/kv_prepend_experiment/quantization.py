from __future__ import annotations

import numpy as np


def fit_trinary_thresholds(x: np.ndarray, nonzero_fraction: float, axis: int | tuple[int, ...] | None = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    abs_x = np.abs(x)
    q = 1.0 - float(nonzero_fraction)
    if axis is None:
        return np.asarray(np.quantile(abs_x, q), dtype=np.float32)
    return np.asarray(np.quantile(abs_x, q, axis=axis, keepdims=True), dtype=np.float32)


def trinarize_array(
    x: np.ndarray,
    threshold_percentile: float = 33.0,
    axis: int | None = None,
    thresholds: np.ndarray | None = None,
    positive_only: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if thresholds is None:
        abs_x = np.abs(x)
        if axis is None:
            threshold = np.quantile(abs_x, threshold_percentile / 100.0)
        else:
            threshold = np.quantile(abs_x, threshold_percentile / 100.0, axis=axis, keepdims=True)
    else:
        threshold = thresholds
    result = np.zeros_like(x, dtype=np.int8)
    result[x > threshold] = 1
    if not positive_only:
        result[x < -threshold] = -1
    return result


def mean_pool_trinary(token_vectors: np.ndarray) -> np.ndarray:
    token_vectors = np.asarray(token_vectors, dtype=np.float32)
    if token_vectors.ndim != 2:
        raise ValueError(f"Expected 2D token vectors, got shape {token_vectors.shape}")
    pooled = token_vectors.mean(axis=0)
    norm = np.linalg.norm(pooled)
    return pooled if norm == 0 else pooled / norm


def mean_pool_float(token_vectors: np.ndarray) -> np.ndarray:
    token_vectors = np.asarray(token_vectors, dtype=np.float32)
    pooled = token_vectors.mean(axis=0)
    norm = np.linalg.norm(pooled)
    return pooled if norm == 0 else pooled / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_token_cosine(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> np.ndarray:
    q = np.asarray(query_tokens, dtype=np.float32)
    d = np.asarray(doc_tokens, dtype=np.float32)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    d_norm = np.linalg.norm(d, axis=1, keepdims=True)
    q_norm = np.where(q_norm > 0, q_norm, 1.0)
    d_norm = np.where(d_norm > 0, d_norm, 1.0)
    return (q / q_norm) @ (d / d_norm).T


def maxsim(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    sims = pairwise_token_cosine(query_tokens, doc_tokens)
    if sims.size == 0:
        return 0.0
    return float(sims.max(axis=1).sum())
