from __future__ import annotations

import os
import json
import logging
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .capture_io import iter_payload_bundles
from .evaluation import _content_row_mask, _find_layer_capture, _tensor_to_numpy
from .quantization import cosine_similarity
from .types import CaptureCondition, ExampleCaptureBundle

logger = logging.getLogger(__name__)


def _memory_limit_bytes() -> int:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value > 0:
            return value
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * page_count)
    except (OSError, ValueError, AttributeError):
        return 0


def _recommended_worker_cap(capture_paths: list[Path]) -> tuple[int, dict[str, float]]:
    memory_limit = _memory_limit_bytes()
    if memory_limit <= 0:
        return max(1, min(4, len(capture_paths))), {
            "memory_limit_gib": 0.0,
            "largest_shard_gib": 0.0,
            "estimated_worker_gib": 0.0,
        }
    largest_shard = max(path.stat().st_size for path in capture_paths)
    factor = float(os.environ.get("KV_PREPEND_ANALYSIS_WORKER_MEMORY_FACTOR", "1.0") or "1.0")
    estimated_worker_bytes = max(int(largest_shard * factor), 1536 * 1024**2)
    usable_bytes = int(memory_limit * 0.80)
    worker_cap = max(1, usable_bytes // estimated_worker_bytes)
    return int(max(1, min(len(capture_paths), worker_cap))), {
        "memory_limit_gib": round(memory_limit / (1024**3), 2),
        "largest_shard_gib": round(largest_shard / (1024**3), 2),
        "estimated_worker_gib": round(estimated_worker_bytes / (1024**3), 2),
    }


def _analysis_start_method() -> str:
    configured = os.environ.get("KV_PREPEND_ANALYSIS_START_METHOD", "").strip().lower()
    if configured in {"fork", "spawn", "forkserver"}:
        return configured
    if os.name == "posix":
        return "fork"
    return "spawn"


_METRIC_PREFIXES = (
    "delta_norm",
    "router_entropy",
    "m_rank",
    "d_rank",
    "bias_sensitivity",
    "bias_flip_count",
)


def _analysis_rank_feature_width() -> int:
    configured = int(os.environ.get("KV_PREPEND_ANALYSIS_RANK_FEATURE_WIDTH", "32") or "32")
    return max(0, configured)


def _analysis_compute_pairwise_cosine() -> bool:
    return os.environ.get("KV_PREPEND_ANALYSIS_PAIRWISE_COSINE", "").strip().lower() in {"1", "true", "yes", "on"}


def _analysis_supports_batch_native() -> bool:
    return os.environ.get("KV_PREPEND_ANALYSIS_BATCH_NATIVE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _recommended_blas_threads(worker_count: int) -> int:
    configured = int(os.environ.get("KV_PREPEND_ANALYSIS_BLAS_THREADS", "0") or "0")
    if configured > 0:
        return max(1, configured)
    cpu_count = max(1, os.cpu_count() or 1)
    if worker_count <= 1:
        return min(8, cpu_count)
    return max(1, min(4, cpu_count // worker_count))


def _configure_analysis_runtime(blas_threads: int) -> None:
    thread_value = str(max(1, blas_threads))
    for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = thread_value
    try:
        import torch

        torch.set_num_threads(max(1, blas_threads))
        torch.set_num_interop_threads(1)
    except Exception:
        return


def _analysis_worker_initializer(blas_threads: int) -> None:
    _configure_analysis_runtime(blas_threads)


def _empty_summary() -> dict[str, Any]:
    return {
        "analysis_version": 2,
        "analysis_mode": "batch_native_streaming",
        "rank_method": "subsampled_effective_rank",
        "rank_feature_width": _analysis_rank_feature_width(),
        "pairwise_cosine_enabled": _analysis_compute_pairwise_cosine(),
        "num_bundles": 0,
        "layers": {},
    }


def _empty_layer_stats() -> dict[str, float | int]:
    stats: dict[str, float | int] = {}
    for prefix in _METRIC_PREFIXES:
        stats[f"{prefix}_sum"] = 0.0
        stats[f"{prefix}_count"] = 0
    return stats


def _ensure_layer_stats(summary: dict[str, Any], layer_key: str) -> dict[str, float | int]:
    layers = summary["layers"]
    layer_stats = layers.get(layer_key)
    if layer_stats is None:
        layer_stats = _empty_layer_stats()
        layers[layer_key] = layer_stats
    return layer_stats


def _accumulate_metric_values(layer_stats: dict[str, float | int], prefix: str, values: Any) -> None:
    if values is None:
        return
    try:
        import torch

        if isinstance(values, torch.Tensor):
            tensor = values.detach()
            if tensor.numel() == 0:
                return
            tensor = tensor.reshape(-1).to(dtype=torch.float32)
            tensor = tensor[torch.isfinite(tensor)]
            if tensor.numel() == 0:
                return
            layer_stats[f"{prefix}_sum"] = float(layer_stats.get(f"{prefix}_sum", 0.0)) + float(tensor.sum().item())
            layer_stats[f"{prefix}_count"] = int(layer_stats.get(f"{prefix}_count", 0)) + int(tensor.numel())
            return
    except ModuleNotFoundError:  # pragma: no cover
        pass
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return
    array = array[np.isfinite(array)]
    if array.size == 0:
        return
    layer_stats[f"{prefix}_sum"] = float(layer_stats.get(f"{prefix}_sum", 0.0)) + float(array.sum())
    layer_stats[f"{prefix}_count"] = int(layer_stats.get(f"{prefix}_count", 0)) + int(array.size)


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary["num_bundles"] = int(summary.get("num_bundles", 0))
    for layer_stats in summary["layers"].values():
        for prefix in _METRIC_PREFIXES:
            count = int(layer_stats.get(f"{prefix}_count", 0))
            total = float(layer_stats.get(f"{prefix}_sum", 0.0))
            layer_stats[f"{prefix}_mean"] = (total / count) if count else 0.0
    return summary


def _merge_summary_partials(partials: list[dict[str, Any]]) -> dict[str, Any]:
    merged = _empty_summary()
    for partial in partials:
        merged["num_bundles"] += int(partial.get("num_bundles", 0))
        for layer_key, layer_stats in partial.get("layers", {}).items():
            merged_layer = _ensure_layer_stats(merged, layer_key)
            for prefix in _METRIC_PREFIXES:
                merged_layer[f"{prefix}_sum"] = float(merged_layer.get(f"{prefix}_sum", 0.0)) + float(
                    layer_stats.get(f"{prefix}_sum", 0.0)
                )
                merged_layer[f"{prefix}_count"] = int(merged_layer.get(f"{prefix}_count", 0)) + int(
                    layer_stats.get(f"{prefix}_count", 0)
                )
    return _finalize_summary(merged)


def _torch_load_payload(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _capture_dicts_by_layer(pass_payload: dict[str, Any], condition: str) -> dict[int, dict[str, Any]]:
    return {
        int(capture["layer_idx"]): capture
        for capture in pass_payload.get("captures_by_condition", {}).get(condition, [])
        if isinstance(capture, dict) and "layer_idx" in capture
    }


def _content_mask_batch(examples: list[dict[str, Any]], seq_len: int):
    import torch

    mask = torch.zeros((len(examples), seq_len), dtype=torch.bool)
    for row_idx, example in enumerate(examples):
        row_mask = example.get("content_token_mask", [])
        if not row_mask:
            continue
        valid = min(seq_len, len(row_mask))
        if valid > 0:
            mask[row_idx, :valid] = torch.as_tensor(row_mask[:valid], dtype=torch.bool)
    return mask


@lru_cache(maxsize=32)
def _rank_feature_indices(width: int, target_width: int) -> tuple[int, ...]:
    if target_width <= 0 or width <= target_width:
        return tuple(range(width))
    positions = np.linspace(0, width - 1, num=target_width, dtype=np.int64)
    return tuple(int(index) for index in np.unique(positions))


def _effective_rank_tensor(tokens) -> float:
    import torch

    if tokens.ndim != 2 or tokens.numel() == 0:
        return 0.0
    finite_rows = torch.isfinite(tokens).all(dim=1)
    tokens = tokens[finite_rows]
    if tokens.numel() == 0:
        return 0.0
    tokens = torch.nan_to_num(tokens.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    target_width = _analysis_rank_feature_width()
    if tokens.shape[1] > target_width > 0:
        indices = torch.as_tensor(_rank_feature_indices(tokens.shape[1], target_width), dtype=torch.long, device=tokens.device)
        tokens = tokens.index_select(1, indices)
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    scale = centered.abs().max()
    if torch.isfinite(scale) and float(scale.item()) > 0.0:
        centered = centered / scale
    try:
        singular_values = torch.linalg.svdvals(centered)
    except RuntimeError:
        logger.warning("analysis_svd_nonconverged rows=%s width=%s", tokens.shape[0], tokens.shape[1])
        return 0.0
    energy = singular_values.square()
    total = energy.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return 0.0
    probs = (energy / total).clamp_min(1e-12)
    return float(torch.exp(-(probs * probs.log()).sum()).item())


def _summarize_schema3_payload(payload: dict[str, Any]) -> dict[str, Any]:
    import torch

    summary = _empty_summary()
    examples = payload.get("examples", [])
    batch_size = len(examples)
    if batch_size == 0:
        return summary
    summary["num_bundles"] += batch_size
    pass_map = {
        str(pass_payload.get("pass_name")): pass_payload
        for pass_payload in payload.get("passes", [])
        if isinstance(pass_payload, dict)
    }
    pass1 = pass_map.get("pass1")
    if not isinstance(pass1, dict):
        return summary
    causal_by_layer = _capture_dicts_by_layer(pass1, CaptureCondition.CAUSAL.value)
    local_by_layer = _capture_dicts_by_layer(pass1, CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE.value)
    mask_cache: dict[int, Any] = {}
    with torch.inference_mode():
        for layer_idx in sorted(set(causal_by_layer) & set(local_by_layer)):
            causal_capture = causal_by_layer[layer_idx]
            local_capture = local_by_layer[layer_idx]
            causal_z = causal_capture.get("z_attn")
            local_z = local_capture.get("z_attn")
            if causal_z is None or local_z is None:
                continue
            seq_len = min(int(causal_z.shape[1]), int(local_z.shape[1]))
            if seq_len <= 0:
                continue
            mask = mask_cache.get(seq_len)
            if mask is None:
                mask = _content_mask_batch(examples, seq_len)
                mask_cache[seq_len] = mask
            if not bool(mask.any()):
                continue
            layer_stats = _ensure_layer_stats(summary, str(layer_idx))
            causal_tokens = causal_z[:, :seq_len].to(dtype=torch.float32)
            local_tokens = local_z[:, :seq_len].to(dtype=torch.float32)
            delta = local_tokens - causal_tokens
            masked_delta = delta.masked_fill(~mask.unsqueeze(-1), 0.0)
            delta_norms = torch.linalg.vector_norm(masked_delta.reshape(batch_size, -1), dim=1)
            valid_rows = mask.any(dim=1)
            _accumulate_metric_values(layer_stats, "delta_norm", delta_norms[valid_rows])

            router_logits = local_capture.get("router_logits_pre_softmax")
            if router_logits is not None:
                logits = router_logits[:, :seq_len].to(dtype=torch.float32)
                logits = logits - logits.amax(dim=-1, keepdim=True)
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                token_counts = mask.sum(dim=1)
                row_entropy = (entropy * mask).sum(dim=1) / token_counts.clamp_min(1)
                _accumulate_metric_values(layer_stats, "router_entropy", row_entropy[token_counts > 0])

            for row_idx in torch.nonzero(valid_rows, as_tuple=False).flatten().tolist():
                row_mask = mask[row_idx]
                base_tokens = causal_tokens[row_idx, :seq_len][row_mask]
                treated_tokens = local_tokens[row_idx, :seq_len][row_mask]
                if base_tokens.numel() == 0 or treated_tokens.numel() == 0:
                    continue
                mean_tokens = 0.5 * (base_tokens + treated_tokens)
                delta_tokens = treated_tokens - base_tokens
                _accumulate_metric_values(layer_stats, "m_rank", [_effective_rank_tensor(mean_tokens)])
                _accumulate_metric_values(layer_stats, "d_rank", [_effective_rank_tensor(delta_tokens)])

            metadata = local_capture.get("metadata", {})
            per_example_metadata = metadata.get("_per_example_metadata") if isinstance(metadata, dict) else None
            if isinstance(per_example_metadata, list):
                for row_meta in per_example_metadata[:batch_size]:
                    if not isinstance(row_meta, dict):
                        continue
                    bias_meta = row_meta.get("bias_spectrum_signature")
                    if not isinstance(bias_meta, dict):
                        continue
                    _accumulate_metric_values(layer_stats, "bias_sensitivity", bias_meta.get("sensitivity", []))
                    _accumulate_metric_values(layer_stats, "bias_flip_count", bias_meta.get("flip_count", []))
    return summary


def _accumulate_bundle(summary: dict[str, Any], bundle: ExampleCaptureBundle) -> None:
    summary["num_bundles"] += 1
    bundle_count = int(summary["num_bundles"])
    if bundle_count % 10 == 0:
        logger.info("analysis_progress bundles=%s", bundle_count)
    for layer_idx in range(48):
        try:
            causal = _find_layer_capture(bundle, "pass1", CaptureCondition.CAUSAL.value, layer_idx)
            local = _find_layer_capture(bundle, "pass1", CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE.value, layer_idx)
        except KeyError:
            continue
        layer_stats = _ensure_layer_stats(summary, str(layer_idx))
        causal_z = _tensor_to_numpy(causal.z_attn, dtype=np.float32)[0]
        local_z = _tensor_to_numpy(local.z_attn, dtype=np.float32)[0]
        expected_len = min(causal_z.shape[0], local_z.shape[0])
        mask = _content_row_mask(bundle, expected_len)
        base_tokens = causal_z[:expected_len][mask]
        treated_tokens = local_z[:expected_len][mask]
        if base_tokens.size == 0 or treated_tokens.size == 0:
            continue
        delta = treated_tokens - base_tokens
        _accumulate_metric_values(layer_stats, "delta_norm", [float(np.linalg.norm(delta))])
        _accumulate_metric_values(layer_stats, "m_rank", [token_collapse_panel((base_tokens + treated_tokens) / 2.0)["effective_rank"]])
        _accumulate_metric_values(layer_stats, "d_rank", [token_collapse_panel(delta)["effective_rank"]])
        if local.router_logits_pre_softmax is not None:
            local_router = _tensor_to_numpy(local.router_logits_pre_softmax, dtype=np.float32)[0]
            routing = routing_entropy_and_divergence(local_router[:expected_len][mask], None)
            _accumulate_metric_values(layer_stats, "router_entropy", [routing["entropy_mean"]])
        if "bias_spectrum_signature" in local.metadata:
            bias_meta = local.metadata["bias_spectrum_signature"]
            _accumulate_metric_values(layer_stats, "bias_sensitivity", bias_meta.get("sensitivity", []))
            _accumulate_metric_values(layer_stats, "bias_flip_count", bias_meta.get("flip_count", []))


def _summarize_capture_path(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    payload = _torch_load_payload(path)
    if int(payload.get("schema_version", 2)) == 3 and _analysis_supports_batch_native():
        return _summarize_schema3_payload(payload)
    partial = _empty_summary()
    partial["analysis_mode"] = "bundle_fallback"
    for bundle in iter_payload_bundles(payload):
        _accumulate_bundle(partial, bundle)
    return partial


def compute_m_d_u(base_tokens: np.ndarray, treated_tokens: np.ndarray, uptake: np.ndarray | None = None):
    m = (base_tokens + treated_tokens) / 2.0
    d = treated_tokens - base_tokens
    u = uptake
    return {"m": m, "d": d, "u": u}


def compute_gamma_metrics(delta_causal: np.ndarray, delta_prop: np.ndarray, eps: float = 1e-6):
    norm_causal = np.linalg.norm(delta_causal)
    norm_prop = np.linalg.norm(delta_prop)
    gain = float(norm_prop / (norm_causal + eps))
    angle = float(cosine_similarity(delta_prop, delta_causal))
    normalized = delta_causal / (norm_causal + eps)
    gamma_parallel = float(np.dot(delta_prop - delta_causal, normalized))
    return {"gain": gain, "angle": angle, "gamma_parallel": gamma_parallel}


def radial_tangential_decomposition(h_base: np.ndarray, delta_z: np.ndarray, eps: float = 1e-6):
    h_base = np.asarray(h_base, dtype=np.float32)
    delta_z = np.asarray(delta_z, dtype=np.float32)
    unit = h_base / (np.linalg.norm(h_base, axis=-1, keepdims=True) + eps)
    radial_scale = np.sum(delta_z * unit, axis=-1, keepdims=True)
    radial = radial_scale * unit
    tangential = delta_z - radial
    return {"radial": radial, "tangential": tangential}


def token_collapse_panel(tokens: np.ndarray):
    import torch

    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or tokens.size == 0:
        return {"effective_rank": 0.0, "pairwise_cosine_mean": 0.0, "pairwise_cosine_std": 0.0}
    effective_rank = _effective_rank_tensor(torch.as_tensor(tokens))
    if not _analysis_compute_pairwise_cosine():
        return {"effective_rank": effective_rank, "pairwise_cosine_mean": 0.0, "pairwise_cosine_std": 0.0}
    finite_rows = np.all(np.isfinite(tokens), axis=1)
    tokens = tokens[finite_rows]
    if tokens.shape[0] == 0:
        return {"effective_rank": effective_rank, "pairwise_cosine_mean": 0.0, "pairwise_cosine_std": 0.0}
    tokens = np.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(tokens, axis=-1, keepdims=True)
    unit = np.nan_to_num(tokens / np.clip(norms, 1e-6, None), nan=0.0, posinf=0.0, neginf=0.0)
    cosine_matrix = unit @ unit.T
    if cosine_matrix.shape[0] > 1:
        mask = ~np.eye(cosine_matrix.shape[0], dtype=bool)
        cosine_values = cosine_matrix[mask]
    else:
        cosine_values = cosine_matrix.reshape(-1)
    cosine_values = np.nan_to_num(cosine_values, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "effective_rank": effective_rank,
        "pairwise_cosine_mean": float(cosine_values.mean()),
        "pairwise_cosine_std": float(cosine_values.std()),
    }


def routing_entropy_and_divergence(router_logits: np.ndarray, top_k_indices: np.ndarray):
    logits = np.asarray(router_logits, dtype=np.float32)
    if logits.size == 0:
        return {"entropy_mean": 0.0, "topk_unique_mean": 0.0}
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1)
    if top_k_indices is None:
        return {"entropy_mean": float(entropy.mean()), "topk_unique_mean": 0.0}
    unique_counts = np.array([len(set(row.tolist())) for row in np.asarray(top_k_indices)])
    return {"entropy_mean": float(entropy.mean()), "topk_unique_mean": float(unique_counts.mean())}


def bias_spectrum_from_margin(summary_beta: np.ndarray, bias_values: np.ndarray):
    summary_beta = np.asarray(summary_beta, dtype=np.float32)
    bias_values = np.asarray(bias_values, dtype=np.float32)
    summary_beta = np.clip(summary_beta, 1e-6, 1 - 1e-6)
    margin0 = np.log(summary_beta / (1.0 - summary_beta))
    beta = 1.0 / (1.0 + np.exp(-(margin0[:, None] + bias_values[None, :])))
    grad = np.gradient(beta, bias_values, axis=1)
    curvature = np.gradient(grad, bias_values, axis=1)
    return {
        "margin0": margin0,
        "beta": beta,
        "dbeta_db": grad,
        "d2beta_db2": curvature,
    }


def exact_bias_spectrum_reconstruction(
    *,
    causal_attn_weights: np.ndarray,
    value_states: np.ndarray,
    summary_value: np.ndarray,
    summary_beta: np.ndarray,
    bias_values: np.ndarray,
    o_proj_weight: np.ndarray,
    resid_pre_attn: np.ndarray,
    rmsnorm_weight: np.ndarray,
    rmsnorm_eps: float,
    router_weight: np.ndarray,
):
    causal_attn_weights = np.asarray(causal_attn_weights, dtype=np.float32)
    value_states = np.asarray(value_states, dtype=np.float32)
    summary_value = np.asarray(summary_value, dtype=np.float32)
    resid_pre_attn = np.asarray(resid_pre_attn, dtype=np.float32)
    o_proj_weight = np.asarray(o_proj_weight, dtype=np.float32)
    rmsnorm_weight = np.asarray(rmsnorm_weight, dtype=np.float32)
    router_weight = np.asarray(router_weight, dtype=np.float32)
    sweep = bias_spectrum_from_margin(summary_beta, bias_values)
    causal_headwise = np.einsum("hts,shd->thd", causal_attn_weights, value_states)
    mixed_outputs = []
    router_logits = []
    for beta_column in sweep["beta"].T:
        beta_t = beta_column[:, None, None]
        mixed_headwise = (1.0 - beta_t) * causal_headwise + beta_t * summary_value[None, :, :]
        concat = mixed_headwise.reshape(mixed_headwise.shape[0], -1)
        z_b = concat @ o_proj_weight.T
        h_base = resid_pre_attn + z_b
        variance = np.mean(h_base.astype(np.float32) ** 2, axis=-1, keepdims=True)
        h_pre_moe = rmsnorm_weight[None, :] * (h_base * (1.0 / np.sqrt(variance + rmsnorm_eps)))
        mixed_outputs.append(h_pre_moe)
        router_logits.append(h_pre_moe @ router_weight.T)
    sweep["h_pre_moe"] = np.stack(mixed_outputs, axis=1)
    sweep["router_logits"] = np.stack(router_logits, axis=1)
    sweep["dr_db"] = np.gradient(sweep["router_logits"], bias_values, axis=1)
    sweep["d2r_db2"] = np.gradient(sweep["dr_db"], bias_values, axis=1)
    return sweep


def two_nn_intrinsic_dimensionality(points: np.ndarray, eps: float = 1e-8):
    points = np.asarray(points, dtype=np.float32)
    distances = np.sqrt(np.maximum(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1), 0.0))
    np.fill_diagonal(distances, np.inf)
    nearest = np.sort(distances, axis=1)[:, :2]
    ratios = nearest[:, 1] / np.clip(nearest[:, 0], eps, None)
    ratios = ratios[np.isfinite(ratios) & (ratios > 1)]
    if len(ratios) == 0:
        return 0.0
    return float(1.0 / np.mean(np.log(ratios)))


def shared_private_variance(x: np.ndarray, y: np.ndarray, ridge: float = 1e-4):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    xtx = x.T @ x + ridge * np.eye(x.shape[1], dtype=np.float32)
    weights = np.linalg.solve(xtx, x.T @ y)
    pred = x @ weights
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum()
    r2 = 1.0 - float(ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"r2": r2, "private_variance": float(max(ss_res, 0.0))}


def specificity_index(own_effect: np.ndarray, alt_effect: np.ndarray):
    own = np.asarray(own_effect, dtype=np.float32)
    alt = np.asarray(alt_effect, dtype=np.float32)
    return float(np.linalg.norm(own) - np.linalg.norm(alt))


def sparse_transport_coeff(local_delta: np.ndarray, final_delta: np.ndarray, eps: float = 1e-6):
    local_delta = np.asarray(local_delta, dtype=np.float32)
    final_delta = np.asarray(final_delta, dtype=np.float32)
    direction = local_delta / (np.linalg.norm(local_delta) + eps)
    return float(np.dot(final_delta, direction) / (np.linalg.norm(local_delta) + eps))


def sparse_transport_summary(local_delta: np.ndarray, final_delta: np.ndarray):
    local_delta = np.asarray(local_delta, dtype=np.float32)
    final_delta = np.asarray(final_delta, dtype=np.float32)
    return {
        "local_norm": float(np.linalg.norm(local_delta)),
        "final_norm": float(np.linalg.norm(final_delta)),
        "cosine": float(cosine_similarity(local_delta, final_delta)),
        "transport_coeff": sparse_transport_coeff(local_delta, final_delta),
    }


def summarize_bundles(bundles):
    summary: dict[str, Any] = {"num_bundles": 0, "layers": {}}
    for bundle in bundles:
        _accumulate_bundle(summary, bundle)
    return _finalize_summary(summary)


def analyze_capture_directory(
    capture_dir: str | Path,
    output_path: str | Path,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
):
    capture_dir = Path(capture_dir)
    logger.info("analysis_start capture_dir=%s", capture_dir)
    capture_paths = sorted(capture_dir.glob("*.pt"))
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "analysis_start",
                "capture_dir": str(capture_dir),
                "total_shards": len(capture_paths),
                "completed_shards": 0,
            }
        )
    if not capture_paths:
        summary = _empty_summary()
    else:
        # The accelerated path now uses mmap shard loads, schema-3 batch reduction,
        # and sketched rank metrics. We still bound workers by memory and cap per-worker
        # BLAS threads to keep CPU utilization high without oversubscription.
        configured_workers = int(os.environ.get("KV_PREPEND_ANALYSIS_WORKERS", "0") or "0")
        worker_count = min(max(1, os.cpu_count() or 1), len(capture_paths))
        if configured_workers > 0:
            worker_count = min(len(capture_paths), configured_workers)
        memory_cap, memory_meta = _recommended_worker_cap(capture_paths)
        worker_count = max(1, min(worker_count, memory_cap))
        blas_threads = _recommended_blas_threads(worker_count)
        _configure_analysis_runtime(blas_threads)
        start_method = _analysis_start_method()
        ctx = mp.get_context(start_method)
        logger.info(
            "analysis_parallel workers=%s shards=%s start_method=%s blas_threads=%s memory_limit_gib=%s largest_shard_gib=%s estimated_worker_gib=%s batch_native=%s rank_width=%s",
            worker_count,
            len(capture_paths),
            ctx.get_start_method(),
            blas_threads,
            memory_meta["memory_limit_gib"],
            memory_meta["largest_shard_gib"],
            memory_meta["estimated_worker_gib"],
            _analysis_supports_batch_native(),
            _analysis_rank_feature_width(),
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "analysis_parallel",
                    "worker_count": worker_count,
                    "worker_count_memory_cap": memory_cap,
                    "worker_start_method": ctx.get_start_method(),
                    "blas_threads": blas_threads,
                    "batch_native": _analysis_supports_batch_native(),
                    "rank_feature_width": _analysis_rank_feature_width(),
                    **memory_meta,
                    "total_shards": len(capture_paths),
                    "completed_shards": 0,
                }
            )
        partials: list[dict[str, Any]] = []
        completed = 0
        if worker_count == 1:
            logger.info("analysis_serial shards=%s", len(capture_paths))
            for path in capture_paths:
                partial = _summarize_capture_path(str(path))
                completed += 1
                logger.info(
                    "analysis_shard_done completed=%s/%s path=%s bundles=%s",
                    completed,
                    len(capture_paths),
                    path,
                    partial.get("num_bundles", 0),
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "analysis_shard_done",
                            "completed_shards": completed,
                            "total_shards": len(capture_paths),
                            "last_path": str(path),
                            "last_bundles": int(partial.get("num_bundles", 0)),
                        }
                    )
                partials.append(partial)
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=ctx,
                initializer=_analysis_worker_initializer,
                initargs=(blas_threads,),
            ) as executor:
                future_to_path = {
                    executor.submit(_summarize_capture_path, str(path)): path
                    for path in capture_paths
                }
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    partial = future.result()
                    completed += 1
                    logger.info(
                        "analysis_shard_done completed=%s/%s path=%s bundles=%s",
                        completed,
                        len(capture_paths),
                        path,
                        partial.get("num_bundles", 0),
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "stage": "analysis_shard_done",
                                "completed_shards": completed,
                                "total_shards": len(capture_paths),
                                "last_path": str(path),
                                "last_bundles": int(partial.get("num_bundles", 0)),
                            }
                        )
                    partials.append(partial)
        summary = _merge_summary_partials(partials)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "analysis_done output=%s bundles=%s layers=%s",
        output_path,
        summary["num_bundles"],
        len(summary["layers"]),
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "analysis_done",
                "completed_shards": len(capture_paths),
                "total_shards": len(capture_paths),
                "output_path": str(output_path),
                "num_bundles": int(summary.get("num_bundles", 0)),
                "num_layers": len(summary.get("layers", {})),
            }
        )
    return summary
