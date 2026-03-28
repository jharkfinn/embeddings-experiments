from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import _find_layer_capture, _tensor_to_numpy, load_capture_payloads
from .quantization import cosine_similarity
from .types import CaptureCondition, ExampleCaptureBundle

logger = logging.getLogger(__name__)


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
    if tokens.size == 0:
        return {"effective_rank": 0.0, "pairwise_cosine_mean": 0.0, "pairwise_cosine_std": 0.0}
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, tokens.shape[0] - 1)
    singular_values = np.linalg.svd(covariance, compute_uv=False)
    effective_rank = float(np.exp(-(p := singular_values / singular_values.sum() if singular_values.sum() > 0 else singular_values).dot(np.log(np.clip(p, 1e-12, None))))) if singular_values.sum() > 0 else 0.0
    cosine_matrix = tokens @ tokens.T
    return {
        "effective_rank": effective_rank,
        "pairwise_cosine_mean": float(cosine_matrix.mean()),
        "pairwise_cosine_std": float(cosine_matrix.std()),
    }


def routing_entropy_and_divergence(router_logits: np.ndarray, top_k_indices: np.ndarray):
    logits = np.asarray(router_logits, dtype=np.float32)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=-1)
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
    bundle_count = 0
    for bundle in bundles:
        bundle_count += 1
        if bundle_count % 10 == 0:
            logger.info("analysis_progress bundles=%s", bundle_count)
        for layer_idx in range(48):
            try:
                causal = _find_layer_capture(bundle, "pass1", CaptureCondition.CAUSAL.value, layer_idx)
                local = _find_layer_capture(bundle, "pass1", CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE.value, layer_idx)
            except KeyError:
                continue
            layer_key = str(layer_idx)
            summary["layers"].setdefault(layer_key, {"delta_norms": [], "router_entropy": [], "m_rank": [], "d_rank": []})
            mask = np.asarray(bundle.content_token_mask, dtype=bool)
            delta = _tensor_to_numpy(local.z_attn, dtype=np.float32)[0][mask] - _tensor_to_numpy(causal.z_attn, dtype=np.float32)[0][mask]
            mdu = compute_m_d_u(
                _tensor_to_numpy(causal.z_attn, dtype=np.float32)[0][mask],
                _tensor_to_numpy(local.z_attn, dtype=np.float32)[0][mask],
                _tensor_to_numpy(local.beta, dtype=np.float32).squeeze(-1)[0][:, mask] if local.beta is not None else None,
            )
            summary["layers"][layer_key]["delta_norms"].append(float(np.linalg.norm(delta)))
            summary["layers"][layer_key]["m_rank"].append(token_collapse_panel(mdu["m"])["effective_rank"])
            summary["layers"][layer_key]["d_rank"].append(token_collapse_panel(mdu["d"])["effective_rank"])
            if local.router_logits_pre_softmax is not None and local.top_k_indices is not None:
                routing = routing_entropy_and_divergence(
                    _tensor_to_numpy(local.router_logits_pre_softmax, dtype=np.float32)[0][mask],
                    _tensor_to_numpy(local.top_k_indices, dtype=np.int16)[0][mask],
                )
                summary["layers"][layer_key]["router_entropy"].append(routing["entropy_mean"])
            if "bias_spectrum_signature" in local.metadata:
                bias_meta = local.metadata["bias_spectrum_signature"]
                summary["layers"][layer_key].setdefault("bias_sensitivity", []).extend(bias_meta.get("sensitivity", []))
                summary["layers"][layer_key].setdefault("bias_flip_count", []).extend(bias_meta.get("flip_count", []))
    summary["num_bundles"] = bundle_count
    for layer_stats in summary["layers"].values():
        layer_stats["delta_norm_mean"] = float(np.mean(layer_stats["delta_norms"])) if layer_stats["delta_norms"] else 0.0
        layer_stats["router_entropy_mean"] = float(np.mean(layer_stats["router_entropy"])) if layer_stats["router_entropy"] else 0.0
        layer_stats["m_rank_mean"] = float(np.mean(layer_stats["m_rank"])) if layer_stats["m_rank"] else 0.0
        layer_stats["d_rank_mean"] = float(np.mean(layer_stats["d_rank"])) if layer_stats["d_rank"] else 0.0
        layer_stats["bias_sensitivity_mean"] = float(np.mean(layer_stats.get("bias_sensitivity", []))) if layer_stats.get("bias_sensitivity") else 0.0
        layer_stats["bias_flip_count_mean"] = float(np.mean(layer_stats.get("bias_flip_count", []))) if layer_stats.get("bias_flip_count") else 0.0
    return summary


def analyze_capture_directory(capture_dir: str | Path, output_path: str | Path):
    capture_dir = Path(capture_dir)
    logger.info("analysis_start capture_dir=%s", capture_dir)

    def _bundle_stream():
        shard_count = 0
        for path, payload in load_capture_payloads(capture_dir):
            shard_count += 1
            bundles = payload.get("bundles", [])
            logger.info(
                "analysis_shard_loaded shard_index=%s path=%s bundles=%s",
                shard_count,
                path,
                len(bundles),
            )
            for bundle in bundles:
                yield bundle

    summary = summarize_bundles(_bundle_stream())
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "analysis_done output=%s bundles=%s layers=%s",
        output_path,
        summary["num_bundles"],
        len(summary["layers"]),
    )
    return summary
