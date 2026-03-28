from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .nano_beir import build_synthetic_late_evidence_task, load_nanobeir_task
from .quantization import (
    cosine_similarity,
    fit_trinary_thresholds,
    maxsim,
    mean_pool_float,
    mean_pool_trinary,
    trinarize_array,
)
from .capture_io import iter_payload_bundles
from .types import CaptureCondition, ExampleCaptureBundle, RetrievalTask


def load_capture_payloads(capture_dir: str | Path):
    import torch

    capture_dir = Path(capture_dir)
    for path in sorted(capture_dir.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        yield path, payload


def iter_bundles(capture_dir: str | Path):
    for _, payload in load_capture_payloads(capture_dir):
        for bundle in iter_payload_bundles(payload):
            yield bundle


def _tensor_to_numpy(value, dtype=np.float32):
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy().astype(dtype, copy=False)
    except ModuleNotFoundError:  # pragma: no cover - optional local env
        pass
    return np.asarray(value, dtype=dtype)


def _get_pass(bundle: ExampleCaptureBundle, pass_name: str):
    for pass_capture in bundle.passes:
        if pass_capture.pass_name == pass_name:
            return pass_capture
    raise KeyError(f"Pass {pass_name} not found for {bundle.text_id}")


def _find_layer_capture(bundle: ExampleCaptureBundle, pass_name: str, condition: str, layer_idx: int):
    pass_capture = _get_pass(bundle, pass_name)
    captures = pass_capture.captures_by_condition.get(condition, [])
    for capture in captures:
        if capture.layer_idx == layer_idx:
            return capture
    raise KeyError(f"Missing {pass_name}/{condition}/layer {layer_idx} for {bundle.text_id}")


def _has_layer_capture(bundle: ExampleCaptureBundle, pass_name: str, condition: str, layer_idx: int) -> bool:
    try:
        _find_layer_capture(bundle, pass_name, condition, layer_idx)
        return True
    except KeyError:
        return False


def _topk_to_indicator(top_k_tensor, num_experts: int = 128):
    top_k = np.asarray(top_k_tensor, dtype=np.int64)
    seq_len, topk = top_k.shape
    out = np.zeros((seq_len, num_experts), dtype=np.float32)
    rows = np.repeat(np.arange(seq_len), topk)
    cols = top_k.reshape(-1)
    out[rows, cols] = 1.0
    return out


def _content_row_mask(bundle: ExampleCaptureBundle, expected_len: int) -> np.ndarray:
    mask = np.asarray(bundle.content_token_mask, dtype=np.int8)
    if mask.shape[0] < expected_len:
        pad = np.zeros(expected_len - mask.shape[0], dtype=np.int8)
        mask = np.concatenate([mask, pad], axis=0)
    return mask[:expected_len].astype(bool)


def _signal_percentile(signal_name: str, quantization_spec) -> float:
    if signal_name in {"router_logits", "router_logits_positive"}:
        return float(quantization_spec.router_percentile)
    if signal_name in ("attention_output", "delta_attention"):
        return float(quantization_spec.attention_percentile)
    if signal_name in {"summary_key_rot", "summary_key_raw", "summary_memory"}:
        return float(quantization_spec.attention_percentile)
    if signal_name == "pre_moe":
        return float(quantization_spec.hidden_percentile)
    if signal_name in {"value_vectors", "summary_value"}:
        return float(quantization_spec.value_percentile)
    return float(quantization_spec.default_percentile)


def _raw_signal_tokens(
    bundle: ExampleCaptureBundle,
    *,
    pass_name: str,
    condition: str,
    signal_name: str,
    layer_indices: list[int],
):
    tokens = []
    summary_signal_names = {"summary_value", "summary_key_rot", "summary_key_raw", "summary_memory"}
    for layer_idx in layer_indices:
        capture = _find_layer_capture(bundle, pass_name, condition, layer_idx)
        if signal_name == "attention_output":
            tensor = _tensor_to_numpy(capture.z_attn, dtype=np.float32)[0]
        elif signal_name == "pre_moe":
            tensor = _tensor_to_numpy(capture.h_pre_moe, dtype=np.float32)[0]
        elif signal_name in {"router_logits", "router_logits_positive"}:
            tensor = _tensor_to_numpy(capture.router_logits_pre_softmax, dtype=np.float32)[0]
        elif signal_name == "top_k_binary":
            tensor = _topk_to_indicator(_tensor_to_numpy(capture.top_k_indices, dtype=np.int16)[0])
        elif signal_name == "value_vectors":
            raw = _tensor_to_numpy(capture.v_raw, dtype=np.float32)[0]
            tensor = raw.transpose(1, 0, 2).reshape(raw.shape[1], -1)
        elif signal_name == "summary_value":
            tensor = _tensor_to_numpy(capture.final_token_v, dtype=np.float32)[0]
        elif signal_name == "summary_key_rot":
            tensor = _tensor_to_numpy(capture.final_token_k_rot, dtype=np.float32)[0]
        elif signal_name == "summary_key_raw":
            tensor = _tensor_to_numpy(capture.final_token_k_raw, dtype=np.float32)[0]
        elif signal_name == "summary_memory":
            tensor = np.concatenate(
                [
                    _tensor_to_numpy(capture.final_token_k_rot, dtype=np.float32)[0],
                    _tensor_to_numpy(capture.final_token_v, dtype=np.float32)[0],
                ],
                axis=-1,
            )
        elif signal_name == "delta_attention":
            if condition == CaptureCondition.CAUSAL.value:
                treated = _find_layer_capture(bundle, pass_name, CaptureCondition.LOCAL_PREPEND_CAUSAL_BASE.value, layer_idx)
            else:
                treated = _find_layer_capture(bundle, pass_name, CaptureCondition.PROPAGATED.value, layer_idx)
            tensor = _tensor_to_numpy(treated.z_attn, dtype=np.float32)[0] - _tensor_to_numpy(capture.z_attn, dtype=np.float32)[0]
        else:
            raise ValueError(f"Unsupported signal: {signal_name}")
        if signal_name not in summary_signal_names:
            mask = _content_row_mask(bundle, tensor.shape[0])
            tensor = tensor[mask]
        tokens.append(tensor)
    if not tokens:
        return np.zeros((0, 0), dtype=np.float32)
    if signal_name in summary_signal_names:
        return np.concatenate(tokens, axis=1)
    return np.concatenate(tokens, axis=1)


def extract_signal_tokens(
    bundle: ExampleCaptureBundle,
    *,
    pass_name: str,
    condition: str,
    signal_name: str,
    layer_indices: list[int],
    quantization_spec,
    thresholds: np.ndarray | None = None,
):
    raw = _raw_signal_tokens(
        bundle,
        pass_name=pass_name,
        condition=condition,
        signal_name=signal_name,
        layer_indices=layer_indices,
    )
    if signal_name == "top_k_binary":
        return raw.astype(np.float32)
    return trinarize_array(
        raw,
        threshold_percentile=_signal_percentile(signal_name, quantization_spec),
        thresholds=thresholds,
        positive_only=signal_name == "router_logits_positive",
    )


def ndcg_at_k(results: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]], k: int = 10):
    scores = []
    for query_id, rels in qrels.items():
        ranked = sorted(results.get(query_id, {}).items(), key=lambda item: item[1], reverse=True)[:k]
        dcg = 0.0
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            dcg += rels.get(doc_id, 0) / math.log2(rank + 1)
        ideal = sorted(rels.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)) if scores else 0.0


def score_single_vector(query_vectors, doc_vectors):
    results = {}
    for query_id, query_vec in query_vectors.items():
        results[query_id] = {doc_id: cosine_similarity(query_vec, doc_vec) for doc_id, doc_vec in doc_vectors.items()}
    return results


def score_multivector(query_vectors, doc_vectors):
    results = {}
    for query_id, query_tokens in query_vectors.items():
        results[query_id] = {doc_id: maxsim(query_tokens, doc_tokens) for doc_id, doc_tokens in doc_vectors.items()}
    return results


def _split_doc_query_bundles(task: RetrievalTask, bundle_lookup: dict[str, ExampleCaptureBundle]):
    doc_bundles = {doc_id: bundle_lookup[doc_id] for doc_id in task.corpus if doc_id in bundle_lookup}
    query_bundles = {query_id: bundle_lookup[query_id] for query_id in task.queries if query_id in bundle_lookup}
    return doc_bundles, query_bundles


def evaluate_signal(
    *,
    capture_dir: str | Path,
    task_name: str,
    repo_name: str,
    pass_name: str,
    condition: str,
    signal_name: str,
    layer_indices: list[int],
    quantization_spec,
):
    task = load_nanobeir_task(repo_name, task_name)
    bundle_lookup = {bundle.text_id: bundle for bundle in iter_bundles(capture_dir)}
    doc_bundles, query_bundles = _split_doc_query_bundles(task, bundle_lookup)

    raw_doc_tokens = {
        doc_id: _raw_signal_tokens(
            bundle,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            layer_indices=layer_indices,
        )
        for doc_id, bundle in doc_bundles.items()
    }
    raw_query_tokens = {
        query_id: _raw_signal_tokens(
            bundle,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            layer_indices=layer_indices,
        )
        for query_id, bundle in query_bundles.items()
    }

    threshold_source = [tokens for tokens in list(raw_doc_tokens.values()) + list(raw_query_tokens.values()) if tokens.size > 0]
    thresholds = None
    if signal_name != "top_k_binary" and threshold_source:
        concat = np.concatenate(threshold_source, axis=0)
        nonzero_fraction = 1.0 - (_signal_percentile(signal_name, quantization_spec) / 100.0)
        thresholds = fit_trinary_thresholds(concat, nonzero_fraction=nonzero_fraction, axis=0)

    doc_token_vectors = {
        doc_id: extract_signal_tokens(
            bundle,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            layer_indices=layer_indices,
            quantization_spec=quantization_spec,
            thresholds=thresholds,
        )
        for doc_id, bundle in doc_bundles.items()
    }
    query_token_vectors = {
        query_id: extract_signal_tokens(
            bundle,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            layer_indices=layer_indices,
            quantization_spec=quantization_spec,
            thresholds=thresholds,
        )
        for query_id, bundle in query_bundles.items()
    }

    doc_single_qtp = {doc_id: mean_pool_trinary(tokens) for doc_id, tokens in doc_token_vectors.items()}
    query_single_qtp = {query_id: mean_pool_trinary(tokens) for query_id, tokens in query_token_vectors.items()}
    single_scores_qtp = score_single_vector(query_single_qtp, doc_single_qtp)
    multi_scores_trinary = score_multivector(query_token_vectors, doc_token_vectors)

    doc_single_float = {doc_id: mean_pool_float(tokens) for doc_id, tokens in raw_doc_tokens.items()}
    query_single_float = {query_id: mean_pool_float(tokens) for query_id, tokens in raw_query_tokens.items()}
    single_scores_float = score_single_vector(query_single_float, doc_single_float)
    multi_scores_float = score_multivector(raw_query_tokens, raw_doc_tokens)

    pooled_doc_trinary = {
        doc_id: trinarize_array(vec, threshold_percentile=_signal_percentile(signal_name, quantization_spec), thresholds=thresholds).astype(np.float32)
        if signal_name != "top_k_binary"
        else vec.astype(np.float32)
        for doc_id, vec in doc_single_float.items()
    }
    pooled_query_trinary = {
        query_id: trinarize_array(vec, threshold_percentile=_signal_percentile(signal_name, quantization_spec), thresholds=thresholds).astype(np.float32)
        if signal_name != "top_k_binary"
        else vec.astype(np.float32)
        for query_id, vec in query_single_float.items()
    }
    single_scores_ptq = score_single_vector(pooled_query_trinary, pooled_doc_trinary)

    return {
        "task_name": task_name,
        "signal_name": signal_name,
        "encoding_variant": "positive_only_trinary" if signal_name == "router_logits_positive" else "signed_trinary_or_float",
        "condition": condition,
        "layer_indices": layer_indices,
        "thresholds_shape": None if thresholds is None else list(np.asarray(thresholds).shape),
        "float_single_vector_ndcg_at_10": ndcg_at_k(single_scores_float, task.qrels, k=10),
        "float_multivector_ndcg_at_10": ndcg_at_k(multi_scores_float, task.qrels, k=10),
        "trinary_single_vector_quantize_then_pool_ndcg_at_10": ndcg_at_k(single_scores_qtp, task.qrels, k=10),
        "trinary_single_vector_pool_then_quantize_ndcg_at_10": ndcg_at_k(single_scores_ptq, task.qrels, k=10),
        "trinary_multivector_ndcg_at_10": ndcg_at_k(multi_scores_trinary, task.qrels, k=10),
        "single_vector_ndcg_at_10": ndcg_at_k(single_scores_qtp, task.qrels, k=10),
        "multivector_ndcg_at_10": ndcg_at_k(multi_scores_trinary, task.qrels, k=10),
        "float_single_vector_scores": single_scores_float,
        "float_multivector_scores": multi_scores_float,
        "single_vector_scores": single_scores_qtp,
        "multivector_scores": multi_scores_trinary,
    }


def evaluate_signal_family(
    *,
    capture_dir: str | Path,
    task_name: str,
    repo_name: str,
    pass_name: str,
    condition: str,
    signal_name: str,
    selected_layers: list[int],
    topn_layers_for_grouping: int,
    layer_grouping_policy: str,
    fixed_group_layers: list[int],
    quantization_spec,
):
    bundle_lookup = {bundle.text_id: bundle for bundle in iter_bundles(capture_dir)}
    sample_bundle = next(iter(bundle_lookup.values()), None)
    available_layers = [
        layer_idx
        for layer_idx in selected_layers
        if sample_bundle is not None and _has_layer_capture(sample_bundle, pass_name, condition, layer_idx)
    ]
    if not available_layers:
        return {
            "skipped": True,
            "reason": f"No available layers for signal={signal_name}, pass={pass_name}, condition={condition}",
            "signal_name": signal_name,
            "condition": condition,
            "pass_name": pass_name,
            "requested_layers": selected_layers,
        }
    per_layer = []
    for layer_idx in available_layers:
        layer_result = evaluate_signal(
            capture_dir=capture_dir,
            task_name=task_name,
            repo_name=repo_name,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            layer_indices=[layer_idx],
            quantization_spec=quantization_spec,
        )
        per_layer.append(layer_result)
    ranked_single = sorted(per_layer, key=lambda item: item["single_vector_ndcg_at_10"], reverse=True)
    ranked_multi = sorted(per_layer, key=lambda item: item["multivector_ndcg_at_10"], reverse=True)
    topn = max(1, int(topn_layers_for_grouping))
    topn_single_layers = [entry["layer_indices"][0] for entry in ranked_single[:topn]]
    topn_multi_layers = [entry["layer_indices"][0] for entry in ranked_multi[:topn]]
    if layer_grouping_policy == "in_task_topn_from_per_layer_scores":
        grouped_single_layers = topn_single_layers
        grouped_multi_layers = topn_multi_layers
    elif layer_grouping_policy == "fixed_group_layers":
        fixed_layers = [layer for layer in fixed_group_layers if layer in available_layers]
        if not fixed_layers:
            raise ValueError("layer_grouping_policy=fixed_group_layers requires at least one available fixed_group_layer")
        grouped_single_layers = fixed_layers
        grouped_multi_layers = fixed_layers
    elif layer_grouping_policy == "all_selected_layers":
        grouped_single_layers = available_layers
        grouped_multi_layers = available_layers
    else:
        raise ValueError(f"Unknown layer_grouping_policy: {layer_grouping_policy}")
    grouped = {
        "per_layer": per_layer,
        "layer_grouping_topn": topn,
        "layer_selection_policy": layer_grouping_policy,
        "diagnostic_in_task_topn_single_layers": topn_single_layers,
        "diagnostic_in_task_topn_multi_layers": topn_multi_layers,
        "grouped_single_layers": grouped_single_layers,
        "grouped_multi_layers": grouped_multi_layers,
        "grouped_single": evaluate_signal(
            capture_dir=capture_dir,
            task_name=task_name,
            repo_name=repo_name,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            layer_indices=grouped_single_layers,
            quantization_spec=quantization_spec,
        ),
        "grouped_multi": evaluate_signal(
            capture_dir=capture_dir,
            task_name=task_name,
            repo_name=repo_name,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            layer_indices=grouped_multi_layers,
            quantization_spec=quantization_spec,
        ),
        "all_layers": evaluate_signal(
            capture_dir=capture_dir,
            task_name=task_name,
            repo_name=repo_name,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            layer_indices=available_layers,
            quantization_spec=quantization_spec,
        ),
    }
    return grouped


def linear_fusion(scores_a, scores_b, alpha: float):
    fused = {}
    for query_id in scores_a:
        fused[query_id] = {}
        for doc_id in scores_a[query_id]:
            fused[query_id][doc_id] = scores_a[query_id].get(doc_id, 0.0) + alpha * scores_b[query_id].get(doc_id, 0.0)
    return fused


def reciprocal_rank_fusion(scores_a, scores_b, k: int = 60):
    fused = {}
    for query_id in scores_a:
        rank_a = {doc_id: rank for rank, (doc_id, _) in enumerate(sorted(scores_a[query_id].items(), key=lambda item: item[1], reverse=True), start=1)}
        rank_b = {doc_id: rank for rank, (doc_id, _) in enumerate(sorted(scores_b[query_id].items(), key=lambda item: item[1], reverse=True), start=1)}
        docs = set(rank_a) | set(rank_b)
        fused[query_id] = {doc_id: 1.0 / (k + rank_a.get(doc_id, 10**9)) + 1.0 / (k + rank_b.get(doc_id, 10**9)) for doc_id in docs}
    return fused


def interleave_rankings(scores_a, scores_b, limit: int = 10, attn_first: bool = True):
    fused = {}
    for query_id in scores_a:
        ranked_a = [doc_id for doc_id, _ in sorted(scores_a[query_id].items(), key=lambda item: item[1], reverse=True)]
        ranked_b = [doc_id for doc_id, _ in sorted(scores_b[query_id].items(), key=lambda item: item[1], reverse=True)]
        streams = (ranked_a, ranked_b) if attn_first else (ranked_b, ranked_a)
        chosen = []
        seen = set()
        index = 0
        while len(chosen) < limit and (index < len(streams[0]) or index < len(streams[1])):
            for stream in streams:
                if index >= len(stream):
                    continue
                doc_id = stream[index]
                if doc_id in seen:
                    continue
                chosen.append(doc_id)
                seen.add(doc_id)
                if len(chosen) >= limit:
                    break
            index += 1
        fused[query_id] = {doc_id: float(limit - rank) for rank, doc_id in enumerate(chosen)}
    return fused


def candidate_union_rerank(scores_retriever, scores_reranker, candidate_pool_k: int = 100):
    fused = {}
    for query_id in scores_retriever:
        ranked_retriever = [doc_id for doc_id, _ in sorted(scores_retriever[query_id].items(), key=lambda item: item[1], reverse=True)[:candidate_pool_k]]
        ranked_other = [doc_id for doc_id, _ in sorted(scores_reranker[query_id].items(), key=lambda item: item[1], reverse=True)[:candidate_pool_k]]
        union = list(dict.fromkeys(ranked_retriever + ranked_other))
        fused[query_id] = {doc_id: scores_reranker[query_id].get(doc_id, float("-inf")) for doc_id in union}
    return fused


def oracle_analysis(scores_a, scores_b, qrels):
    weak = {}
    strong = {}
    overlap = {"shared": 0, "unique_to_a": 0, "unique_to_b": 0, "missed_by_both": 0}
    for query_id, rels in qrels.items():
        ranked_a = [doc_id for doc_id, _ in sorted(scores_a[query_id].items(), key=lambda item: item[1], reverse=True)[:10]]
        ranked_b = [doc_id for doc_id, _ in sorted(scores_b[query_id].items(), key=lambda item: item[1], reverse=True)[:10]]
        relevant_a = {doc_id for doc_id in ranked_a if rels.get(doc_id, 0) > 0}
        relevant_b = {doc_id for doc_id in ranked_b if rels.get(doc_id, 0) > 0}
        shared = relevant_a & relevant_b
        unique_a = relevant_a - relevant_b
        unique_b = relevant_b - relevant_a
        overlap["shared"] += len(shared)
        overlap["unique_to_a"] += len(unique_a)
        overlap["unique_to_b"] += len(unique_b)
        overlap["missed_by_both"] += max(0, len(rels) - len(shared) - len(unique_a) - len(unique_b))
        weak[query_id] = scores_a[query_id] if ndcg_at_k({query_id: scores_a[query_id]}, {query_id: rels}) >= ndcg_at_k({query_id: scores_b[query_id]}, {query_id: rels}) else scores_b[query_id]
        union_rank = list(dict.fromkeys([doc_id for doc_id in ranked_a + ranked_b if rels.get(doc_id, 0) > 0]))
        strong[query_id] = {doc_id: float(10 - rank) for rank, doc_id in enumerate(union_rank[:10])}
    return {
        "weak_oracle_ndcg_at_10": ndcg_at_k(weak, qrels, 10),
        "strong_union_oracle_ndcg_at_10": ndcg_at_k(strong, qrels, 10),
        "candidate_overlap": overlap,
    }


def evaluate_pairwise_fusion(scores_a, scores_b, qrels, fusion_weights, rrf_k: int = 60, candidate_pool_k: int = 100):
    best_linear = None
    for alpha in fusion_weights:
        fused = linear_fusion(scores_a, scores_b, alpha)
        ndcg = ndcg_at_k(fused, qrels, 10)
        row = {"alpha": float(alpha), "ndcg_at_10": ndcg}
        if best_linear is None or ndcg > best_linear["ndcg_at_10"]:
            best_linear = row
    rrf_scores = reciprocal_rank_fusion(scores_a, scores_b, rrf_k)
    interleave_a = interleave_rankings(scores_a, scores_b, limit=10, attn_first=True)
    interleave_b = interleave_rankings(scores_a, scores_b, limit=10, attn_first=False)
    return {
        "best_linear": best_linear,
        "rrf_ndcg_at_10": ndcg_at_k(rrf_scores, qrels, 10),
        "interleave_a_first_ndcg_at_10": ndcg_at_k(interleave_a, qrels, 10),
        "interleave_b_first_ndcg_at_10": ndcg_at_k(interleave_b, qrels, 10),
        "a_retrieve_b_rerank_ndcg_at_10": ndcg_at_k(candidate_union_rerank(scores_a, scores_b, candidate_pool_k), qrels, 10),
        "b_retrieve_a_rerank_ndcg_at_10": ndcg_at_k(candidate_union_rerank(scores_b, scores_a, candidate_pool_k), qrels, 10),
        "oracle": oracle_analysis(scores_a, scores_b, qrels),
    }


def evaluate_task_suite(
    *,
    capture_dir: str | Path,
    repo_name: str,
    task_name: str,
    quantization_spec,
    selected_layers: list[int],
    fusion_weights: list[float],
    rrf_k: int,
    candidate_pool_k: int,
    topn_layers_for_grouping: int,
    layer_grouping_policy: str,
    fixed_group_layers: list[int],
):
    signals = [
        ("attention_output", "pass1", CaptureCondition.CAUSAL.value),
        ("attention_output", "pass2", CaptureCondition.PROPAGATED.value),
        ("pre_moe", "pass1", CaptureCondition.CAUSAL.value),
        ("pre_moe", "pass2", CaptureCondition.PROPAGATED.value),
        ("router_logits", "pass1", CaptureCondition.CAUSAL.value),
        ("router_logits", "pass2", CaptureCondition.PROPAGATED.value),
        ("router_logits_positive", "pass1", CaptureCondition.CAUSAL.value),
        ("router_logits_positive", "pass2", CaptureCondition.PROPAGATED.value),
        ("top_k_binary", "pass1", CaptureCondition.CAUSAL.value),
        ("top_k_binary", "pass2", CaptureCondition.PROPAGATED.value),
        ("value_vectors", "pass1", CaptureCondition.CAUSAL.value),
        ("value_vectors", "pass2", CaptureCondition.PROPAGATED.value),
        ("summary_value", "pass1", CaptureCondition.CAUSAL.value),
        ("summary_value", "pass2", CaptureCondition.PROPAGATED.value),
        ("summary_key_rot", "pass1", CaptureCondition.CAUSAL.value),
        ("summary_key_rot", "pass2", CaptureCondition.PROPAGATED.value),
        ("summary_key_raw", "pass1", CaptureCondition.CAUSAL.value),
        ("summary_key_raw", "pass2", CaptureCondition.PROPAGATED.value),
        ("summary_memory", "pass1", CaptureCondition.CAUSAL.value),
        ("summary_memory", "pass2", CaptureCondition.PROPAGATED.value),
        ("delta_attention", "pass1", CaptureCondition.CAUSAL.value),
        ("delta_attention", "pass2", CaptureCondition.LOCAL_NOPREPEND_PROPAGATED_BASE.value),
    ]
    suite = {"task_name": task_name, "signals": {}, "pairwise_fusions": {}}
    for signal_name, pass_name, condition in signals:
        signal_key = f"{signal_name}__{pass_name}__{condition}"
        suite["signals"][signal_key] = evaluate_signal_family(
            capture_dir=capture_dir,
            task_name=task_name,
            repo_name=repo_name,
            pass_name=pass_name,
            condition=condition,
            signal_name=signal_name,
            selected_layers=selected_layers,
            topn_layers_for_grouping=topn_layers_for_grouping,
            layer_grouping_policy=layer_grouping_policy,
            fixed_group_layers=fixed_group_layers,
            quantization_spec=quantization_spec,
        )
    task = load_nanobeir_task(repo_name, task_name)
    keys = [key for key, value in suite["signals"].items() if not value.get("skipped")]
    for left_index, left_signal in enumerate(keys):
        left_scores = suite["signals"][left_signal]["grouped_multi"]["multivector_scores"]
        for right_signal in keys[left_index + 1 :]:
            right_scores = suite["signals"][right_signal]["grouped_multi"]["multivector_scores"]
            suite["pairwise_fusions"][f"{left_signal}__{right_signal}"] = evaluate_pairwise_fusion(
                left_scores,
                right_scores,
                task.qrels,
                fusion_weights=fusion_weights,
                rrf_k=rrf_k,
                candidate_pool_k=candidate_pool_k,
            )
    late_task = build_synthetic_late_evidence_task(task)
    suite["late_evidence_task"] = late_task.dataset_name
    return suite


def save_json(path: str | Path, payload: dict[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
