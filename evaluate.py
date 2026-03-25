from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Callable

from runtime_bootstrap import bootstrap_workspace_env

bootstrap_workspace_env()

import numpy as np
import pandas as pd
import torch
from beir.retrieval.evaluation import EvaluateRetrieval

from experiment_utils import (
    EPS,
    ensure_project_dirs,
    human_bytes,
    load_architecture,
    load_manifest,
    load_npz_fields,
    load_scifact,
    l2_normalize_array,
    normalize_multivector,
    pack_multivectors,
)


EVAL_TOP_K = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved Qwen3.5 MoE extraction signals on SciFact.")
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/kv_moee_experiment"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--maxsim-batch-size", type=int, default=48)
    parser.add_argument("--maxsim-query-batch-size", type=int, default=16)
    return parser.parse_args()


def pass_rows(project_root: Path, pass_name: str, kind: str) -> list[dict]:
    rows = [row for row in load_manifest(project_root) if row["kind"] == kind]
    base_dir = project_root / "embeddings" / pass_name
    for row in rows:
        row["path"] = str(base_dir / row["filename"])
    return rows


def final_last_token(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["hs_last_token"][-1].astype(np.float32, copy=False)


def final_mean_pool(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["hs_mean"][-1].astype(np.float32, copy=False)


def final_hybrid_pool(data: dict[str, np.ndarray]) -> np.ndarray:
    return 0.5 * (final_last_token(data) + final_mean_pool(data))


def moee_dense(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["routing_full_last"].astype(np.float32, copy=False).reshape(-1)


def moee_sparse_dense(data: dict[str, np.ndarray], architecture: dict) -> np.ndarray:
    num_layers = architecture["num_layers"]
    num_experts = architecture["num_experts"]
    last_indices = data["routing_indices"][:, -1, :8].astype(np.int64, copy=False)
    last_weights = data["routing_weights"][:, -1, :8].astype(np.float32, copy=False)
    vec = np.zeros(num_layers * num_experts, dtype=np.float32)
    cols = np.repeat(np.arange(num_layers) * num_experts, 8) + last_indices.reshape(-1)
    vec[cols] = last_weights.reshape(-1)
    return vec


def select_attn_offsets(architecture: dict, selector: str) -> list[int]:
    total = len(architecture["attn_layer_indices"])
    if selector == "all":
        return list(range(total))
    if selector == "last":
        return [total - 1]
    if selector == "second_half":
        return list(range(total // 2, total))
    raise ValueError(f"Unsupported attention selector: {selector}")


def pool_va_mean(data: dict[str, np.ndarray], offsets: list[int]) -> np.ndarray:
    return data["va_mean"][offsets].astype(np.float32, copy=False).mean(axis=0)


def build_aligned_wva(data: dict[str, np.ndarray], architecture: dict, offsets: list[int]) -> np.ndarray:
    num_heads = architecture["num_attention_heads"]
    head_dim = architecture["head_dim"]
    layer_vectors = []
    for offset in offsets:
        weights = data["attn_weights_last"][offset].astype(np.float32, copy=False)
        values = data["va_all_tokens"][offset].astype(np.float32, copy=False)
        values = values.reshape(values.shape[0], num_heads, head_dim).transpose(1, 0, 2)
        weights = weights / np.clip(weights.sum(axis=-1, keepdims=True), EPS, None)
        layer_vector = (weights[:, :, None] * values).sum(axis=1).reshape(-1)
        layer_vectors.append(layer_vector)
    return np.mean(np.stack(layer_vectors, axis=0), axis=0)


def build_expert_out_mean(data: dict[str, np.ndarray], layer_indices: list[int]) -> np.ndarray:
    pooled = data["expert_out_pool"][layer_indices].astype(np.float32, copy=False)
    counts = data["expert_out_counts"][layer_indices].astype(np.float32, copy=False)
    layer_vectors = []
    for layer_pool, layer_counts in zip(pooled, counts, strict=True):
        total = float(layer_counts.sum())
        if total <= 0:
            continue
        layer_vectors.append((layer_pool * layer_counts[:, None]).sum(axis=0) / total)
    if not layer_vectors:
        return np.zeros(pooled.shape[-1], dtype=np.float32)
    return np.mean(np.stack(layer_vectors, axis=0), axis=0)


def ensure_nonempty(vectors: list[np.ndarray], dim: int) -> np.ndarray:
    if not vectors:
        return np.zeros((1, dim), dtype=np.float32)
    return np.stack(vectors, axis=0).astype(np.float32, copy=False)


def build_value_expert_pool(
    data: dict[str, np.ndarray],
    offsets: list[int],
    min_count: int,
    value_dim: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for offset in offsets:
        counts = data["va_expert_counts"][offset]
        pooled = data["va_expert_pool"][offset]
        for slot in range(pooled.shape[0]):
            if counts[slot] >= min_count:
                vectors.append(pooled[slot].astype(np.float32, copy=False))
    if not vectors:
        for offset in offsets:
            counts = data["va_expert_counts"][offset]
            pooled = data["va_expert_pool"][offset]
            for slot in range(pooled.shape[0]):
                if counts[slot] > 0:
                    vectors.append(pooled[slot].astype(np.float32, copy=False))
    return ensure_nonempty(vectors, value_dim)


def build_expert_out_pool(
    data: dict[str, np.ndarray],
    layer_indices: list[int],
    min_count: int,
    hidden_size: int,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for layer_idx in layer_indices:
        counts = data["expert_out_counts"][layer_idx]
        pooled = data["expert_out_pool"][layer_idx]
        for slot in range(pooled.shape[0]):
            if counts[slot] >= min_count:
                vectors.append(pooled[slot].astype(np.float32, copy=False))
    if not vectors:
        for layer_idx in layer_indices:
            counts = data["expert_out_counts"][layer_idx]
            pooled = data["expert_out_pool"][layer_idx]
            for slot in range(pooled.shape[0]):
                if counts[slot] > 0:
                    vectors.append(pooled[slot].astype(np.float32, copy=False))
    return ensure_nonempty(vectors, hidden_size)


def build_token_level_final(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["hs_final_all_tokens"].astype(np.float32, copy=False)


def load_dense_embeddings(
    rows: list[dict],
    fields: list[str],
    builder: Callable[[dict[str, np.ndarray]], np.ndarray],
) -> tuple[list[str], np.ndarray, float, float]:
    ids = []
    embeddings = []
    build_start = time.perf_counter()
    for row in rows:
        data = load_npz_fields(row["path"], fields)
        vec = builder(data)
        vec = l2_normalize_array(vec.astype(np.float32, copy=False), axis=-1)
        ids.append(str(row["text_id"]))
        embeddings.append(vec)
    matrix = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    index_bytes = float(matrix.nbytes)
    build_time = time.perf_counter() - build_start
    return ids, matrix, build_time, index_bytes


def load_dense_group_embeddings(
    rows: list[dict],
    specs: list[tuple[str, list[str], Callable[[dict[str, np.ndarray]], np.ndarray], int]],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    ids: list[str] = []
    embeddings: dict[str, list[np.ndarray]] = {name: [] for name, *_ in specs}
    method_times: dict[str, float] = {name: 0.0 for name, *_ in specs}
    all_fields = sorted({field for _, fields, _, _ in specs for field in fields})
    load_time = 0.0

    for row in rows:
        load_started = time.perf_counter()
        data = load_npz_fields(row["path"], all_fields)
        load_time += time.perf_counter() - load_started
        ids.append(str(row["text_id"]))

        for name, _fields, builder, _vec_dim in specs:
            build_started = time.perf_counter()
            vec = builder(data)
            vec = l2_normalize_array(vec.astype(np.float32, copy=False), axis=-1)
            embeddings[name].append(vec)
            method_times[name] += time.perf_counter() - build_started

    matrices = {
        name: np.stack(vectors, axis=0).astype(np.float32, copy=False) for name, vectors in embeddings.items()
    }
    load_share = load_time / max(len(specs), 1)
    metadata = {
        name: {
            "build_time": method_times[name] + load_share,
            "index_bytes": float(matrices[name].nbytes),
            "vec_dim": float(vec_dim),
        }
        for name, _fields, _builder, vec_dim in specs
    }
    return ids, matrices, metadata


def load_multivectors(
    rows: list[dict],
    fields: list[str],
    builder: Callable[[dict[str, np.ndarray]], np.ndarray],
) -> tuple[list[str], list[np.ndarray], float, float, float]:
    ids = []
    representations = []
    build_start = time.perf_counter()
    for row in rows:
        data = load_npz_fields(row["path"], fields)
        rep = normalize_multivector(builder(data))
        ids.append(str(row["text_id"]))
        representations.append(rep.astype(np.float32, copy=False))
    build_time = time.perf_counter() - build_start
    index_bytes = float(sum(rep.nbytes for rep in representations))
    avg_vectors = float(np.mean([rep.shape[0] for rep in representations]))
    return ids, representations, build_time, index_bytes, avg_vectors


def load_multivector_group(
    rows: list[dict],
    specs: list[tuple[str, list[str], Callable[[dict[str, np.ndarray]], np.ndarray], int]],
) -> tuple[list[str], dict[str, list[np.ndarray]], dict[str, dict[str, float]]]:
    ids: list[str] = []
    representations: dict[str, list[np.ndarray]] = {name: [] for name, *_ in specs}
    method_times: dict[str, float] = {name: 0.0 for name, *_ in specs}
    all_fields = sorted({field for _, fields, _, _ in specs for field in fields})
    load_time = 0.0

    for row in rows:
        load_started = time.perf_counter()
        data = load_npz_fields(row["path"], all_fields)
        load_time += time.perf_counter() - load_started
        ids.append(str(row["text_id"]))
        for name, _fields, builder, _vec_dim in specs:
            build_started = time.perf_counter()
            rep = normalize_multivector(builder(data)).astype(np.float32, copy=False)
            representations[name].append(rep)
            method_times[name] += time.perf_counter() - build_started

    load_share = load_time / max(len(specs), 1)
    metadata = {
        name: {
            "build_time": method_times[name] + load_share,
            "index_bytes": float(sum(rep.nbytes for rep in representations[name])),
            "avg_vectors": float(np.mean([rep.shape[0] for rep in representations[name]])),
            "vec_dim": float(vec_dim),
        }
        for name, _fields, _builder, vec_dim in specs
    }
    return ids, representations, metadata


def prepare_doc_batches(doc_vectors: list[np.ndarray], batch_size: int):
    batches = []
    for start in range(0, len(doc_vectors), batch_size):
        end = min(len(doc_vectors), start + batch_size)
        packed, mask = pack_multivectors(doc_vectors[start:end])
        batches.append((start, end, packed, mask))
    return batches


def prepare_query_batches(query_vectors: list[np.ndarray], batch_size: int):
    batches = []
    for start in range(0, len(query_vectors), batch_size):
        end = min(len(query_vectors), start + batch_size)
        packed, mask = pack_multivectors(query_vectors[start:end])
        batches.append((start, end, packed, mask))
    return batches


def dense_cosine_scores_torch(query_matrix: np.ndarray, doc_matrix: np.ndarray, device: torch.device) -> np.ndarray:
    if device.type == "cpu":
        return query_matrix @ doc_matrix.T
    with torch.no_grad():
        query_tensor = torch.from_numpy(query_matrix).to(device=device, dtype=torch.float32)
        doc_tensor = torch.from_numpy(doc_matrix).to(device=device, dtype=torch.float32)
        scores = torch.matmul(query_tensor, doc_tensor.T)
        return scores.cpu().numpy().astype(np.float32, copy=False)


def topk_results_dict(query_ids: list[str], doc_ids: list[str], scores: np.ndarray, top_k: int = EVAL_TOP_K) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    limit = min(top_k, scores.shape[1])
    for row_idx, query_id in enumerate(query_ids):
        row = scores[row_idx]
        if limit >= row.shape[0]:
            order = np.argsort(-row)
        else:
            candidates = np.argpartition(-row, limit - 1)[:limit]
            order = candidates[np.argsort(-row[candidates])]
        results[query_id] = {doc_ids[col_idx]: float(row[col_idx]) for col_idx in order}
    return results


def compute_maxsim_scores(
    query_vectors: list[np.ndarray],
    doc_vectors: list[np.ndarray],
    device: torch.device,
    batch_size: int,
    query_batch_size: int,
) -> np.ndarray:
    scores = np.zeros((len(query_vectors), len(doc_vectors)), dtype=np.float32)
    doc_batches = prepare_doc_batches(doc_vectors, batch_size)
    query_batches = prepare_query_batches(query_vectors, query_batch_size)
    with torch.no_grad():
        for start, end, packed, mask in doc_batches:
            doc_tensor = torch.from_numpy(packed).to(device=device, dtype=torch.float32)
            mask_tensor = torch.from_numpy(mask).to(device=device)
            for q_start, q_end, q_packed, q_mask in query_batches:
                query_tensor = torch.from_numpy(q_packed).to(device=device, dtype=torch.float32)
                query_mask = torch.from_numpy(q_mask).to(device=device)
                sims = torch.einsum("aqd,bkd->abqk", query_tensor, doc_tensor)
                sims = sims.masked_fill(~mask_tensor[None, :, None, :], -1e30)
                batch_scores = sims.max(dim=-1).values
                batch_scores = batch_scores.masked_fill(~query_mask[:, None, :], 0.0).sum(dim=-1)
                scores[q_start:q_end, start:end] = batch_scores.cpu().numpy()
    return scores


def compute_maxsim_rerank(
    query_vectors: list[np.ndarray],
    doc_vectors: list[np.ndarray],
    candidate_indices: np.ndarray,
    device: torch.device,
    query_batch_size: int,
) -> list[dict[int, float]]:
    reranked: list[dict[int, float]] = [{} for _ in range(len(query_vectors))]
    with torch.no_grad():
        for q_start in range(0, len(query_vectors), query_batch_size):
            q_end = min(len(query_vectors), q_start + query_batch_size)
            batch_queries = query_vectors[q_start:q_end]
            batch_candidates = candidate_indices[q_start:q_end]
            packed_queries, query_mask = pack_multivectors(batch_queries)
            max_doc_len = max(doc_vectors[int(doc_idx)].shape[0] for row in batch_candidates for doc_idx in row.tolist())
            doc_dim = batch_queries[0].shape[1]
            packed_docs = np.zeros((len(batch_queries), batch_candidates.shape[1], max_doc_len, doc_dim), dtype=np.float32)
            doc_mask = np.zeros((len(batch_queries), batch_candidates.shape[1], max_doc_len), dtype=bool)
            for local_idx, candidate_row in enumerate(batch_candidates):
                for cand_idx, doc_idx in enumerate(candidate_row.tolist()):
                    rep = doc_vectors[int(doc_idx)]
                    packed_docs[local_idx, cand_idx, : rep.shape[0]] = rep
                    doc_mask[local_idx, cand_idx, : rep.shape[0]] = True

            query_tensor = torch.from_numpy(packed_queries).to(device=device, dtype=torch.float32)
            query_mask_tensor = torch.from_numpy(query_mask).to(device=device)
            doc_tensor = torch.from_numpy(packed_docs).to(device=device, dtype=torch.float32)
            doc_mask_tensor = torch.from_numpy(doc_mask).to(device=device)
            sims = torch.einsum("aqd,abkd->abqk", query_tensor, doc_tensor)
            sims = sims.masked_fill(~doc_mask_tensor[:, :, None, :], -1e30)
            batch_scores = sims.max(dim=-1).values
            batch_scores = batch_scores.masked_fill(~query_mask_tensor[:, None, :], 0.0).sum(dim=-1).cpu().numpy()
            for local_idx, candidate_row in enumerate(batch_candidates):
                reranked[q_start + local_idx] = {
                    int(doc_idx): float(score)
                    for doc_idx, score in zip(candidate_row.tolist(), batch_scores[local_idx].tolist(), strict=True)
                }
    return reranked


def evaluate_scores(
    evaluator: EvaluateRetrieval,
    qrels: dict,
    query_ids: list[str],
    doc_ids: list[str],
    scores: np.ndarray,
) -> dict[str, float]:
    results = topk_results_dict(query_ids, doc_ids, scores, top_k=EVAL_TOP_K)
    ndcg, _map, recall, _precision = evaluator.evaluate(qrels, results, [10, 100])
    return {
        "nDCG@10": float(ndcg["NDCG@10"]),
        "nDCG@100": float(ndcg["NDCG@100"]),
        "Recall@10": float(recall["Recall@10"]),
        "Recall@100": float(recall["Recall@100"]),
    }


def evaluate_reranked(
    evaluator: EvaluateRetrieval,
    qrels: dict,
    query_ids: list[str],
    doc_ids: list[str],
    candidate_scores: list[dict[int, float]],
) -> dict[str, float]:
    results = {}
    for q_idx, query_id in enumerate(query_ids):
        ranked = sorted(candidate_scores[q_idx].items(), key=lambda item: item[1], reverse=True)
        results[query_id] = {doc_ids[doc_idx]: score for doc_idx, score in ranked}
    ndcg, _map, recall, _precision = evaluator.evaluate(qrels, results, [10, 100])
    return {
        "nDCG@10": float(ndcg["NDCG@10"]),
        "nDCG@100": float(ndcg["NDCG@100"]),
        "Recall@10": float(recall["Recall@10"]),
        "Recall@100": float(recall["Recall@100"]),
    }


def main() -> None:
    args = parse_args()
    dirs = ensure_project_dirs(args.project_root)
    architecture = load_architecture(args.project_root)
    _corpus, _queries, qrels = load_scifact(dirs["datasets"])
    evaluator = EvaluateRetrieval()
    device = torch.device(args.device)

    attn_second_half = select_attn_offsets(architecture, "second_half")
    attn_all = select_attn_offsets(architecture, "all")
    attn_last = select_attn_offsets(architecture, "last")
    all_layers = list(range(architecture["num_layers"]))
    attn_layers = architecture["attn_layer_indices"]
    delta_layers = architecture["delta_layer_indices"]

    results_rows: list[dict] = []
    stats_lines: list[str] = []
    score_cache: dict[str, np.ndarray] = {}

    def record_method(
        method: str,
        metrics: dict[str, float],
        avg_vectors: float,
        vec_dim: int,
        pass_name: str,
        build_time: float,
        index_bytes: float,
        avg_nonzero_dims: float | None,
    ) -> None:
        results_rows.append(
            {
                "Method": method,
                "nDCG@10": metrics["nDCG@10"],
                "nDCG@100": metrics["nDCG@100"],
                "Recall@10": metrics["Recall@10"],
                "Recall@100": metrics["Recall@100"],
                "Num Vectors": avg_vectors,
                "Vec Dim": vec_dim,
                "Pass": pass_name,
            }
        )
        line = (
            f"{method}: build_time={build_time:.3f}s, index_size={human_bytes(index_bytes)}, "
            f"avg_vectors={avg_vectors:.3f}"
        )
        if avg_nonzero_dims is not None:
            line += f", avg_nonzero_dims={avg_nonzero_dims:.3f}"
        stats_lines.append(line)

    pass_variants = [
        {
            "pass_name": "rerouted",
            "suffix": "allattn",
            "query_rows": pass_rows(args.project_root, "rerouted", "query"),
            "doc_rows": pass_rows(args.project_root, "rerouted", "doc"),
        },
        {
            "pass_name": "paper_rerouted",
            "suffix": "paperlayers",
            "query_rows": pass_rows(args.project_root, "paper_rerouted", "query"),
            "doc_rows": pass_rows(args.project_root, "paper_rerouted", "doc"),
        },
    ]
    for variant in pass_variants:
        if not variant["query_rows"] or not variant["doc_rows"]:
            raise RuntimeError(f"Missing embeddings for pass {variant['pass_name']} in {args.project_root}")

    dense_bases = [
        ("KV-Embedding", ["hs_last_token", "hs_mean"], final_hybrid_pool, architecture["hidden_size"]),
        ("LastToken-HS", ["hs_last_token"], final_last_token, architecture["hidden_size"]),
        ("MeanPool-HS", ["hs_mean"], final_mean_pool, architecture["hidden_size"]),
        ("MoEE", ["routing_full_last"], moee_dense, architecture["num_layers"] * architecture["num_experts"]),
        ("VA", ["va_mean"], lambda data: pool_va_mean(data, attn_second_half), architecture["value_dim"]),
        ("VA-all-attn", ["va_mean"], lambda data: pool_va_mean(data, attn_all), architecture["value_dim"]),
        ("VA-last-attn", ["va_mean"], lambda data: pool_va_mean(data, attn_last), architecture["value_dim"]),
        (
            "AlignedWVA",
            ["attn_weights_last", "va_all_tokens"],
            lambda data: build_aligned_wva(data, architecture, attn_second_half),
            architecture["value_dim"],
        ),
        (
            "ExpertOut-mean",
            ["expert_out_pool", "expert_out_counts"],
            lambda data: build_expert_out_mean(data, all_layers),
            architecture["hidden_size"],
        ),
        (
            "ExpertOut-mean-attn-only",
            ["expert_out_pool", "expert_out_counts"],
            lambda data: build_expert_out_mean(data, attn_layers),
            architecture["hidden_size"],
        ),
        (
            "ExpertOut-mean-delta-only",
            ["expert_out_pool", "expert_out_counts"],
            lambda data: build_expert_out_mean(data, delta_layers),
            architecture["hidden_size"],
        ),
    ]

    dense_metadata: dict[str, dict[str, float]] = {}
    for variant in pass_variants:
        suffix = variant["suffix"]
        pass_name = variant["pass_name"]
        dense_specs = [(f"{base_name}-{suffix}", fields, builder, vec_dim) for base_name, fields, builder, vec_dim in dense_bases]
        print(f"[Evaluate] Building dense representations for {pass_name} ({len(dense_specs)} methods)")
        query_ids, query_matrices, query_metadata = load_dense_group_embeddings(variant["query_rows"], dense_specs)
        doc_ids, doc_matrices, doc_metadata = load_dense_group_embeddings(variant["doc_rows"], dense_specs)
        for method, _fields, _builder, vec_dim in dense_specs:
            query_matrix = query_matrices[method]
            doc_matrix = doc_matrices[method]
            scores = dense_cosine_scores_torch(query_matrix, doc_matrix, device)
            metrics = evaluate_scores(evaluator, qrels, query_ids, doc_ids, scores)
            score_cache[method] = scores
            build_time = query_metadata[method]["build_time"] + doc_metadata[method]["build_time"]
            index_bytes = doc_metadata[method]["index_bytes"]
            record_method(method, metrics, 1.0, vec_dim, pass_name, build_time, index_bytes, None)
            dense_metadata[method] = {"build_time": build_time, "index_bytes": index_bytes, "vec_dim": vec_dim}
            del query_matrix, doc_matrix, scores
            gc.collect()
        del query_matrices, doc_matrices
        gc.collect()

    for variant in pass_variants:
        suffix = variant["suffix"]
        pass_name = variant["pass_name"]
        print(f"[Evaluate] Building route-signature sparse proxy for {pass_name}")
        query_ids, query_matrix, q_time, _ = load_dense_embeddings(
            variant["query_rows"], ["routing_indices", "routing_weights"], lambda data: moee_sparse_dense(data, architecture)
        )
        doc_ids, doc_matrix, d_time, index_bytes = load_dense_embeddings(
            variant["doc_rows"], ["routing_indices", "routing_weights"], lambda data: moee_sparse_dense(data, architecture)
        )
        scores = dense_cosine_scores_torch(query_matrix, doc_matrix, device)
        metrics = evaluate_scores(evaluator, qrels, query_ids, doc_ids, scores)
        record_method(
            f"MoEE-sparse-{suffix}",
            metrics,
            1.0,
            architecture["num_layers"] * architecture["num_experts"],
            pass_name,
            q_time + d_time,
            index_bytes,
            float(architecture["num_layers"] * 8),
        )
        del query_matrix, doc_matrix, scores
        gc.collect()

    fusion_bases = [
        ("MoEE+VA-0.3", 0.3, 0.7),
        ("MoEE+VA-0.5", 0.5, 0.5),
        ("MoEE+VA-0.7", 0.7, 0.3),
    ]
    for variant in pass_variants:
        suffix = variant["suffix"]
        pass_name = variant["pass_name"]
        moee_scores = score_cache[f"MoEE-{suffix}"]
        va_scores = score_cache[f"VA-{suffix}"]
        expertout_scores = score_cache[f"ExpertOut-mean-{suffix}"]
        query_ids = [row["text_id"] for row in variant["query_rows"]]
        doc_ids = [row["text_id"] for row in variant["doc_rows"]]

        for base_name, moee_weight, va_weight in fusion_bases:
            method = f"{base_name}-{suffix}"
            scores = moee_weight * moee_scores + va_weight * va_scores
            metrics = evaluate_scores(evaluator, qrels, query_ids, doc_ids, scores)
            score_cache[method] = scores
            build_time = dense_metadata[f"MoEE-{suffix}"]["build_time"] + dense_metadata[f"VA-{suffix}"]["build_time"]
            index_bytes = dense_metadata[f"MoEE-{suffix}"]["index_bytes"] + dense_metadata[f"VA-{suffix}"]["index_bytes"]
            vec_dim = max(dense_metadata[f"MoEE-{suffix}"]["vec_dim"], dense_metadata[f"VA-{suffix}"]["vec_dim"])
            record_method(method, metrics, 1.0, int(vec_dim), pass_name, float(build_time), float(index_bytes), None)

        method = f"MoEE+ExpertOut-0.5-{suffix}"
        scores = 0.5 * moee_scores + 0.5 * expertout_scores
        metrics = evaluate_scores(evaluator, qrels, query_ids, doc_ids, scores)
        score_cache[method] = scores
        build_time = (
            dense_metadata[f"MoEE-{suffix}"]["build_time"] + dense_metadata[f"ExpertOut-mean-{suffix}"]["build_time"]
        )
        index_bytes = (
            dense_metadata[f"MoEE-{suffix}"]["index_bytes"] + dense_metadata[f"ExpertOut-mean-{suffix}"]["index_bytes"]
        )
        vec_dim = max(
            dense_metadata[f"MoEE-{suffix}"]["vec_dim"],
            dense_metadata[f"ExpertOut-mean-{suffix}"]["vec_dim"],
        )
        record_method(method, metrics, 1.0, int(vec_dim), pass_name, float(build_time), float(index_bytes), None)

    multivector_bases = [
        (
            "ExpertPool-Value-attn",
            ["va_expert_counts", "va_expert_pool"],
            lambda data: build_value_expert_pool(data, attn_second_half, min_count=2, value_dim=architecture["value_dim"]),
            architecture["value_dim"],
        ),
        (
            "ExpertPool-Value-attn-all",
            ["va_expert_counts", "va_expert_pool"],
            lambda data: build_value_expert_pool(data, attn_all, min_count=2, value_dim=architecture["value_dim"]),
            architecture["value_dim"],
        ),
        (
            "ExpertPool-ExpertOut-all",
            ["expert_out_counts", "expert_out_pool"],
            lambda data: build_expert_out_pool(data, all_layers, min_count=1, hidden_size=architecture["hidden_size"]),
            architecture["hidden_size"],
        ),
        (
            "ExpertPool-ExpertOut-attn",
            ["expert_out_counts", "expert_out_pool"],
            lambda data: build_expert_out_pool(data, attn_layers, min_count=1, hidden_size=architecture["hidden_size"]),
            architecture["hidden_size"],
        ),
        (
            "ExpertPool-ExpertOut-delta",
            ["expert_out_counts", "expert_out_pool"],
            lambda data: build_expert_out_pool(data, delta_layers, min_count=1, hidden_size=architecture["hidden_size"]),
            architecture["hidden_size"],
        ),
        ("TokenLevel-HS-final", ["hs_final_all_tokens"], build_token_level_final, architecture["hidden_size"]),
    ]

    multivector_cache: dict[str, tuple[list[str], list[np.ndarray], list[str], list[np.ndarray]]] = {}
    cache_base_names = {"ExpertPool-Value-attn", "ExpertPool-ExpertOut-all", "TokenLevel-HS-final"}
    for variant in pass_variants:
        suffix = variant["suffix"]
        pass_name = variant["pass_name"]
        grouped_specs: dict[tuple[str, ...], list[tuple[str, list[str], Callable[[dict[str, np.ndarray]], np.ndarray], int]]] = {}
        for base_name, fields, builder, vec_dim in multivector_bases:
            grouped_specs.setdefault(tuple(fields), []).append((f"{base_name}-{suffix}", fields, builder, vec_dim))
        for field_group, method_specs in grouped_specs.items():
            print(
                f"[Evaluate] Building multivector group for {pass_name} "
                f"fields={list(field_group)} methods={len(method_specs)}"
            )
            query_ids, query_reps, query_metadata = load_multivector_group(variant["query_rows"], method_specs)
            doc_ids, doc_reps, doc_metadata = load_multivector_group(variant["doc_rows"], method_specs)
            for method, _fields, _builder, vec_dim in method_specs:
                print(f"[Evaluate] Scoring multivector method {method}")
                query_vectors = query_reps[method]
                doc_vectors = doc_reps[method]
                scores = compute_maxsim_scores(
                    query_vectors,
                    doc_vectors,
                    device=device,
                    batch_size=args.maxsim_batch_size,
                    query_batch_size=args.maxsim_query_batch_size,
                )
                metrics = evaluate_scores(evaluator, qrels, query_ids, doc_ids, scores)
                base_name = method[: -(len(suffix) + 1)]
                if base_name in cache_base_names:
                    multivector_cache[method] = (query_ids, query_vectors, doc_ids, doc_vectors)
                score_cache[method] = scores
                record_method(
                    method,
                    metrics,
                    doc_metadata[method]["avg_vectors"],
                    vec_dim,
                    pass_name,
                    query_metadata[method]["build_time"] + doc_metadata[method]["build_time"],
                    doc_metadata[method]["index_bytes"],
                    None,
                )
                if base_name not in cache_base_names:
                    del query_vectors, doc_vectors
                del scores
                gc.collect()
            del query_reps, doc_reps
            gc.collect()

    rerank_bases = [
        ("RouteSig-then-ExpertPool-Value", "ExpertPool-Value-attn"),
        ("RouteSig-then-ExpertPool-ExpertOut", "ExpertPool-ExpertOut-all"),
        ("RouteSig-then-TokenLevel-HS", "TokenLevel-HS-final"),
    ]
    for variant in pass_variants:
        suffix = variant["suffix"]
        pass_name = variant["pass_name"]
        candidate_indices = np.argsort(-score_cache[f"MoEE-{suffix}"], axis=1)[:, :100]
        for rerank_name, base_name in rerank_bases:
            method = f"{rerank_name}-{suffix}"
            base_method = f"{base_name}-{suffix}"
            query_ids, query_vectors, doc_ids, doc_vectors = multivector_cache[base_method]
            reranked = compute_maxsim_rerank(
                query_vectors,
                doc_vectors,
                candidate_indices,
                device=device,
                query_batch_size=args.maxsim_query_batch_size,
            )
            metrics = evaluate_reranked(evaluator, qrels, query_ids, doc_ids, reranked)
            avg_vectors = float(np.mean([rep.shape[0] for rep in doc_vectors]))
            vec_dim = int(doc_vectors[0].shape[1])
            record_method(method, metrics, avg_vectors, vec_dim, pass_name, 0.0, 0.0, None)

    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(dirs["results"] / "results.csv", index=False)
    (dirs["results"] / "extraction_stats.txt").write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
