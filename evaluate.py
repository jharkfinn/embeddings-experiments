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
import scipy.sparse as sp
import torch
from beir.retrieval.evaluation import EvaluateRetrieval

from experiment_utils import (
    EPS,
    dense_cosine_scores,
    ensure_project_dirs,
    human_bytes,
    load_architecture,
    load_manifest,
    load_npz_fields,
    load_scifact,
    l2_normalize_array,
    make_results_dict,
    normalize_multivector,
    pack_multivectors,
    sparse_cosine_scores,
    sparse_stack,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved Qwen3.5 MoE extraction signals on SciFact.")
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/kv_moee_experiment"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--maxsim-batch-size", type=int, default=48)
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


def moee_sparse(data: dict[str, np.ndarray], architecture: dict) -> sp.csr_matrix:
    num_layers = architecture["num_layers"]
    num_experts = architecture["num_experts"]
    last_indices = data["routing_indices"][:, -1, :8].astype(np.int64, copy=False)
    last_weights = data["routing_weights"][:, -1, :8].astype(np.float32, copy=False)
    cols = np.repeat(np.arange(num_layers) * num_experts, 8) + last_indices.reshape(-1)
    vals = last_weights.reshape(-1)
    rows = np.zeros_like(cols)
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(1, num_layers * num_experts), dtype=np.float32)
    norm = np.sqrt(mat.multiply(mat).sum())
    if norm > 0:
        mat = mat / norm
    return mat


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


def load_sparse_embeddings(
    rows: list[dict],
    fields: list[str],
    builder: Callable[[dict[str, np.ndarray]], sp.csr_matrix],
) -> tuple[list[str], sp.csr_matrix, float, float, float]:
    ids = []
    vectors = []
    nnz = []
    build_start = time.perf_counter()
    for row in rows:
        data = load_npz_fields(row["path"], fields)
        vec = builder(data)
        ids.append(str(row["text_id"]))
        vectors.append(vec)
        nnz.append(float(vec.nnz))
    matrix = sparse_stack(vectors)
    index_bytes = float(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    build_time = time.perf_counter() - build_start
    return ids, matrix, build_time, index_bytes, float(np.mean(nnz))


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


def prepare_doc_batches(doc_vectors: list[np.ndarray], batch_size: int):
    batches = []
    for start in range(0, len(doc_vectors), batch_size):
        end = min(len(doc_vectors), start + batch_size)
        packed, mask = pack_multivectors(doc_vectors[start:end])
        batches.append((start, end, packed, mask))
    return batches


def compute_maxsim_scores(
    query_vectors: list[np.ndarray],
    doc_vectors: list[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores = np.zeros((len(query_vectors), len(doc_vectors)), dtype=np.float32)
    doc_batches = prepare_doc_batches(doc_vectors, batch_size)
    with torch.no_grad():
        for q_idx, query in enumerate(query_vectors):
            query_tensor = torch.from_numpy(query).to(device=device, dtype=torch.float32)
            for start, end, packed, mask in doc_batches:
                doc_tensor = torch.from_numpy(packed).to(device=device, dtype=torch.float32)
                mask_tensor = torch.from_numpy(mask).to(device=device)
                sims = torch.einsum("qd,bkd->bqk", query_tensor, doc_tensor)
                sims = sims.masked_fill(~mask_tensor[:, None, :], -1e30)
                batch_scores = sims.max(dim=-1).values.sum(dim=-1)
                scores[q_idx, start:end] = batch_scores.cpu().numpy()
    return scores


def compute_maxsim_rerank(
    query_vectors: list[np.ndarray],
    doc_vectors: list[np.ndarray],
    candidate_indices: list[np.ndarray],
    device: torch.device,
) -> list[dict[int, float]]:
    reranked: list[dict[int, float]] = []
    with torch.no_grad():
        for q_idx, query in enumerate(query_vectors):
            query_tensor = torch.from_numpy(query).to(device=device, dtype=torch.float32)
            candidates = candidate_indices[q_idx]
            subset = [doc_vectors[int(doc_idx)] for doc_idx in candidates.tolist()]
            packed, mask = pack_multivectors(subset)
            doc_tensor = torch.from_numpy(packed).to(device=device, dtype=torch.float32)
            mask_tensor = torch.from_numpy(mask).to(device=device)
            sims = torch.einsum("qd,bkd->bqk", query_tensor, doc_tensor)
            sims = sims.masked_fill(~mask_tensor[:, None, :], -1e30)
            batch_scores = sims.max(dim=-1).values.sum(dim=-1).cpu().numpy()
            reranked.append({int(doc_idx): float(score) for doc_idx, score in zip(candidates, batch_scores, strict=True)})
    return reranked


def evaluate_scores(
    evaluator: EvaluateRetrieval,
    qrels: dict,
    query_ids: list[str],
    doc_ids: list[str],
    scores: np.ndarray,
) -> dict[str, float]:
    results = make_results_dict(query_ids, doc_ids, scores)
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

    causal_doc_rows = pass_rows(args.project_root, "causal", "doc")
    causal_query_rows = pass_rows(args.project_root, "causal", "query")
    rerouted_doc_rows = pass_rows(args.project_root, "rerouted", "doc")
    rerouted_query_rows = pass_rows(args.project_root, "rerouted", "query")
    paper_causal_doc_rows = pass_rows(args.project_root, "paper_causal", "doc")
    paper_causal_query_rows = pass_rows(args.project_root, "paper_causal", "query")
    paper_rerouted_doc_rows = pass_rows(args.project_root, "paper_rerouted", "doc")
    paper_rerouted_query_rows = pass_rows(args.project_root, "paper_rerouted", "query")

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

    dense_specs = [
        (
            "LastToken-HS",
            "causal",
            causal_query_rows,
            causal_doc_rows,
            ["hs_last_token"],
            final_last_token,
            architecture["hidden_size"],
        ),
        (
            "MeanPool-HS",
            "causal",
            causal_query_rows,
            causal_doc_rows,
            ["hs_mean"],
            final_mean_pool,
            architecture["hidden_size"],
        ),
        (
            "MoEE-causal",
            "causal",
            causal_query_rows,
            causal_doc_rows,
            ["routing_full_last"],
            moee_dense,
            architecture["num_layers"] * architecture["num_experts"],
        ),
        (
            "LastToken-HS-rerouted",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["hs_last_token"],
            final_last_token,
            architecture["hidden_size"],
        ),
        (
            "MeanPool-HS-rerouted",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["hs_mean"],
            final_mean_pool,
            architecture["hidden_size"],
        ),
        (
            "MoEE-rerouted",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["routing_full_last"],
            moee_dense,
            architecture["num_layers"] * architecture["num_experts"],
        ),
        (
            "VA-rerouted",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["va_mean"],
            lambda data: pool_va_mean(data, attn_second_half),
            architecture["value_dim"],
        ),
        (
            "VA-rerouted-all-attn",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["va_mean"],
            lambda data: pool_va_mean(data, attn_all),
            architecture["value_dim"],
        ),
        (
            "VA-rerouted-last-attn",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["va_mean"],
            lambda data: pool_va_mean(data, attn_last),
            architecture["value_dim"],
        ),
        (
            "AlignedWVA-rerouted",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["attn_weights_last", "va_all_tokens"],
            lambda data: build_aligned_wva(data, architecture, attn_second_half),
            architecture["value_dim"],
        ),
        (
            "ExpertOut-mean-rerouted",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["expert_out_pool", "expert_out_counts"],
            lambda data: build_expert_out_mean(data, all_layers),
            architecture["hidden_size"],
        ),
        (
            "ExpertOut-mean-rerouted-attn-only",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["expert_out_pool", "expert_out_counts"],
            lambda data: build_expert_out_mean(data, attn_layers),
            architecture["hidden_size"],
        ),
        (
            "ExpertOut-mean-rerouted-delta-only",
            "rerouted",
            rerouted_query_rows,
            rerouted_doc_rows,
            ["expert_out_pool", "expert_out_counts"],
            lambda data: build_expert_out_mean(data, delta_layers),
            architecture["hidden_size"],
        ),
    ]

    paper_dense_specs = [
        (
            "KV-Embedding-paper-causal",
            "paper_causal",
            paper_causal_query_rows,
            paper_causal_doc_rows,
            ["hs_last_token", "hs_mean"],
            final_hybrid_pool,
            architecture["hidden_size"],
        ),
        (
            "KV-Embedding-paper-rerouted",
            "paper_rerouted",
            paper_rerouted_query_rows,
            paper_rerouted_doc_rows,
            ["hs_last_token", "hs_mean"],
            final_hybrid_pool,
            architecture["hidden_size"],
        ),
    ]

    for method, pass_name, _query_rows, _doc_rows, fields, builder, vec_dim in list(dense_specs):
        if pass_name == "causal":
            paper_pass_name = "paper_causal"
            paper_query_rows = paper_causal_query_rows
            paper_doc_rows = paper_causal_doc_rows
        else:
            paper_pass_name = "paper_rerouted"
            paper_query_rows = paper_rerouted_query_rows
            paper_doc_rows = paper_rerouted_doc_rows
        paper_dense_specs.append(
            (
                f"{method}-papercond",
                paper_pass_name,
                paper_query_rows,
                paper_doc_rows,
                fields,
                builder,
                vec_dim,
            )
        )

    dense_specs.extend(paper_dense_specs)

    dense_metadata: dict[str, dict[str, float]] = {}
    for method, pass_name, query_rows, doc_rows, fields, builder, vec_dim in dense_specs:
        query_ids, query_matrix, q_time, _ = load_dense_embeddings(query_rows, fields, builder)
        doc_ids, doc_matrix, d_time, index_bytes = load_dense_embeddings(doc_rows, fields, builder)
        scores = dense_cosine_scores(query_matrix, doc_matrix)
        metrics = evaluate_scores(evaluator, qrels, query_ids, doc_ids, scores)
        score_cache[method] = scores
        build_time = q_time + d_time
        record_method(method, metrics, 1.0, vec_dim, pass_name, build_time, index_bytes, None)
        dense_metadata[method] = {"build_time": build_time, "index_bytes": index_bytes, "vec_dim": vec_dim}
        del query_matrix, doc_matrix, scores
        gc.collect()

    sparse_query_ids, sparse_query_matrix, sparse_q_time, _, sparse_q_nnz = load_sparse_embeddings(
        causal_query_rows, ["routing_indices", "routing_weights"], lambda data: moee_sparse(data, architecture)
    )
    sparse_doc_ids, sparse_doc_matrix, sparse_d_time, sparse_index_bytes, sparse_doc_nnz = load_sparse_embeddings(
        causal_doc_rows, ["routing_indices", "routing_weights"], lambda data: moee_sparse(data, architecture)
    )
    sparse_scores = sparse_cosine_scores(sparse_query_matrix, sparse_doc_matrix)
    sparse_metrics = evaluate_scores(evaluator, qrels, sparse_query_ids, sparse_doc_ids, sparse_scores)
    record_method(
        "MoEE-causal-sparse",
        sparse_metrics,
        1.0,
        architecture["num_layers"] * architecture["num_experts"],
        "causal",
        sparse_q_time + sparse_d_time,
        sparse_index_bytes,
        sparse_doc_nnz,
    )
    del sparse_query_matrix, sparse_doc_matrix, sparse_scores
    gc.collect()

    paper_sparse_query_ids, paper_sparse_query_matrix, paper_sparse_q_time, _, _ = load_sparse_embeddings(
        paper_causal_query_rows, ["routing_indices", "routing_weights"], lambda data: moee_sparse(data, architecture)
    )
    paper_sparse_doc_ids, paper_sparse_doc_matrix, paper_sparse_d_time, paper_sparse_index_bytes, paper_sparse_doc_nnz = load_sparse_embeddings(
        paper_causal_doc_rows, ["routing_indices", "routing_weights"], lambda data: moee_sparse(data, architecture)
    )
    paper_sparse_scores = sparse_cosine_scores(paper_sparse_query_matrix, paper_sparse_doc_matrix)
    paper_sparse_metrics = evaluate_scores(evaluator, qrels, paper_sparse_query_ids, paper_sparse_doc_ids, paper_sparse_scores)
    record_method(
        "MoEE-causal-sparse-papercond",
        paper_sparse_metrics,
        1.0,
        architecture["num_layers"] * architecture["num_experts"],
        "paper_causal",
        paper_sparse_q_time + paper_sparse_d_time,
        paper_sparse_index_bytes,
        paper_sparse_doc_nnz,
    )
    del paper_sparse_query_matrix, paper_sparse_doc_matrix, paper_sparse_scores
    gc.collect()

    moee_scores = score_cache["MoEE-rerouted"]
    va_scores = score_cache["VA-rerouted"]
    expertout_scores = score_cache["ExpertOut-mean-rerouted"]
    fusion_specs = [
        (
            "MoEE+VA-0.3",
            0.3 * moee_scores + 0.7 * va_scores,
            dense_metadata["MoEE-rerouted"]["build_time"] + dense_metadata["VA-rerouted"]["build_time"],
            dense_metadata["MoEE-rerouted"]["index_bytes"] + dense_metadata["VA-rerouted"]["index_bytes"],
            max(dense_metadata["MoEE-rerouted"]["vec_dim"], dense_metadata["VA-rerouted"]["vec_dim"]),
        ),
        (
            "MoEE+VA-0.5",
            0.5 * moee_scores + 0.5 * va_scores,
            dense_metadata["MoEE-rerouted"]["build_time"] + dense_metadata["VA-rerouted"]["build_time"],
            dense_metadata["MoEE-rerouted"]["index_bytes"] + dense_metadata["VA-rerouted"]["index_bytes"],
            max(dense_metadata["MoEE-rerouted"]["vec_dim"], dense_metadata["VA-rerouted"]["vec_dim"]),
        ),
        (
            "MoEE+VA-0.7",
            0.7 * moee_scores + 0.3 * va_scores,
            dense_metadata["MoEE-rerouted"]["build_time"] + dense_metadata["VA-rerouted"]["build_time"],
            dense_metadata["MoEE-rerouted"]["index_bytes"] + dense_metadata["VA-rerouted"]["index_bytes"],
            max(dense_metadata["MoEE-rerouted"]["vec_dim"], dense_metadata["VA-rerouted"]["vec_dim"]),
        ),
        (
            "MoEE+ExpertOut-0.5",
            0.5 * moee_scores + 0.5 * expertout_scores,
            dense_metadata["MoEE-rerouted"]["build_time"] + dense_metadata["ExpertOut-mean-rerouted"]["build_time"],
            dense_metadata["MoEE-rerouted"]["index_bytes"] + dense_metadata["ExpertOut-mean-rerouted"]["index_bytes"],
            max(
                dense_metadata["MoEE-rerouted"]["vec_dim"],
                dense_metadata["ExpertOut-mean-rerouted"]["vec_dim"],
            ),
        ),
    ]

    paper_moee_scores = score_cache["MoEE-rerouted-papercond"]
    paper_va_scores = score_cache["VA-rerouted-papercond"]
    paper_expertout_scores = score_cache["ExpertOut-mean-rerouted-papercond"]
    fusion_specs.extend(
        [
            (
                "MoEE+VA-0.3-papercond",
                0.3 * paper_moee_scores + 0.7 * paper_va_scores,
                dense_metadata["MoEE-rerouted-papercond"]["build_time"] + dense_metadata["VA-rerouted-papercond"]["build_time"],
                dense_metadata["MoEE-rerouted-papercond"]["index_bytes"] + dense_metadata["VA-rerouted-papercond"]["index_bytes"],
                max(dense_metadata["MoEE-rerouted-papercond"]["vec_dim"], dense_metadata["VA-rerouted-papercond"]["vec_dim"]),
            ),
            (
                "MoEE+VA-0.5-papercond",
                0.5 * paper_moee_scores + 0.5 * paper_va_scores,
                dense_metadata["MoEE-rerouted-papercond"]["build_time"] + dense_metadata["VA-rerouted-papercond"]["build_time"],
                dense_metadata["MoEE-rerouted-papercond"]["index_bytes"] + dense_metadata["VA-rerouted-papercond"]["index_bytes"],
                max(dense_metadata["MoEE-rerouted-papercond"]["vec_dim"], dense_metadata["VA-rerouted-papercond"]["vec_dim"]),
            ),
            (
                "MoEE+VA-0.7-papercond",
                0.7 * paper_moee_scores + 0.3 * paper_va_scores,
                dense_metadata["MoEE-rerouted-papercond"]["build_time"] + dense_metadata["VA-rerouted-papercond"]["build_time"],
                dense_metadata["MoEE-rerouted-papercond"]["index_bytes"] + dense_metadata["VA-rerouted-papercond"]["index_bytes"],
                max(dense_metadata["MoEE-rerouted-papercond"]["vec_dim"], dense_metadata["VA-rerouted-papercond"]["vec_dim"]),
            ),
            (
                "MoEE+ExpertOut-0.5-papercond",
                0.5 * paper_moee_scores + 0.5 * paper_expertout_scores,
                dense_metadata["MoEE-rerouted-papercond"]["build_time"] + dense_metadata["ExpertOut-mean-rerouted-papercond"]["build_time"],
                dense_metadata["MoEE-rerouted-papercond"]["index_bytes"] + dense_metadata["ExpertOut-mean-rerouted-papercond"]["index_bytes"],
                max(
                    dense_metadata["MoEE-rerouted-papercond"]["vec_dim"],
                    dense_metadata["ExpertOut-mean-rerouted-papercond"]["vec_dim"],
                ),
            ),
        ]
    )

    rerouted_query_ids = [row["text_id"] for row in rerouted_query_rows]
    rerouted_doc_ids = [row["text_id"] for row in rerouted_doc_rows]
    for method, scores, build_time, index_bytes, vec_dim in fusion_specs:
        if method.endswith("-papercond"):
            query_ids = [row["text_id"] for row in paper_rerouted_query_rows]
            doc_ids = [row["text_id"] for row in paper_rerouted_doc_rows]
            pass_name = "paper_rerouted"
        else:
            query_ids = rerouted_query_ids
            doc_ids = rerouted_doc_ids
            pass_name = "rerouted"
        metrics = evaluate_scores(evaluator, qrels, query_ids, doc_ids, scores)
        score_cache[method] = scores
        record_method(method, metrics, 1.0, int(vec_dim), pass_name, float(build_time), float(index_bytes), None)

    multivector_specs = [
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

    paper_multivector_specs = []
    for method, fields, builder, vec_dim in multivector_specs:
        paper_multivector_specs.append((f"{method}-papercond", fields, builder, vec_dim))
    multivector_specs.extend(paper_multivector_specs)

    multivector_cache: dict[str, tuple[list[str], list[np.ndarray], list[str], list[np.ndarray]]] = {}
    cache_methods = {
        "ExpertPool-Value-attn",
        "ExpertPool-ExpertOut-all",
        "TokenLevel-HS-final",
        "ExpertPool-Value-attn-papercond",
        "ExpertPool-ExpertOut-all-papercond",
        "TokenLevel-HS-final-papercond",
    }
    for method, fields, builder, vec_dim in multivector_specs:
        if method.endswith("-papercond"):
            source_query_rows = paper_rerouted_query_rows
            source_doc_rows = paper_rerouted_doc_rows
            pass_name = "paper_rerouted"
        else:
            source_query_rows = rerouted_query_rows
            source_doc_rows = rerouted_doc_rows
            pass_name = "rerouted"
        query_ids, query_vectors, q_time, _, _ = load_multivectors(source_query_rows, fields, builder)
        doc_ids, doc_vectors, d_time, index_bytes, avg_vectors = load_multivectors(source_doc_rows, fields, builder)
        scores = compute_maxsim_scores(query_vectors, doc_vectors, device=device, batch_size=args.maxsim_batch_size)
        metrics = evaluate_scores(evaluator, qrels, query_ids, doc_ids, scores)
        if method in cache_methods:
            multivector_cache[method] = (query_ids, query_vectors, doc_ids, doc_vectors)
        score_cache[method] = scores
        record_method(method, metrics, avg_vectors, vec_dim, pass_name, q_time + d_time, index_bytes, None)
        if method not in cache_methods:
            del query_vectors, doc_vectors
        del scores
        gc.collect()

    top100_candidates = np.argsort(-moee_scores, axis=1)[:, :100]
    rerank_specs = [
        ("RouteSig-then-ExpertPool-Value", "ExpertPool-Value-attn"),
        ("RouteSig-then-ExpertPool-ExpertOut", "ExpertPool-ExpertOut-all"),
        ("RouteSig-then-TokenLevel-HS", "TokenLevel-HS-final"),
        ("RouteSig-then-ExpertPool-Value-papercond", "ExpertPool-Value-attn-papercond"),
        ("RouteSig-then-ExpertPool-ExpertOut-papercond", "ExpertPool-ExpertOut-all-papercond"),
        ("RouteSig-then-TokenLevel-HS-papercond", "TokenLevel-HS-final-papercond"),
    ]
    for method, base_method in rerank_specs:
        if method.endswith("-papercond"):
            candidate_scores = np.argsort(-score_cache["MoEE-rerouted-papercond"], axis=1)[:, :100]
            pass_name = "paper_rerouted"
        else:
            candidate_scores = top100_candidates
            pass_name = "rerouted"
        query_ids, query_vectors, doc_ids, doc_vectors = multivector_cache[base_method]
        reranked = compute_maxsim_rerank(query_vectors, doc_vectors, candidate_scores, device=device)
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
