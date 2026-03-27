from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_binary_notebook_analysis import (
    _POPCOUNT_TABLE,
    _chunked,
    load_all_texts,
    load_scifact_qrels,
)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))

_WORKER_DOCS = None
_WORKER_QUERIES = None
_WORKER_DOC_IDS = None
_WORKER_QUERY_IDS = None
_WORKER_CANDIDATE_DOCS = None
_WORKER_CONFIGS = None
_WORKER_POSITION_CACHE = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--prefilter-group", type=str, default="attn_out_sign_35")
    parser.add_argument("--prefilter-top-k", type=int, default=100)
    parser.add_argument("--alpha-grid", type=str, default="0.0,0.005,0.01,0.02,0.03,0.05")
    parser.add_argument("--beta-grid", type=str, default="0.0,0.005,0.01,0.02,0.05")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", type=str, default="order_aware_router_fusion")
    return parser.parse_args()


def _parse_float_grid(spec: str):
    values = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Grid must contain at least one value")
    return values


def _get_executor(max_workers: int):
    if max_workers <= 1:
        return None
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)


def define_groups():
    groups = []
    for name in (
        "attn_out_sign_35",
        "router_full_20",
        "router_full_8",
        "router_full_13",
    ):
        if name.startswith("attn_out_sign_"):
            layer = int(name.rsplit("_", 1)[1])
            groups.append(
                {
                    "name": name,
                    "type": "sign",
                    "field": f"attn_out_{layer}",
                    "layer": layer,
                    "num_bits": 2048,
                    "packed_bytes": 256,
                }
            )
        elif name.startswith("router_full_"):
            layer = int(name.rsplit("_", 1)[1])
            groups.append(
                {
                    "name": name,
                    "type": "router_full",
                    "field": name,
                    "layer": layer,
                    "num_bits": 256,
                    "packed_bytes": 32,
                }
            )
        else:
            raise ValueError(f"Unsupported group name: {name}")
    return groups


def _configure_worker_state(*, docs, queries, doc_ids, query_ids, candidate_docs, configs):
    global _WORKER_DOCS
    global _WORKER_QUERIES
    global _WORKER_DOC_IDS
    global _WORKER_QUERY_IDS
    global _WORKER_CANDIDATE_DOCS
    global _WORKER_CONFIGS
    global _WORKER_POSITION_CACHE

    _WORKER_DOCS = docs
    _WORKER_QUERIES = queries
    _WORKER_DOC_IDS = doc_ids
    _WORKER_QUERY_IDS = query_ids
    _WORKER_CANDIDATE_DOCS = candidate_docs
    _WORKER_CONFIGS = configs
    _WORKER_POSITION_CACHE = {}


def _pairwise_binary_similarity(q_bits, d_bits):
    xor = np.bitwise_xor(q_bits[:, None, :], d_bits[None, :, :])
    hamming = _POPCOUNT_TABLE[xor].astype(np.uint16).sum(axis=2)
    total_bits = q_bits.shape[1] * 8
    return (total_bits - hamming).astype(np.float32) / float(total_bits)


def _position_penalty(q_len: int, d_len: int):
    key = (q_len, d_len)
    cached = _WORKER_POSITION_CACHE.get(key)
    if cached is not None:
        return cached
    if q_len <= 1:
        q_pos = np.zeros((q_len, 1), dtype=np.float32)
    else:
        q_pos = np.linspace(0.0, 1.0, q_len, dtype=np.float32)[:, None]
    if d_len <= 1:
        d_pos = np.zeros((1, d_len), dtype=np.float32)
    else:
        d_pos = np.linspace(0.0, 1.0, d_len, dtype=np.float32)[None, :]
    penalty = np.abs(q_pos - d_pos, dtype=np.float32)
    _WORKER_POSITION_CACHE[key] = penalty
    return penalty


def _score_query_batch(query_indices):
    num_docs = len(_WORKER_DOC_IDS)
    config_names = [config["name"] for config in _WORKER_CONFIGS]
    batch_outputs = {
        name: np.zeros((len(query_indices), num_docs), dtype=np.float32) for name in config_names
    }

    for batch_row, query_index in enumerate(query_indices):
        query_id = _WORKER_QUERY_IDS[query_index]
        query_bits = _WORKER_QUERIES[query_id]
        q_attn = np.asarray(query_bits["attn_out_sign_35"], dtype=np.uint8)
        q_router_20 = np.asarray(query_bits["router_full_20"], dtype=np.uint8)
        q_router_8 = np.asarray(query_bits["router_full_8"], dtype=np.uint8)
        q_router_13 = np.asarray(query_bits["router_full_13"], dtype=np.uint8)

        for doc_index in _WORKER_CANDIDATE_DOCS[query_index]:
            doc_id = _WORKER_DOC_IDS[int(doc_index)]
            doc_bits = _WORKER_DOCS[doc_id]
            d_attn = np.asarray(doc_bits["attn_out_sign_35"], dtype=np.uint8)
            d_router_20 = np.asarray(doc_bits["router_full_20"], dtype=np.uint8)
            d_router_8 = np.asarray(doc_bits["router_full_8"], dtype=np.uint8)
            d_router_13 = np.asarray(doc_bits["router_full_13"], dtype=np.uint8)

            attn_sim = _pairwise_binary_similarity(q_attn, d_attn)
            router_sim_20 = _pairwise_binary_similarity(q_router_20, d_router_20)
            router_sim_top3 = (
                router_sim_20
                + _pairwise_binary_similarity(q_router_8, d_router_8)
                + _pairwise_binary_similarity(q_router_13, d_router_13)
            ) / 3.0
            penalty = _position_penalty(attn_sim.shape[0], attn_sim.shape[1])

            for config in _WORKER_CONFIGS:
                if config["router_variant"] == "router_full_20":
                    router_sim = router_sim_20
                elif config["router_variant"] == "router_full_top3_avg":
                    router_sim = router_sim_top3
                else:
                    raise ValueError(f"Unknown router variant: {config['router_variant']}")

                fused = attn_sim + config["alpha"] * router_sim - config["beta"] * penalty
                batch_outputs[config["name"]][batch_row, int(doc_index)] = fused.max(axis=1).mean()

    return list(query_indices), batch_outputs


def ndcg_at_k(scores_matrix, query_ids, doc_ids, qrels, k=10, query_subset=None):
    values = []
    if query_subset is None:
        query_subset = range(len(query_ids))
    for query_index in query_subset:
        query_id = query_ids[int(query_index)]
        rels = qrels.get(query_id)
        if not rels:
            continue
        ranked = np.argsort(-scores_matrix[int(query_index)])[:k]
        dcg = sum(rels.get(doc_ids[int(doc_index)], 0) / np.log2(rank + 2) for rank, doc_index in enumerate(ranked))
        ideal = sorted(rels.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal))
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else 0.0


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = project_root / "results"
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else (results_dir / "checkpoints").resolve()
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    groups = define_groups()
    docs, queries = load_all_texts(project_root, groups, args.max_doc_tokens, args.max_query_tokens)
    qrels = load_scifact_qrels(project_root / "datasets")
    doc_ids = sorted(docs.keys())
    query_ids = sorted(queries.keys())
    num_q = len(query_ids)
    num_d = len(doc_ids)
    print(f"Loaded {len(docs)} docs and {len(queries)} queries")

    prefilter_path = checkpoint_dir / "group_scores" / f"{args.prefilter_group}.npy"
    if not prefilter_path.exists():
        raise FileNotFoundError(f"Missing prefilter score matrix: {prefilter_path}")
    prefilter_scores = np.load(prefilter_path, mmap_mode="r")
    if prefilter_scores.shape != (num_q, num_d):
        raise RuntimeError(
            f"Unexpected prefilter score shape {prefilter_scores.shape}, expected {(num_q, num_d)}"
        )

    top_k = min(args.prefilter_top_k, num_d)
    if top_k >= num_d:
        candidate_docs = np.broadcast_to(np.arange(num_d, dtype=np.int32), (num_q, num_d))
    else:
        candidate_docs = np.argpartition(-prefilter_scores, top_k - 1, axis=1)[:, :top_k]

    alphas = _parse_float_grid(args.alpha_grid)
    betas = _parse_float_grid(args.beta_grid)
    router_variants = OrderedDict(
        [
            ("router_full_20", ["router_full_20"]),
            ("router_full_top3_avg", ["router_full_20", "router_full_8", "router_full_13"]),
        ]
    )
    configs = []
    for router_variant in router_variants:
        for alpha in alphas:
            for beta in betas:
                configs.append(
                    {
                        "name": f"{router_variant}__alpha_{alpha:g}__beta_{beta:g}",
                        "router_variant": router_variant,
                        "alpha": float(alpha),
                        "beta": float(beta),
                    }
                )

    workers = max(1, min(args.workers, os.cpu_count() or 1))
    _configure_worker_state(
        docs=docs,
        queries=queries,
        doc_ids=doc_ids,
        query_ids=query_ids,
        candidate_docs=candidate_docs,
        configs=configs,
    )

    score_matrices = {
        config["name"]: np.zeros((num_q, num_d), dtype=np.float32) for config in configs
    }
    query_batches = [list(batch) for batch in _chunked(list(range(num_q)), max(1, args.query_batch_size))]
    t0 = time.perf_counter()
    executor = _get_executor(workers) if workers > 1 else None
    if executor is not None:
        with executor:
            future_map = {
                executor.submit(_score_query_batch, batch): batch for batch in query_batches
            }
            processed_queries = 0
            for count, future in enumerate(as_completed(future_map), start=1):
                batch_indices, batch_outputs = future.result()
                for config_name, batch_scores in batch_outputs.items():
                    for row_idx, query_index in enumerate(batch_indices):
                        score_matrices[config_name][query_index] = batch_scores[row_idx]
                processed_queries += len(batch_indices)
                if count % 5 == 0 or count == len(query_batches):
                    elapsed = time.perf_counter() - t0
                    print(f"  processed {min(processed_queries, num_q)}/{num_q} queries ({elapsed:.1f}s)")
    else:
        for count, batch in enumerate(query_batches, start=1):
            batch_indices, batch_outputs = _score_query_batch(batch)
            for config_name, batch_scores in batch_outputs.items():
                for row_idx, query_index in enumerate(batch_indices):
                    score_matrices[config_name][query_index] = batch_scores[row_idx]
            if count % 5 == 0 or count == len(query_batches):
                elapsed = time.perf_counter() - t0
                print(f"  processed {min(count * len(batch), num_q)}/{num_q} queries ({elapsed:.1f}s)")

    rows = []
    for config in configs:
        ndcg = ndcg_at_k(score_matrices[config["name"]], query_ids, doc_ids, qrels)
        rows.append(
            {
                "variant": config["router_variant"],
                "alpha": config["alpha"],
                "beta": config["beta"],
                "ndcg_at_10": ndcg,
            }
        )
    rows.sort(key=lambda row: row["ndcg_at_10"], reverse=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(num_q)
    folds = [np.array(sorted(fold.tolist())) for fold in np.array_split(perm, args.num_folds)]
    cv_summary = {}
    for router_variant in router_variants:
        variant_configs = [config for config in configs if config["router_variant"] == router_variant]
        fold_scores = []
        chosen = []
        for test_idx in folds:
            train_idx = np.array(sorted(set(range(num_q)) - set(test_idx.tolist())))
            best_score = -1.0
            best_config = None
            for config in variant_configs:
                score = ndcg_at_k(
                    score_matrices[config["name"]],
                    query_ids,
                    doc_ids,
                    qrels,
                    query_subset=train_idx,
                )
                if score > best_score:
                    best_score = score
                    best_config = config
            chosen.append({"alpha": best_config["alpha"], "beta": best_config["beta"]})
            fold_scores.append(
                ndcg_at_k(
                    score_matrices[best_config["name"]],
                    query_ids,
                    doc_ids,
                    qrels,
                    query_subset=test_idx,
                )
            )
        cv_summary[router_variant] = {
            "mean_ndcg_at_10": float(np.mean(fold_scores)),
            "fold_scores": [float(score) for score in fold_scores],
            "chosen_configs": chosen,
        }

    best_by_variant = {}
    for router_variant in router_variants:
        variant_rows = [row for row in rows if row["variant"] == router_variant]
        best_by_variant[router_variant] = variant_rows[0]

    summary = {
        "prefilter_group": args.prefilter_group,
        "prefilter_top_k": top_k,
        "num_queries": num_q,
        "num_docs": num_d,
        "best_by_variant": best_by_variant,
        "cv_by_variant": cv_summary,
    }

    output_prefix = args.output_prefix
    summary_path = results_dir / f"{output_prefix}_summary.json"
    grid_path = results_dir / f"{output_prefix}_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "alpha", "beta", "ndcg_at_10"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {summary_path}")
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
