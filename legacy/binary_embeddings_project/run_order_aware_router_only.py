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

NUM_LAYERS = 40
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))

_WORKER_DOCS = None
_WORKER_QUERIES = None
_WORKER_DOC_IDS = None
_WORKER_QUERY_IDS = None
_WORKER_VARIANTS = None
_WORKER_CONFIGS = None
_WORKER_CANDIDATE_DOCS = None
_WORKER_POSITION_CACHE = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--prefilter-top-k", type=int, default=500)
    parser.add_argument("--beta-grid", type=str, default="0.0,0.005,0.01,0.02,0.05")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--query-batch-size", type=int, default=2)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-prefix", type=str, default="order_aware_router_only")
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
    for layer_idx in range(NUM_LAYERS):
        groups.append(
            {
                "name": f"router_full_{layer_idx}",
                "type": "router_full",
                "field": f"router_full_{layer_idx}",
                "layer": layer_idx,
                "num_bits": 256,
                "packed_bytes": 32,
            }
        )
        groups.append(
            {
                "name": f"router_topk_{layer_idx}",
                "type": "router_topk",
                "field": f"router_topk_{layer_idx}",
                "layer": layer_idx,
                "num_bits": 256,
                "packed_bytes": 32,
            }
        )
    return groups


def define_variants():
    return OrderedDict(
        [
            (
                "router_full_20",
                {"family": "router_full", "layers": [20], "prefilter_source": "router_full_20"},
            ),
            (
                "router_full_top3_avg",
                {
                    "family": "router_full",
                    "layers": [20, 8, 13],
                    "prefilter_source": ["router_full_20", "router_full_8", "router_full_13"],
                },
            ),
            (
                "router_full_all_layers_avg",
                {
                    "family": "router_full",
                    "layers": list(range(NUM_LAYERS)),
                    "prefilter_source": [f"router_full_{layer_idx}" for layer_idx in range(NUM_LAYERS)],
                },
            ),
            (
                "router_topk_2",
                {"family": "router_topk", "layers": [2], "prefilter_source": "router_topk_2"},
            ),
            (
                "router_topk_top3_avg",
                {
                    "family": "router_topk",
                    "layers": [2, 20, 1],
                    "prefilter_source": ["router_topk_2", "router_topk_20", "router_topk_1"],
                },
            ),
            (
                "router_topk_all_layers_avg",
                {
                    "family": "router_topk",
                    "layers": list(range(NUM_LAYERS)),
                    "prefilter_source": [f"router_topk_{layer_idx}" for layer_idx in range(NUM_LAYERS)],
                },
            ),
        ]
    )


def _configure_worker_state(*, docs, queries, doc_ids, query_ids, variants, configs, candidate_docs):
    global _WORKER_DOCS
    global _WORKER_QUERIES
    global _WORKER_DOC_IDS
    global _WORKER_QUERY_IDS
    global _WORKER_VARIANTS
    global _WORKER_CONFIGS
    global _WORKER_CANDIDATE_DOCS
    global _WORKER_POSITION_CACHE

    _WORKER_DOCS = docs
    _WORKER_QUERIES = queries
    _WORKER_DOC_IDS = doc_ids
    _WORKER_QUERY_IDS = query_ids
    _WORKER_VARIANTS = variants
    _WORKER_CONFIGS = configs
    _WORKER_CANDIDATE_DOCS = candidate_docs
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


def _variant_similarity(query_bits, doc_bits, variant_name):
    variant = _WORKER_VARIANTS[variant_name]
    family = variant["family"]
    layers = variant["layers"]
    accumulator = None
    for layer_idx in layers:
        q_layer = np.asarray(query_bits[f"{family}_{layer_idx}"], dtype=np.uint8)
        d_layer = np.asarray(doc_bits[f"{family}_{layer_idx}"], dtype=np.uint8)
        sim = _pairwise_binary_similarity(q_layer, d_layer)
        if accumulator is None:
            accumulator = sim
        else:
            accumulator += sim
    return accumulator / float(len(layers))


def _score_query_batch(query_indices):
    num_docs = len(_WORKER_DOC_IDS)
    config_names = [config["name"] for config in _WORKER_CONFIGS]
    batch_outputs = {
        name: np.zeros((len(query_indices), num_docs), dtype=np.float32) for name in config_names
    }

    variant_to_configs = OrderedDict()
    for config in _WORKER_CONFIGS:
        variant_to_configs.setdefault(config["variant"], []).append(config)

    for batch_row, query_index in enumerate(query_indices):
        query_id = _WORKER_QUERY_IDS[query_index]
        query_bits = _WORKER_QUERIES[query_id]

        for variant_name, variant_configs in variant_to_configs.items():
            for doc_index in _WORKER_CANDIDATE_DOCS[variant_name][query_index]:
                doc_id = _WORKER_DOC_IDS[int(doc_index)]
                doc_bits = _WORKER_DOCS[doc_id]
                router_sim = _variant_similarity(query_bits, doc_bits, variant_name)
                penalty = _position_penalty(router_sim.shape[0], router_sim.shape[1])

                for config in variant_configs:
                    fused = router_sim - config["beta"] * penalty
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
        dcg = sum(
            rels.get(doc_ids[int(doc_index)], 0) / np.log2(rank + 2)
            for rank, doc_index in enumerate(ranked)
        )
        ideal = sorted(rels.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal))
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else 0.0


def _load_group_score(checkpoint_dir: Path, group_name: str):
    path = checkpoint_dir / "group_scores" / f"{group_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing group score matrix: {path}")
    return np.load(path, mmap_mode="r")


def _build_prefilter_scores(checkpoint_dir: Path, source):
    if isinstance(source, str):
        return np.asarray(_load_group_score(checkpoint_dir, source), dtype=np.float32)

    accumulator = None
    for group_name in source:
        matrix = np.asarray(_load_group_score(checkpoint_dir, group_name), dtype=np.float32)
        if accumulator is None:
            accumulator = np.zeros_like(matrix, dtype=np.float32)
        accumulator += matrix
    return accumulator / float(len(source))


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
    variants = define_variants()
    docs, queries = load_all_texts(project_root, groups, args.max_doc_tokens, args.max_query_tokens)
    qrels = load_scifact_qrels(project_root / "datasets")
    doc_ids = sorted(docs.keys())
    query_ids = sorted(queries.keys())
    num_q = len(query_ids)
    num_d = len(doc_ids)
    print(f"Loaded {len(docs)} docs and {len(queries)} queries")

    top_k = min(args.prefilter_top_k, num_d)
    candidate_docs = {}
    for variant_name, variant in variants.items():
        prefilter_scores = _build_prefilter_scores(checkpoint_dir, variant["prefilter_source"])
        if prefilter_scores.shape != (num_q, num_d):
            raise RuntimeError(
                f"Unexpected prefilter score shape {prefilter_scores.shape} for {variant_name}, expected {(num_q, num_d)}"
            )
        if top_k >= num_d:
            selected = np.broadcast_to(np.arange(num_d, dtype=np.int32), (num_q, num_d))
        else:
            selected = np.argpartition(-prefilter_scores, top_k - 1, axis=1)[:, :top_k]
        candidate_docs[variant_name] = np.asarray(selected, dtype=np.int32)
        print(f"Prepared router-only candidates for {variant_name}: top {top_k}")

    betas = _parse_float_grid(args.beta_grid)
    configs = []
    for variant_name in variants:
        for beta in betas:
            configs.append(
                {
                    "name": f"{variant_name}__beta_{beta:g}",
                    "variant": variant_name,
                    "beta": float(beta),
                }
            )

    workers = max(1, min(args.workers, os.cpu_count() or 1))
    _configure_worker_state(
        docs=docs,
        queries=queries,
        doc_ids=doc_ids,
        query_ids=query_ids,
        variants=variants,
        configs=configs,
        candidate_docs=candidate_docs,
    )

    score_matrices = {
        config["name"]: np.zeros((num_q, num_d), dtype=np.float32) for config in configs
    }
    query_batches = [list(batch) for batch in _chunked(list(range(num_q)), max(1, args.query_batch_size))]
    t0 = time.perf_counter()
    executor = _get_executor(workers) if workers > 1 else None
    if executor is not None:
        with executor:
            future_map = {executor.submit(_score_query_batch, batch): batch for batch in query_batches}
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
                "variant": config["variant"],
                "beta": config["beta"],
                "ndcg_at_10": ndcg,
            }
        )
    rows.sort(key=lambda row: row["ndcg_at_10"], reverse=True)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(num_q)
    folds = [np.array(sorted(fold.tolist())) for fold in np.array_split(perm, args.num_folds)]
    cv_summary = {}
    for variant_name in variants:
        variant_configs = [config for config in configs if config["variant"] == variant_name]
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
            chosen.append({"beta": best_config["beta"]})
            fold_scores.append(
                ndcg_at_k(
                    score_matrices[best_config["name"]],
                    query_ids,
                    doc_ids,
                    qrels,
                    query_subset=test_idx,
                )
            )
        cv_summary[variant_name] = {
            "mean_ndcg_at_10": float(np.mean(fold_scores)),
            "fold_scores": [float(score) for score in fold_scores],
            "chosen_configs": chosen,
        }

    best_by_variant = {}
    for variant_name in variants:
        variant_rows = [row for row in rows if row["variant"] == variant_name]
        best_by_variant[variant_name] = variant_rows[0]

    summary = {
        "prefilter_top_k": top_k,
        "num_queries": num_q,
        "num_docs": num_d,
        "best_by_variant": best_by_variant,
        "cv_by_variant": cv_summary,
        "variant_definitions": variants,
    }

    output_prefix = args.output_prefix
    summary_path = results_dir / f"{output_prefix}_summary.json"
    grid_path = results_dir / f"{output_prefix}_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "beta", "ndcg_at_10"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {summary_path}")
    print(f"Saved: {grid_path}")


if __name__ == "__main__":
    main()
