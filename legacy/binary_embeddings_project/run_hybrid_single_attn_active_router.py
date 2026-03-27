from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_binary_notebook_analysis import _POPCOUNT_TABLE, _chunked, load_all_texts, load_scifact_qrels
from run_supervised_group_fusion import build_label_matrices, fit_weighted_ridge, make_query_folds

NUM_LAYERS = 40
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))

_WORKER_DOCS = None
_WORKER_QUERIES = None
_WORKER_DOC_IDS = None
_WORKER_QUERY_IDS = None
_WORKER_CANDIDATE_DOCS = None
_WORKER_VARIANTS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--prefilter-top-k", type=int, default=200)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-grid", type=str, default="0.01,0.1,1,10,100")
    parser.add_argument("--output-prefix", type=str, default="hybrid_single_attn_active_router")
    return parser.parse_args()


def parse_csv(spec: str, cast):
    values = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(cast(item))
    if not values:
        raise ValueError("Expected at least one value")
    return values


def read_ids(results_dir: Path):
    records = []
    with (results_dir / "records.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(row)
    doc_ids = sorted({row["text_id"] for row in records if row["kind"] == "doc"})
    query_ids = sorted({row["text_id"] for row in records if row["kind"] == "query"})
    return doc_ids, query_ids


def define_groups():
    groups = [
        {
            "name": "attn_out_sign_35",
            "type": "sign",
            "field": "attn_out_35",
            "layer": 35,
            "num_bits": 2048,
            "packed_bytes": 256,
        }
    ]
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


def define_router_variants():
    return {
        "router_full_top3": ["router_full_20", "router_full_8", "router_full_13"],
        "router_topk_top3": ["router_topk_2", "router_topk_20", "router_topk_1"],
        "router_full_all_layers": [f"router_full_{idx}" for idx in range(NUM_LAYERS)],
        "router_topk_all_layers": [f"router_topk_{idx}" for idx in range(NUM_LAYERS)],
    }


def mean_pool_packed(packed):
    unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1).astype(np.float32)
    return unpacked.mean(axis=0)


def l2norm(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return matrix / norms


def q_zscore(scores):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (scores - mean) / std


def ndcg_at_10(scores, query_ids, doc_ids, qrels):
    values = []
    for query_index, query_id in enumerate(query_ids):
        rels = qrels.get(query_id)
        if not rels:
            continue
        ranked = np.argsort(-scores[query_index])[:10]
        dcg = 0.0
        for rank, doc_index in enumerate(ranked):
            dcg += rels.get(doc_ids[int(doc_index)], 0) / math.log2(rank + 2)
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal))
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else 0.0


def ndcg_from_rankings(rankings, query_ids, qrels):
    values = []
    for query_id in query_ids:
        rels = qrels.get(query_id)
        if not rels:
            continue
        ranked = rankings[query_id][:10]
        dcg = 0.0
        for rank, doc_id in enumerate(ranked):
            dcg += rels.get(doc_id, 0) / math.log2(rank + 2)
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal))
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else 0.0


def top_lists(scores, query_ids, doc_ids, limit):
    rankings = {}
    for query_index, query_id in enumerate(query_ids):
        ranked = np.argsort(-scores[query_index])[:limit]
        rankings[query_id] = [doc_ids[int(doc_index)] for doc_index in ranked]
    return rankings


def interleave(rank_a, rank_b, first: str, out_top_k: int = 10):
    rankings = {}
    for query_id in rank_a:
        left = rank_a[query_id]
        right = rank_b[query_id]
        streams = (left, right) if first == "a" else (right, left)
        chosen = []
        seen = set()
        index = 0
        while len(chosen) < out_top_k and (index < len(streams[0]) or index < len(streams[1])):
            for stream in streams:
                if index >= len(stream):
                    continue
                doc_id = stream[index]
                if doc_id in seen:
                    continue
                chosen.append(doc_id)
                seen.add(doc_id)
                if len(chosen) >= out_top_k:
                    break
            index += 1
        rankings[query_id] = chosen
    return rankings


def build_single_attn_scores(docs, queries, doc_ids, query_ids):
    doc_matrix = [mean_pool_packed(docs[doc_id]["attn_out_sign_35"]) for doc_id in doc_ids]
    query_matrix = [mean_pool_packed(queries[query_id]["attn_out_sign_35"]) for query_id in query_ids]
    doc_emb = l2norm(np.stack(doc_matrix).astype(np.float32))
    query_emb = l2norm(np.stack(query_matrix).astype(np.float32))
    return query_emb @ doc_emb.T


def _load_score_matrix(checkpoint_dir: Path, group_name: str):
    path = checkpoint_dir / "group_scores" / f"{group_name}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing score matrix: {path}")
    return np.load(path, mmap_mode="r")


def _average_score_matrix(checkpoint_dir: Path, group_names):
    accumulator = None
    for group_name in group_names:
        matrix = np.asarray(_load_score_matrix(checkpoint_dir, group_name), dtype=np.float32)
        if accumulator is None:
            accumulator = np.zeros_like(matrix, dtype=np.float32)
        accumulator += matrix
    return accumulator / float(len(group_names))


def _count_bits(packed):
    return _POPCOUNT_TABLE[np.asarray(packed, dtype=np.uint8)].sum(axis=1).astype(np.float32)


def _pairwise_active_cosine(q_bits, d_bits):
    inter = _POPCOUNT_TABLE[np.bitwise_and(q_bits[:, None, :], d_bits[None, :, :])].astype(np.uint16).sum(axis=2)
    q_count = _count_bits(q_bits)[:, None]
    d_count = _count_bits(d_bits)[None, :]
    denom = np.sqrt(q_count * d_count, dtype=np.float32)
    return np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float32), where=denom > 0)


def _configure_worker_state(docs, queries, doc_ids, query_ids, candidate_docs, variants):
    global _WORKER_DOCS
    global _WORKER_QUERIES
    global _WORKER_DOC_IDS
    global _WORKER_QUERY_IDS
    global _WORKER_CANDIDATE_DOCS
    global _WORKER_VARIANTS

    _WORKER_DOCS = docs
    _WORKER_QUERIES = queries
    _WORKER_DOC_IDS = doc_ids
    _WORKER_QUERY_IDS = query_ids
    _WORKER_CANDIDATE_DOCS = candidate_docs
    _WORKER_VARIANTS = variants


def _score_query_batch(query_indices):
    num_docs = len(_WORKER_DOC_IDS)
    outputs = {
        variant_name: np.zeros((len(query_indices), num_docs), dtype=np.float32)
        for variant_name in _WORKER_VARIANTS
    }
    for batch_row, query_index in enumerate(query_indices):
        query_id = _WORKER_QUERY_IDS[int(query_index)]
        query_bits = _WORKER_QUERIES[query_id]
        for variant_name, group_names in _WORKER_VARIANTS.items():
            for doc_index in _WORKER_CANDIDATE_DOCS[variant_name][int(query_index)]:
                doc_id = _WORKER_DOC_IDS[int(doc_index)]
                doc_bits = _WORKER_DOCS[doc_id]
                accumulator = None
                for group_name in group_names:
                    q_bits = np.asarray(query_bits[group_name], dtype=np.uint8)
                    d_bits = np.asarray(doc_bits[group_name], dtype=np.uint8)
                    sim = _pairwise_active_cosine(q_bits, d_bits)
                    accumulator = sim if accumulator is None else (accumulator + sim)
                fused = accumulator / float(len(group_names))
                outputs[variant_name][batch_row, int(doc_index)] = float(fused.max(axis=1).mean())
    return list(query_indices), outputs


def compute_router_scores(docs, queries, doc_ids, query_ids, checkpoint_dir: Path, variants, prefilter_top_k: int, workers: int, query_batch_size: int):
    top_k = min(prefilter_top_k, len(doc_ids))
    candidate_docs = {}
    for variant_name, group_names in variants.items():
        prefilter_scores = _average_score_matrix(checkpoint_dir, group_names)
        if top_k >= len(doc_ids):
            selected = np.broadcast_to(np.arange(len(doc_ids), dtype=np.int32), prefilter_scores.shape)
        else:
            selected = np.argpartition(-prefilter_scores, top_k - 1, axis=1)[:, :top_k]
        candidate_docs[variant_name] = np.asarray(selected, dtype=np.int32)

    query_batches = [list(batch) for batch in _chunked(list(range(len(query_ids))), max(1, query_batch_size))]
    score_matrices = {
        variant_name: np.zeros((len(query_ids), len(doc_ids)), dtype=np.float32)
        for variant_name in variants
    }
    if workers > 1:
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_configure_worker_state,
            initargs=(docs, queries, doc_ids, query_ids, candidate_docs, variants),
        ) as executor:
            futures = [executor.submit(_score_query_batch, batch) for batch in query_batches]
            for future in as_completed(futures):
                batch_indices, batch_outputs = future.result()
                for variant_name, batch_scores in batch_outputs.items():
                    for row_index, query_index in enumerate(batch_indices):
                        score_matrices[variant_name][query_index] = batch_scores[row_index]
    else:
        _configure_worker_state(
            docs=docs,
            queries=queries,
            doc_ids=doc_ids,
            query_ids=query_ids,
            candidate_docs=candidate_docs,
            variants=variants,
        )
        for batch in query_batches:
            batch_indices, batch_outputs = _score_query_batch(batch)
            for variant_name, batch_scores in batch_outputs.items():
                for row_index, query_index in enumerate(batch_indices):
                    score_matrices[variant_name][query_index] = batch_scores[row_index]
    return score_matrices


def compute_ranks(scores):
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    row_indices = np.arange(scores.shape[0])[:, None]
    ranks[row_indices, order] = np.arange(1, scores.shape[1] + 1, dtype=np.int32)
    return ranks


def fit_ridge_cv(attn_scores, router_scores, query_ids, doc_ids, qrels, alpha_grid, num_folds, seed):
    labels, sample_weights = build_label_matrices(query_ids, doc_ids, qrels)
    attn_z = q_zscore(attn_scores)
    router_z = q_zscore(router_scores)
    attn_ranks = compute_ranks(attn_scores)
    router_ranks = compute_ranks(router_scores)
    features = np.stack(
        [
            attn_z,
            router_z,
            1.0 / attn_ranks.astype(np.float32),
            1.0 / router_ranks.astype(np.float32),
            (attn_ranks <= 10).astype(np.float32),
            (router_ranks <= 10).astype(np.float32),
        ],
        axis=2,
    )

    folds = make_query_folds(len(query_ids), num_folds, seed)
    best = None
    rows = []
    for alpha in alpha_grid:
        fold_scores = []
        for fold_index, val_queries in enumerate(folds, start=1):
            train_queries = np.concatenate(
                [fold for other_index, fold in enumerate(folds, start=1) if other_index != fold_index]
            )
            X_train = features[train_queries].reshape(-1, features.shape[2])
            y_train = labels[train_queries].reshape(-1)
            w_train = sample_weights[train_queries].reshape(-1)
            weights = fit_weighted_ridge(X_train, y_train, w_train, alpha=alpha, batch_size=65536)
            val_scores = np.tensordot(features[val_queries], weights, axes=([2], [0]))
            val_query_ids = [query_ids[int(index)] for index in val_queries]
            fold_scores.append(ndcg_at_10(val_scores, val_query_ids, doc_ids, qrels))
        mean_ndcg = float(np.mean(fold_scores))
        row = {"alpha": float(alpha), "mean_ndcg_at_10": mean_ndcg, "fold_scores": fold_scores}
        rows.append(row)
        if best is None or mean_ndcg > best["mean_ndcg_at_10"]:
            best = row

    final_weights = fit_weighted_ridge(
        features.reshape(-1, features.shape[2]),
        labels.reshape(-1),
        sample_weights.reshape(-1),
        alpha=best["alpha"],
        batch_size=65536,
    )
    full_scores = np.tensordot(features, final_weights, axes=([2], [0]))
    return {
        "best_cv": best,
        "fullset_ndcg_at_10": ndcg_at_10(full_scores, query_ids, doc_ids, qrels),
        "weights": [float(x) for x in final_weights],
        "feature_names": [
            "attn_zscore",
            "router_zscore",
            "attn_recip_rank",
            "router_recip_rank",
            "attn_top10_flag",
            "router_top10_flag",
        ],
        "grid": rows,
    }


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = project_root / "results"
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else (results_dir / "checkpoints").resolve()
    )
    alpha_grid = parse_csv(args.alpha_grid, float)

    doc_ids, query_ids = read_ids(results_dir)
    qrels = load_scifact_qrels(project_root / "datasets")
    groups = define_groups()
    docs, queries = load_all_texts(project_root, groups, args.max_doc_tokens, args.max_query_tokens)

    attn_scores = build_single_attn_scores(docs, queries, doc_ids, query_ids)
    router_variants = define_router_variants()
    router_scores = compute_router_scores(
        docs,
        queries,
        doc_ids,
        query_ids,
        checkpoint_dir,
        router_variants,
        args.prefilter_top_k,
        max(1, min(args.workers, os.cpu_count() or 1)),
        args.query_batch_size,
    )

    summary = {
        "prefilter_top_k": int(min(args.prefilter_top_k, len(doc_ids))),
        "single_attn_ndcg_at_10": ndcg_at_10(attn_scores, query_ids, doc_ids, qrels),
        "variants": {},
    }

    attn_top = top_lists(attn_scores, query_ids, doc_ids, min(args.prefilter_top_k, len(doc_ids)))
    for variant_name, scores in router_scores.items():
        router_top = top_lists(scores, query_ids, doc_ids, min(args.prefilter_top_k, len(doc_ids)))
        interleaved = interleave(attn_top, router_top, first="a", out_top_k=10)
        z_alpha_best = (-1.0, None)
        attn_z = q_zscore(attn_scores)
        router_z = q_zscore(scores)
        for alpha in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]:
            fused = attn_z + alpha * router_z
            ndcg = ndcg_at_10(fused, query_ids, doc_ids, qrels)
            if ndcg > z_alpha_best[0]:
                z_alpha_best = (ndcg, alpha)

        ridge = fit_ridge_cv(
            attn_scores,
            scores,
            query_ids,
            doc_ids,
            qrels,
            alpha_grid=alpha_grid,
            num_folds=args.num_folds,
            seed=args.seed,
        )
        summary["variants"][variant_name] = {
            "router_ndcg_at_10": ndcg_at_10(scores, query_ids, doc_ids, qrels),
            "interleave_attn_first_ndcg_at_10": ndcg_from_rankings(interleaved, query_ids, qrels),
            "best_zscore_sum": {
                "ndcg_at_10": z_alpha_best[0],
                "alpha": z_alpha_best[1],
            },
            "ridge_rerank": ridge,
        }

    summary_path = results_dir / f"{args.output_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
