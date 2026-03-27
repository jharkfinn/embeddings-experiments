from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_binary_notebook_analysis import _POPCOUNT_TABLE, load_all_texts, load_scifact_qrels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--prefilter-top-k", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--alpha-grid", type=str, default="0.001,0.002,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.5,1,2,5")
    parser.add_argument("--rrf-k-grid", type=str, default="10,20,60,100")
    parser.add_argument("--output-prefix", type=str, default="rank_fusion_ablation")
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


def ndcg_at_10(scores: np.ndarray, query_ids, doc_ids, qrels):
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


def q_zscore(scores: np.ndarray):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (scores - mean) / std


def q_minmax(scores: np.ndarray):
    mins = scores.min(axis=1, keepdims=True)
    maxs = scores.max(axis=1, keepdims=True)
    spans = np.where(maxs - mins > 1e-8, maxs - mins, 1.0)
    return (scores - mins) / spans


def top_lists(scores: np.ndarray, query_ids, doc_ids, limit: int):
    rankings = {}
    for query_index, query_id in enumerate(query_ids):
        ranked = np.argsort(-scores[query_index])[:limit]
        rankings[query_id] = [doc_ids[int(doc_index)] for doc_index in ranked]
    return rankings


def reciprocal_rank_fusion(rank_a, rank_b, k: int, out_top_k: int = 10):
    rankings = {}
    for query_id in rank_a:
        scores = {}
        for rank, doc_id in enumerate(rank_a[query_id], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for rank, doc_id in enumerate(rank_b[query_id], start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        rankings[query_id] = [doc_id for doc_id, _ in ordered[:out_top_k]]
    return rankings


def interleave(rank_a, rank_b, first: str, out_top_k: int = 10):
    rankings = {}
    for query_id in rank_a:
        left = rank_a[query_id]
        right = rank_b[query_id]
        streams = (left, right) if first == "a" else (right, left)
        chosen = []
        seen = set()
        i = 0
        while len(chosen) < out_top_k and (i < len(streams[0]) or i < len(streams[1])):
            for stream in streams:
                if i >= len(stream):
                    continue
                doc_id = stream[i]
                if doc_id in seen:
                    continue
                chosen.append(doc_id)
                seen.add(doc_id)
                if len(chosen) >= out_top_k:
                    break
            i += 1
        rankings[query_id] = chosen
    return rankings


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
    for layer_idx in range(40):
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


def build_single_vector_scores(docs, queries, doc_ids, query_ids):
    variants = {
        "attn_out_sign_35": ["attn_out_sign_35"],
        "router_full_top3_avg": ["router_full_20", "router_full_8", "router_full_13"],
        "router_topk_top3_avg": ["router_topk_2", "router_topk_20", "router_topk_1"],
        "router_full_all_layers_avg": [f"router_full_{idx}" for idx in range(40)],
        "router_topk_all_layers_avg": [f"router_topk_{idx}" for idx in range(40)],
    }
    outputs = {}
    for variant, group_names in variants.items():
        doc_matrix = []
        for doc_id in doc_ids:
            pooled = [
                np.unpackbits(np.asarray(docs[doc_id][group_name], dtype=np.uint8), axis=1).astype(np.float32).mean(axis=0)
                for group_name in group_names
            ]
            doc_matrix.append(np.mean(pooled, axis=0))
        query_matrix = []
        for query_id in query_ids:
            pooled = [
                np.unpackbits(np.asarray(queries[query_id][group_name], dtype=np.uint8), axis=1).astype(np.float32).mean(axis=0)
                for group_name in group_names
            ]
            query_matrix.append(np.mean(pooled, axis=0))
        doc_mat = np.stack(doc_matrix).astype(np.float32)
        query_mat = np.stack(query_matrix).astype(np.float32)
        doc_norm = np.linalg.norm(doc_mat, axis=1, keepdims=True)
        doc_norm = np.where(doc_norm > 1e-8, doc_norm, 1.0)
        query_norm = np.linalg.norm(query_mat, axis=1, keepdims=True)
        query_norm = np.where(query_norm > 1e-8, query_norm, 1.0)
        outputs[variant] = (query_mat / query_norm) @ (doc_mat / doc_norm).T
    return outputs


def pairwise_binary_similarity(q_bits, d_bits):
    xor = np.bitwise_xor(q_bits[:, None, :], d_bits[None, :, :])
    hamming = _POPCOUNT_TABLE[xor].astype(np.uint16).sum(axis=2)
    total_bits = q_bits.shape[1] * 8
    return (total_bits - hamming).astype(np.float32) / float(total_bits)


_WORKER_DOCS = None
_WORKER_QUERIES = None
_WORKER_DOC_IDS = None
_WORKER_QUERY_IDS = None


def configure_worker(docs, queries, doc_ids, query_ids):
    global _WORKER_DOCS
    global _WORKER_QUERIES
    global _WORKER_DOC_IDS
    global _WORKER_QUERY_IDS
    _WORKER_DOCS = docs
    _WORKER_QUERIES = queries
    _WORKER_DOC_IDS = doc_ids
    _WORKER_QUERY_IDS = query_ids


def score_router_batch(query_indices, family, layers, candidates):
    outputs = {}
    for query_index in query_indices:
        query_id = _WORKER_QUERY_IDS[query_index]
        query_bits = _WORKER_QUERIES[query_id]
        row = np.zeros(len(_WORKER_DOC_IDS), dtype=np.float32)
        for doc_index in candidates[query_index]:
            doc_id = _WORKER_DOC_IDS[int(doc_index)]
            doc_bits = _WORKER_DOCS[doc_id]
            accum = None
            for layer in layers:
                sim = pairwise_binary_similarity(
                    np.asarray(query_bits[f"{family}_{layer}"], dtype=np.uint8),
                    np.asarray(doc_bits[f"{family}_{layer}"], dtype=np.uint8),
                )
                accum = sim if accum is None else accum + sim
            accum /= float(len(layers))
            row[int(doc_index)] = accum.max(axis=1).mean()
        outputs[query_index] = row
    return outputs


def run_router_scores(docs, queries, doc_ids, query_ids, checkpoint_dir: Path, family: str, layers, prefilter_top_k: int, workers: int, query_batch_size: int):
    prefilter_names = [f"{family}_{layer}" for layer in layers]
    prefilter_parts = [
        np.load(checkpoint_dir / "group_scores" / f"{name}.npy", mmap_mode="r")
        for name in prefilter_names
    ]
    prefilter = np.zeros_like(np.asarray(prefilter_parts[0], dtype=np.float32))
    for part in prefilter_parts:
        prefilter += np.asarray(part, dtype=np.float32)
    prefilter /= float(len(prefilter_parts))
    candidates = np.argpartition(-prefilter, prefilter_top_k - 1, axis=1)[:, :prefilter_top_k]

    ctx = mp.get_context("fork")
    query_batches = [
        list(range(start, min(start + query_batch_size, len(query_ids))))
        for start in range(0, len(query_ids), query_batch_size)
    ]
    score_matrix = np.zeros((len(query_ids), len(doc_ids)), dtype=np.float32)
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=configure_worker,
        initargs=(docs, queries, doc_ids, query_ids),
    ) as executor:
        futures = [
            executor.submit(score_router_batch, batch, family, layers, candidates)
            for batch in query_batches
        ]
        for future in as_completed(futures):
            batch_scores = future.result()
            for query_index, row in batch_scores.items():
                score_matrix[query_index] = row
    return score_matrix


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = project_root / "results"
    checkpoint_dir = args.checkpoint_dir or (results_dir / "checkpoints")
    checkpoint_dir = checkpoint_dir.resolve()
    alpha_grid = parse_csv(args.alpha_grid, float)
    rrf_k_grid = parse_csv(args.rrf_k_grid, int)

    doc_ids, query_ids = read_ids(results_dir)
    qrels = load_scifact_qrels(project_root / "datasets")
    docs, queries = load_all_texts(
        project_root,
        define_groups(),
        args.max_doc_tokens,
        args.max_query_tokens,
    )

    single_scores = build_single_vector_scores(docs, queries, doc_ids, query_ids)
    multivector_scores = {
        "attn_out_sign_35": np.load(checkpoint_dir / "group_scores" / "attn_out_sign_35.npy", mmap_mode="r"),
        "router_full_top3_avg": run_router_scores(
            docs,
            queries,
            doc_ids,
            query_ids,
            checkpoint_dir,
            "router_full",
            [20, 8, 13],
            args.prefilter_top_k,
            args.workers,
            args.query_batch_size,
        ),
        "router_topk_top3_avg": run_router_scores(
            docs,
            queries,
            doc_ids,
            query_ids,
            checkpoint_dir,
            "router_topk",
            [2, 20, 1],
            args.prefilter_top_k,
            args.workers,
            args.query_batch_size,
        ),
        "router_full_all_layers_avg": run_router_scores(
            docs,
            queries,
            doc_ids,
            query_ids,
            checkpoint_dir,
            "router_full",
            list(range(40)),
            args.prefilter_top_k,
            args.workers,
            args.query_batch_size,
        ),
        "router_topk_all_layers_avg": run_router_scores(
            docs,
            queries,
            doc_ids,
            query_ids,
            checkpoint_dir,
            "router_topk",
            list(range(40)),
            args.prefilter_top_k,
            args.workers,
            args.query_batch_size,
        ),
    }

    experiments = {
        "single_attn_vs_router_full_top3": (
            single_scores["attn_out_sign_35"],
            single_scores["router_full_top3_avg"],
        ),
        "single_attn_vs_router_topk_top3": (
            single_scores["attn_out_sign_35"],
            single_scores["router_topk_top3_avg"],
        ),
        "single_attn_vs_router_full_all_layers": (
            single_scores["attn_out_sign_35"],
            single_scores["router_full_all_layers_avg"],
        ),
        "single_attn_vs_router_topk_all_layers": (
            single_scores["attn_out_sign_35"],
            single_scores["router_topk_all_layers_avg"],
        ),
        "multivector_attn_vs_router_full_top3": (
            multivector_scores["attn_out_sign_35"],
            multivector_scores["router_full_top3_avg"],
        ),
        "multivector_attn_vs_router_topk_top3": (
            multivector_scores["attn_out_sign_35"],
            multivector_scores["router_topk_top3_avg"],
        ),
        "multivector_attn_vs_router_full_all_layers": (
            multivector_scores["attn_out_sign_35"],
            multivector_scores["router_full_all_layers_avg"],
        ),
        "multivector_attn_vs_router_topk_all_layers": (
            multivector_scores["attn_out_sign_35"],
            multivector_scores["router_topk_all_layers_avg"],
        ),
    }

    summary = {}
    grid_rows = []
    for name, (attn_scores, router_scores) in experiments.items():
        experiment_summary = {
            "attn_ndcg_at_10": ndcg_at_10(attn_scores, query_ids, doc_ids, qrels),
            "router_ndcg_at_10": ndcg_at_10(router_scores, query_ids, doc_ids, qrels),
        }

        attn_top = top_lists(attn_scores, query_ids, doc_ids, args.prefilter_top_k)
        router_top = top_lists(router_scores, query_ids, doc_ids, args.prefilter_top_k)

        for label, a_scores, b_scores in (
            ("raw_sum", attn_scores, router_scores),
            ("zscore_sum", q_zscore(attn_scores), q_zscore(router_scores)),
            ("minmax_sum", q_minmax(attn_scores), q_minmax(router_scores)),
        ):
            best = (-1.0, None)
            for alpha in alpha_grid:
                fused = a_scores + alpha * b_scores
                ndcg = ndcg_at_10(fused, query_ids, doc_ids, qrels)
                grid_rows.append(
                    {
                        "experiment": name,
                        "method": label,
                        "param": f"{alpha:g}",
                        "ndcg_at_10": ndcg,
                    }
                )
                if ndcg > best[0]:
                    best = (ndcg, alpha)
            experiment_summary[label] = {"ndcg_at_10": best[0], "alpha": best[1]}

        best_rrf = (-1.0, None)
        for k in rrf_k_grid:
            ranking = reciprocal_rank_fusion(attn_top, router_top, k=k, out_top_k=10)
            ndcg = ndcg_from_rankings(ranking, query_ids, qrels)
            grid_rows.append(
                {"experiment": name, "method": "rrf", "param": str(k), "ndcg_at_10": ndcg}
            )
            if ndcg > best_rrf[0]:
                best_rrf = (ndcg, k)
        experiment_summary["rrf"] = {"ndcg_at_10": best_rrf[0], "k": best_rrf[1]}

        for first in ("attn_first", "router_first"):
            ranking = interleave(
                attn_top,
                router_top,
                first="a" if first == "attn_first" else "b",
                out_top_k=10,
            )
            ndcg = ndcg_from_rankings(ranking, query_ids, qrels)
            experiment_summary[f"interleave_{first}"] = {"ndcg_at_10": ndcg}
            grid_rows.append(
                {"experiment": name, "method": f"interleave_{first}", "param": "-", "ndcg_at_10": ndcg}
            )

        summary[name] = experiment_summary

    summary_path = results_dir / f"{args.output_prefix}_summary.json"
    grid_path = results_dir / f"{args.output_prefix}_grid.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with grid_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "method", "param", "ndcg_at_10"])
        writer.writeheader()
        writer.writerows(grid_rows)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {grid_path}")


if __name__ == "__main__":
    main()
