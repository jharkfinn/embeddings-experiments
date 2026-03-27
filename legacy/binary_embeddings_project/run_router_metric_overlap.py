from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_binary_notebook_analysis import _POPCOUNT_TABLE, _chunked, load_all_texts, load_scifact_qrels

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

NUM_LAYERS = 40
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 1))

_WORKER_DOCS = None
_WORKER_QUERIES = None
_WORKER_DOC_IDS = None
_WORKER_QUERY_IDS = None
_WORKER_CANDIDATE_DOCS = None
_WORKER_VARIANTS = None
_WORKER_METRIC = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--prefilter-top-k", type=int, default=200)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--output-prefix", type=str, default="router_metric_overlap")
    parser.add_argument("--metrics", type=str, default="active_cosine,jaccard")
    return parser.parse_args()


def _get_executor(max_workers: int):
    if max_workers <= 1:
        return None
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)


def _read_ids(results_dir: Path):
    records = []
    with (results_dir / "records.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(row)
    doc_ids = sorted({row["text_id"] for row in records if row["kind"] == "doc"})
    query_ids = sorted({row["text_id"] for row in records if row["kind"] == "query"})
    return doc_ids, query_ids


def define_groups():
    groups = []
    groups.append(
        {
            "name": "attn_out_sign_35",
            "type": "sign",
            "field": "attn_out_35",
            "layer": 35,
            "num_bits": 2048,
            "packed_bytes": 256,
        }
    )
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
        "router_full_20": ["router_full_20"],
        "router_full_top3_avg": ["router_full_20", "router_full_8", "router_full_13"],
        "router_full_all_layers_avg": [f"router_full_{idx}" for idx in range(NUM_LAYERS)],
        "router_topk_2": ["router_topk_2"],
        "router_topk_top3_avg": ["router_topk_2", "router_topk_20", "router_topk_1"],
        "router_topk_all_layers_avg": [f"router_topk_{idx}" for idx in range(NUM_LAYERS)],
    }


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
    inter = _POPCOUNT_TABLE[np.bitwise_and(q_bits[:, None, :], d_bits[None, :, :])].astype(
        np.uint16
    ).sum(axis=2)
    q_count = _count_bits(q_bits)[:, None]
    d_count = _count_bits(d_bits)[None, :]
    denom = np.sqrt(q_count * d_count, dtype=np.float32)
    return np.divide(inter, denom, out=np.zeros_like(inter, dtype=np.float32), where=denom > 0)


def _pairwise_jaccard(q_bits, d_bits):
    inter = _POPCOUNT_TABLE[np.bitwise_and(q_bits[:, None, :], d_bits[None, :, :])].astype(
        np.uint16
    ).sum(axis=2)
    union = _POPCOUNT_TABLE[np.bitwise_or(q_bits[:, None, :], d_bits[None, :, :])].astype(
        np.uint16
    ).sum(axis=2)
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def _score_variant_pair(query_bits_by_group, doc_bits_by_group, group_names, metric_name):
    accumulator = None
    for group_name in group_names:
        q_bits = np.asarray(query_bits_by_group[group_name], dtype=np.uint8)
        d_bits = np.asarray(doc_bits_by_group[group_name], dtype=np.uint8)
        if metric_name == "active_cosine":
            sim = _pairwise_active_cosine(q_bits, d_bits)
        elif metric_name == "jaccard":
            sim = _pairwise_jaccard(q_bits, d_bits)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")
        if accumulator is None:
            accumulator = sim
        else:
            accumulator += sim
    fused = accumulator / float(len(group_names))
    return float(fused.max(axis=1).mean())


def _configure_worker_state(*, docs, queries, doc_ids, query_ids, candidate_docs, variants, metric):
    global _WORKER_DOCS
    global _WORKER_QUERIES
    global _WORKER_DOC_IDS
    global _WORKER_QUERY_IDS
    global _WORKER_CANDIDATE_DOCS
    global _WORKER_VARIANTS
    global _WORKER_METRIC

    _WORKER_DOCS = docs
    _WORKER_QUERIES = queries
    _WORKER_DOC_IDS = doc_ids
    _WORKER_QUERY_IDS = query_ids
    _WORKER_CANDIDATE_DOCS = candidate_docs
    _WORKER_VARIANTS = variants
    _WORKER_METRIC = metric


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
                outputs[variant_name][batch_row, int(doc_index)] = _score_variant_pair(
                    query_bits, doc_bits, group_names, _WORKER_METRIC
                )
    return list(query_indices), outputs


def _ndcg_at_10(scores_matrix, query_ids, doc_ids, qrels):
    values = []
    for query_index, query_id in enumerate(query_ids):
        rels = qrels.get(query_id)
        if not rels:
            continue
        ranked = np.argsort(-scores_matrix[query_index])[:10]
        dcg = 0.0
        for rank, doc_index in enumerate(ranked):
            dcg += rels.get(doc_ids[int(doc_index)], 0) / np.log2(rank + 2)
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else 0.0


def _per_query_ndcg(scores_matrix, query_ids, doc_ids, qrels):
    values = []
    for query_index, query_id in enumerate(query_ids):
        rels = qrels.get(query_id)
        if not rels:
            values.append(0.0)
            continue
        ranked = np.argsort(-scores_matrix[query_index])[:10]
        dcg = 0.0
        for rank, doc_index in enumerate(ranked):
            dcg += rels.get(doc_ids[int(doc_index)], 0) / np.log2(rank + 2)
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
        values.append(float(dcg / idcg) if idcg > 0 else 0.0)
    return np.asarray(values, dtype=np.float32)


def _hit_at_10_set(scores_matrix, query_ids, doc_ids, qrels):
    hit_queries = set()
    top10_docs = {}
    top1_docs = {}
    for query_index, query_id in enumerate(query_ids):
        rels = qrels.get(query_id)
        ranked = np.argsort(-scores_matrix[query_index])[:10]
        docs = [doc_ids[int(doc_index)] for doc_index in ranked]
        top10_docs[query_id] = docs
        top1_docs[query_id] = docs[0] if docs else None
        if rels and any(doc_id in rels for doc_id in docs):
            hit_queries.add(query_id)
    return hit_queries, top10_docs, top1_docs


def _overlap_summary(attn_scores, router_scores, query_ids, doc_ids, qrels):
    attn_ndcg = _per_query_ndcg(attn_scores, query_ids, doc_ids, qrels)
    router_ndcg = _per_query_ndcg(router_scores, query_ids, doc_ids, qrels)
    attn_hits, attn_top10, attn_top1 = _hit_at_10_set(attn_scores, query_ids, doc_ids, qrels)
    router_hits, router_top10, router_top1 = _hit_at_10_set(router_scores, query_ids, doc_ids, qrels)

    both = attn_hits & router_hits
    only_attn = attn_hits - router_hits
    only_router = router_hits - attn_hits
    neither = set(query_ids) - (attn_hits | router_hits)

    top10_jaccards = []
    same_top1 = 0
    for query_id in query_ids:
        set_a = set(attn_top10[query_id])
        set_b = set(router_top10[query_id])
        union = set_a | set_b
        if union:
            top10_jaccards.append(len(set_a & set_b) / len(union))
        if attn_top1[query_id] == router_top1[query_id]:
            same_top1 += 1

    return {
        "attn_ndcg_at_10": float(np.mean(attn_ndcg)),
        "router_ndcg_at_10": float(np.mean(router_ndcg)),
        "attn_better_queries": int(np.sum(attn_ndcg > router_ndcg)),
        "router_better_queries": int(np.sum(router_ndcg > attn_ndcg)),
        "ties": int(np.sum(attn_ndcg == router_ndcg)),
        "hit_at_10_overlap": {
            "both": len(both),
            "only_attn": len(only_attn),
            "only_router": len(only_router),
            "neither": len(neither),
            "jaccard": (len(both) / len(attn_hits | router_hits)) if (attn_hits | router_hits) else 0.0,
        },
        "avg_top10_doc_jaccard": float(np.mean(top10_jaccards)) if top10_jaccards else 0.0,
        "same_top1_doc_queries": same_top1,
        "per_query_oracle_ndcg_at_10": float(np.mean(np.maximum(attn_ndcg, router_ndcg))),
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

    groups = define_groups()
    docs, queries = load_all_texts(project_root, groups, args.max_doc_tokens, args.max_query_tokens)
    qrels = load_scifact_qrels(project_root / "datasets")
    doc_ids, query_ids = _read_ids(results_dir)
    variants = define_router_variants()

    top_k = min(args.prefilter_top_k, len(doc_ids))
    candidate_docs = {}
    for variant_name, group_names in variants.items():
        prefilter_scores = _average_score_matrix(checkpoint_dir, group_names)
        if top_k >= len(doc_ids):
            selected = np.broadcast_to(np.arange(len(doc_ids), dtype=np.int32), prefilter_scores.shape)
        else:
            selected = np.argpartition(-prefilter_scores, top_k - 1, axis=1)[:, :top_k]
        candidate_docs[variant_name] = np.asarray(selected, dtype=np.int32)

    metric_results = {}
    workers = max(1, min(args.workers, os.cpu_count() or 1))
    query_batches = [list(batch) for batch in _chunked(list(range(len(query_ids))), max(1, args.query_batch_size))]

    metric_names = [item.strip() for item in args.metrics.split(",") if item.strip()]
    for metric_name in metric_names:
        print(f"Running multivector router metric: {metric_name}")
        _configure_worker_state(
            docs=docs,
            queries=queries,
            doc_ids=doc_ids,
            query_ids=query_ids,
            candidate_docs=candidate_docs,
            variants=variants,
            metric=metric_name,
        )
        score_matrices = {
            variant_name: np.zeros((len(query_ids), len(doc_ids)), dtype=np.float32)
            for variant_name in variants
        }
        t0 = time.perf_counter()
        executor = _get_executor(workers) if workers > 1 else None
        if executor is not None:
            with executor:
                future_map = {executor.submit(_score_query_batch, batch): batch for batch in query_batches}
                processed = 0
                for count, future in enumerate(as_completed(future_map), start=1):
                    batch_indices, batch_outputs = future.result()
                    for variant_name, batch_scores in batch_outputs.items():
                        for row_idx, query_index in enumerate(batch_indices):
                            score_matrices[variant_name][query_index] = batch_scores[row_idx]
                    processed += len(batch_indices)
                    if count % 5 == 0 or count == len(query_batches):
                        print(
                            f"  {metric_name}: processed {processed}/{len(query_ids)} queries ({time.perf_counter() - t0:.1f}s)"
                        )
        else:
            for count, batch in enumerate(query_batches, start=1):
                batch_indices, batch_outputs = _score_query_batch(batch)
                for variant_name, batch_scores in batch_outputs.items():
                    for row_idx, query_index in enumerate(batch_indices):
                        score_matrices[variant_name][query_index] = batch_scores[row_idx]
                if count % 5 == 0 or count == len(query_batches):
                    print(
                        f"  {metric_name}: processed {min(count * len(batch), len(query_ids))}/{len(query_ids)} queries ({time.perf_counter() - t0:.1f}s)"
                    )

        metric_results[metric_name] = {
            variant_name: {
                "ndcg_at_10": _ndcg_at_10(score_matrices[variant_name], query_ids, doc_ids, qrels)
            }
            for variant_name in variants
        }

        if metric_name == "active_cosine":
            attn_scores = np.asarray(_load_score_matrix(checkpoint_dir, "attn_out_sign_35"), dtype=np.float32)
            overlap = {}
            for variant_name in ("router_full_top3_avg", "router_topk_top3_avg", "router_full_all_layers_avg", "router_topk_all_layers_avg"):
                overlap[variant_name] = _overlap_summary(
                    attn_scores, score_matrices[variant_name], query_ids, doc_ids, qrels
                )
            metric_results[metric_name]["overlap_vs_attn_out_sign_35"] = overlap

    single_vector_path = results_dir / "single_vector_mean_pool_summary.json"
    single_vector_summary = {}
    if single_vector_path.exists():
        single_vector_summary = json.loads(single_vector_path.read_text(encoding="utf-8"))

    output_prefix = args.output_prefix
    summary_path = results_dir / f"{output_prefix}_summary.json"
    summary = {
        "prefilter_top_k": top_k,
        "single_vector_mean_pool_existing": single_vector_summary,
        "multivector_router_metrics": metric_results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
