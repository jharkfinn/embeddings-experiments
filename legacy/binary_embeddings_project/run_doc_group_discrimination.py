from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_binary_notebook_analysis import (
    DEFAULT_WORKERS,
    define_feature_groups,
    load_all_texts,
    load_scifact_qrels,
    maxsim_one_pair,
)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

_WORKER_DOCS = None
_WORKER_GROUP = None
_WORKER_ANCHOR_TASKS = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-negative-docs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-prefix", type=str, default="doc_group_discrimination"
    )
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


def _stable_sample(items, size, seed_text: str):
    if size >= len(items):
        return list(items)
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(items), size=size, replace=False)
    return [items[int(idx)] for idx in indices]


def build_anchor_tasks(qrels, doc_ids, max_negative_docs: int):
    doc_set = set(doc_ids)
    tasks = []
    group_sizes = {}
    for query_id, rels in sorted(qrels.items()):
        positives = sorted(rels.keys())
        if len(positives) < 2:
            continue
        group_sizes[query_id] = len(positives)
        negatives = sorted(doc_set - set(positives))
        for anchor_doc_id in positives:
            positive_targets = [doc_id for doc_id in positives if doc_id != anchor_doc_id]
            sampled_negatives = _stable_sample(
                negatives,
                max_negative_docs,
                seed_text=f"{query_id}::{anchor_doc_id}::{max_negative_docs}",
            )
            candidate_doc_ids = positive_targets + sampled_negatives
            relevance = {doc_id: 1 for doc_id in positive_targets}
            tasks.append(
                {
                    "query_id": query_id,
                    "anchor_doc_id": anchor_doc_id,
                    "candidate_doc_ids": candidate_doc_ids,
                    "relevance": relevance,
                }
            )
    return tasks, group_sizes


def _l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return matrix / norms


def _mean_pool_packed(packed):
    unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1).astype(np.float32)
    return unpacked.mean(axis=0)


def _average_precision(scores, candidate_doc_ids, relevance):
    ranked = np.argsort(-scores)
    hit_count = 0
    precisions = []
    for rank, idx in enumerate(ranked, start=1):
        doc_id = candidate_doc_ids[int(idx)]
        if relevance.get(doc_id, 0) > 0:
            hit_count += 1
            precisions.append(hit_count / rank)
    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def _ndcg_at_10(scores, candidate_doc_ids, relevance):
    ranked = np.argsort(-scores)[:10]
    dcg = 0.0
    for rank, idx in enumerate(ranked, start=1):
        doc_id = candidate_doc_ids[int(idx)]
        dcg += relevance.get(doc_id, 0) / np.log2(rank + 1)
    ideal = sorted(relevance.values(), reverse=True)[:10]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _configure_worker_state(*, docs, group, anchor_tasks):
    global _WORKER_DOCS
    global _WORKER_GROUP
    global _WORKER_ANCHOR_TASKS

    _WORKER_DOCS = docs
    _WORKER_GROUP = group
    _WORKER_ANCHOR_TASKS = anchor_tasks


def _score_anchor_batch(anchor_indices):
    group_name = _WORKER_GROUP["name"]

    needed_doc_ids = set()
    for anchor_index in anchor_indices:
        task = _WORKER_ANCHOR_TASKS[int(anchor_index)]
        needed_doc_ids.add(task["anchor_doc_id"])
        needed_doc_ids.update(task["candidate_doc_ids"])

    single_vectors = {}
    for doc_id in needed_doc_ids:
        single_vectors[doc_id] = _mean_pool_packed(_WORKER_DOCS[doc_id][group_name])
    doc_matrix = np.stack([single_vectors[doc_id] for doc_id in sorted(needed_doc_ids)]).astype(
        np.float32
    )
    doc_matrix = _l2_normalize(doc_matrix)
    vector_lookup = {
        doc_id: doc_matrix[idx] for idx, doc_id in enumerate(sorted(needed_doc_ids))
    }

    outputs = []
    for anchor_index in anchor_indices:
        task = _WORKER_ANCHOR_TASKS[int(anchor_index)]
        anchor_doc_id = task["anchor_doc_id"]
        candidate_doc_ids = task["candidate_doc_ids"]
        relevance = task["relevance"]

        anchor_bits = np.asarray(_WORKER_DOCS[anchor_doc_id][group_name], dtype=np.uint8)
        token_scores = np.array(
            [
                maxsim_one_pair(anchor_bits, np.asarray(_WORKER_DOCS[doc_id][group_name], dtype=np.uint8))
                for doc_id in candidate_doc_ids
            ],
            dtype=np.float32,
        )

        anchor_vec = vector_lookup[anchor_doc_id]
        single_scores = np.array(
            [float(anchor_vec @ vector_lookup[doc_id]) for doc_id in candidate_doc_ids],
            dtype=np.float32,
        )

        outputs.append(
            {
                "anchor_index": int(anchor_index),
                "token_ap": _average_precision(token_scores, candidate_doc_ids, relevance),
                "token_ndcg_at_10": _ndcg_at_10(token_scores, candidate_doc_ids, relevance),
                "single_ap": _average_precision(single_scores, candidate_doc_ids, relevance),
                "single_ndcg_at_10": _ndcg_at_10(single_scores, candidate_doc_ids, relevance),
            }
        )

    return outputs


def evaluate_group(docs, group, anchor_tasks, workers, batch_size=4):
    _configure_worker_state(docs=docs, group=group, anchor_tasks=anchor_tasks)
    anchor_batches = [
        list(range(start, min(start + batch_size, len(anchor_tasks))))
        for start in range(0, len(anchor_tasks), batch_size)
    ]

    token_ap_values = []
    token_ndcg_values = []
    single_ap_values = []
    single_ndcg_values = []

    executor = _get_executor(workers) if workers > 1 else None
    if executor is not None:
        with executor:
            future_map = {
                executor.submit(_score_anchor_batch, batch): batch for batch in anchor_batches
            }
            for future in as_completed(future_map):
                outputs = future.result()
                for row in outputs:
                    token_ap_values.append(row["token_ap"])
                    token_ndcg_values.append(row["token_ndcg_at_10"])
                    single_ap_values.append(row["single_ap"])
                    single_ndcg_values.append(row["single_ndcg_at_10"])
    else:
        for batch in anchor_batches:
            outputs = _score_anchor_batch(batch)
            for row in outputs:
                token_ap_values.append(row["token_ap"])
                token_ndcg_values.append(row["token_ndcg_at_10"])
                single_ap_values.append(row["single_ap"])
                single_ndcg_values.append(row["single_ndcg_at_10"])

    return {
        "token_map": float(np.mean(token_ap_values)) if token_ap_values else 0.0,
        "token_ndcg_at_10": float(np.mean(token_ndcg_values)) if token_ndcg_values else 0.0,
        "single_map": float(np.mean(single_ap_values)) if single_ap_values else 0.0,
        "single_ndcg_at_10": float(np.mean(single_ndcg_values)) if single_ndcg_values else 0.0,
    }


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = project_root / "results"

    groups = define_feature_groups()
    docs, _queries = load_all_texts(project_root, groups, args.max_doc_tokens, args.max_query_tokens)
    doc_ids, _query_ids = _read_ids(results_dir)
    qrels = load_scifact_qrels(project_root / "datasets")
    anchor_tasks, group_sizes = build_anchor_tasks(qrels, doc_ids, args.max_negative_docs)

    print(
        f"Loaded {len(docs)} docs; using {len(group_sizes)} multi-doc groups and {len(anchor_tasks)} anchors"
    )

    rows = []
    t0 = time.perf_counter()
    workers = max(1, min(args.workers, os.cpu_count() or 1))

    for idx, group in enumerate(groups, start=1):
        metrics = evaluate_group(docs, group, anchor_tasks, workers)
        row = {
            "group_name": group["name"],
            "group_type": group["type"],
            "layer": group["layer"],
            "num_bits": group["num_bits"],
            "token_ndcg_at_10": metrics["token_ndcg_at_10"],
            "token_map": metrics["token_map"],
            "single_ndcg_at_10": metrics["single_ndcg_at_10"],
            "single_map": metrics["single_map"],
        }
        rows.append(row)
        if idx % 10 == 0 or idx == len(groups):
            elapsed = time.perf_counter() - t0
            print(f"  scored {idx}/{len(groups)} groups ({elapsed:.1f}s)")

    rows.sort(key=lambda row: row["token_ndcg_at_10"], reverse=True)

    output_prefix = args.output_prefix
    csv_path = results_dir / f"{output_prefix}.csv"
    summary_path = results_dir / f"{output_prefix}_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_name",
                "group_type",
                "layer",
                "num_bits",
                "token_ndcg_at_10",
                "token_map",
                "single_ndcg_at_10",
                "single_map",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    single_sorted = sorted(rows, key=lambda row: row["single_ndcg_at_10"], reverse=True)
    summary = {
        "num_doc_groups": len(group_sizes),
        "num_anchors": len(anchor_tasks),
        "max_negative_docs": args.max_negative_docs,
        "top_token_groups": rows[:20],
        "top_single_vector_groups": single_sorted[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
