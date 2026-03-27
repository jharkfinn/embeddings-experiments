from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from run_binary_notebook_analysis import load_all_texts, load_scifact_qrels


def _read_ids(results_dir: Path):
    records = []
    with (results_dir / "records.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(row)
    doc_ids = sorted({row["text_id"] for row in records if row["kind"] == "doc"})
    query_ids = sorted({row["text_id"] for row in records if row["kind"] == "query"})
    return doc_ids, query_ids


def _ndcg_at_10(scores, query_ids, doc_ids, qrels):
    values = []
    for query_index, query_id in enumerate(query_ids):
        rels = qrels.get(query_id)
        if not rels:
            continue
        ranked = np.argsort(-scores[query_index])[:10]
        dcg = 0.0
        for rank, doc_index in enumerate(ranked):
            dcg += rels.get(doc_ids[int(doc_index)], 0) / np.log2(rank + 2)
        ideal = sorted(rels.values(), reverse=True)[:10]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
        if idcg > 0:
            values.append(dcg / idcg)
    return float(np.mean(values)) if values else 0.0


def _qnorm(scores):
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (scores - mean) / std


def _l2norm(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return matrix / norms


def _mean_pool_packed(packed):
    unpacked = np.unpackbits(np.asarray(packed, dtype=np.uint8), axis=1).astype(np.float32)
    return unpacked.mean(axis=0)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_single_vector_ablation.py <project_root>")

    project_root = Path(sys.argv[1]).resolve()
    results_dir = project_root / "results"

    doc_ids, query_ids = _read_ids(results_dir)
    qrels = load_scifact_qrels(project_root / "datasets")

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

    docs, queries = load_all_texts(project_root, groups, 512, 64)

    variants = {
        "attn_out_sign_35": ["attn_out_sign_35"],
        "router_full_top3_avg": ["router_full_20", "router_full_8", "router_full_13"],
        "router_full_all_layers_avg": [f"router_full_{idx}" for idx in range(40)],
        "router_topk_top3_avg": ["router_topk_2", "router_topk_20", "router_topk_1"],
        "router_topk_all_layers_avg": [f"router_topk_{idx}" for idx in range(40)],
    }

    doc_embeddings = {}
    query_embeddings = {}
    for variant, group_names in variants.items():
        doc_matrix = []
        for doc_id in doc_ids:
            pooled = [_mean_pool_packed(docs[doc_id][group_name]) for group_name in group_names]
            doc_matrix.append(np.mean(pooled, axis=0))
        query_matrix = []
        for query_id in query_ids:
            pooled = [_mean_pool_packed(queries[query_id][group_name]) for group_name in group_names]
            query_matrix.append(np.mean(pooled, axis=0))
        doc_embeddings[variant] = _l2norm(np.stack(doc_matrix).astype(np.float32))
        query_embeddings[variant] = _l2norm(np.stack(query_matrix).astype(np.float32))

    score_matrices = {}
    for variant in variants:
        score_matrices[variant] = query_embeddings[variant] @ doc_embeddings[variant].T

    alphas = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
    base_scores = _qnorm(score_matrices["attn_out_sign_35"])

    summary = {"single_vector_mean_pool": {}}
    summary["single_vector_mean_pool"]["attn_out_sign_35"] = _ndcg_at_10(
        score_matrices["attn_out_sign_35"], query_ids, doc_ids, qrels
    )

    for variant in (
        "router_full_top3_avg",
        "router_full_all_layers_avg",
        "router_topk_top3_avg",
        "router_topk_all_layers_avg",
    ):
        router_scores = score_matrices[variant]
        summary["single_vector_mean_pool"][variant] = _ndcg_at_10(
            router_scores, query_ids, doc_ids, qrels
        )
        router_z = _qnorm(router_scores)
        best_value = (-1.0, None)
        for alpha in alphas:
            combined = base_scores + alpha * router_z
            ndcg = _ndcg_at_10(combined, query_ids, doc_ids, qrels)
            if ndcg > best_value[0]:
                best_value = (ndcg, alpha)
        summary["single_vector_mean_pool"][f"attn_out_sign_35_plus_{variant}"] = {
            "alpha": best_value[1],
            "ndcg_at_10": best_value[0],
        }

    output_path = results_dir / "single_vector_mean_pool_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
