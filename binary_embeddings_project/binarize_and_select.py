"""
Binary feature discovery for late-interaction retrieval on SciFact.

Fast approach: compute per-group MaxSim scores independently for each
feature group, then find the best weighted combination using supervision.

Pipeline:
1. Load extracted shards, binarize all signals
2. Compute per-group MaxSim score matrices (group × query × doc)
3. Use qrels to find best group combinations via greedy forward selection
   on precomputed score matrices (instant per evaluation)
4. Report best feature subsets at various bit budgets
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
NUM_LAYERS = 40
NUM_EXPERTS = 256

# Popcount lookup table
_POPCOUNT_TABLE = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--max-select-steps", type=int, default=30)
    p.add_argument("--eval-budgets", type=str, default="256,512,1024,2048,4096,8192")
    p.add_argument("--max-doc-tokens", type=int, default=512)
    p.add_argument("--max-query-tokens", type=int, default=64)
    p.add_argument("--prefilter-top-n", type=int, default=100,
                   help="Prefilter docs per query for MaxSim")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

def define_feature_groups():
    groups = []
    for l in range(NUM_LAYERS):
        groups.append({
            "name": f"router_{l}", "type": "router", "layer": l,
            "num_bits": NUM_EXPERTS, "packed_bytes": NUM_EXPERTS // 8,
        })
    for l in ATTN_LAYERS:
        groups.append({
            "name": f"q_sign_{l}", "type": "sign", "field": f"q_{l}",
            "layer": l, "num_bits": 4096, "packed_bytes": 512,
        })
        groups.append({
            "name": f"k_sign_{l}", "type": "sign", "field": f"k_{l}",
            "layer": l, "num_bits": 512, "packed_bytes": 64,
        })
        groups.append({
            "name": f"v_sign_{l}", "type": "sign", "field": f"v_{l}",
            "layer": l, "num_bits": 512, "packed_bytes": 64,
        })
        groups.append({
            "name": f"attn_out_sign_{l}", "type": "sign",
            "field": f"attn_out_{l}", "layer": l,
            "num_bits": 2048, "packed_bytes": 256,
        })
    groups.append({
        "name": "hs_final_sign", "type": "sign", "field": "hs_final",
        "layer": -1, "num_bits": 2048, "packed_bytes": 256,
    })
    return groups


# ---------------------------------------------------------------------------
# Shard loading + binarization
# ---------------------------------------------------------------------------

def binarize_router_topk(tensors, meta, offset, length, layer_idx):
    field_name = f"router_topk_ids_{layer_idx}"
    ids = tensors[meta["fields"][field_name]["data_key"]][offset:offset + length]
    seq_len = ids.shape[0]
    bits = np.zeros((seq_len, NUM_EXPERTS), dtype=np.uint8)
    for k in range(ids.shape[1]):
        col = ids[:, k].astype(np.int32)
        valid = (col >= 0) & (col < NUM_EXPERTS)
        rows = np.arange(seq_len)[valid]
        bits[rows, col[valid]] = 1
    return np.packbits(bits, axis=1)


def binarize_sign_int8(tensors, meta, offset, length, field_name):
    field_info = meta["fields"][field_name]
    if field_info["encoding"] == "int8_rowwise":
        data = tensors[field_info["data_key"]][offset:offset + length]
        sign_bits = (data > 0).astype(np.uint8)
    else:
        data = tensors[field_info["data_key"]][offset:offset + length]
        sign_bits = (data > 0).astype(np.uint8)
    return np.packbits(sign_bits, axis=1)


def load_all_texts(project_root, groups, max_doc_tokens, max_query_tokens):
    """Load shards and binarize. Returns per-group packed bits per text."""
    from safetensors.numpy import load_file

    shard_dir = project_root / "embeddings" / "echo" / "shards"
    results_dir = project_root / "results"

    records = []
    with open(results_dir / "records.csv") as f:
        for row in csv.DictReader(f):
            row["ordinal"] = int(row["ordinal"])
            row["record_index"] = int(row["record_index"])
            row["token_offset"] = int(row["token_offset"])
            row["seq_len"] = int(row["seq_len"])
            records.append(row)

    shard_records = defaultdict(list)
    for rec in records:
        shard_records[rec["shard_stem"]].append(rec)

    # text_id -> {group_name: packed_bits (seq_len, packed_bytes)}
    docs = {}
    queries = {}
    done = 0
    total = len(records)
    t0 = time.perf_counter()

    for shard_stem, recs in sorted(shard_records.items()):
        shard_path = shard_dir / f"{shard_stem}.safetensors"
        meta_path = shard_dir / f"{shard_stem}.json"
        meta = json.loads(meta_path.read_text())
        tensors = load_file(str(shard_path))

        for rec in recs:
            max_tok = max_query_tokens if rec["kind"] == "query" else max_doc_tokens
            length = min(rec["seq_len"], max_tok)
            offset = rec["token_offset"]

            bits = {}
            for g in groups:
                if g["type"] == "router":
                    bits[g["name"]] = binarize_router_topk(
                        tensors, meta, offset, length, g["layer"])
                elif g["type"] == "sign":
                    bits[g["name"]] = binarize_sign_int8(
                        tensors, meta, offset, length, g["field"])

            if rec["kind"] == "doc":
                docs[rec["text_id"]] = bits
            else:
                queries[rec["text_id"]] = bits
            done += 1

        if done % 500 == 0 or done == total:
            print(f"  Binarized {done}/{total} ({time.perf_counter()-t0:.1f}s)")
        del tensors

    return docs, queries


# ---------------------------------------------------------------------------
# Per-group MaxSim score computation
# ---------------------------------------------------------------------------

def maxsim_one_pair(q_bits, d_bits):
    """MaxSim between one query and one doc for a single feature group.
    q_bits: (q_len, packed_bytes), d_bits: (d_len, packed_bytes)
    Returns: float score = mean_q max_d hamming_agreement(q,d)

    Uses uint64 view for fast XOR + byte popcount lookup.
    """
    q_len, nbytes = q_bits.shape
    d_len = d_bits.shape[0]

    # For small groups (≤64 bytes = 512 bits), use vectorized approach
    # XOR all pairs at once: (q_len, d_len, nbytes)
    xor = np.bitwise_xor(q_bits[:, None, :], d_bits[None, :, :])
    # Byte-level popcount via lookup, sum across bytes
    hamming_dist = _POPCOUNT_TABLE[xor].astype(np.uint16).sum(axis=2)
    # Agreement = total_bits - hamming_distance
    # MaxSim = mean over q of max over d of agreement
    # Since agreement = C - dist, max(agreement) = C - min(dist)
    min_dist = hamming_dist.min(axis=1)  # (q_len,)
    total_bits = nbytes * 8
    return (total_bits - min_dist.astype(np.float32)).mean()


def maxsim_batch_query(q_bits, d_bits_list, d_indices):
    """Compute MaxSim for one query against multiple docs at once.
    q_bits: (q_len, nbytes)
    d_bits_list: list of (d_len_i, nbytes) arrays
    d_indices: which doc indices these correspond to
    Returns: dict of {doc_index: score}
    """
    results = {}
    q_len, nbytes = q_bits.shape
    total_bits = nbytes * 8
    for idx, d_bits in zip(d_indices, d_bits_list):
        xor = np.bitwise_xor(q_bits[:, None, :], d_bits[None, :, :])
        hamming_dist = _POPCOUNT_TABLE[xor].astype(np.uint16).sum(axis=2)
        min_dist = hamming_dist.min(axis=1)
        results[idx] = (total_bits - min_dist.astype(np.float32)).mean()
    return results


def compute_group_score_matrix(docs, queries, doc_ids, query_ids,
                               group_name, prefilter_top_n):
    """Compute MaxSim scores for one feature group across all query-doc pairs.

    Uses single-vector prefiltering to top-N docs per query.
    Returns: score_matrix (num_queries, num_docs), prefilter_indices.
    """
    num_q = len(query_ids)
    num_d = len(doc_ids)

    # Precompute single-vector representations for prefiltering
    # Use bit-frequency vector: for each bit position, fraction of tokens with 1
    packed_bytes = docs[doc_ids[0]][group_name].shape[1]
    num_bits = packed_bytes * 8

    # Vectorized single-vector computation
    doc_vecs = np.zeros((num_d, num_bits), dtype=np.float32)
    for i, did in enumerate(doc_ids):
        bits = docs[did][group_name]
        doc_vecs[i] = np.unpackbits(bits, axis=1).astype(np.float32).mean(axis=0)

    query_vecs = np.zeros((num_q, num_bits), dtype=np.float32)
    for i, qid in enumerate(query_ids):
        bits = queries[qid][group_name]
        query_vecs[i] = np.unpackbits(bits, axis=1).astype(np.float32).mean(axis=0)

    # Dot product prefilter
    sv_scores = query_vecs @ doc_vecs.T  # (num_q, num_d)
    top_n = min(prefilter_top_n, num_d)
    prefilter_idx = np.argpartition(-sv_scores, top_n, axis=1)[:, :top_n]

    # Precompute doc packed bit arrays indexed by position
    doc_bits_by_idx = [docs[doc_ids[di]][group_name] for di in range(num_d)]

    # MaxSim on prefiltered pairs
    scores = np.zeros((num_q, num_d), dtype=np.float32)
    for qi in range(num_q):
        qid = query_ids[qi]
        q_bits = queries[qid][group_name]
        candidates = prefilter_idx[qi]
        for di in candidates:
            scores[qi, di] = maxsim_one_pair(q_bits, doc_bits_by_idx[di])

    return scores, prefilter_idx


# ---------------------------------------------------------------------------
# nDCG computation on score matrices
# ---------------------------------------------------------------------------

def ndcg_at_k_from_scores(scores, query_ids, doc_ids, qrels, k=10):
    """Compute nDCG@k from a score matrix."""
    ndcg_values = []
    for qi, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
        rels = qrels[qid]
        row = scores[qi]
        ranked_indices = np.argsort(-row)[:k]

        dcg = 0.0
        for rank, di in enumerate(ranked_indices):
            did = doc_ids[di]
            rel = rels.get(did, 0)
            dcg += rel / np.log2(rank + 2)

        ideal_rels = sorted(rels.values(), reverse=True)[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))
        if idcg > 0:
            ndcg_values.append(dcg / idcg)

    return np.mean(ndcg_values) if ndcg_values else 0.0


# ---------------------------------------------------------------------------
# Greedy forward selection on precomputed score matrices
# ---------------------------------------------------------------------------

def greedy_select(group_scores, group_names, groups_lookup,
                  query_ids, doc_ids, qrels, max_steps):
    """Greedy forward selection using precomputed per-group score matrices.

    Combined score = sum of selected group scores.
    Each step tries adding each remaining group and picks the one
    that maximizes nDCG@10. This is instant because we just add matrices.
    """
    selected = []
    combined = np.zeros_like(group_scores[group_names[0]])
    best_ndcg = 0.0
    remaining = set(group_names)

    for step in range(max_steps):
        best_cand = None
        best_cand_ndcg = best_ndcg

        for cand in sorted(remaining):
            trial_scores = combined + group_scores[cand]
            ndcg = ndcg_at_k_from_scores(
                trial_scores, query_ids, doc_ids, qrels)
            if ndcg > best_cand_ndcg:
                best_cand_ndcg = ndcg
                best_cand = cand

        if best_cand is None:
            print(f"  Step {step+1}: no improvement, stopping")
            break

        selected.append(best_cand)
        remaining.remove(best_cand)
        combined = combined + group_scores[best_cand]
        best_ndcg = best_cand_ndcg
        g = groups_lookup[best_cand]
        total_bits = sum(groups_lookup[s]["num_bits"] for s in selected)
        print(f"  Step {step+1}: +{best_cand} ({g['num_bits']}b) → "
              f"nDCG@10={best_ndcg:.4f} (total {total_bits} bits)")

    return selected, best_ndcg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    groups = define_feature_groups()
    groups_lookup = {g["name"]: g for g in groups}
    total_bits = sum(g["num_bits"] for g in groups)
    print(f"Feature groups: {len(groups)}, total candidate bits: {total_bits}")

    # Step 1: Load and binarize
    print("\nStep 1: Loading and binarizing...")
    docs, queries = load_all_texts(
        project_root, groups, args.max_doc_tokens, args.max_query_tokens)

    # Load qrels
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    datasets_dir = project_root / "datasets"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, str(datasets_dir))
    _, _, qrels = GenericDataLoader(data_path).load(split="test")
    print(f"Loaded {len(docs)} docs, {len(queries)} queries, "
          f"{len(qrels)} queries with qrels")

    doc_ids = sorted(docs.keys())
    query_ids = sorted(queries.keys())

    # Step 2: Compute per-group MaxSim score matrices
    print(f"\nStep 2: Computing per-group MaxSim scores "
          f"(prefilter={args.prefilter_top_n})...")
    group_scores = {}
    t0 = time.perf_counter()
    for i, g in enumerate(groups):
        gt0 = time.perf_counter()
        scores, _ = compute_group_score_matrix(
            docs, queries, doc_ids, query_ids,
            g["name"], args.prefilter_top_n)
        group_scores[g["name"]] = scores
        elapsed = time.perf_counter() - gt0

        # Also compute individual nDCG
        ndcg = ndcg_at_k_from_scores(scores, query_ids, doc_ids, qrels)
        g["individual_ndcg"] = ndcg

        if (i + 1) % 10 == 0 or i == len(groups) - 1:
            total_elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{len(groups)} groups ({total_elapsed:.1f}s) "
                  f"last: {g['name']} nDCG={ndcg:.4f} ({elapsed:.1f}s)")

    # Report individual rankings
    ranked_groups = sorted(groups, key=lambda g: g["individual_ndcg"],
                           reverse=True)
    print("\nTop 20 individual feature groups (MaxSim):")
    for g in ranked_groups[:20]:
        print(f"  {g['name']:30s}  nDCG@10={g['individual_ndcg']:.4f}  "
              f"({g['num_bits']} bits)")

    # Save screening results
    screen_path = results_dir / "feature_screening_maxsim.csv"
    with open(screen_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_name", "ndcg_at_10", "num_bits", "type", "layer"])
        for g in ranked_groups:
            w.writerow([g["name"], f"{g['individual_ndcg']:.6f}",
                        g["num_bits"], g["type"], g["layer"]])
    print(f"Saved: {screen_path}")

    # Step 3: Greedy forward selection
    print(f"\nStep 3: Greedy forward selection (max {args.max_select_steps} steps)...")
    t0 = time.perf_counter()
    all_names = [g["name"] for g in groups]
    selected, best_ndcg = greedy_select(
        group_scores, all_names, groups_lookup,
        query_ids, doc_ids, qrels, args.max_select_steps)
    print(f"Selection done in {time.perf_counter()-t0:.1f}s")

    # Step 4: Evaluate at bit budgets
    budgets = [int(x) for x in args.eval_budgets.split(",")]
    print(f"\nStep 4: Bit budget evaluation...")
    budget_results = []
    for budget in budgets:
        subset = []
        bits_used = 0
        combined = np.zeros_like(group_scores[selected[0]])
        for name in selected:
            g = groups_lookup[name]
            if bits_used + g["num_bits"] <= budget:
                subset.append(name)
                bits_used += g["num_bits"]
                combined = combined + group_scores[name]
        if not subset:
            print(f"  {budget:6d} bits: no features fit")
            continue
        ndcg = ndcg_at_k_from_scores(combined, query_ids, doc_ids, qrels)
        print(f"  {budget:6d} bits ({bits_used:5d} used, "
              f"{len(subset):2d} groups): nDCG@10={ndcg:.4f}")
        budget_results.append({
            "budget": budget, "bits_used": bits_used,
            "num_groups": len(subset), "ndcg_at_10": ndcg,
            "groups": subset,
        })

    # Save results
    result_path = results_dir / "selected_features.json"
    result = {
        "selected_groups_ordered": selected,
        "best_ndcg_at_10": best_ndcg,
        "budget_results": budget_results,
        "individual_top_20": [
            {"name": g["name"], "ndcg": g["individual_ndcg"],
             "bits": g["num_bits"], "type": g["type"], "layer": g["layer"]}
            for g in ranked_groups[:20]
        ],
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
