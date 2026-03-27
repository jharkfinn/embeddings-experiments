from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

ATTN_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
NUM_LAYERS = 40
NUM_EXPERTS = 256
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
DEFAULT_WORKERS = max(1, min(12, os.cpu_count() or 1))

_WORKER_DOCS = None
_WORKER_QUERIES = None
_WORKER_DOC_IDS = None
_WORKER_QUERY_IDS = None
_WORKER_QRELS = None
_WORKER_GROUP_ORDER = None
_WORKER_TOTAL_BITS = None
_WORKER_TOTAL_BYTES = None
_WORKER_IDF_QUERY_FLAT = None
_WORKER_IDF_QUERY_OR_VECTORS = None
_WORKER_IDF_DOC_OR_VECTORS = None
_WORKER_IDF_WEIGHTED_TABLE = None
_WORKER_IDF_BYTE_INDEX = None
_WORKER_IDF_PREFILTER_K = None
_WORKER_IDF = None
_WORKER_IDF_DOC_FLAT = None
_WORKER_IDF_DOC_FLAT_CACHE = None
_WORKER_CONCAT_DOC_VALUES = None
_WORKER_CONCAT_QUERY_VALUES = None
_WORKER_CONCAT_PREFILTER_IDX = None
_WORKER_CONCAT_LAST_TOKEN = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--prefilter-top-n", type=int, default=100)
    parser.add_argument("--idf-prefilter-k", type=int, default=1000)
    parser.add_argument("--max-select-steps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--skip-idf", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def define_feature_groups():
    groups = []
    for layer_idx in range(NUM_LAYERS):
        groups.append(
            {
                "name": f"router_topk_{layer_idx}",
                "type": "router_topk",
                "field": f"router_topk_{layer_idx}",
                "layer": layer_idx,
                "num_bits": NUM_EXPERTS,
                "packed_bytes": NUM_EXPERTS // 8,
            }
        )
        groups.append(
            {
                "name": f"router_full_{layer_idx}",
                "type": "router_full",
                "field": f"router_full_{layer_idx}",
                "layer": layer_idx,
                "num_bits": NUM_EXPERTS,
                "packed_bytes": NUM_EXPERTS // 8,
            }
        )
    for layer_idx in ATTN_LAYERS:
        groups.extend(
            [
                {
                    "name": f"q_sign_{layer_idx}",
                    "type": "sign",
                    "field": f"q_{layer_idx}",
                    "layer": layer_idx,
                    "num_bits": 4096,
                    "packed_bytes": 512,
                },
                {
                    "name": f"k_sign_{layer_idx}",
                    "type": "sign",
                    "field": f"k_{layer_idx}",
                    "layer": layer_idx,
                    "num_bits": 512,
                    "packed_bytes": 64,
                },
                {
                    "name": f"v_sign_{layer_idx}",
                    "type": "sign",
                    "field": f"v_{layer_idx}",
                    "layer": layer_idx,
                    "num_bits": 512,
                    "packed_bytes": 64,
                },
                {
                    "name": f"attn_out_sign_{layer_idx}",
                    "type": "sign",
                    "field": f"attn_out_{layer_idx}",
                    "layer": layer_idx,
                    "num_bits": 2048,
                    "packed_bytes": 256,
                },
            ]
        )
    groups.append(
        {
            "name": "hs_final_sign",
            "type": "sign",
            "field": "hs_final",
            "layer": -1,
            "num_bits": 2048,
            "packed_bytes": 256,
        }
    )
    return groups


def load_scifact_qrels(dataset_dir: Path):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, str(dataset_dir))
    _, _, qrels = GenericDataLoader(data_path).load(split="test")
    return qrels


def load_packed_sign_bits(tensors, meta, offset, length, field_name):
    field_info = meta["fields"].get(field_name)
    if field_info is None:
        raise KeyError(f"Missing field in shard metadata: {field_name}")
    data = tensors[field_info["data_key"]][offset : offset + length]
    if data.shape[0] != length:
        raise RuntimeError(f"Field {field_name} has {data.shape[0]} rows, expected {length}")
    return np.asarray(data, dtype=np.uint8)


def load_all_texts(project_root: Path, groups, max_doc_tokens: int, max_query_tokens: int):
    from safetensors.numpy import load_file

    shard_dir = project_root / "embeddings" / "echo" / "shards"
    results_dir = project_root / "results"

    records = []
    with (results_dir / "records.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["ordinal"] = int(row["ordinal"])
            row["record_index"] = int(row["record_index"])
            row["token_offset"] = int(row["token_offset"])
            row["seq_len"] = int(row["seq_len"])
            records.append(row)

    shard_records = defaultdict(list)
    for record in records:
        shard_records[record["shard_stem"]].append(record)

    docs = {}
    queries = {}
    done = 0
    total = len(records)
    t0 = time.perf_counter()

    for shard_stem, shard_recs in sorted(shard_records.items()):
        shard_path = shard_dir / f"{shard_stem}.safetensors"
        meta_path = shard_dir / f"{shard_stem}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tensors = load_file(str(shard_path))

        for record in shard_recs:
            max_tokens = max_query_tokens if record["kind"] == "query" else max_doc_tokens
            length = min(record["seq_len"], max_tokens)
            offset = record["token_offset"]

            bits = {}
            for group in groups:
                bits[group["name"]] = load_packed_sign_bits(
                    tensors, meta, offset, length, group["field"]
                )

            target = docs if record["kind"] == "doc" else queries
            target[record["text_id"]] = bits
            done += 1

        if done % 500 == 0 or done == total:
            print(f"  Binarized {done}/{total} ({time.perf_counter() - t0:.1f}s)")
        del tensors

    print(f"Loaded {len(docs)} docs and {len(queries)} queries")
    return docs, queries


def _configure_worker_state(
    docs=None,
    queries=None,
    doc_ids=None,
    query_ids=None,
    qrels=None,
    group_order=None,
    total_bits=None,
    total_bytes=None,
    idf_query_flat=None,
    idf_query_or_vectors=None,
    idf_doc_or_vectors=None,
    idf_doc_flat=None,
    idf_weighted_table=None,
    idf_byte_index=None,
    idf_prefilter_k=None,
    idf=None,
    concat_doc_values=None,
    concat_query_values=None,
    concat_prefilter_idx=None,
    concat_last_token=None,
):
    global _WORKER_DOCS
    global _WORKER_QUERIES
    global _WORKER_DOC_IDS
    global _WORKER_QUERY_IDS
    global _WORKER_QRELS
    global _WORKER_GROUP_ORDER
    global _WORKER_TOTAL_BITS
    global _WORKER_TOTAL_BYTES
    global _WORKER_IDF_QUERY_FLAT
    global _WORKER_IDF_QUERY_OR_VECTORS
    global _WORKER_IDF_DOC_OR_VECTORS
    global _WORKER_IDF_DOC_FLAT
    global _WORKER_IDF_WEIGHTED_TABLE
    global _WORKER_IDF_BYTE_INDEX
    global _WORKER_IDF_PREFILTER_K
    global _WORKER_IDF
    global _WORKER_IDF_DOC_FLAT_CACHE
    global _WORKER_CONCAT_DOC_VALUES
    global _WORKER_CONCAT_QUERY_VALUES
    global _WORKER_CONCAT_PREFILTER_IDX
    global _WORKER_CONCAT_LAST_TOKEN

    if docs is not None:
        _WORKER_DOCS = docs
    if queries is not None:
        _WORKER_QUERIES = queries
    if doc_ids is not None:
        _WORKER_DOC_IDS = doc_ids
    if query_ids is not None:
        _WORKER_QUERY_IDS = query_ids
    if qrels is not None:
        _WORKER_QRELS = qrels
    if group_order is not None:
        _WORKER_GROUP_ORDER = group_order
    if total_bits is not None:
        _WORKER_TOTAL_BITS = total_bits
    if total_bytes is not None:
        _WORKER_TOTAL_BYTES = total_bytes
    if idf_query_flat is not None:
        _WORKER_IDF_QUERY_FLAT = idf_query_flat
    if idf_query_or_vectors is not None:
        _WORKER_IDF_QUERY_OR_VECTORS = idf_query_or_vectors
    if idf_doc_or_vectors is not None:
        _WORKER_IDF_DOC_OR_VECTORS = idf_doc_or_vectors
    if idf_doc_flat is not None:
        _WORKER_IDF_DOC_FLAT = idf_doc_flat
    if idf_weighted_table is not None:
        _WORKER_IDF_WEIGHTED_TABLE = idf_weighted_table
    if idf_byte_index is not None:
        _WORKER_IDF_BYTE_INDEX = idf_byte_index
    if idf_prefilter_k is not None:
        _WORKER_IDF_PREFILTER_K = idf_prefilter_k
    if idf is not None:
        _WORKER_IDF = idf
    if concat_doc_values is not None:
        _WORKER_CONCAT_DOC_VALUES = concat_doc_values
    if concat_query_values is not None:
        _WORKER_CONCAT_QUERY_VALUES = concat_query_values
    if concat_prefilter_idx is not None:
        _WORKER_CONCAT_PREFILTER_IDX = concat_prefilter_idx
    if concat_last_token is not None:
        _WORKER_CONCAT_LAST_TOKEN = bool(concat_last_token)
    _WORKER_IDF_DOC_FLAT_CACHE = OrderedDict()


def _write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    tmp_file = Path(tmp_path)
    try:
        tmp_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_file.replace(path)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def _save_npy_atomic(path: Path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".npy.tmp")
    os.close(fd)
    tmp_file = Path(tmp_path)
    try:
        with tmp_file.open("wb") as handle:
            np.save(handle, array)
        tmp_file.replace(path)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def _get_executor(max_workers: int):
    if max_workers <= 1:
        return None
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return None
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)


def _score_path(checkpoint_dir: Path, group_name: str):
    return checkpoint_dir / "group_scores" / f"{group_name}.npy"


def _valid_score_file(path: Path, num_q: int, num_d: int):
    if not path.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return array.shape == (num_q, num_d)


def _chunked(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def maxsim_one_pair(q_bits, d_bits):
    xor = np.bitwise_xor(q_bits[:, None, :], d_bits[None, :, :])
    hamming_dist = _POPCOUNT_TABLE[xor].astype(np.uint16).sum(axis=2)
    total_bits = q_bits.shape[1] * 8
    return (total_bits - hamming_dist.min(axis=1).astype(np.float32)).mean()


def maxsim_one_pair_dense(q_vecs, d_vecs):
    similarities = q_vecs @ d_vecs.T
    return similarities.max(axis=1).astype(np.float32).mean()


def ndcg_at_k(scores_matrix, query_ids, doc_ids, qrels, k=10):
    ndcg_values = []
    for query_index, query_id in enumerate(query_ids):
        if query_id not in qrels:
            continue
        rels = qrels[query_id]
        ranked = np.argsort(-scores_matrix[query_index])[:k]
        dcg = sum(rels.get(doc_ids[doc_index], 0) / np.log2(rank + 2) for rank, doc_index in enumerate(ranked))
        ideal = sorted(rels.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal))
        if idcg > 0:
            ndcg_values.append(dcg / idcg)
    return float(np.mean(ndcg_values)) if ndcg_values else 0.0


def compute_group_scores(docs, queries_bin, doc_ids, query_ids, group, prefilter_n):
    group_name = group["name"]
    num_q = len(query_ids)
    num_d = len(doc_ids)

    sample = docs[doc_ids[0]][group_name]
    num_bits = sample.shape[1] * 8
    doc_vecs = np.zeros((num_d, num_bits), dtype=np.float32)
    for doc_index, doc_id in enumerate(doc_ids):
        doc_vecs[doc_index] = np.unpackbits(
            docs[doc_id][group_name], axis=1
        ).astype(np.float32).mean(axis=0)

    query_vecs = np.zeros((num_q, num_bits), dtype=np.float32)
    for query_index, query_id in enumerate(query_ids):
        query_vecs[query_index] = np.unpackbits(
            queries_bin[query_id][group_name], axis=1
        ).astype(np.float32).mean(axis=0)

    sv_scores = query_vecs @ doc_vecs.T
    top_n = min(prefilter_n, num_d)
    if top_n >= num_d:
        prefilter_idx = np.broadcast_to(np.arange(num_d, dtype=np.int32), (num_q, num_d))
    else:
        prefilter_idx = np.argpartition(-sv_scores, top_n - 1, axis=1)[:, :top_n]

    scores = np.zeros((num_q, num_d), dtype=np.float32)
    for query_index, query_id in enumerate(query_ids):
        query_value = queries_bin[query_id][group_name]
        for doc_index in prefilter_idx[query_index]:
            doc_value = docs[doc_ids[doc_index]][group_name]
            scores[query_index, doc_index] = maxsim_one_pair(query_value, doc_value)
    return scores


def _compute_group_scores_worker(task):
    group, prefilter_n, checkpoint_path = task
    scores = compute_group_scores(
        _WORKER_DOCS,
        _WORKER_QUERIES,
        _WORKER_DOC_IDS,
        _WORKER_QUERY_IDS,
        group,
        prefilter_n,
    )
    ndcg = ndcg_at_k(scores, _WORKER_QUERY_IDS, _WORKER_DOC_IDS, _WORKER_QRELS)
    _save_npy_atomic(checkpoint_path, scores.astype(np.float32, copy=False))
    return {
        "name": group["name"],
        "ndcg_at_10": float(ndcg),
        "path": str(checkpoint_path),
    }


def greedy_select(group_scores, groups_lookup, query_ids, doc_ids, qrels, max_steps):
    selected = []
    combined = np.zeros_like(next(iter(group_scores.values())))
    best_ndcg = 0.0
    remaining = set(group_scores.keys())
    selection_log = []

    print(f"Greedy forward selection (max {max_steps} steps)...")
    print(f"{'Step':>4s} {'Added':>35s} {'Bits':>6s} {'Total Bits':>10s} {'nDCG@10':>8s}")
    print("-" * 75)

    for step in range(max_steps):
        best_candidate = None
        best_candidate_ndcg = best_ndcg
        for candidate in remaining:
            trial = combined + group_scores[candidate]
            ndcg = ndcg_at_k(trial, query_ids, doc_ids, qrels)
            if ndcg > best_candidate_ndcg:
                best_candidate_ndcg = ndcg
                best_candidate = candidate

        if best_candidate is None:
            print(f"  Step {step + 1}: no improvement, stopping")
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        combined = combined + group_scores[best_candidate]
        best_ndcg = best_candidate_ndcg
        total_bits = sum(groups_lookup[name]["num_bits"] for name in selected)
        group = groups_lookup[best_candidate]
        print(
            f"  {step + 1:3d} {best_candidate:>35s} {group['num_bits']:6d} "
            f"{total_bits:10d} {best_ndcg:8.4f}"
        )
        selection_log.append(
            {
                "step": step + 1,
                "group": best_candidate,
                "group_bits": group["num_bits"],
                "total_bits": total_bits,
                "ndcg_at_10": best_ndcg,
            }
        )

    return selected, combined, best_ndcg, selection_log


def analyze_group_family(group_scores, groups_lookup, query_ids, doc_ids, qrels, group_type, max_steps):
    family_names = [
        name for name, group in groups_lookup.items() if group["type"] == group_type
    ]
    family_scores = {name: group_scores[name] for name in family_names}
    family_results = [
        (
            name,
            ndcg_at_k(group_scores[name], query_ids, doc_ids, qrels),
            groups_lookup[name]["num_bits"],
            groups_lookup[name]["layer"],
        )
        for name in family_names
    ]
    family_results.sort(key=lambda item: item[1], reverse=True)

    aggregate_scores = np.zeros_like(next(iter(group_scores.values())))
    for name in family_names:
        aggregate_scores += group_scores[name]
    aggregate_ndcg = ndcg_at_k(aggregate_scores, query_ids, doc_ids, qrels)

    print()
    print(f"{group_type} only: top 10 individual layers")
    print(f"{'Group':35s} {'nDCG@10':>8s} {'Bits':>6s} {'Layer':>6s}")
    print("-" * 62)
    for name, ndcg, nbits, layer in family_results[:10]:
        print(f"  {name:33s} {ndcg:8.4f} {nbits:6d} {layer:6d}")
    print(f"  aggregate_all_layers{'':>16s} {aggregate_ndcg:8.4f}")

    print()
    print(f"{group_type} only greedy selection")
    selected, combined, best_ndcg, selection_log = greedy_select(
        family_scores, groups_lookup, query_ids, doc_ids, qrels, max_steps
    )

    return {
        "group_type": group_type,
        "aggregate_all_layers_ndcg_at_10": float(aggregate_ndcg),
        "best_single_group": family_results[0][0] if family_results else None,
        "best_single_group_ndcg_at_10": float(family_results[0][1]) if family_results else None,
        "individual_results": [
            {
                "name": name,
                "ndcg_at_10": float(ndcg),
                "num_bits": int(nbits),
                "layer": int(layer),
            }
            for name, ndcg, nbits, layer in family_results
        ],
        "greedy_selected_groups": selected,
        "greedy_best_ndcg_at_10": float(best_ndcg),
        "greedy_selection_log": selection_log,
    }


def flatten_token_bits(text_bits, group_order):
    arrays = [np.asarray(text_bits[group_name], dtype=np.uint8) for group_name in group_order]
    seq_lens = {array.shape[0] for array in arrays}
    if len(seq_lens) != 1:
        raise ValueError(f"Inconsistent token counts across groups: {sorted(seq_lens)}")
    return np.concatenate(arrays, axis=1)


def flatten_or_vector(text_bits, group_order):
    arrays = [
        np.bitwise_or.reduce(np.asarray(text_bits[group_name], dtype=np.uint8), axis=0)
        for group_name in group_order
    ]
    return np.concatenate(arrays, axis=0)


def _concat_last_token_bits(text_bits, group_order):
    arrays = [np.asarray(text_bits[group_name], dtype=np.uint8)[-1] for group_name in group_order]
    return np.concatenate(arrays, axis=0)


def binary_agreement_score(q_bits, d_bits):
    xor = np.bitwise_xor(q_bits, d_bits)
    total_bits = q_bits.shape[0] * 8
    return float(total_bits - _POPCOUNT_TABLE[xor].astype(np.uint16).sum(dtype=np.uint32))


def _concat_score_path(checkpoint_dir: Path, variant_name: str):
    return checkpoint_dir / "concat_scores" / f"{variant_name}.npy"


def _compute_concat_query_batch(query_indices):
    batch_scores = np.zeros((len(query_indices), len(_WORKER_DOC_IDS)), dtype=np.float32)
    for batch_row, query_index in enumerate(query_indices):
        query_value = _WORKER_CONCAT_QUERY_VALUES[int(query_index)]
        for doc_index in _WORKER_CONCAT_PREFILTER_IDX[int(query_index)]:
            doc_value = _WORKER_CONCAT_DOC_VALUES[int(doc_index)]
            if _WORKER_CONCAT_LAST_TOKEN:
                score = binary_agreement_score(query_value, doc_value)
            else:
                score = maxsim_one_pair(query_value, doc_value)
            batch_scores[batch_row, int(doc_index)] = score
    return list(query_indices), batch_scores


def compute_concat_scores(
    docs,
    queries_bin,
    doc_ids,
    query_ids,
    group_order,
    prefilter_n,
    last_token=False,
    workers=1,
    query_batch_size=8,
):
    num_q = len(query_ids)
    num_d = len(doc_ids)

    if last_token:
        sample = _concat_last_token_bits(docs[doc_ids[0]], group_order)
        doc_vecs = np.zeros((num_d, sample.shape[0] * 8), dtype=np.float32)
        doc_values = {}
        for doc_index, doc_id in enumerate(doc_ids):
            packed = _concat_last_token_bits(docs[doc_id], group_order)
            doc_values[doc_id] = packed
            doc_vecs[doc_index] = np.unpackbits(packed[None, :], axis=1)[0].astype(np.float32)

        query_vecs = np.zeros((num_q, sample.shape[0] * 8), dtype=np.float32)
        query_values = {}
        for query_index, query_id in enumerate(query_ids):
            packed = _concat_last_token_bits(queries_bin[query_id], group_order)
            query_values[query_id] = packed
            query_vecs[query_index] = np.unpackbits(packed[None, :], axis=1)[0].astype(np.float32)
    else:
        sample = flatten_token_bits(docs[doc_ids[0]], group_order)
        doc_vecs = np.zeros((num_d, sample.shape[1] * 8), dtype=np.float32)
        doc_values = {}
        for doc_index, doc_id in enumerate(doc_ids):
            packed = flatten_token_bits(docs[doc_id], group_order)
            doc_values[doc_id] = packed
            doc_vecs[doc_index] = np.unpackbits(packed, axis=1).astype(np.float32).mean(axis=0)

        query_vecs = np.zeros((num_q, sample.shape[1] * 8), dtype=np.float32)
        query_values = {}
        for query_index, query_id in enumerate(query_ids):
            packed = flatten_token_bits(queries_bin[query_id], group_order)
            query_values[query_id] = packed
            query_vecs[query_index] = np.unpackbits(packed, axis=1).astype(np.float32).mean(axis=0)

    sv_scores = query_vecs @ doc_vecs.T
    top_n = min(prefilter_n, num_d)
    if top_n >= num_d:
        prefilter_idx = np.broadcast_to(np.arange(num_d, dtype=np.int32), (num_q, num_d))
    else:
        prefilter_idx = np.argpartition(-sv_scores, top_n - 1, axis=1)[:, :top_n]

    scores = np.zeros((num_q, num_d), dtype=np.float32)
    doc_value_list = [doc_values[doc_id] for doc_id in doc_ids]
    query_value_list = [query_values[query_id] for query_id in query_ids]

    if workers > 1:
        _configure_worker_state(
            concat_doc_values=doc_value_list,
            concat_query_values=query_value_list,
            concat_prefilter_idx=prefilter_idx,
            concat_last_token=last_token,
        )
        executor = _get_executor(workers)
        query_batches = list(_chunked(list(range(num_q)), max(1, query_batch_size)))
        if executor is not None:
            with executor:
                future_map = {
                    executor.submit(_compute_concat_query_batch, batch): batch for batch in query_batches
                }
                for future in as_completed(future_map):
                    batch_indices, batch_scores = future.result()
                    for row_idx, query_index in enumerate(batch_indices):
                        scores[query_index] = batch_scores[row_idx]
        else:
            for batch in query_batches:
                batch_indices, batch_scores = _compute_concat_query_batch(batch)
                for row_idx, query_index in enumerate(batch_indices):
                    scores[query_index] = batch_scores[row_idx]
    else:
        for query_index, query_value in enumerate(query_value_list):
            for doc_index in prefilter_idx[query_index]:
                doc_value = doc_value_list[int(doc_index)]
                if last_token:
                    score = binary_agreement_score(query_value, doc_value)
                else:
                    score = maxsim_one_pair(query_value, doc_value)
                scores[query_index, int(doc_index)] = score
    return scores


def compute_router_concat_variants(
    docs,
    queries_bin,
    doc_ids,
    query_ids,
    qrels,
    checkpoint_dir: Path,
    prefilter_n: int,
    workers: int,
    query_batch_size: int,
):
    concat_dir = checkpoint_dir / "concat_scores"
    concat_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        {
            "name": "router_topk_concat_token_all_layers",
            "group_order": [f"router_topk_{layer_idx}" for layer_idx in range(NUM_LAYERS)],
            "last_token": False,
            "type": "router_topk_concat",
            "num_bits": NUM_LAYERS * NUM_EXPERTS,
        },
        {
            "name": "router_full_concat_token_all_layers",
            "group_order": [f"router_full_{layer_idx}" for layer_idx in range(NUM_LAYERS)],
            "last_token": False,
            "type": "router_full_concat",
            "num_bits": NUM_LAYERS * NUM_EXPERTS,
        },
        {
            "name": "router_topk_concat_last_token_all_layers",
            "group_order": [f"router_topk_{layer_idx}" for layer_idx in range(NUM_LAYERS)],
            "last_token": True,
            "type": "router_topk_last_token_concat",
            "num_bits": NUM_LAYERS * NUM_EXPERTS,
        },
        {
            "name": "router_full_concat_last_token_all_layers",
            "group_order": [f"router_full_{layer_idx}" for layer_idx in range(NUM_LAYERS)],
            "last_token": True,
            "type": "router_full_last_token_concat",
            "num_bits": NUM_LAYERS * NUM_EXPERTS,
        },
    ]

    results = []
    for variant in variants:
        checkpoint_path = _concat_score_path(checkpoint_dir, variant["name"])
        if _valid_score_file(checkpoint_path, len(query_ids), len(doc_ids)):
            scores = np.load(checkpoint_path, mmap_mode="r")
        else:
            t0 = time.perf_counter()
            print(f"Computing {variant['name']}...")
            scores = compute_concat_scores(
                docs,
                queries_bin,
                doc_ids,
                query_ids,
                variant["group_order"],
                prefilter_n,
                last_token=variant["last_token"],
                workers=workers,
                query_batch_size=query_batch_size,
            )
            _save_npy_atomic(checkpoint_path, scores.astype(np.float32, copy=False))
            print(f"  done in {time.perf_counter() - t0:.1f}s")
            scores = np.load(checkpoint_path, mmap_mode="r")

        ndcg = ndcg_at_k(scores, query_ids, doc_ids, qrels)
        results.append(
            {
                "name": variant["name"],
                "type": variant["type"],
                "mode": "last_token" if variant["last_token"] else "token_maxsim",
                "num_bits": variant["num_bits"],
                "num_layers": NUM_LAYERS,
                "ndcg_at_10": float(ndcg),
                "path": str(checkpoint_path),
            }
        )

    results.sort(key=lambda row: row["ndcg_at_10"], reverse=True)
    return results


def _count_idf_chunk(doc_batch):
    token_freq = np.zeros(_WORKER_TOTAL_BITS, dtype=np.float64)
    total_tokens = 0
    for doc_id in doc_batch:
        flat = flatten_token_bits(_WORKER_DOCS[doc_id], _WORKER_GROUP_ORDER)
        unpacked = np.unpackbits(flat, axis=1)[:, :_WORKER_TOTAL_BITS]
        token_freq += unpacked.sum(axis=0, dtype=np.float64)
        total_tokens += flat.shape[0]
    return token_freq, total_tokens, len(doc_batch)


def build_weighted_byte_table(copies, total_bytes):
    padded = np.pad(copies, (0, total_bytes * 8 - len(copies)))
    byte_patterns = ((np.arange(256, dtype=np.uint16)[:, None] >> np.arange(7, -1, -1)) & 1).astype(np.uint8)
    weights_by_byte = padded.reshape(total_bytes, 8)
    return (byte_patterns[None, :, :] * weights_by_byte[:, None, :]).sum(axis=2).astype(np.uint8), byte_patterns


def weighted_overlap_scores(query_vec, doc_matrix, weighted_table, byte_index, chunk_size=256):
    scores = np.zeros(doc_matrix.shape[0], dtype=np.uint32)
    for start in range(0, doc_matrix.shape[0], chunk_size):
        end = min(start + chunk_size, doc_matrix.shape[0])
        shared = np.bitwise_and(query_vec, doc_matrix[start:end])
        scores[start:end] = weighted_table[byte_index, shared].sum(axis=1, dtype=np.uint32)
    return scores


def maxsim_expanded_equivalent(q_packed, d_packed, weighted_table, byte_index, doc_chunk=128):
    q_max = np.zeros(q_packed.shape[0], dtype=np.uint32)
    for query_index, q_tok in enumerate(q_packed):
        best = 0
        for start in range(0, d_packed.shape[0], doc_chunk):
            end = min(start + doc_chunk, d_packed.shape[0])
            shared = np.bitwise_and(q_tok, d_packed[start:end])
            chunk_scores = weighted_table[byte_index, shared].sum(axis=1, dtype=np.uint32)
            chunk_best = int(chunk_scores.max()) if chunk_scores.size else 0
            if chunk_best > best:
                best = chunk_best
        q_max[query_index] = best
    return float(q_max.sum(dtype=np.uint64))


def _get_doc_flat(doc_index):
    if _WORKER_IDF_DOC_FLAT is not None:
        return _WORKER_IDF_DOC_FLAT[int(doc_index)]

    doc_id = _WORKER_DOC_IDS[int(doc_index)]
    cached = _WORKER_IDF_DOC_FLAT_CACHE.get(doc_id)
    if cached is not None:
        _WORKER_IDF_DOC_FLAT_CACHE.move_to_end(doc_id)
        return cached
    flat = flatten_token_bits(_WORKER_DOCS[doc_id], _WORKER_GROUP_ORDER)
    _WORKER_IDF_DOC_FLAT_CACHE[doc_id] = flat
    _WORKER_IDF_DOC_FLAT_CACHE.move_to_end(doc_id)
    while len(_WORKER_IDF_DOC_FLAT_CACHE) > 256:
        _WORKER_IDF_DOC_FLAT_CACHE.popitem(last=False)
    return flat


def _compute_idf_query_batch(query_indices):
    batch_scores = np.zeros((len(query_indices), len(_WORKER_DOC_IDS)), dtype=np.float32)
    for batch_row, query_index in enumerate(query_indices):
        prefilter_scores = weighted_overlap_scores(
            _WORKER_IDF_QUERY_OR_VECTORS[query_index],
            _WORKER_IDF_DOC_OR_VECTORS,
            _WORKER_IDF_WEIGHTED_TABLE,
            _WORKER_IDF_BYTE_INDEX,
        ).astype(np.float32)
        if _WORKER_IDF_PREFILTER_K >= len(_WORKER_DOC_IDS):
            prefilter_idx = np.arange(len(_WORKER_DOC_IDS), dtype=np.int32)
        else:
            prefilter_idx = np.argpartition(
                -prefilter_scores,
                _WORKER_IDF_PREFILTER_K - 1,
            )[: _WORKER_IDF_PREFILTER_K]

        q_packed = _WORKER_IDF_QUERY_FLAT[_WORKER_QUERY_IDS[query_index]]
        for doc_index in prefilter_idx:
            batch_scores[batch_row, int(doc_index)] = maxsim_expanded_equivalent(
                q_packed,
                _get_doc_flat(int(doc_index)),
                _WORKER_IDF_WEIGHTED_TABLE,
                _WORKER_IDF_BYTE_INDEX,
            )
    return list(query_indices), batch_scores


def _run_threshold_task(threshold):
    copies_t = np.rint(_WORKER_IDF).astype(np.int32)
    copies_t[_WORKER_IDF < threshold] = 0
    copies_t = np.clip(copies_t, 0, 8)
    bits_kept = int((copies_t > 0).sum())
    expanded_t = int(copies_t.sum())
    if bits_kept == 0:
        return {
            "threshold": threshold,
            "bits_kept": 0,
            "expanded_bits": 0,
            "ndcg_at_10": None,
        }

    weighted_table_t, _ = build_weighted_byte_table(copies_t, _WORKER_TOTAL_BYTES)
    scores = np.zeros((len(_WORKER_QUERY_IDS), len(_WORKER_DOC_IDS)), dtype=np.float32)
    for query_index in range(len(_WORKER_QUERY_IDS)):
        scores[query_index] = weighted_overlap_scores(
            _WORKER_IDF_QUERY_OR_VECTORS[query_index],
            _WORKER_IDF_DOC_OR_VECTORS,
            weighted_table_t,
            _WORKER_IDF_BYTE_INDEX,
        ).astype(np.float32)
    ndcg = ndcg_at_k(scores, _WORKER_QUERY_IDS, _WORKER_DOC_IDS, _WORKER_QRELS)
    return {
        "threshold": threshold,
        "bits_kept": bits_kept,
        "expanded_bits": expanded_t,
        "ndcg_at_10": float(ndcg),
    }


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    datasets_dir = project_root / "datasets"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else (results_dir / "checkpoints").resolve()
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    groups = define_feature_groups()
    groups_lookup = {group["name"]: group for group in groups}
    docs, queries_bin = load_all_texts(project_root, groups, args.max_doc_tokens, args.max_query_tokens)
    qrels = load_scifact_qrels(datasets_dir)

    doc_ids = sorted(docs.keys())
    query_ids = sorted(queries_bin.keys())
    num_q = len(query_ids)
    num_d = len(doc_ids)
    workers = max(1, min(args.workers, os.cpu_count() or 1))

    _configure_worker_state(
        docs=docs,
        queries=queries_bin,
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
    )

    screening_progress_path = checkpoint_dir / "screening_progress.json"
    screening_progress = {}
    if screening_progress_path.exists():
        try:
            screening_progress = json.loads(screening_progress_path.read_text(encoding="utf-8"))
        except Exception:
            screening_progress = {}
    screening_completed = screening_progress.get("completed", {})

    pending_groups = []
    completed_groups = {}
    for group in groups:
        score_path = _score_path(checkpoint_dir, group["name"])
        if _valid_score_file(score_path, num_q, num_d):
            ndcg = screening_completed.get(group["name"], {}).get("ndcg_at_10")
            if ndcg is None:
                scores = np.load(score_path, mmap_mode="r")
                ndcg = ndcg_at_k(scores, query_ids, doc_ids, qrels)
            completed_groups[group["name"]] = {
                "name": group["name"],
                "ndcg_at_10": float(ndcg),
                "path": str(score_path),
            }
        else:
            pending_groups.append((group, args.prefilter_top_n, score_path))

    if pending_groups:
        t0 = time.perf_counter()
        executor = _get_executor(workers) if workers > 1 else None
        if executor is not None:
            with executor:
                future_map = {
                    executor.submit(_compute_group_scores_worker, task): task[0]["name"]
                    for task in pending_groups
                }
                for index, future in enumerate(as_completed(future_map), start=len(completed_groups) + 1):
                    result = future.result()
                    completed_groups[result["name"]] = result
                    screening_completed[result["name"]] = {
                        "ndcg_at_10": float(result["ndcg_at_10"]),
                        "path": result["path"],
                    }
                    _write_json_atomic(screening_progress_path, {"completed": screening_completed})
                    if index % 10 == 0 or index == len(groups):
                        print(f"  {index}/{len(groups)} ({time.perf_counter() - t0:.1f}s total)")
        else:
            for index, task in enumerate(pending_groups, start=len(completed_groups) + 1):
                result = _compute_group_scores_worker(task)
                completed_groups[result["name"]] = result
                screening_completed[result["name"]] = {
                    "ndcg_at_10": float(result["ndcg_at_10"]),
                    "path": result["path"],
                }
                _write_json_atomic(screening_progress_path, {"completed": screening_completed})
                if index % 10 == 0 or index == len(groups):
                    print(f"  {index}/{len(groups)} ({time.perf_counter() - t0:.1f}s total)")

    group_scores = {
        group["name"]: np.load(_score_path(checkpoint_dir, group["name"]), mmap_mode="r")
        for group in groups
    }

    individual_results = []
    for group in groups:
        ndcg = float(screening_completed[group["name"]]["ndcg_at_10"])
        individual_results.append(
            (group["name"], ndcg, group["num_bits"], group["type"], group["layer"])
        )
    individual_results.sort(key=lambda item: item[1], reverse=True)

    print("Top 20 individual feature groups (MaxSim nDCG@10):")
    print(f"{'Group':35s} {'nDCG@10':>8s} {'Bits':>6s} {'Type':>8s} {'Layer':>6s}")
    print("-" * 70)
    for name, ndcg, nbits, group_type, layer in individual_results[:20]:
        print(f"  {name:33s} {ndcg:8.4f} {nbits:6d} {group_type:>8s} {layer:6d}")

    router_topk_analysis = analyze_group_family(
        group_scores, groups_lookup, query_ids, doc_ids, qrels, "router_topk", args.max_select_steps
    )
    router_full_analysis = analyze_group_family(
        group_scores, groups_lookup, query_ids, doc_ids, qrels, "router_full", args.max_select_steps
    )
    router_concat_analysis = compute_router_concat_variants(
        docs,
        queries_bin,
        doc_ids,
        query_ids,
        qrels,
        checkpoint_dir,
        args.prefilter_top_n,
        workers,
        args.query_batch_size,
    )
    print()
    print("Router concatenation experiments")
    print(f"{'Variant':40s} {'Mode':>12s} {'Bits':>8s} {'nDCG@10':>8s}")
    print("-" * 75)
    for row in router_concat_analysis:
        print(f"  {row['name']:38s} {row['mode']:>12s} {row['num_bits']:8d} {row['ndcg_at_10']:8.4f}")

    selected, combined, best_ndcg, selection_log = greedy_select(
        group_scores, groups_lookup, query_ids, doc_ids, qrels, args.max_select_steps
    )

    budgets = [256, 512, 1024, 2048, 4096, 8192, 16384]
    print(f"\n{'Budget':>7s} {'Used':>6s} {'Groups':>6s} {'nDCG@10':>8s}  Groups")
    print("-" * 80)
    budget_results = []
    for budget in budgets:
        subset = []
        bits_used = 0
        score_sum = np.zeros((num_q, num_d), dtype=np.float32)
        for name in selected:
            group = groups_lookup[name]
            if bits_used + group["num_bits"] <= budget:
                subset.append(name)
                bits_used += group["num_bits"]
                score_sum += group_scores[name]
        if not subset:
            print(f"  {budget:6d}  {'—':>6s} {'—':>6s} {'—':>8s}")
            budget_results.append({"budget": budget, "used_bits": 0, "num_groups": 0, "ndcg_at_10": None})
            continue
        ndcg = ndcg_at_k(score_sum, query_ids, doc_ids, qrels)
        print(f"  {budget:6d} {bits_used:6d} {len(subset):6d} {ndcg:8.4f}  {', '.join(subset)}")
        budget_results.append(
            {
                "budget": budget,
                "used_bits": bits_used,
                "num_groups": len(subset),
                "ndcg_at_10": ndcg,
                "groups": subset,
            }
        )

    idf_ndcg = None
    idf_recall = None
    threshold_results = []
    contribution_rows = []
    total_bits = None
    expanded_dim = None

    if not args.skip_idf:
        binary_groups = groups
        group_order = [group["name"] for group in binary_groups]
        group_offsets = {}
        total_bits = 0
        total_bytes = 0
        for group_name in group_order:
            group_offsets[group_name] = total_bits
            total_bits += groups_lookup[group_name]["num_bits"]
            total_bytes += groups_lookup[group_name]["packed_bytes"]

        idf_token_freq = np.zeros(total_bits, dtype=np.float64)
        idf_total_doc_tokens = 0
        _configure_worker_state(group_order=group_order, total_bits=total_bits, total_bytes=total_bytes)
        doc_batches = list(_chunked(doc_ids, 256))
        counted_docs = 0
        t0 = time.perf_counter()
        executor = _get_executor(workers) if workers > 1 else None
        if executor is not None:
            with executor:
                future_map = {executor.submit(_count_idf_chunk, batch): len(batch) for batch in doc_batches}
                for future in as_completed(future_map):
                    token_freq_chunk, total_tokens_chunk, batch_len = future.result()
                    idf_token_freq += token_freq_chunk
                    idf_total_doc_tokens += total_tokens_chunk
                    counted_docs += batch_len
                    if counted_docs % 500 == 0 or counted_docs >= len(doc_ids):
                        print(f"  Counted {counted_docs}/{len(docs)} docs ({time.perf_counter() - t0:.1f}s)")
        else:
            for batch in doc_batches:
                token_freq_chunk, total_tokens_chunk, batch_len = _count_idf_chunk(batch)
                idf_token_freq += token_freq_chunk
                idf_total_doc_tokens += total_tokens_chunk
                counted_docs += batch_len
                if counted_docs % 500 == 0 or counted_docs >= len(doc_ids):
                    print(f"  Counted {counted_docs}/{len(docs)} docs ({time.perf_counter() - t0:.1f}s)")

        idf = np.log(idf_total_doc_tokens / (idf_token_freq + 1.0))
        copies = np.rint(idf).astype(np.int32)
        copies = np.clip(copies, 0, 8)
        expanded_dim = int(copies.sum())
        weighted_table, _byte_patterns = build_weighted_byte_table(copies, total_bytes)
        byte_index = np.arange(total_bytes, dtype=np.int32)

        idf_query_flat = {query_id: flatten_token_bits(queries_bin[query_id], group_order) for query_id in query_ids}
        idf_query_or_vectors = np.stack(
            [flatten_or_vector(queries_bin[query_id], group_order) for query_id in query_ids],
            axis=0,
        )
        print("Precomputing flattened doc token matrices for IDF stage...")
        t_doc_flat = time.perf_counter()
        idf_doc_flat = []
        for index, doc_id in enumerate(doc_ids, start=1):
            idf_doc_flat.append(flatten_token_bits(docs[doc_id], group_order))
            if index % 500 == 0 or index == len(doc_ids):
                print(f"  Flattened {index}/{len(doc_ids)} docs ({time.perf_counter() - t_doc_flat:.1f}s)")
        idf_doc_or_vectors = np.stack(
            [flatten_or_vector(docs[doc_id], group_order) for doc_id in doc_ids],
            axis=0,
        )

        idf_prefilter_k = min(args.idf_prefilter_k, num_d)
        _configure_worker_state(
            idf=idf,
            idf_query_flat=idf_query_flat,
            idf_query_or_vectors=idf_query_or_vectors,
            idf_doc_or_vectors=idf_doc_or_vectors,
            idf_doc_flat=idf_doc_flat,
            idf_weighted_table=weighted_table,
            idf_byte_index=byte_index,
            idf_prefilter_k=idf_prefilter_k,
        )

        idf_scores_path = checkpoint_dir / "idf_scores.npy"
        idf_progress_path = checkpoint_dir / "idf_progress.json"
        if idf_scores_path.exists():
            idf_scores = np.load(idf_scores_path, mmap_mode="r+")
            if idf_scores.shape != (num_q, num_d):
                raise RuntimeError(
                    f"Unexpected IDF score shape {idf_scores.shape}, expected {(num_q, num_d)}"
                )
        else:
            idf_scores = np.lib.format.open_memmap(
                idf_scores_path,
                mode="w+",
                dtype=np.float32,
                shape=(num_q, num_d),
            )
            idf_scores[:] = 0.0

        idf_progress = {"completed_queries": []}
        if idf_progress_path.exists():
            try:
                idf_progress = json.loads(idf_progress_path.read_text(encoding="utf-8"))
            except Exception:
                idf_progress = {"completed_queries": []}
        completed_queries = set(int(idx) for idx in idf_progress.get("completed_queries", []))
        query_batches = [list(batch) for batch in _chunked(list(range(num_q)), max(1, args.query_batch_size))]
        pending_batches = [batch for batch in query_batches if any(qi not in completed_queries for qi in batch)]

        t0 = time.perf_counter()
        if pending_batches:
            executor = _get_executor(workers) if workers > 1 else None
            if executor is not None:
                with executor:
                    future_map = {
                        executor.submit(_compute_idf_query_batch, batch): batch
                        for batch in pending_batches
                    }
                    for future in as_completed(future_map):
                        batch_indices, batch_scores = future.result()
                        for row_idx, query_index in enumerate(batch_indices):
                            idf_scores[query_index] = batch_scores[row_idx]
                            completed_queries.add(int(query_index))
                        idf_scores.flush()
                        _write_json_atomic(
                            idf_progress_path,
                            {"completed_queries": sorted(completed_queries)},
                        )
                        elapsed = time.perf_counter() - t0
                        done_queries = len(completed_queries)
                        eta = elapsed / max(done_queries, 1) * len(query_ids) - elapsed
                        print(f"  Query {done_queries}/{len(query_ids)} ({elapsed:.1f}s, ETA {eta:.1f}s)")
            else:
                for batch in pending_batches:
                    batch_indices, batch_scores = _compute_idf_query_batch(batch)
                    for row_idx, query_index in enumerate(batch_indices):
                        idf_scores[query_index] = batch_scores[row_idx]
                        completed_queries.add(int(query_index))
                    idf_scores.flush()
                    _write_json_atomic(
                        idf_progress_path,
                        {"completed_queries": sorted(completed_queries)},
                    )
                    elapsed = time.perf_counter() - t0
                    done_queries = len(completed_queries)
                    eta = elapsed / max(done_queries, 1) * len(query_ids) - elapsed
                    print(f"  Query {done_queries}/{len(query_ids)} ({elapsed:.1f}s, ETA {eta:.1f}s)")

        from beir.retrieval.evaluation import EvaluateRetrieval

        idf_results = {}
        for query_index, query_id in enumerate(query_ids):
            row = {}
            nz = np.flatnonzero(idf_scores[query_index] > 0)
            for doc_index in nz:
                row[doc_ids[int(doc_index)]] = float(idf_scores[query_index, int(doc_index)])
            idf_results[query_id] = row

        evaluator = EvaluateRetrieval()
        idf_ndcg, idf_map, idf_recall, idf_precision = evaluator.evaluate(qrels, idf_results, [10, 100])
        print()
        print(f"IDF-Expanded Binary MaxSim (all {total_bits} bits, expanded to {expanded_dim}):")
        print(f"  nDCG@10:    {idf_ndcg['NDCG@10']:.4f}")
        print(f"  nDCG@100:   {idf_ndcg['NDCG@100']:.4f}")
        print(f"  Recall@10:  {idf_recall['Recall@10']:.4f}")
        print(f"  Recall@100: {idf_recall['Recall@100']:.4f}")

        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        threshold_progress_path = checkpoint_dir / "threshold_results.json"
        threshold_completed = {}
        if threshold_progress_path.exists():
            try:
                saved_thresholds = json.loads(threshold_progress_path.read_text(encoding="utf-8"))
                threshold_completed = {float(row["threshold"]): row for row in saved_thresholds}
            except Exception:
                threshold_completed = {}

        print(f"{'Threshold':>10s} {'Bits kept':>10s} {'Expanded':>10s} {'nDCG@10':>8s}")
        print("-" * 45)
        pending_thresholds = [threshold for threshold in thresholds if threshold not in threshold_completed]
        if pending_thresholds:
            executor = _get_executor(min(workers, len(pending_thresholds))) if workers > 1 else None
            if executor is not None:
                with executor:
                    future_map = {
                        executor.submit(_run_threshold_task, threshold): threshold
                        for threshold in pending_thresholds
                    }
                    for future in as_completed(future_map):
                        row = future.result()
                        threshold_completed[float(row["threshold"])] = row
                        _write_json_atomic(
                            threshold_progress_path,
                            [threshold_completed[key] for key in sorted(threshold_completed)],
                        )
            else:
                for threshold in pending_thresholds:
                    row = _run_threshold_task(threshold)
                    threshold_completed[float(row["threshold"])] = row
                    _write_json_atomic(
                        threshold_progress_path,
                        [threshold_completed[key] for key in sorted(threshold_completed)],
                    )

        for threshold in thresholds:
            row = threshold_completed[float(threshold)]
            threshold_results.append(row)
            if row["ndcg_at_10"] is None:
                print(f"{threshold:10.1f} {0:10d} {0:10d} {'n/a':>8s}")
            else:
                print(
                    f"{threshold:10.1f} {row['bits_kept']:10d} "
                    f"{row['expanded_bits']:10d} {row['ndcg_at_10']:8.4f}"
                )

        print()
        print("Bit contribution to expanded vector by group:")
        print(f"{'Group':>30s} {'Orig bits':>10s} {'Expanded':>10s} {'Pct':>6s} {'Mean IDF':>9s}")
        print("-" * 70)
        offset = 0
        for group_name in group_order:
            group = groups_lookup[group_name]
            nbits = group["num_bits"]
            g_copies = copies[offset : offset + nbits]
            g_idf = idf[offset : offset + nbits]
            g_expanded = int(g_copies.sum())
            pct = (g_expanded / expanded_dim * 100.0) if expanded_dim else 0.0
            print(f"{group_name:>30s} {nbits:10d} {g_expanded:10d} {pct:5.1f}% {g_idf.mean():9.2f}")
            contribution_rows.append(
                {
                    "group": group_name,
                    "orig_bits": nbits,
                    "expanded_bits": g_expanded,
                    "pct_expanded": pct,
                    "mean_idf": float(g_idf.mean()),
                }
            )
            offset += nbits

    summary = {
        "selected_groups_ordered": selected,
        "best_ndcg_at_10": float(best_ndcg),
        "selection_log": selection_log,
        "individual_top_20": [
            {"name": name, "ndcg": float(ndcg), "bits": nbits, "type": group_type, "layer": layer}
            for name, ndcg, nbits, group_type, layer in individual_results[:20]
        ],
        "bit_budget_results": budget_results,
        "idf_maxsim": (
            {
                "total_bits": total_bits,
                "expanded_dim": expanded_dim,
                "ndcg_at_10": float(idf_ndcg["NDCG@10"]),
                "ndcg_at_100": float(idf_ndcg["NDCG@100"]),
                "recall_at_10": float(idf_recall["Recall@10"]),
                "recall_at_100": float(idf_recall["Recall@100"]),
            }
            if idf_ndcg is not None
            else None
        ),
        "router_topk_analysis": router_topk_analysis,
        "router_full_analysis": router_full_analysis,
        "router_concat_analysis": router_concat_analysis,
        "idf_threshold_sweep": threshold_results,
        "group_contributions": contribution_rows,
    }
    (results_dir / "binary_feature_discovery_results.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    screening_path = results_dir / "feature_screening_maxsim.csv"
    with screening_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_name", "ndcg_at_10", "num_bits", "type", "layer"])
        for name, ndcg, nbits, group_type, layer in individual_results:
            writer.writerow([name, f"{ndcg:.6f}", nbits, group_type, layer])

    for group_type, analysis in (
        ("router_topk", router_topk_analysis),
        ("router_full", router_full_analysis),
    ):
        family_screening_path = results_dir / f"{group_type}_screening_maxsim.csv"
        with family_screening_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["group_name", "ndcg_at_10", "num_bits", "layer"])
            for row in analysis["individual_results"]:
                writer.writerow(
                    [
                        row["name"],
                        f"{row['ndcg_at_10']:.6f}",
                        row["num_bits"],
                        row["layer"],
                    ]
                )

    concat_summary_path = results_dir / "router_concat_analysis.csv"
    with concat_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "mode", "type", "num_bits", "num_layers", "ndcg_at_10"])
        for row in router_concat_analysis:
            writer.writerow(
                [
                    row["name"],
                    row["mode"],
                    row["type"],
                    row["num_bits"],
                    row["num_layers"],
                    f"{row['ndcg_at_10']:.6f}",
                ]
            )

    family_summary_path = results_dir / "router_family_summary.csv"
    with family_summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "family",
                "aggregate_all_layers_ndcg_at_10",
                "best_single_group",
                "best_single_group_ndcg_at_10",
                "greedy_best_ndcg_at_10",
                "greedy_num_selected",
                "greedy_selected_groups",
            ]
        )
        for analysis in (router_topk_analysis, router_full_analysis):
            writer.writerow(
                [
                    analysis["group_type"],
                    f"{analysis['aggregate_all_layers_ndcg_at_10']:.6f}",
                    analysis["best_single_group"],
                    f"{analysis['best_single_group_ndcg_at_10']:.6f}",
                    f"{analysis['greedy_best_ndcg_at_10']:.6f}",
                    len(analysis["greedy_selected_groups"]),
                    json.dumps(analysis["greedy_selected_groups"]),
                ]
            )

    threshold_path = results_dir / "idf_threshold_sweep.csv"
    if threshold_results:
        with threshold_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(threshold_results[0].keys()))
            writer.writeheader()
            writer.writerows(threshold_results)

    contribution_path = results_dir / "idf_group_contributions.csv"
    if contribution_rows:
        with contribution_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(contribution_rows[0].keys()))
            writer.writeheader()
            writer.writerows(contribution_rows)

    if not args.skip_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        top_n = 25
        names = [item[0] for item in individual_results[:top_n]]
        ndcgs = [item[1] for item in individual_results[:top_n]]
        colors = [
            "#1f77b4"
            if "router_full" in name
            else "#3498db"
            if "router_topk" in name
            else "#e74c3c"
            if "attn_out" in name
            else "#2ecc71"
            if "v_sign" in name
            else "#9b59b6"
            if "k_sign" in name
            else "#f39c12"
            if "q_sign" in name
            else "#1abc9c"
            for name in names
        ]
        ax.barh(range(top_n), ndcgs, color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("nDCG@10")
        ax.set_title("Individual Feature Group Performance")
        ax.invert_yaxis()

        ax = axes[1]
        if selection_log:
            bits = [item["total_bits"] for item in selection_log]
            ndcgs = [item["ndcg_at_10"] for item in selection_log]
            ax.plot(bits, ndcgs, "bo-", markersize=6)
            for index, item in enumerate(selection_log[:10]):
                ax.annotate(item["group"], (bits[index], ndcgs[index]), fontsize=6, rotation=30, ha="left")
            ax.set_xlabel("Total Bits")
            ax.set_ylabel("nDCG@10")
            ax.set_title("Greedy Selection: nDCG@10 vs Bit Budget")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(results_dir / "feature_discovery.png"), dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
