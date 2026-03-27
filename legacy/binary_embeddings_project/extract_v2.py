"""
Signal extraction v2 — optimized for throughput and shard-based storage.

Changes from v1:
- Stable prompt-only capture with `prompt_token_ids`
- Chunked prefill re-enabled with explicit token budget
- Bulk GPU->CPU transfer (one copy per field, not per sequence)
- Columnar shard output instead of one file per text
- Optional int8 row-wise activation quantization
- Multi-threaded async shard writes with sidecar metadata
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"
MAX_TOKENS = 512
ECHO_SEP = " "

EXPECTED_ARCH = {
    "num_layers": 40,
    "attn_layer_indices": [3, 7, 11, 15, 19, 23, 27, 31, 35, 39],
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "num_experts": 256,
    "num_experts_per_tok": 8,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path("."))
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--writer-threads", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--save-queue-size", type=int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--gpu-packer-threads", type=int, default=2)
    p.add_argument("--disk-writer-threads", type=int, default=2)
    p.add_argument("--pack-queue-size", type=int, default=3)
    p.add_argument("--write-queue-size", type=int, default=6)
    p.add_argument("--batches-per-shard", type=int, default=4)
    p.add_argument("--compute-prenorm-attn-dist", action="store_true")
    p.add_argument("--shard-format", type=str, choices=["auto", "torch", "safetensors"],
                   default="auto")
    p.add_argument("--activation-quantization", type=str,
                   choices=["none", "int8"], default="int8")
    p.add_argument("--staging-dir", type=Path, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit-docs", type=int, default=None)
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--inspect-only", action="store_true")
    return p.parse_args()


def load_scifact(dataset_dir):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, str(dataset_dir))
    return GenericDataLoader(data_path).load(split="test")


def _sort_key(text_id):
    if text_id.isdigit():
        return (0, f"{int(text_id):020d}")
    return (1, text_id)


def build_texts(corpus, queries, limit_docs, limit_queries):
    texts = []
    sorted_docs = sorted(corpus.items(), key=lambda x: _sort_key(str(x[0])))
    if limit_docs is not None:
        sorted_docs = sorted_docs[:limit_docs]
    for ordinal, (doc_id, doc) in enumerate(sorted_docs, start=1):
        title = doc.get("title", "").strip()
        body = doc.get("text", "").strip()
        texts.append(("doc", ordinal, str(doc_id),
                      f"{title}. {body}".strip(), f"doc_{ordinal:07d}.pt"))

    sorted_queries = sorted(queries.items(),
                            key=lambda x: _sort_key(str(x[0])))
    if limit_queries is not None:
        sorted_queries = sorted_queries[:limit_queries]
    for ordinal, (qid, text) in enumerate(sorted_queries, start=1):
        texts.append(("query", ordinal, str(qid), str(text),
                      f"query_{ordinal:06d}.pt"))
    return texts


def build_echo_token_ids(tokenizer, text, max_tokens):
    original_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    sep_ids = tokenizer.encode(ECHO_SEP, add_special_tokens=False)
    return original_ids + sep_ids + original_ids, len(original_ids), len(sep_ids)


def prompt_key(token_ids):
    arr = np.asarray(token_ids, dtype=np.int32)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def resolve_shard_format(requested):
    if requested == "torch":
        return "torch"
    if requested == "safetensors":
        return "safetensors"
    try:
        import safetensors.torch  # noqa: F401
        return "safetensors"
    except ImportError:
        return "torch"


def shard_paths(shard_dir, batch_idx, shard_format, dataset_shard_index, dataset_num_shards):
    stem = f"batch_{batch_idx:06d}"
    if dataset_num_shards > 1:
        stem = f"dataset_shard_{dataset_shard_index:02d}_of_{dataset_num_shards:02d}_{stem}"
    suffix = ".safetensors" if shard_format == "safetensors" else ".pt"
    return (
        shard_dir / f"{stem}{suffix}",
        shard_dir / f"{stem}.json",
        stem,
    )


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.batches_per_shard < 1:
        raise ValueError("--batches-per-shard must be >= 1")

    if args.writer_threads is not None:
        args.gpu_packer_threads = args.writer_threads
    if args.save_queue_size is not None:
        args.pack_queue_size = args.save_queue_size
        args.write_queue_size = max(args.write_queue_size, args.save_queue_size)
    if args.gpu_packer_threads < 1:
        raise ValueError("--gpu-packer-threads must be >= 1")
    if args.disk_writer_threads < 1:
        raise ValueError("--disk-writer-threads must be >= 1")
    if args.pack_queue_size < 1:
        raise ValueError("--pack-queue-size must be >= 1")
    if args.write_queue_size < 1:
        raise ValueError("--write-queue-size must be >= 1")

    project_root = args.project_root.resolve()
    embeddings_dir = project_root / "embeddings" / "echo"
    shard_dir = (args.staging_dir.resolve() if args.staging_dir is not None
                 else embeddings_dir / "shards")
    datasets_dir = project_root / "datasets"
    results_dir = project_root / "results"
    for d in (embeddings_dir, shard_dir, datasets_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from vllm import LLM, SamplingParams, TokensPrompt

    shard_format = resolve_shard_format(args.shard_format)

    print(f"Loading model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2 * args.max_tokens + 32,
        trust_remote_code=True,
        worker_extension_cls="worker_extension_v2.SignalExtractorExtension",
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    tokenizer = llm.get_tokenizer()

    sampling = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)

    assert len(llm.collective_rpc("get_architecture")) == 1, \
        "Only tensor_parallel_size=1 supported"

    print("Discovering architecture...")
    arch = json.loads(llm.collective_rpc("get_architecture")[0])
    for key, expected in EXPECTED_ARCH.items():
        if arch.get(key) != expected:
            raise RuntimeError(f"Arch mismatch: {key}={arch.get(key)}, "
                               f"expected {expected}")
    print("Architecture verified OK")
    (results_dir / "architecture.json").write_text(
        json.dumps(arch, indent=2), encoding="utf-8")

    print(f"Layers: {arch['num_layers']}, "
          f"Attn: {arch['attn_layer_indices']}, "
          f"Hidden: {arch['hidden_size']}, "
          f"Experts: {arch['num_experts']}")

    hook_config = {
        "arch": arch,
        "options": {
            "gpu_packer_threads": args.gpu_packer_threads,
            "disk_writer_threads": args.disk_writer_threads,
            "pack_queue_size": args.pack_queue_size,
            "write_queue_size": args.write_queue_size,
            "compute_prenorm_attn_dist": args.compute_prenorm_attn_dist,
            "shard_format": shard_format,
            "activation_quantization": args.activation_quantization,
        },
    }
    print(llm.collective_rpc("setup_hooks", args=(json.dumps(hook_config),))[0])
    print(
        f"Shard output: {shard_dir} ({shard_format}, "
        f"activation_quantization={args.activation_quantization}, "
        f"gpu_packers={args.gpu_packer_threads}, "
        f"disk_writers={args.disk_writer_threads}, "
        f"batches_per_shard={args.batches_per_shard})"
    )

    # Test
    print("\n--- Test ---")
    echo_ids, N, sep_len = build_echo_token_ids(
        tokenizer, "Test sentence for extraction.", args.max_tokens)
    assert echo_ids[:N] == echo_ids[N + sep_len:], "Echo token mismatch"
    test_batch_info = {
        "shard_stem": "inspect",
        "shard_data_path": "/dev/null",
        "shard_meta_path": "/dev/null",
        "shard_piece_index": 0,
        "shard_piece_count": 1,
        "records": [{
            "prompt_key": prompt_key(echo_ids),
            "echo_start": N + sep_len,
            "echo_end": N + sep_len + N,
            "text_id": "test",
            "kind": "test",
            "ordinal": 0,
            "filename": "inspect",
            "record_index": 0,
        }],
    }

    llm.collective_rpc("clear_capture")
    llm.collective_rpc("prepare_batch", args=(json.dumps(test_batch_info),))
    llm.generate([TokensPrompt(prompt_token_ids=echo_ids)],
                 sampling_params=sampling)

    tp = llm.collective_rpc("get_captured_payload")[0]

    print(f"Tokens: {N}, Echo: {len(echo_ids)}, Keys: {len(tp)}")
    for key in sorted(tp):
        v = tp[key]
        if isinstance(v, np.ndarray) and v.ndim > 0:
            print(f"  {key}: {v.shape} {v.dtype}")

    if args.inspect_only:
        llm.collective_rpc("remove_hooks")
        print("--inspect-only: done.")
        return

    print("\nLoading SciFact...")
    corpus, queries, qrels = load_scifact(datasets_dir)
    texts = build_texts(corpus, queries, args.limit_docs, args.limit_queries)
    if args.num_shards > 1:
        texts = [
            text for idx, text in enumerate(texts)
            if idx % args.num_shards == args.shard_index
        ]
    total = len(texts)
    if args.num_shards > 1:
        print(f"Total: {total} texts on shard {args.shard_index}/{args.num_shards}")
    else:
        print(f"Total: {total} texts")

    prepared_records = []
    for kind, ordinal, text_id, text, filename in texts:
        echo_ids, original_len, sep_len = build_echo_token_ids(
            tokenizer, text, args.max_tokens)
        echo_ids = np.asarray(echo_ids, dtype=np.int32)
        echo_start = original_len + sep_len
        echo_end = echo_start + original_len
        prepared_records.append({
            "kind": kind,
            "ordinal": ordinal,
            "text_id": text_id,
            "filename": filename,
            "echo_ids": echo_ids,
            "prompt_key": prompt_key(echo_ids),
            "echo_start": echo_start,
            "echo_end": echo_end,
            "seq_len": echo_end - echo_start,
        })
    print(f"Prepared: {len(prepared_records)} texts")

    batch_size = args.batch_size
    print(
        f"\nExtracting (batch_size={batch_size}, "
        f"max_num_batched_tokens={args.max_num_batched_tokens})..."
    )
    t_start = time.perf_counter()
    done = 0
    skipped = 0
    manifest_rows = []
    shard_rows_by_stem = {}
    num_batches = ((len(prepared_records) + batch_size - 1) // batch_size
                   if prepared_records else 0)

    for batch_idx, batch_start in enumerate(range(0, len(prepared_records), batch_size)):
        batch = prepared_records[batch_start:batch_start + batch_size]
        if not batch:
            continue

        shard_batch_idx = batch_idx // args.batches_per_shard
        shard_piece_index = batch_idx % args.batches_per_shard
        shard_piece_count = min(
            args.batches_per_shard,
            num_batches - shard_batch_idx * args.batches_per_shard,
        )
        data_path, meta_path, shard_stem = shard_paths(
            shard_dir, shard_batch_idx, shard_format, args.shard_index, args.num_shards)
        shard_row = shard_rows_by_stem.setdefault(
            shard_stem,
            {
                "shard_stem": shard_stem,
                "shard_file": str(data_path),
                "meta_file": str(meta_path),
                "num_records": 0,
                "num_tokens": 0,
                "storage_format": shard_format,
                "piece_count": shard_piece_count,
            },
        )
        record_base = int(shard_row["num_records"])
        token_base = int(shard_row["num_tokens"])
        seq_offset = 0
        for record_idx, record in enumerate(batch):
            manifest_rows.append({
                "kind": record["kind"],
                "ordinal": record["ordinal"],
                "text_id": record["text_id"],
                "filename": record["filename"],
                "shard_stem": shard_stem,
                "shard_file": str(data_path),
                "meta_file": str(meta_path),
                "record_index": record_base + record_idx,
                "token_offset": token_base + seq_offset,
                "seq_len": record["seq_len"],
                "storage_format": shard_format,
            })
            seq_offset += record["seq_len"]
        shard_row["num_records"] = record_base + len(batch)
        shard_row["num_tokens"] = token_base + seq_offset

        if data_path.exists() and meta_path.exists():
            skipped += len(batch)
            done += len(batch)
            continue

        prompt_ids_list = []
        batch_records = []
        for record_idx, record in enumerate(batch):
            echo_ids = record["echo_ids"]
            prompt_ids_list.append(echo_ids.tolist())
            batch_records.append({
                "prompt_key": record["prompt_key"],
                "echo_start": record["echo_start"],
                "echo_end": record["echo_end"],
                "text_id": record["text_id"],
                "kind": record["kind"],
                "ordinal": record["ordinal"],
                "filename": record["filename"],
                "record_index": record_idx,
                "shard_record_index": record_base + record_idx,
            })

        llm.collective_rpc("clear_capture")
        batch_info = {
            "shard_stem": shard_stem,
            "shard_data_path": str(data_path),
            "shard_meta_path": str(meta_path),
            "shard_piece_index": shard_piece_index,
            "shard_piece_count": shard_piece_count,
            "records": batch_records,
        }
        llm.collective_rpc("prepare_batch", args=(json.dumps(batch_info),))
        prompts = [TokensPrompt(prompt_token_ids=ids)
                   for ids in prompt_ids_list]
        llm.generate(prompts, sampling_params=sampling)
        llm.collective_rpc("transfer_and_save_async")
        done += len(batch)

        elapsed = time.perf_counter() - t_start
        processed = done - skipped
        rate = processed / max(elapsed, 1e-6)
        remaining = ((total - done) / max(rate, 1e-6)) if rate > 0 else 0
        print(f"  [{done}/{total}] {elapsed:.0f}s, "
              f"~{remaining:.0f}s left, {rate:.1f} t/s")

    llm.collective_rpc("flush_saves")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. {done} texts in {elapsed:.1f}s ({skipped} resumed)")

    if args.num_shards > 1:
        manifest_path = results_dir / f"records_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
        shard_manifest_path = results_dir / f"shards_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.csv"
    else:
        manifest_path = results_dir / "records.csv"
        shard_manifest_path = results_dir / "shards.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "ordinal",
                "text_id",
                "filename",
                "shard_stem",
                "shard_file",
                "meta_file",
                "record_index",
                "token_offset",
                "seq_len",
                "storage_format",
            ],
        )
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)
    print(f"Manifest: {manifest_path}")

    with shard_manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "shard_stem",
                "shard_file",
                "meta_file",
                "num_records",
                "num_tokens",
                "storage_format",
                "piece_count",
            ],
        )
        w.writeheader()
        for row in shard_rows_by_stem.values():
            w.writerow(row)
    print(f"Shard manifest: {shard_manifest_path}")

    llm.collective_rpc("remove_hooks")
    print("Done.")


if __name__ == "__main__":
    main()
