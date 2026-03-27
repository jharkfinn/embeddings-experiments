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
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-doc-tokens", type=int, default=512)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--batch-token-budget", type=int, default=None)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--skip-smoke-test", action="store_true")
    return parser.parse_args()


def load_scifact(dataset_dir: Path):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, str(dataset_dir))
    return GenericDataLoader(data_path).load(split="test")


def _sort_key(text_id: str):
    if text_id.isdigit():
        return (0, f"{int(text_id):020d}")
    return (1, text_id)


def build_texts(corpus, queries, limit_docs=None, limit_queries=None):
    texts = []
    sorted_docs = sorted(corpus.items(), key=lambda item: _sort_key(str(item[0])))
    if limit_docs is not None:
        sorted_docs = sorted_docs[:limit_docs]
    for ordinal, (doc_id, doc) in enumerate(sorted_docs, start=1):
        title = doc.get("title", "").strip()
        body = doc.get("text", "").strip()
        text = f"{title}. {body}".strip()
        texts.append(("doc", ordinal, str(doc_id), text, f"doc_{ordinal:07d}.pt"))

    sorted_queries = sorted(queries.items(), key=lambda item: _sort_key(str(item[0])))
    if limit_queries is not None:
        sorted_queries = sorted_queries[:limit_queries]
    for ordinal, (query_id, text) in enumerate(sorted_queries, start=1):
        texts.append(("query", ordinal, str(query_id), str(text), f"query_{ordinal:06d}.pt"))
    return texts


def build_echo_token_ids(tokenizer, text: str, max_tokens: int):
    original_ids = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
    sep_ids = tokenizer.encode(ECHO_SEP, add_special_tokens=False)
    return original_ids + sep_ids + original_ids, len(original_ids), len(sep_ids)


def prompt_key(token_ids):
    array = np.asarray(token_ids, dtype=np.int32)
    return hashlib.sha1(array.tobytes()).hexdigest()


def shard_paths(shard_dir: Path, batch_idx: int):
    stem = f"batch_{batch_idx:06d}"
    return (
        str(shard_dir / f"{stem}.safetensors"),
        str(shard_dir / f"{stem}.json"),
        stem,
    )


def batch_records(records, max_batch_size: int, token_budget: int):
    current = []
    current_tokens = 0
    for record in records:
        prompt_tokens = int(record["prompt_len"])
        if prompt_tokens > token_budget:
            raise RuntimeError(
                f"Prompt for text_id={record['text_id']} has {prompt_tokens} tokens, "
                f"exceeding batch token budget {token_budget}"
            )
        would_exceed_size = len(current) >= max_batch_size
        would_exceed_tokens = current and current_tokens + prompt_tokens > token_budget
        if would_exceed_size or would_exceed_tokens:
            yield current
            current = []
            current_tokens = 0
        current.append(record)
        current_tokens += prompt_tokens
    if current:
        yield current


def run_smoke_test(llm, tokenizer, sampling, max_tokens: int):
    from vllm import TokensPrompt

    echo_ids, n_tokens, sep_len = build_echo_token_ids(tokenizer, "Test sentence.", max_tokens)
    assert echo_ids[:n_tokens] == echo_ids[n_tokens + sep_len :]

    test_info = {
        "shard_data_path": "/dev/null",
        "shard_meta_path": "/dev/null",
        "records": [
            {
                "prompt_key": prompt_key(echo_ids),
                "echo_start": n_tokens + sep_len,
                "echo_end": n_tokens + sep_len + n_tokens,
                "text_id": "test",
                "kind": "test",
                "ordinal": 0,
                "filename": "test",
                "record_index": 0,
            }
        ],
    }
    llm.collective_rpc("clear_capture")
    llm.collective_rpc("prepare_batch", args=(json.dumps(test_info),))
    llm.generate([TokensPrompt(prompt_token_ids=echo_ids)], sampling_params=sampling)
    payload = llm.collective_rpc("get_captured_payload")[0]
    print(f"Smoke test OK: {len(payload)} payload keys")


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    shard_dir = project_root / "embeddings" / "echo" / "shards"
    datasets_dir = project_root / "datasets"
    results_dir = project_root / "results"
    for directory in (project_root, shard_dir, datasets_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

    from vllm import LLM, SamplingParams, TokensPrompt

    corpus, queries, qrels = load_scifact(datasets_dir)
    texts = build_texts(corpus, queries, args.limit_docs, args.limit_queries)
    print(
        f"Loaded SciFact: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels; "
        f"prepared {len(texts)} texts"
    )

    batch_token_budget = (
        args.batch_token_budget
        if args.batch_token_budget is not None
        else args.max_num_batched_tokens
    )

    print(f"Loading model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2 * args.max_tokens + 32,
        trust_remote_code=True,
        worker_extension_cls="worker_extension_nb.SignalExtractorExtension",
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)

    arch = json.loads(llm.collective_rpc("get_architecture")[0])
    for key, expected in EXPECTED_ARCH.items():
        if arch.get(key) != expected:
            raise RuntimeError(f"Arch mismatch: {key}={arch.get(key)}, expected {expected}")
    print("Architecture verified OK")

    hook_config = {
        "arch": arch,
        "options": {
            "shard_format": "safetensors",
            "write_queue_size": 4,
        },
    }
    print(llm.collective_rpc("setup_hooks", args=(json.dumps(hook_config),))[0])

    if not args.skip_smoke_test:
        run_smoke_test(llm, tokenizer, sampling, args.max_tokens)

    prepared = []
    for kind, ordinal, text_id, text, filename in texts:
        max_len = args.max_query_tokens if kind == "query" else args.max_doc_tokens
        echo_ids, orig_len, sep_len = build_echo_token_ids(tokenizer, text, min(args.max_tokens, max_len))
        echo_ids_array = np.asarray(echo_ids, dtype=np.int32)
        echo_start = orig_len + sep_len
        echo_end = echo_start + orig_len
        prepared.append(
            {
                "kind": kind,
                "ordinal": ordinal,
                "text_id": text_id,
                "filename": filename,
                "echo_ids": echo_ids_array,
                "prompt_key": prompt_key(echo_ids),
                "echo_start": echo_start,
                "echo_end": echo_end,
                "seq_len": echo_end - echo_start,
                "prompt_len": len(echo_ids),
            }
        )
    print(f"Prepared {len(prepared)} texts")

    manifest_rows = []
    shard_rows = []
    done = 0
    skipped = 0
    t_start = time.perf_counter()

    for batch_idx, batch in enumerate(batch_records(prepared, args.batch_size, batch_token_budget)):
        data_path, meta_path, shard_stem = shard_paths(shard_dir, batch_idx)

        seq_offset = 0
        for record_index, record in enumerate(batch):
            manifest_rows.append(
                {
                    "kind": record["kind"],
                    "ordinal": record["ordinal"],
                    "text_id": record["text_id"],
                    "filename": record["filename"],
                    "shard_stem": shard_stem,
                    "shard_file": data_path,
                    "meta_file": meta_path,
                    "record_index": record_index,
                    "token_offset": seq_offset,
                    "seq_len": record["seq_len"],
                    "storage_format": "safetensors",
                }
            )
            seq_offset += record["seq_len"]
        shard_rows.append(
            {
                "shard_stem": shard_stem,
                "shard_file": data_path,
                "meta_file": meta_path,
                "num_records": len(batch),
                "num_tokens": seq_offset,
                "storage_format": "safetensors",
            }
        )

        if os.path.exists(data_path) and os.path.exists(meta_path):
            skipped += len(batch)
            done += len(batch)
            continue

        batch_records_payload = []
        prompts = []
        for record_index, record in enumerate(batch):
            prompts.append(TokensPrompt(prompt_token_ids=record["echo_ids"].tolist()))
            batch_records_payload.append(
                {
                    "prompt_key": record["prompt_key"],
                    "echo_start": record["echo_start"],
                    "echo_end": record["echo_end"],
                    "text_id": record["text_id"],
                    "kind": record["kind"],
                    "ordinal": record["ordinal"],
                    "filename": record["filename"],
                    "record_index": record_index,
                }
            )

        batch_info = {
            "shard_data_path": data_path,
            "shard_meta_path": meta_path,
            "records": batch_records_payload,
        }

        llm.collective_rpc("clear_capture")
        llm.collective_rpc("prepare_batch", args=(json.dumps(batch_info),))
        llm.generate(prompts, sampling_params=sampling)
        saved_count = llm.collective_rpc("transfer_and_save_async")[0]
        if saved_count != len(batch):
            raise RuntimeError(f"Worker saved {saved_count} records for a batch of {len(batch)}")

        done += len(batch)
        elapsed = time.perf_counter() - t_start
        processed = done - skipped
        rate = processed / max(elapsed, 1e-6)
        remaining = (len(prepared) - done) / max(rate, 1e-6) if rate > 0 else 0.0
        if done % max(args.batch_size * 4, 1) == 0 or done >= len(prepared):
            print(
                f"  [{done}/{len(prepared)}] elapsed={elapsed:.0f}s "
                f"remaining~{remaining:.0f}s rate={rate:.2f} texts/s"
            )

    llm.collective_rpc("flush_saves")
    llm.collective_rpc("remove_hooks")

    elapsed = time.perf_counter() - t_start
    print(f"Done. {done} texts in {elapsed:.1f}s ({skipped} resumed)")

    manifest_path = results_dir / "records.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    shard_manifest_path = results_dir / "shards.csv"
    with shard_manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(shard_rows[0].keys()))
        writer.writeheader()
        writer.writerows(shard_rows)

    run_config = {
        "model_name": args.model_name,
        "max_tokens": args.max_tokens,
        "max_doc_tokens": args.max_doc_tokens,
        "max_query_tokens": args.max_query_tokens,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "batch_token_budget": batch_token_budget,
        "limit_docs": args.limit_docs,
        "limit_queries": args.limit_queries,
    }
    (results_dir / "extraction_run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest_path}")
    print(f"Shard manifest: {shard_manifest_path}")


if __name__ == "__main__":
    main()
