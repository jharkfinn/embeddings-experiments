"""
Signal extraction from Qwen3.5-35B-A3B-FP8 using vLLM with Echo.

Uses worker_extension_cls to hook into the model running inside vLLM's
worker process. GPU->CPU transfer overlaps with next batch's compute.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.98)
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
    echo_ids = original_ids + sep_ids + original_ids
    return echo_ids, len(original_ids), len(sep_ids)


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    embeddings_dir = project_root / "embeddings" / "echo"
    datasets_dir = project_root / "datasets"
    results_dir = project_root / "results"
    for d in (embeddings_dir, datasets_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from vllm import LLM, SamplingParams, TokensPrompt

    print(f"Loading model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2 * args.max_tokens + 32,
        trust_remote_code=True,
        worker_extension_cls="worker_extension.SignalExtractorExtension",
        enable_chunked_prefill=False,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(max_tokens=1, temperature=0.0)

    # Assert single GPU
    assert len(llm.collective_rpc("get_architecture")) == 1, \
        "Only tensor_parallel_size=1 supported"

    # Discover + verify architecture
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

    # Install hooks
    print(llm.collective_rpc("setup_hooks", args=(json.dumps(arch),))[0])

    # Test forward pass
    print("\n--- Test ---")
    echo_ids, N, sep_len = build_echo_token_ids(
        tokenizer, "Test sentence for extraction.", args.max_tokens)
    assert echo_ids[:N] == echo_ids[N + sep_len:], "Echo token mismatch"

    llm.collective_rpc("clear_capture")
    llm.generate([TokensPrompt(prompt_token_ids=echo_ids)],
                 sampling_params=sampling)

    echo_start, echo_end = N + sep_len, N + sep_len + N
    test_payload = llm.collective_rpc(
        "get_captured_payload",
        args=(echo_start, echo_end, echo_ids[echo_start:echo_end]))[0]

    print(f"Tokens: {N}, Echo: {len(echo_ids)}, "
          f"Payload keys: {len(test_payload)}")
    for key in sorted(test_payload):
        v = test_payload[key]
        if isinstance(v, np.ndarray) and v.ndim > 0:
            print(f"  {key}: {v.shape} {v.dtype}")

    if args.inspect_only:
        llm.collective_rpc("remove_hooks")
        print("--inspect-only: done.")
        return

    # Load dataset
    print("\nLoading SciFact...")
    corpus, queries, qrels = load_scifact(datasets_dir)
    texts = build_texts(corpus, queries, args.limit_docs, args.limit_queries)
    total = len(texts)
    print(f"Total: {total} texts")

    # Pre-compute sep length
    sep_ids = tokenizer.encode(ECHO_SEP, add_special_tokens=False)
    sep_len = len(sep_ids)

    # Extract with async saves
    batch_size = args.batch_size
    print(f"\nExtracting (batch_size={batch_size})...")
    t_start = time.perf_counter()
    done = 0
    skipped = 0

    for batch_start in range(0, total, batch_size):
        batch = texts[batch_start:batch_start + batch_size]

        pending = []
        for item in batch:
            if (embeddings_dir / item[4]).exists():
                skipped += 1
                done += 1
            else:
                pending.append(item)

        if not pending:
            continue

        # Build echo prompts
        prompt_ids_list = []
        original_lengths = []
        for kind, ordinal, text_id, text, filename in pending:
            echo_ids, N, _ = build_echo_token_ids(
                tokenizer, text, args.max_tokens)
            prompt_ids_list.append(echo_ids)
            original_lengths.append(N)

        # Compute flattened boundaries
        boundaries = []
        offset = 0
        for i, echo_ids in enumerate(prompt_ids_list):
            N = original_lengths[i]
            echo_start = offset + N + sep_len
            echo_end = echo_start + N
            boundaries.append((echo_start, echo_end))
            offset += len(echo_ids)

        # Forward pass
        llm.collective_rpc("clear_capture")
        prompts = [TokensPrompt(prompt_token_ids=ids)
                   for ids in prompt_ids_list]
        llm.generate(prompts, sampling_params=sampling)

        # Build batch info and kick off async save
        batch_info = []
        for i, (kind, ordinal, text_id, text, filename) in enumerate(pending):
            es, ee = boundaries[i]
            N = original_lengths[i]
            batch_info.append([
                es, ee,
                prompt_ids_list[i][es:ee],
                str(embeddings_dir / filename),
                text_id, kind,
            ])

        # This transfers GPU->CPU synchronously, then saves in background
        llm.collective_rpc(
            "transfer_and_save_async", args=(json.dumps(batch_info),))
        done += len(pending)

        elapsed = time.perf_counter() - t_start
        processed = done - skipped
        rate = processed / max(elapsed, 1e-6)
        remaining = ((total - done) / max(rate, 1e-6)) if rate > 0 else 0
        print(f"  [{done}/{total}] {elapsed:.0f}s, "
              f"~{remaining:.0f}s left, {rate:.1f} t/s")

    # Wait for last save
    llm.collective_rpc("flush_saves")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone. {done} texts in {elapsed:.1f}s ({skipped} resumed)")

    # Manifest
    manifest_path = results_dir / "records.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "ordinal", "text_id", "filename"])
        w.writeheader()
        for kind, ordinal, text_id, text, filename in texts:
            w.writerow({"kind": kind, "ordinal": ordinal,
                         "text_id": text_id, "filename": filename})
    print(f"Manifest: {manifest_path}")

    llm.collective_rpc("remove_hooks")
    print("Done.")


if __name__ == "__main__":
    main()
