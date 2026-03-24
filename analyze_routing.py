from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from runtime_bootstrap import bootstrap_workspace_env

bootstrap_workspace_env()

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from experiment_utils import (
    EPS,
    MODEL_NAME,
    ensure_project_dirs,
    jaccard_from_binary_matrix,
    load_architecture,
    load_manifest,
    load_npz_fields,
    load_scifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze routing deltas between all-attention and paper-layer rerouted extraction passes."
    )
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/kv_moee_experiment"))
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def pass_doc_rows(project_root: Path, pass_name: str) -> list[dict]:
    rows = [row for row in load_manifest(project_root) if row["kind"] == "doc"]
    base_dir = project_root / "embeddings" / pass_name
    for row in rows:
        row["path"] = str(base_dir / row["filename"])
    return rows


def routed_distributions(data: dict[str, np.ndarray], architecture: dict) -> np.ndarray:
    num_layers = architecture["num_layers"]
    num_experts = architecture["num_experts"]
    indices = data["routing_indices"][:, :, :8].astype(np.int64, copy=False)
    weights = data["routing_weights"][:, :, :8].astype(np.float32, copy=False)
    full_last = data["routing_full_last"].astype(np.float32, copy=False)

    seq_len = indices.shape[1]
    dists = np.zeros((num_layers, seq_len, num_experts), dtype=np.float32)
    token_rows = np.arange(seq_len, dtype=np.int64)[:, None]
    for layer_idx in range(num_layers):
        dists[layer_idx, token_rows, indices[layer_idx]] = weights[layer_idx]
        dists[layer_idx, -1] = full_last[layer_idx]
    return dists


def topk_changed_fraction(causal_indices: np.ndarray, rerouted_indices: np.ndarray) -> np.ndarray:
    causal_sorted = np.sort(causal_indices[:, :, :8], axis=-1)
    rerouted_sorted = np.sort(rerouted_indices[:, :, :8], axis=-1)
    return np.any(causal_sorted != rerouted_sorted, axis=-1).mean(axis=1)


def layerwise_plot(
    architecture: dict,
    kl_values: np.ndarray,
    top1_values: np.ndarray,
    topk_values: np.ndarray,
    output_path: Path,
) -> None:
    layer_indices = np.arange(architecture["num_layers"])
    attn_mask = np.zeros(architecture["num_layers"], dtype=bool)
    attn_mask[architecture["attn_layer_indices"]] = True
    delta_mask = ~attn_mask

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    metrics = [
        ("Routing KL", kl_values),
        ("Top-1 Changed Fraction", top1_values),
        ("Top-k Changed Fraction", topk_values),
    ]
    for axis, (title, values) in zip(axes, metrics, strict=True):
        axis.plot(layer_indices, values, color="0.5", linewidth=1.5)
        axis.scatter(layer_indices[delta_mask], values[delta_mask], color="#1f77b4", marker="o", label="DeltaNet")
        axis.scatter(layer_indices[attn_mask], values[attn_mask], color="#ff7f0e", marker="s", label="Attention")
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Layer Depth")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dirs = ensure_project_dirs(args.project_root)
    architecture = load_architecture(args.project_root)
    _corpus, _queries, qrels = load_scifact(dirs["datasets"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    allattn_rows = pass_doc_rows(args.project_root, "rerouted")
    paper_rows = pass_doc_rows(args.project_root, "paper_rerouted")
    if len(allattn_rows) != len(paper_rows):
        raise RuntimeError("All-attention and paper-layer document counts do not match.")

    rng = np.random.default_rng(args.seed)
    sample_indices = set(rng.choice(len(allattn_rows), size=min(10, len(allattn_rows)), replace=False).tolist())

    kl_sum = np.zeros(architecture["num_layers"], dtype=np.float64)
    top1_sum = np.zeros(architecture["num_layers"], dtype=np.float64)
    topk_sum = np.zeros(architecture["num_layers"], dtype=np.float64)
    paper_binary = np.zeros(
        (len(paper_rows), architecture["num_layers"] * architecture["num_experts"]),
        dtype=bool,
    )
    routing_examples: list[str] = []

    for row_idx, (allattn_row, paper_row) in enumerate(zip(allattn_rows, paper_rows, strict=True)):
        allattn = load_npz_fields(
            allattn_row["path"],
            ["routing_indices", "routing_weights", "routing_full_last", "token_ids"],
        )
        paper = load_npz_fields(
            paper_row["path"],
            ["routing_indices", "routing_weights", "routing_full_last", "token_ids"],
        )

        allattn_dist = routed_distributions(allattn, architecture)
        paper_dist = routed_distributions(paper, architecture)

        kl_per_token = np.sum(
            allattn_dist * (np.log(allattn_dist + EPS) - np.log(paper_dist + EPS)),
            axis=-1,
        )
        kl_sum += kl_per_token.mean(axis=1)

        allattn_indices = allattn["routing_indices"].astype(np.int64, copy=False)
        paper_indices = paper["routing_indices"].astype(np.int64, copy=False)
        top1_sum += (allattn_indices[:, :, 0] != paper_indices[:, :, 0]).mean(axis=1)
        topk_sum += topk_changed_fraction(allattn_indices, paper_indices)

        for layer_idx in range(architecture["num_layers"]):
            active = np.unique(paper_indices[layer_idx, :, :8])
            paper_binary[row_idx, layer_idx * architecture["num_experts"] + active] = True

        if row_idx in sample_indices:
            token_ids = allattn["token_ids"].astype(np.int32, copy=False)
            token_strings = tokenizer.convert_ids_to_tokens(token_ids.tolist())
            mean_token_kl = kl_per_token.mean(axis=0)
            top_token_indices = np.argsort(-mean_token_kl)[:10]
            routing_examples.append(f"Document {allattn_row['text_id']}")
            for token_idx in top_token_indices.tolist():
                left = max(0, token_idx - 5)
                right = min(len(token_ids), token_idx + 6)
                context = tokenizer.decode(token_ids[left:right].tolist(), skip_special_tokens=False)
                routing_examples.append(
                    f"  token_idx={token_idx:03d} kl={mean_token_kl[token_idx]:.6f} "
                    f"token={token_strings[token_idx]!r} context={context!r}"
                )
            routing_examples.append("")

    doc_count = max(len(allattn_rows), 1)
    kl_mean = kl_sum / doc_count
    top1_mean = top1_sum / doc_count
    topk_mean = topk_sum / doc_count

    layerwise_plot(
        architecture,
        kl_mean,
        top1_mean,
        topk_mean,
        dirs["results"] / "routing_delta_by_layer.png",
    )
    (dirs["results"] / "routing_delta_examples.txt").write_text(
        "\n".join(routing_examples) + "\n",
        encoding="utf-8",
    )

    doc_index = {row["text_id"]: idx for idx, row in enumerate(paper_rows)}
    relevant_mask = np.zeros((len(paper_rows), len(paper_rows)), dtype=bool)
    for relevant_docs in qrels.values():
        present = [doc_index[doc_id] for doc_id in relevant_docs if doc_id in doc_index]
        for left_idx, right_idx in itertools.combinations(present, 2):
            relevant_mask[left_idx, right_idx] = True
            relevant_mask[right_idx, left_idx] = True

    jaccard = jaccard_from_binary_matrix(paper_binary)
    upper_mask = np.triu(np.ones_like(jaccard, dtype=bool), k=1)
    relevant_upper = upper_mask & relevant_mask
    nonrelevant_upper = upper_mask & ~relevant_mask

    relevant_mean = float(jaccard[relevant_upper].mean()) if relevant_upper.any() else float("nan")
    nonrelevant_mean = float(jaccard[nonrelevant_upper].mean()) if nonrelevant_upper.any() else float("nan")
    expert_text = [
        f"Relevant-pair mean Jaccard similarity: {relevant_mean:.6f}",
        f"Non-relevant-pair mean Jaccard similarity: {nonrelevant_mean:.6f}",
        f"Relevant pair count: {int(relevant_upper.sum())}",
        f"Non-relevant pair count: {int(nonrelevant_upper.sum())}",
    ]
    (dirs["results"] / "expert_coactivation.txt").write_text("\n".join(expert_text) + "\n", encoding="utf-8")

    print("\n".join(expert_text))


if __name__ == "__main__":
    main()
