from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader

from run_binary_notebook_analysis import _score_path, ndcg_at_k


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-n-options", type=str, default="1,5,10,20,40,80,121")
    parser.add_argument("--alpha-options", type=str, default="0.1,1,10,100")
    parser.add_argument("--clip-zscore", type=float, default=0.0)
    parser.add_argument("--fit-batch-size", type=int, default=200_000)
    parser.add_argument("--output-prefix", type=str, default="supervised_group_fusion")
    return parser.parse_args()


def parse_csv_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def load_scifact(dataset_dir: Path):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    data_path = util.download_and_unzip(url, str(dataset_dir))
    return GenericDataLoader(data_path).load(split="test")


def load_screening_rows(results_dir: Path):
    rows = []
    with (results_dir / "feature_screening_maxsim.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if "name" not in row and "group_name" in row:
                row["name"] = row["group_name"]
            row["ndcg_at_10"] = float(row["ndcg_at_10"])
            row["num_bits"] = int(row["num_bits"])
            row["layer"] = int(row["layer"])
            rows.append(row)
    rows.sort(key=lambda row: row["ndcg_at_10"], reverse=True)
    return rows


def build_label_matrices(query_ids, doc_ids, qrels):
    num_q = len(query_ids)
    num_d = len(doc_ids)
    doc_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}

    labels = np.zeros((num_q, num_d), dtype=np.float32)
    sample_weights = np.zeros((num_q, num_d), dtype=np.float32)

    for query_index, query_id in enumerate(query_ids):
        rels = qrels.get(query_id, {})
        positive_indices = [
            doc_to_index[doc_id]
            for doc_id, score in rels.items()
            if score > 0 and doc_id in doc_to_index
        ]
        if positive_indices:
            labels[query_index, positive_indices] = 1.0

        num_pos = len(positive_indices)
        num_neg = num_d - num_pos
        neg_weight = 0.5 / max(num_neg, 1)
        pos_weight = 0.5 / max(num_pos, 1) if num_pos else neg_weight
        sample_weights[query_index, :] = neg_weight
        if positive_indices:
            sample_weights[query_index, positive_indices] = pos_weight

    return labels, sample_weights


def make_query_folds(num_queries: int, num_folds: int, seed: int):
    indices = np.arange(num_queries, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [np.sort(fold.astype(np.int32)) for fold in np.array_split(indices, num_folds)]


def load_feature_tensor(group_rows, checkpoint_dir: Path, num_queries: int, num_docs: int, clip_zscore: float):
    num_groups = len(group_rows)
    features = np.empty((num_queries, num_docs, num_groups), dtype=np.float32)
    group_names = [row["name"] for row in group_rows]

    t0 = time.perf_counter()
    for group_index, row in enumerate(group_rows, start=1):
        score_path = _score_path(checkpoint_dir, row["name"])
        score_matrix = np.load(score_path, mmap_mode="r")
        if score_matrix.shape != (num_queries, num_docs):
            raise RuntimeError(
                f"Unexpected shape for {row['name']}: {score_matrix.shape}, expected {(num_queries, num_docs)}"
            )
        score_matrix = np.asarray(score_matrix, dtype=np.float32)
        mean = score_matrix.mean(axis=1, keepdims=True)
        std = score_matrix.std(axis=1, keepdims=True)
        normalized = (score_matrix - mean) / np.maximum(std, 1e-6)
        if clip_zscore > 0:
            np.clip(normalized, -clip_zscore, clip_zscore, out=normalized)
        features[:, :, group_index - 1] = normalized
        if group_index % 10 == 0 or group_index == num_groups:
            print(f"  Loaded {group_index}/{num_groups} groups ({time.perf_counter() - t0:.1f}s)")

    return group_names, features


def rank_groups_for_queries(features, query_indices, group_names, query_ids, doc_ids, qrels):
    ranked = []
    fold_query_ids = [query_ids[int(index)] for index in query_indices]
    for group_index, group_name in enumerate(group_names):
        group_scores = features[query_indices, :, group_index]
        ndcg = ndcg_at_k(group_scores, fold_query_ids, doc_ids, qrels, k=10)
        ranked.append(
            {
                "group_index": int(group_index),
                "group_name": group_name,
                "train_ndcg_at_10": float(ndcg),
            }
        )
    ranked.sort(key=lambda row: row["train_ndcg_at_10"], reverse=True)
    return ranked


def fit_weighted_ridge(X, y, sample_weight, alpha: float, batch_size: int):
    num_features = X.shape[1]
    xtx = np.zeros((num_features, num_features), dtype=np.float64)
    xty = np.zeros(num_features, dtype=np.float64)

    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        X_batch = np.asarray(X[start:end], dtype=np.float64)
        y_batch = np.asarray(y[start:end], dtype=np.float64)
        w_batch = np.asarray(sample_weight[start:end], dtype=np.float64)
        xtx += X_batch.T @ (X_batch * w_batch[:, None])
        xty += X_batch.T @ (y_batch * w_batch)

    xtx.flat[:: num_features + 1] += alpha
    weights = np.linalg.solve(xtx, xty)
    return weights.astype(np.float32)


def evaluate_cv(
    features,
    labels,
    sample_weights,
    group_names,
    query_ids,
    doc_ids,
    qrels,
    top_n_options,
    alpha_options,
    folds,
    batch_size,
):
    cv_rows = []
    best = None

    fold_rankings = []
    for fold_index, val_queries in enumerate(folds, start=1):
        train_queries = np.concatenate(
            [fold for other_index, fold in enumerate(folds, start=1) if other_index != fold_index]
        )
        ranking = rank_groups_for_queries(features, train_queries, group_names, query_ids, doc_ids, qrels)
        fold_rankings.append(ranking)
        print(
            f"Prepared fold {fold_index}/{len(folds)} ranking; "
            f"best train-only group = {ranking[0]['group_name']} ({ranking[0]['train_ndcg_at_10']:.4f})"
        )

    for top_n in top_n_options:
        print(f"\nTop-N = {top_n}")
        for alpha in alpha_options:
            fold_scores = []
            t0 = time.perf_counter()
            for fold_index, val_queries in enumerate(folds, start=1):
                train_queries = np.concatenate(
                    [fold for other_index, fold in enumerate(folds, start=1) if other_index != fold_index]
                )
                selected_indices = [row["group_index"] for row in fold_rankings[fold_index - 1][:top_n]]
                X_train = features[train_queries][:, :, selected_indices].reshape(-1, top_n)
                y_train = labels[train_queries].reshape(-1)
                w_train = sample_weights[train_queries].reshape(-1)
                weights = fit_weighted_ridge(X_train, y_train, w_train, alpha=alpha, batch_size=batch_size)
                val_features = features[val_queries][:, :, selected_indices]
                val_scores = np.tensordot(val_features, weights, axes=([2], [0]))
                val_query_ids = [query_ids[int(index)] for index in val_queries]
                fold_ndcg = ndcg_at_k(val_scores, val_query_ids, doc_ids, qrels, k=10)
                fold_scores.append(float(fold_ndcg))
                print(
                    f"  alpha={alpha:g} fold={fold_index}/{len(folds)} "
                    f"nDCG@10={fold_ndcg:.4f} ({time.perf_counter() - t0:.1f}s)"
                )

            mean_ndcg = float(np.mean(fold_scores))
            std_ndcg = float(np.std(fold_scores))
            row = {
                "top_n": int(top_n),
                "alpha": float(alpha),
                "mean_ndcg_at_10": mean_ndcg,
                "std_ndcg_at_10": std_ndcg,
                "fold_ndcgs": fold_scores,
            }
            cv_rows.append(row)
            print(
                f"  alpha={alpha:g} mean nDCG@10={mean_ndcg:.4f} "
                f"+/- {std_ndcg:.4f}"
            )
            if best is None or mean_ndcg > best["mean_ndcg_at_10"]:
                best = row

    cv_rows.sort(key=lambda row: row["mean_ndcg_at_10"], reverse=True)
    return best, cv_rows


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    results_dir = project_root / "results"
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else (results_dir / "checkpoints").resolve()
    )

    top_n_options = parse_csv_list(args.top_n_options, int)
    alpha_options = parse_csv_list(args.alpha_options, float)

    corpus, queries, qrels = load_scifact(project_root / "datasets")
    doc_ids = sorted(corpus.keys())
    query_ids = sorted(queries.keys())
    num_queries = len(query_ids)
    num_docs = len(doc_ids)

    screening_rows = load_screening_rows(results_dir)
    if top_n_options[-1] > len(screening_rows):
        raise ValueError(f"Top-N option exceeds available groups: {top_n_options[-1]} > {len(screening_rows)}")

    print(f"Loaded SciFact: {len(doc_ids)} docs, {len(query_ids)} queries, {len(qrels)} qrels")
    labels, sample_weights = build_label_matrices(query_ids, doc_ids, qrels)
    folds = make_query_folds(num_queries, args.num_folds, args.seed)

    group_names, features = load_feature_tensor(
        screening_rows,
        checkpoint_dir=checkpoint_dir,
        num_queries=num_queries,
        num_docs=num_docs,
        clip_zscore=args.clip_zscore,
    )

    best, cv_rows = evaluate_cv(
        features=features,
        labels=labels,
        sample_weights=sample_weights,
        group_names=group_names,
        query_ids=query_ids,
        doc_ids=doc_ids,
        qrels=qrels,
        top_n_options=top_n_options,
        alpha_options=alpha_options,
        folds=folds,
        batch_size=args.fit_batch_size,
    )

    print("\nBest CV config:")
    print(json.dumps(best, indent=2))

    best_top_n = int(best["top_n"])
    best_alpha = float(best["alpha"])
    final_ranking = rank_groups_for_queries(
        features,
        np.arange(num_queries, dtype=np.int32),
        group_names,
        query_ids,
        doc_ids,
        qrels,
    )
    final_indices = [row["group_index"] for row in final_ranking[:best_top_n]]
    final_features = features[:, :, final_indices]
    final_names = [group_names[index] for index in final_indices]

    print("\nFitting final model on all queries...")
    t0 = time.perf_counter()
    final_weights = fit_weighted_ridge(
        final_features.reshape(-1, best_top_n),
        labels.reshape(-1),
        sample_weights.reshape(-1),
        alpha=best_alpha,
        batch_size=args.fit_batch_size,
    )
    final_scores = np.tensordot(final_features, final_weights, axes=([2], [0]))
    final_train_ndcg = ndcg_at_k(final_scores, query_ids, doc_ids, qrels, k=10)
    print(f"Final fit done in {time.perf_counter() - t0:.1f}s")
    print(f"Train-set nDCG@10 (optimistic): {final_train_ndcg:.4f}")

    weights_rows = []
    screening_rank = {row["name"]: rank for rank, row in enumerate(screening_rows, start=1)}
    screening_ndcg = {row["name"]: row["ndcg_at_10"] for row in screening_rows}
    for group_name, weight in zip(final_names, final_weights):
        weights_rows.append(
            {
                "group": group_name,
                "weight": float(weight),
                "abs_weight": float(abs(weight)),
                "screening_rank": int(screening_rank[group_name]),
                "screening_ndcg_at_10": float(screening_ndcg[group_name]),
            }
        )
    weights_rows.sort(key=lambda row: row["abs_weight"], reverse=True)

    cv_path = results_dir / f"{args.output_prefix}_cv.csv"
    with cv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["top_n", "alpha", "mean_ndcg_at_10", "std_ndcg_at_10", "fold_ndcgs"],
        )
        writer.writeheader()
        for row in cv_rows:
            writer.writerow(
                {
                    "top_n": row["top_n"],
                    "alpha": row["alpha"],
                    "mean_ndcg_at_10": row["mean_ndcg_at_10"],
                    "std_ndcg_at_10": row["std_ndcg_at_10"],
                    "fold_ndcgs": json.dumps(row["fold_ndcgs"]),
                }
            )

    weights_path = results_dir / f"{args.output_prefix}_weights.csv"
    with weights_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "weight", "abs_weight", "screening_rank", "screening_ndcg_at_10"],
        )
        writer.writeheader()
        writer.writerows(weights_rows)

    results_path = results_dir / f"{args.output_prefix}_results.json"
    results_path.write_text(
        json.dumps(
            {
                "best_cv_config": best,
                "best_cv_ndcg_at_10": float(best["mean_ndcg_at_10"]),
                "num_folds": int(args.num_folds),
                "top_n_options": top_n_options,
                "alpha_options": alpha_options,
                "best_top_n_groups": final_names,
                "final_train_ndcg_at_10": float(final_train_ndcg),
                "top_weights": weights_rows[:25],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved: {cv_path}")
    print(f"Saved: {weights_path}")
    print(f"Saved: {results_path}")


if __name__ == "__main__":
    main()
