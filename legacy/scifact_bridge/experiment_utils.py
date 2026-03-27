from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp


MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"
MAX_LENGTH = 512
DOC_FILENAME_WIDTH = 7
QUERY_FILENAME_WIDTH = 6
SHARED_EXPERT_INDEX = 256
EPS = 1e-10


@dataclass(frozen=True)
class TextRecord:
    kind: str
    ordinal: int
    text_id: str
    text: str
    filename: str


def project_dirs(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root).resolve()
    embeddings = root / "embeddings"
    results = root / "results"
    datasets = root / "datasets"
    return {
        "root": root,
        "embeddings": embeddings,
        "causal": embeddings / "causal",
        "rerouted": embeddings / "rerouted",
        "paper_causal": embeddings / "paper_causal",
        "paper_rerouted": embeddings / "paper_rerouted",
        "results": results,
        "datasets": datasets,
    }


def ensure_project_dirs(project_root: str | Path) -> dict[str, Path]:
    dirs = project_dirs(project_root)
    for key in ("root", "embeddings", "causal", "rerouted", "paper_causal", "paper_rerouted", "results", "datasets"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def _sort_key(text_id: str) -> tuple[int, str]:
    if text_id.isdigit():
        return (0, f"{int(text_id):020d}")
    return (1, text_id)


def make_filename(kind: str, ordinal: int) -> str:
    if kind == "doc":
        return f"doc_{ordinal:0{DOC_FILENAME_WIDTH}d}.npz"
    if kind == "query":
        return f"query_{ordinal:0{QUERY_FILENAME_WIDTH}d}.npz"
    raise ValueError(f"Unsupported record kind: {kind}")


def load_scifact(dataset_root: str | Path):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    dataset = "scifact"
    dataset_root = Path(dataset_root)
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, str(dataset_root))
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return corpus, queries, qrels


def build_records(corpus: dict, queries: dict) -> list[TextRecord]:
    records: list[TextRecord] = []

    sorted_docs = sorted(corpus.items(), key=lambda item: _sort_key(str(item[0])))
    for ordinal, (doc_id, doc) in enumerate(sorted_docs, start=1):
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        joined = f"{title}. {text}".strip()
        records.append(
            TextRecord(
                kind="doc",
                ordinal=ordinal,
                text_id=str(doc_id),
                text=joined,
                filename=make_filename("doc", ordinal),
            )
        )

    sorted_queries = sorted(queries.items(), key=lambda item: _sort_key(str(item[0])))
    for ordinal, (query_id, text) in enumerate(sorted_queries, start=1):
        records.append(
            TextRecord(
                kind="query",
                ordinal=ordinal,
                text_id=str(query_id),
                text=str(text),
                filename=make_filename("query", ordinal),
            )
        )

    return records


def write_manifest(records: Sequence[TextRecord], results_dir: str | Path) -> Path:
    path = Path(results_dir) / "records.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "ordinal", "text_id", "filename"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "kind": record.kind,
                    "ordinal": record.ordinal,
                    "text_id": record.text_id,
                    "filename": record.filename,
                }
            )
    return path


def load_manifest(project_root: str | Path) -> list[dict[str, str]]:
    path = Path(project_root) / "results" / "records.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def manifest_by_kind(project_root: str | Path, kind: str) -> list[dict[str, str]]:
    rows = load_manifest(project_root)
    return [row for row in rows if row["kind"] == kind]


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def architecture_path(project_root: str | Path) -> Path:
    return Path(project_root) / "results" / "architecture.json"


def load_architecture(project_root: str | Path) -> dict:
    return load_json(architecture_path(project_root))


def l2_normalize_array(array: np.ndarray, axis: int = -1, eps: float = EPS) -> np.ndarray:
    denom = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.clip(denom, eps, None)


def dense_cosine_scores(query_matrix: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    query_matrix = np.asarray(query_matrix, dtype=np.float32)
    doc_matrix = np.asarray(doc_matrix, dtype=np.float32)
    return query_matrix @ doc_matrix.T


def sparse_stack(rows: Sequence[sp.csr_matrix]) -> sp.csr_matrix:
    if not rows:
        raise ValueError("Cannot stack an empty sparse row list.")
    return sp.vstack(rows, format="csr")


def sparse_cosine_scores(query_matrix: sp.csr_matrix, doc_matrix: sp.csr_matrix) -> np.ndarray:
    scores = query_matrix @ doc_matrix.T
    return scores.toarray().astype(np.float32, copy=False)


def make_results_dict(query_ids: Sequence[str], doc_ids: Sequence[str], scores: np.ndarray) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for row_idx, query_id in enumerate(query_ids):
        row = scores[row_idx]
        order = np.argsort(-row)
        results[query_id] = {doc_ids[col_idx]: float(row[col_idx]) for col_idx in order}
    return results


def human_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, rem = divmod(minutes, 60)
    return f"{int(hours)}h {int(rem)}m"


def human_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_npz_fields(path: str | Path, fields: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {field: data[field] for field in fields}


def normalize_multivector(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    return l2_normalize_array(vectors, axis=-1)


def pack_multivectors(batch: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not batch:
        raise ValueError("Cannot pack an empty batch.")
    dim = batch[0].shape[1]
    max_len = max(item.shape[0] for item in batch)
    packed = np.zeros((len(batch), max_len, dim), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=bool)
    for idx, item in enumerate(batch):
        packed[idx, : item.shape[0]] = item
        mask[idx, : item.shape[0]] = True
    return packed, mask


def batched(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def routed_distribution_from_sparse(indices: np.ndarray, weights: np.ndarray, num_experts: int) -> np.ndarray:
    dist = np.zeros((indices.shape[0], num_experts), dtype=np.float32)
    routed_indices = indices[:, :8].astype(np.int64, copy=False)
    routed_weights = weights[:, :8].astype(np.float32, copy=False)
    rows = np.arange(indices.shape[0], dtype=np.int64)[:, None]
    dist[rows, routed_indices] = routed_weights
    return dist


def jaccard_from_binary_matrix(binary: np.ndarray) -> np.ndarray:
    ints = binary.astype(np.int32, copy=False)
    intersection = ints @ ints.T
    row_sums = ints.sum(axis=1, dtype=np.int32)
    union = row_sums[:, None] + row_sums[None, :] - intersection
    union = np.maximum(union, 1)
    return intersection.astype(np.float32) / union.astype(np.float32)


def timer() -> tuple[float, callable]:
    start = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - start

    return start, elapsed
