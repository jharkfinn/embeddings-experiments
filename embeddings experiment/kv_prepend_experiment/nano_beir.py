from __future__ import annotations

from typing import Any

from .runtime import import_datasets
from .types import RetrievalTask


_TASK_REPO_OVERRIDES = {
    "scifact": ("zeta-alpha-ai/NanoSciFact", None),
    "fiqa2018": ("zeta-alpha-ai/NanoFiQA2018", None),
    "quoraretrieval": ("zeta-alpha-ai/NanoQuoraRetrieval", None),
}


def _guess_column(row: dict[str, Any], candidates):
    for candidate in candidates:
        if candidate in row:
            return candidate
    raise KeyError(f"None of the candidate columns were found: {candidates}")


def _load_section(datasets, repo_name: str, *, config_name: str | None, section: str):
    attempts = []
    if config_name is not None:
        attempts.append({"name": config_name, "split": section})
    attempts.append({"name": section, "split": "train"})
    attempts.append({"name": section, "split": "test"})
    attempts.append({"name": section, "split": "validation"})

    last_error = None
    for kwargs in attempts:
        try:
            return datasets.load_dataset(repo_name, **kwargs)
        except Exception as exc:  # pragma: no cover - dataset API/version specific
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load {repo_name} section={section}")


def load_nanobeir_task(repo_name: str, dataset_name: str) -> RetrievalTask:
    datasets = import_datasets()
    dataset_name = dataset_name.lower()
    repo_override, config_override = _TASK_REPO_OVERRIDES.get(dataset_name, (repo_name, dataset_name))
    corpus_ds = _load_section(datasets, repo_override, config_name=config_override, section="corpus")
    query_ds = _load_section(datasets, repo_override, config_name=config_override, section="queries")
    qrels_ds = _load_section(datasets, repo_override, config_name=config_override, section="qrels")

    corpus = {}
    for row in corpus_ds:
        doc_id_key = _guess_column(row, ("_id", "doc_id", "id"))
        text_key = _guess_column(row, ("text", "contents", "content"))
        title = row.get("title", "")
        corpus[str(row[doc_id_key])] = {"text": str(row[text_key]), "title": str(title)}

    queries = {}
    for row in query_ds:
        query_id_key = _guess_column(row, ("_id", "query_id", "id"))
        text_key = _guess_column(row, ("text", "query"))
        queries[str(row[query_id_key])] = str(row[text_key])

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        query_id_key = _guess_column(row, ("query-id", "query_id", "query"))
        doc_id_key = _guess_column(row, ("corpus-id", "doc_id", "doc"))
        score_key = _guess_column(row, ("score", "relevance"))
        qrels.setdefault(str(row[query_id_key]), {})[str(row[doc_id_key])] = int(row[score_key])

    return RetrievalTask(dataset_name=dataset_name, corpus=corpus, queries=queries, qrels=qrels)


def build_synthetic_late_evidence_task(task: RetrievalTask, filler_repeat: int = 8) -> RetrievalTask:
    filler = "Background detail with no direct answer. "
    corpus = {}
    for doc_id, row in task.corpus.items():
        corpus[doc_id] = {
            **row,
            "text": filler * filler_repeat + row["text"],
        }
    return RetrievalTask(
        dataset_name=f"{task.dataset_name}_late_evidence",
        corpus=corpus,
        queries=dict(task.queries),
        qrels=dict(task.qrels),
    )
