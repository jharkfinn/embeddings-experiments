from __future__ import annotations

import argparse
import json
from pathlib import Path

from kv_prepend_experiment.nano_beir import load_nanobeir_task


def _doc_text(row: dict[str, str]) -> str:
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def parse_args():
    parser = argparse.ArgumentParser(description="Build a records.json file from NanoBEIR tasks.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact", "fiqa2018", "quoraretrieval"],
        help="NanoBEIR task names to include.",
    )
    parser.add_argument(
        "--repo-name",
        default="zeta-alpha-ai/NanoBEIR",
        help="Default NanoBEIR dataset repo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "records_nanobeir_3tasks.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    records = []
    for dataset_name in args.datasets:
        task = load_nanobeir_task(args.repo_name, dataset_name)
        for doc_id, row in task.corpus.items():
            records.append(
                {
                    "text_id": str(doc_id),
                    "kind": "doc",
                    "dataset_name": task.dataset_name,
                    "text": _doc_text(row),
                }
            )
        for query_id, text in task.queries.items():
            records.append(
                {
                    "text_id": str(query_id),
                    "kind": "query",
                    "dataset_name": task.dataset_name,
                    "text": str(text),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "datasets": list(args.datasets),
                "num_records": len(records),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
