from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from kv_prepend_experiment.logging_utils import configure_logging
from kv_prepend_experiment.nano_beir import load_nanobeir_task

logger = logging.getLogger(__name__)


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
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = args.log_path or (args.output.parent / "artifacts" / "logs" / "build_nanobeir_records.log")
    configure_logging(log_path=log_path, level=args.log_level)
    logger.info("build_records_start datasets=%s repo=%s output=%s", args.datasets, args.repo_name, args.output)
    records = []
    for dataset_name in args.datasets:
        logger.info("load_dataset_start dataset=%s", dataset_name)
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
        logger.info(
            "load_dataset_done dataset=%s docs=%s queries=%s cumulative_records=%s",
            dataset_name,
            len(task.corpus),
            len(task.queries),
            len(records),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("build_records_done output=%s num_records=%s", args.output, len(records))
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
