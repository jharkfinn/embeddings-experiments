from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .analysis import analyze_capture_directory
from .collection import InstrumentedQwen3MoeExperiment
from .config import ExperimentSpec, load_experiment_spec
from .evaluation import evaluate_signal, evaluate_task_suite, save_json
from .logging_utils import configure_logging, default_log_path
from .profiles import describe_run, named_run_specs, write_named_specs
from .runtime import import_torch

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="KV-prepend probing experiment")
    parser.add_argument("--spec", type=Path, default=Path(__file__).resolve().parents[1] / "default_experiment.json")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify-model")
    subparsers.add_parser("write-split-specs")

    describe = subparsers.add_parser("describe-run")
    describe.add_argument(
        "--run-name",
        type=str,
        required=True,
        choices=[
            "main_hf_teacher_forcing_3tasks",
            "calibration_hf_3tasks",
            "controls_hf_3tasks",
            "bridge_hf_3tasks",
        ],
    )

    collect = subparsers.add_parser("collect")
    collect.add_argument("--records-json", type=Path, required=True)
    collect.add_argument("--dataset-name", type=str, default="custom")
    collect.add_argument("--run-controls", action="store_true")
    collect.add_argument("--run-bridge", action="store_true")

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--task-name", type=str, required=True)
    evaluate.add_argument("--signal", type=str, required=True)
    evaluate.add_argument("--condition", type=str, default="causal")
    evaluate.add_argument("--pass-name", type=str, default="pass1")
    evaluate.add_argument("--layers", type=str, default="35")

    evaluate_suite = subparsers.add_parser("evaluate-suite")
    evaluate_suite.add_argument("--task-name", type=str, required=True)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output", type=Path, default=None)

    return parser.parse_args()


def _load_spec(path: Path) -> ExperimentSpec:
    return load_experiment_spec(path)


def _load_records(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_verify_model(spec: ExperimentSpec, root: Path):
    if spec.collection.runtime_backend not in {"hf", "hf_teacher_forcing"}:
        raise NotImplementedError(
            "verify-model currently runs against the HF collector path only. "
            "Use an HF-backed spec for contract verification."
        )
    experiment = InstrumentedQwen3MoeExperiment(spec, root)
    experiment.load()
    contract = experiment.contract
    print(json.dumps(contract.__dict__, indent=2, sort_keys=True))


def cmd_write_split_specs(spec: ExperimentSpec, root: Path):
    paths = write_named_specs(spec, root)
    print(json.dumps({"spec_paths": [str(path) for path in paths]}, indent=2, sort_keys=True))


def cmd_describe_run(spec: ExperimentSpec, args):
    run_spec = named_run_specs(spec)[args.run_name]
    payload = describe_run(run_spec, args.run_name)
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_collect(spec: ExperimentSpec, root: Path, args):
    records = _load_records(args.records_json)
    if spec.collection.runtime_backend == "vllm":
        raise NotImplementedError(
            "The vLLM backend has been moved to embeddings experiment/legacy/vllm_backend. "
            "Use the HF teacher-forcing specs for active runs."
        )
    experiment = InstrumentedQwen3MoeExperiment(spec, root)
    experiment.load()
    spec_path = experiment.save_spec_snapshot()
    need_bundles = bool(args.run_controls or args.run_bridge)
    paths, _ = experiment.collect_examples(records, dataset_name=args.dataset_name, retain_bundles=need_bundles)
    payload = {"main_captures": [str(path) for path in paths], "spec_snapshot": str(spec_path)}
    if args.run_controls:
        payload["controls"] = experiment.run_controls(records, dataset_name=f"{args.dataset_name}_controls")
    if args.run_bridge:
        bridge_paths, _ = experiment.run_bridge_echo(records, dataset_name=f"{args.dataset_name}_bridge")
        payload["bridge"] = [str(path) for path in bridge_paths]
    experiment.flush_writes()
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_evaluate(spec: ExperimentSpec, root: Path, args):
    layers = [int(layer.strip()) for layer in args.layers.split(",") if layer.strip()]
    results = evaluate_signal(
        capture_dir=root / spec.output.captures_dir,
        task_name=args.task_name,
        repo_name=spec.evaluation.nanobeir_repo,
        pass_name=args.pass_name,
        condition=args.condition,
        signal_name=args.signal,
        layer_indices=layers,
        quantization_spec=spec.quantization,
    )
    out_dir = root / spec.output.evaluation_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task_name}_{args.signal}_{args.condition}_{args.pass_name}.json"
    save_json(out_path, results)
    print(json.dumps({"output": str(out_path), "single_vector_ndcg_at_10": results["single_vector_ndcg_at_10"], "multivector_ndcg_at_10": results["multivector_ndcg_at_10"]}, indent=2, sort_keys=True))


def cmd_analyze(spec: ExperimentSpec, root: Path, args):
    output = args.output or (root / spec.output.analysis_dir / "capture_analysis.json")
    summary = analyze_capture_directory(root / spec.output.captures_dir, output)
    print(json.dumps({"output": str(output), "num_bundles": summary["num_bundles"]}, indent=2, sort_keys=True))


def cmd_evaluate_suite(spec: ExperimentSpec, root: Path, args):
    results = evaluate_task_suite(
        capture_dir=root / spec.output.captures_dir,
        repo_name=spec.evaluation.nanobeir_repo,
        task_name=args.task_name,
        quantization_spec=spec.quantization,
        selected_layers=spec.evaluation.selected_layers,
        fusion_weights=spec.evaluation.fusion_weights,
        rrf_k=spec.evaluation.rrf_k,
        candidate_pool_k=spec.evaluation.candidate_pool_k,
        topn_layers_for_grouping=spec.evaluation.topn_layers_for_grouping,
        layer_grouping_policy=spec.evaluation.layer_grouping_policy,
        fixed_group_layers=spec.evaluation.fixed_group_layers,
    )
    out_dir = root / spec.output.evaluation_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task_name}_suite.json"
    save_json(out_path, results)
    print(json.dumps({"output": str(out_path), "signals": list(results["signals"].keys())}, indent=2, sort_keys=True))


def main():
    args = parse_args()
    resolved_log_path = configure_logging(
        log_path=args.log_path or default_log_path(args.root, args.command),
        level=args.log_level,
    )
    logger.info(
        "cli_start command=%s spec=%s root=%s log_path=%s",
        args.command,
        args.spec,
        args.root,
        resolved_log_path,
    )
    spec = _load_spec(args.spec)
    root = args.root
    if args.command == "verify-model":
        return cmd_verify_model(spec, root)
    if args.command == "write-split-specs":
        return cmd_write_split_specs(spec, root)
    if args.command == "describe-run":
        return cmd_describe_run(spec, args)
    if args.command == "collect":
        return cmd_collect(spec, root, args)
    if args.command == "evaluate":
        return cmd_evaluate(spec, root, args)
    if args.command == "evaluate-suite":
        return cmd_evaluate_suite(spec, root, args)
    if args.command == "analyze":
        return cmd_analyze(spec, root, args)
    raise ValueError(f"Unknown command: {args.command}")
