from __future__ import annotations

from typing import Any

from .prompts import PromptExample
from .types import ExampleCaptureBundle, LayerCapture, PassCapture


def _slice_batch_value(value, row_idx: int, batch_size: int):
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] >= batch_size and row_idx < value.shape[0]:
            return value[row_idx : row_idx + 1]
    except ModuleNotFoundError:  # pragma: no cover
        pass
    return value


def _trim_batch_value(value, batch_size: int):
    if value is None:
        return None
    try:
        import torch

        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > batch_size:
            return value[:batch_size]
    except ModuleNotFoundError:  # pragma: no cover
        pass
    return value


def _slice_metadata(metadata: dict[str, Any], row_idx: int, batch_size: int) -> dict[str, Any]:
    out = {key: value for key, value in metadata.items() if key != "_per_example_metadata"}
    per_example = metadata.get("_per_example_metadata")
    if isinstance(per_example, list) and row_idx < min(batch_size, len(per_example)):
        row_meta = per_example[row_idx]
        if isinstance(row_meta, dict):
            out.update(row_meta)
    return out


def _trim_metadata(metadata: dict[str, Any], batch_size: int) -> dict[str, Any]:
    out = dict(metadata)
    per_example = out.get("_per_example_metadata")
    if isinstance(per_example, list) and len(per_example) > batch_size:
        out["_per_example_metadata"] = per_example[:batch_size]
    return out


def slice_layer_capture(capture: LayerCapture, row_idx: int, batch_size: int) -> LayerCapture:
    return LayerCapture(
        layer_idx=capture.layer_idx,
        condition=capture.condition,
        resid_pre_attn=_slice_batch_value(capture.resid_pre_attn, row_idx, batch_size),
        q_pre_rope=_slice_batch_value(capture.q_pre_rope, row_idx, batch_size),
        v_raw=_slice_batch_value(capture.v_raw, row_idx, batch_size),
        attention_weights=_slice_batch_value(capture.attention_weights, row_idx, batch_size),
        beta=_slice_batch_value(capture.beta, row_idx, batch_size),
        z_attn=_slice_batch_value(capture.z_attn, row_idx, batch_size),
        h_pre_moe=_slice_batch_value(capture.h_pre_moe, row_idx, batch_size),
        router_logits_pre_softmax=_slice_batch_value(capture.router_logits_pre_softmax, row_idx, batch_size),
        top_k_indices=_slice_batch_value(capture.top_k_indices, row_idx, batch_size),
        final_token_k_rot=_slice_batch_value(capture.final_token_k_rot, row_idx, batch_size),
        final_token_v=_slice_batch_value(capture.final_token_v, row_idx, batch_size),
        final_token_k_raw=_slice_batch_value(capture.final_token_k_raw, row_idx, batch_size),
        position_ids=_slice_batch_value(capture.position_ids, row_idx, batch_size),
        metadata=_slice_metadata(capture.metadata, row_idx, batch_size),
    )


def trim_layer_capture(capture: LayerCapture, batch_size: int) -> LayerCapture:
    return LayerCapture(
        layer_idx=capture.layer_idx,
        condition=capture.condition,
        resid_pre_attn=_trim_batch_value(capture.resid_pre_attn, batch_size),
        q_pre_rope=_trim_batch_value(capture.q_pre_rope, batch_size),
        v_raw=_trim_batch_value(capture.v_raw, batch_size),
        attention_weights=_trim_batch_value(capture.attention_weights, batch_size),
        beta=_trim_batch_value(capture.beta, batch_size),
        z_attn=_trim_batch_value(capture.z_attn, batch_size),
        h_pre_moe=_trim_batch_value(capture.h_pre_moe, batch_size),
        router_logits_pre_softmax=_trim_batch_value(capture.router_logits_pre_softmax, batch_size),
        top_k_indices=_trim_batch_value(capture.top_k_indices, batch_size),
        final_token_k_rot=_trim_batch_value(capture.final_token_k_rot, batch_size),
        final_token_v=_trim_batch_value(capture.final_token_v, batch_size),
        final_token_k_raw=_trim_batch_value(capture.final_token_k_raw, batch_size),
        position_ids=_trim_batch_value(capture.position_ids, batch_size),
        metadata=_trim_metadata(capture.metadata, batch_size),
    )


def split_pass_capture(pass_capture: PassCapture, batch_size: int) -> list[PassCapture]:
    outputs: list[PassCapture] = []
    for row_idx in range(batch_size):
        captures_by_condition: dict[str, list[LayerCapture]] = {}
        for condition, captures in pass_capture.captures_by_condition.items():
            captures_by_condition[condition] = [slice_layer_capture(capture, row_idx, batch_size) for capture in captures]
        outputs.append(
            PassCapture(
                pass_name=pass_capture.pass_name,
                rope_mode=pass_capture.rope_mode,
                captures_by_condition=captures_by_condition,
            )
        )
    return outputs


def trim_pass_capture(pass_capture: PassCapture, batch_size: int) -> PassCapture:
    captures_by_condition: dict[str, list[LayerCapture]] = {}
    for condition, captures in pass_capture.captures_by_condition.items():
        captures_by_condition[condition] = [trim_layer_capture(capture, batch_size) for capture in captures]
    return PassCapture(
        pass_name=pass_capture.pass_name,
        rope_mode=pass_capture.rope_mode,
        captures_by_condition=captures_by_condition,
    )


def serialize_layer_capture(capture: LayerCapture) -> dict[str, Any]:
    return {
        "layer_idx": capture.layer_idx,
        "condition": capture.condition,
        "resid_pre_attn": capture.resid_pre_attn,
        "q_pre_rope": capture.q_pre_rope,
        "v_raw": capture.v_raw,
        "attention_weights": capture.attention_weights,
        "beta": capture.beta,
        "z_attn": capture.z_attn,
        "h_pre_moe": capture.h_pre_moe,
        "router_logits_pre_softmax": capture.router_logits_pre_softmax,
        "top_k_indices": capture.top_k_indices,
        "final_token_k_rot": capture.final_token_k_rot,
        "final_token_v": capture.final_token_v,
        "final_token_k_raw": capture.final_token_k_raw,
        "position_ids": capture.position_ids,
        "metadata": dict(capture.metadata),
    }


def deserialize_layer_capture(payload: dict[str, Any]) -> LayerCapture:
    return LayerCapture(
        layer_idx=int(payload["layer_idx"]),
        condition=str(payload["condition"]),
        resid_pre_attn=payload.get("resid_pre_attn"),
        q_pre_rope=payload.get("q_pre_rope"),
        v_raw=payload.get("v_raw"),
        attention_weights=payload.get("attention_weights"),
        beta=payload.get("beta"),
        z_attn=payload.get("z_attn"),
        h_pre_moe=payload.get("h_pre_moe"),
        router_logits_pre_softmax=payload.get("router_logits_pre_softmax"),
        top_k_indices=payload.get("top_k_indices"),
        final_token_k_rot=payload.get("final_token_k_rot"),
        final_token_v=payload.get("final_token_v"),
        final_token_k_raw=payload.get("final_token_k_raw"),
        position_ids=payload.get("position_ids"),
        metadata=dict(payload.get("metadata", {})),
    )


def serialize_pass_capture(pass_capture: PassCapture) -> dict[str, Any]:
    return {
        "pass_name": pass_capture.pass_name,
        "rope_mode": pass_capture.rope_mode,
        "captures_by_condition": {
            condition: [serialize_layer_capture(capture) for capture in captures]
            for condition, captures in pass_capture.captures_by_condition.items()
        },
    }


def deserialize_pass_capture(payload: dict[str, Any]) -> PassCapture:
    return PassCapture(
        pass_name=str(payload["pass_name"]),
        rope_mode=str(payload["rope_mode"]),
        captures_by_condition={
            str(condition): [deserialize_layer_capture(capture) for capture in captures]
            for condition, captures in payload.get("captures_by_condition", {}).items()
        },
    )


def build_batched_capture_payload(
    *,
    batch_id: str,
    dataset_name: str,
    batch_examples: list[PromptExample],
    passes: list[PassCapture],
    extra_metadata: dict[str, Any] | None = None,
    multi_slot_by_text_id: dict[str, Any] | None = None,
) -> dict[str, Any]:
    examples_payload = []
    multi_slot_by_text_id = multi_slot_by_text_id or {}
    for example in batch_examples:
        metadata = {"calibration": example.calibration, "tags": list(example.tags)}
        if example.calibration and example.text_id in multi_slot_by_text_id:
            metadata["multi_slot_summaries"] = multi_slot_by_text_id[example.text_id]
        examples_payload.append(
            {
                "text_id": example.text_id,
                "dataset_name": dataset_name,
                "kind": example.kind,
                "prompt": example.prompt,
                "token_ids": list(example.prompt_token_ids or []),
                "content_token_mask": list(example.content_token_mask or []),
                "metadata": metadata,
            }
        )
    return {
        "schema_version": 3,
        "batch_id": batch_id,
        "metadata": extra_metadata or {},
        "examples": examples_payload,
        "passes": [serialize_pass_capture(pass_capture) for pass_capture in passes],
    }


def iter_payload_bundles(payload: dict[str, Any]):
    schema_version = int(payload.get("schema_version", 2))
    if schema_version <= 2:
        for bundle in payload.get("bundles", []):
            yield bundle
        return
    if schema_version != 3:
        raise ValueError(f"Unsupported capture schema_version={schema_version}")
    examples = payload.get("examples", [])
    pass_captures = [deserialize_pass_capture(pass_payload) for pass_payload in payload.get("passes", [])]
    batch_size = len(examples)
    for row_idx, example in enumerate(examples):
        passes = []
        for pass_capture in pass_captures:
            captures_by_condition: dict[str, list[LayerCapture]] = {}
            for condition, captures in pass_capture.captures_by_condition.items():
                captures_by_condition[condition] = [slice_layer_capture(capture, row_idx, batch_size) for capture in captures]
            passes.append(
                PassCapture(
                    pass_name=pass_capture.pass_name,
                    rope_mode=pass_capture.rope_mode,
                    captures_by_condition=captures_by_condition,
                )
            )
        yield ExampleCaptureBundle(
            text_id=str(example["text_id"]),
            dataset_name=str(example["dataset_name"]),
            kind=str(example["kind"]),
            prompt=str(example["prompt"]),
            token_ids=list(example.get("token_ids", [])),
            content_token_mask=list(example.get("content_token_mask", [])),
            passes=passes,
            metadata=dict(example.get("metadata", {})),
        )
