from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RopeMode(str, Enum):
    REUSE_ROTATED = "reuse_rotated"
    REROTATE_ZERO = "rerotate_zero"
    DEROTATE = "derotate"


class CaptureCondition(str, Enum):
    CAUSAL = "causal"
    LOCAL_PREPEND_CAUSAL_BASE = "local_prepend_causal_base"
    PROPAGATED = "propagated"
    LOCAL_NOPREPEND_PROPAGATED_BASE = "local_noprepend_propagated_base"


@dataclass
class LayerCapture:
    layer_idx: int
    condition: str
    resid_pre_attn: Any | None = None
    q_pre_rope: Any | None = None
    v_raw: Any | None = None
    attention_weights: Any | None = None
    beta: Any | None = None
    z_attn: Any | None = None
    h_pre_moe: Any | None = None
    router_logits_pre_softmax: Any | None = None
    top_k_indices: Any | None = None
    final_token_k_rot: Any | None = None
    final_token_v: Any | None = None
    final_token_k_raw: Any | None = None
    position_ids: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PassCapture:
    pass_name: str
    rope_mode: str
    captures_by_condition: dict[str, list[LayerCapture]] = field(default_factory=dict)


@dataclass
class ExampleCaptureBundle:
    text_id: str
    dataset_name: str
    kind: str
    prompt: str
    token_ids: list[int]
    content_token_mask: list[int]
    passes: list[PassCapture]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTask:
    dataset_name: str
    corpus: dict[str, dict[str, Any]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
