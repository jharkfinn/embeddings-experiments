from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass
class PromptExample:
    text_id: str
    kind: str
    text: str
    prompt: str
    dataset_name: str | None = None
    prompt_token_ids: list[int] | None = None
    token_count: int | None = None
    content_token_mask: list[int] | None = None
    calibration: bool = False
    tags: tuple[str, ...] = ()


def format_compression_prompt(text: str, kind: str, prompt_spec) -> str:
    if kind == "query":
        return prompt_spec.query_template.format(text=text)
    return prompt_spec.context_template.format(text=text)


def build_prompt_examples(records, prompt_spec, calibration_ids=None):
    calibration_ids = set(calibration_ids or [])
    examples = []
    for record in records:
        examples.append(
            PromptExample(
                text_id=str(record["text_id"]),
                kind=str(record["kind"]),
                text=str(record["text"]),
                prompt=format_compression_prompt(str(record["text"]), str(record["kind"]), prompt_spec),
                dataset_name=None if record.get("dataset_name") is None else str(record.get("dataset_name")),
                calibration=str(record["text_id"]) in calibration_ids,
                tags=tuple(record.get("tags", ())),
            )
        )
    return examples


_MINIMAL_PAIR_TEMPLATES = [
    ("The committee approved the plan.", "The committee did not approve the plan.", ("negation",)),
    ("The doctor examined the patient with a lamp.", "The doctor examined the patient with a rash.", ("attachment",)),
    ("Jordan visited Paris in spring.", "Jordan visited Paris in autumn.", ("temporal",)),
    ("Alex told Jordan that he won.", "Alex told Jordan that she won.", ("coreference",)),
    ("The bank raised rates.", "The river bank rose overnight.", ("entity_disambiguation",)),
]

_ABSTRACTION_TRIPLETS = [
    (
        "The startup secured funding after strong quarterly growth.",
        "The company raised capital following solid growth.",
        "The startup lost funding after weak quarterly growth.",
        ("paraphrase", "adversarial"),
    ),
    (
        "A vaccine trial reduced severe infections in older adults.",
        "The study showed the vaccine cut serious cases among seniors.",
        "A vaccine trial increased severe infections in older adults.",
        ("paraphrase", "adversarial"),
    ),
]


def generate_semantic_minimal_pairs(prefix: str = "pair"):
    pairs = []
    for index, (left, right, tags) in enumerate(_MINIMAL_PAIR_TEMPLATES):
        pairs.append(
            {
                "text_id": f"{prefix}_{index}_a",
                "kind": "doc",
                "text": left,
                "tags": tags,
                "pair_group": f"{prefix}_{index}",
            }
        )
        pairs.append(
            {
                "text_id": f"{prefix}_{index}_b",
                "kind": "doc",
                "text": right,
                "tags": tags,
                "pair_group": f"{prefix}_{index}",
            }
        )
    return pairs


def generate_semantic_abstraction_triplets(prefix: str = "triplet"):
    triplets = []
    for index, (original, paraphrase, adversarial, tags) in enumerate(_ABSTRACTION_TRIPLETS):
        triplets.extend(
            [
                {
                    "text_id": f"{prefix}_{index}_orig",
                    "kind": "doc",
                    "text": original,
                    "tags": tags + ("original",),
                    "triplet_group": f"{prefix}_{index}",
                },
                {
                    "text_id": f"{prefix}_{index}_para",
                    "kind": "doc",
                    "text": paraphrase,
                    "tags": tags + ("paraphrase",),
                    "triplet_group": f"{prefix}_{index}",
                },
                {
                    "text_id": f"{prefix}_{index}_adv",
                    "kind": "doc",
                    "text": adversarial,
                    "tags": tags + ("adversarial",),
                    "triplet_group": f"{prefix}_{index}",
                },
            ]
        )
    return triplets


def _stable_example_key(example: PromptExample, seed: int) -> tuple[str, str]:
    dataset_name = example.dataset_name or ""
    payload = "\n".join(
        [
            str(seed),
            dataset_name,
            example.kind,
            example.text_id,
            example.text,
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    tiebreak = f"{dataset_name}\t{example.kind}\t{example.text_id}"
    return digest, tiebreak


def _allocate_stratified_quotas(groups: dict[tuple[str, str], list[PromptExample]], total_target: int):
    nonempty = [(key, items) for key, items in groups.items() if items]
    if total_target <= 0 or not nonempty:
        return {key: 0 for key in groups}
    if total_target >= sum(len(items) for _, items in nonempty):
        return {key: len(items) for key, items in groups.items()}

    quotas = {key: 0 for key in groups}
    if total_target >= len(nonempty):
        for key, _items in nonempty:
            quotas[key] = 1
        remaining = total_target - len(nonempty)
    else:
        remaining = total_target

    if remaining <= 0:
        return quotas

    capacities = {key: max(0, len(items) - quotas[key]) for key, items in nonempty}
    total_capacity = sum(capacities.values())
    if total_capacity <= 0:
        return quotas

    fractional = []
    assigned = 0
    for key, _items in nonempty:
        raw = remaining * capacities[key] / total_capacity
        add = min(capacities[key], int(raw))
        quotas[key] += add
        assigned += add
        fractional.append((raw - int(raw), key))

    leftover = remaining - assigned
    for _frac, key in sorted(fractional, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            leftover -= 1
    return quotas


def sample_calibration_ids(examples: list[PromptExample], calibration_subset_size: int, seed: int):
    total_target = min(calibration_subset_size, len(examples))
    groups: dict[tuple[str, str], list[PromptExample]] = {}
    for example in examples:
        key = (example.dataset_name or "", example.kind)
        groups.setdefault(key, []).append(example)
    quotas = _allocate_stratified_quotas(groups, total_target)
    selected: set[str] = set()
    for key, items in groups.items():
        ranked = sorted(items, key=lambda example: _stable_example_key(example, seed))
        for example in ranked[: quotas.get(key, 0)]:
            selected.add(example.text_id)
    return selected


def calibration_manifest(examples: list[PromptExample], calibration_ids: set[str], seed: int):
    selected_examples = [example for example in examples if example.text_id in calibration_ids]
    strata = {}
    for example in selected_examples:
        strata_key = f"{example.dataset_name or 'unknown'}::{example.kind}"
        strata[strata_key] = strata.get(strata_key, 0) + 1
    return {
        "selection_method": "stable_hash_stratified_by_dataset_and_kind",
        "seed": int(seed),
        "num_candidates": len(examples),
        "num_selected": len(selected_examples),
        "selected_ids": sorted(example.text_id for example in selected_examples),
        "strata_counts": dict(sorted(strata.items())),
    }
