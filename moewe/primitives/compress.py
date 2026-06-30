"""Medoid-style compression of retained primitive validation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from .features import FeatureScaleSpec, PrimitiveBehaviourFeature, extract_behaviour_feature, feature_vector
from .generate import PrimitiveCandidate, generate_primitives
from .grammar import PrimitiveGrammarSpec
from .sweep import PrimitiveValidationReport


def _stable_metadata_hash(payload: object) -> str:
    import hashlib

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class CompressionTierBudgets:
    """Per-group medoid budgets for executable primitive library tiers."""

    heavy: int = 8
    balanced: int = 4
    light: int = 2
    super_light: int = 1
    smoke: int = 1

    def validate(self) -> None:
        for name, value in self.to_record().items():
            if int(value) <= 0:
                raise ValueError(f"{name} tier budget must be positive.")

    def to_record(self) -> dict[str, int]:
        return {
            "heavy": int(self.heavy),
            "balanced": int(self.balanced),
            "light": int(self.light),
            "super_light": int(self.super_light),
            "smoke": int(self.smoke),
        }


@dataclass(frozen=True)
class CompressionSpec:
    """Compression settings for retained primitive medoids."""

    tier_budgets: CompressionTierBudgets = CompressionTierBudgets()
    feature_scale: FeatureScaleSpec = FeatureScaleSpec()
    group_by_exit_class: bool = False
    method: str = "deterministic_grouped_medoids"

    def validate(self) -> None:
        self.tier_budgets.validate()
        self.feature_scale.validate()
        if self.method != "deterministic_grouped_medoids":
            raise ValueError("Only deterministic_grouped_medoids compression is implemented.")


@dataclass(frozen=True)
class CompressedPrimitiveEntry:
    """One medoid entry in an executable primitive library tier."""

    primitive_id: str
    family: str
    controller_type: str
    entry_class: str
    exit_class: str
    group_key: str
    represented_primitive_ids: tuple[str, ...]
    scenario_coverage: tuple[str, ...]
    feature: PrimitiveBehaviourFeature

    def to_record(self) -> dict[str, object]:
        return {
            "primitive_id": self.primitive_id,
            "family": self.family,
            "controller_type": self.controller_type,
            "entry_class": self.entry_class,
            "exit_class": self.exit_class,
            "group_key": self.group_key,
            "represented_primitive_ids": list(self.represented_primitive_ids),
            "scenario_coverage": list(self.scenario_coverage),
            "feature": self.feature.to_record(),
        }


@dataclass(frozen=True)
class CompressedPrimitiveTier:
    """Executable primitive tier built from retained medoids."""

    tier_name: str
    entries: tuple[CompressedPrimitiveEntry, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return len(self.entries)

    def to_records(self) -> list[dict[str, object]]:
        return [entry.to_record() for entry in self.entries]

    def to_summary(self) -> dict[str, object]:
        return {
            "tier_name": self.tier_name,
            "candidate_count": self.candidate_count,
            "entry_class_coverage": sorted({entry.entry_class for entry in self.entries}),
            "exit_class_coverage": sorted({entry.exit_class for entry in self.entries}),
            "family_controller_coverage": sorted(
                {f"{entry.family}:{entry.controller_type}" for entry in self.entries}
            ),
            "scenario_coverage": sorted({scenario for entry in self.entries for scenario in entry.scenario_coverage}),
            **self.metadata,
        }


@dataclass(frozen=True)
class CompressedPrimitiveLibrary:
    """Compressed primitive tiers plus optional candidate objects for runtime use."""

    tiers: dict[str, CompressedPrimitiveTier]
    candidate_index: dict[str, PrimitiveCandidate] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def tier(self, tier_name: str) -> CompressedPrimitiveTier:
        if tier_name not in self.tiers:
            raise KeyError(f"Unknown primitive library tier: {tier_name}")
        return self.tiers[tier_name]

    def to_records(self) -> dict[str, list[dict[str, object]]]:
        return {name: self.tiers[name].to_records() for name in sorted(self.tiers)}

    def to_summary(self) -> dict[str, object]:
        return {
            "metadata": dict(self.metadata),
            "tiers": {name: self.tiers[name].to_summary() for name in sorted(self.tiers)},
        }

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.to_summary(),
            "tiers": self.to_records(),
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class _CompressionItem:
    feature: PrimitiveBehaviourFeature
    vector: np.ndarray
    scenario_coverage: tuple[str, ...]


def _group_key(feature: PrimitiveBehaviourFeature, include_exit: bool) -> str:
    parts = [feature.family, feature.controller_type, feature.entry_class]
    if include_exit:
        parts.append(feature.exit_class)
    return "::".join(parts)


def _retained_items(report: PrimitiveValidationReport, spec: CompressionSpec) -> list[_CompressionItem]:
    retained = [result for result in report.results if result.retention.retained]
    grouped: dict[tuple[str, str], list[PrimitiveBehaviourFeature]] = {}
    for result in retained:
        feature = extract_behaviour_feature(result)
        grouped.setdefault((feature.primitive_id, _group_key(feature, spec.group_by_exit_class)), []).append(feature)

    items: list[_CompressionItem] = []
    for (_, _), features in grouped.items():
        features_sorted = sorted(
            features,
            key=lambda item: (
                -float(item.min_safety_margin_m),
                float(item.max_angle_of_attack_rad),
                -float(item.terminal_specific_energy_change_j_kg),
                item.scenario_id,
            ),
        )
        feature = features_sorted[0]
        scenarios = tuple(sorted({item.scenario_id for item in features}))
        items.append(_CompressionItem(feature=feature, vector=feature_vector(feature, spec.feature_scale), scenario_coverage=scenarios))
    return sorted(items, key=lambda item: (item.feature.family, item.feature.controller_type, item.feature.entry_class, item.feature.primitive_id))


def _select_medoid_indices(items: list[_CompressionItem], budget: int) -> list[int]:
    if budget >= len(items):
        return list(range(len(items)))
    vectors = np.asarray([item.vector for item in items], dtype=float)
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    centrality = distances.sum(axis=1)
    selected = [min(range(len(items)), key=lambda index: (centrality[index], items[index].feature.primitive_id))]
    while len(selected) < budget:
        candidates = [index for index in range(len(items)) if index not in selected]
        next_index = max(
            candidates,
            key=lambda index: (
                min(float(distances[index, chosen]) for chosen in selected),
                -centrality[index],
                items[index].feature.primitive_id,
            ),
        )
        selected.append(next_index)
    return sorted(selected, key=lambda index: items[index].feature.primitive_id)


def _assign_represented(
    items: list[_CompressionItem],
    selected_indices: list[int],
) -> dict[str, tuple[str, ...]]:
    selected_vectors = {items[index].feature.primitive_id: items[index].vector for index in selected_indices}
    assignments: dict[str, list[str]] = {items[index].feature.primitive_id: [] for index in selected_indices}
    for item in items:
        nearest_id = min(
            selected_vectors,
            key=lambda primitive_id: (
                float(np.linalg.norm(item.vector - selected_vectors[primitive_id])),
                primitive_id,
            ),
        )
        assignments[nearest_id].append(item.feature.primitive_id)
    return {key: tuple(sorted(set(value))) for key, value in assignments.items()}


def _build_tier(
    tier_name: str,
    groups: dict[str, list[_CompressionItem]],
    budget: int | None,
    spec: CompressionSpec,
) -> CompressedPrimitiveTier:
    entries: list[CompressedPrimitiveEntry] = []
    for key in sorted(groups):
        items = groups[key]
        selected = list(range(len(items))) if budget is None else _select_medoid_indices(items, min(int(budget), len(items)))
        represented = _assign_represented(items, selected)
        for index in selected:
            feature = items[index].feature
            entries.append(
                CompressedPrimitiveEntry(
                    primitive_id=feature.primitive_id,
                    family=feature.family,
                    controller_type=feature.controller_type,
                    entry_class=feature.entry_class,
                    exit_class=feature.exit_class,
                    group_key=key,
                    represented_primitive_ids=represented[feature.primitive_id],
                    scenario_coverage=items[index].scenario_coverage,
                    feature=feature,
                )
            )
    entries = sorted(entries, key=lambda entry: (entry.group_key, entry.primitive_id))
    represented_count = len({primitive_id for entry in entries for primitive_id in entry.represented_primitive_ids})
    metadata = {
        "retention_source_count": represented_count,
        "compression_method": spec.method,
        "compression_seed_or_deterministic_hash": _stable_metadata_hash([entry.to_record() for entry in entries]),
    }
    return CompressedPrimitiveTier(tier_name=tier_name, entries=tuple(entries), metadata=metadata)


def compress_retained_primitives(
    validation_report: PrimitiveValidationReport,
    spec: CompressionSpec | None = None,
    grammar_spec: PrimitiveGrammarSpec | None = None,
) -> CompressedPrimitiveLibrary:
    """Compress retained primitive records into deterministic medoid tiers."""

    compression = CompressionSpec() if spec is None else spec
    compression.validate()
    items = _retained_items(validation_report, compression)
    groups: dict[str, list[_CompressionItem]] = {}
    for item in items:
        groups.setdefault(_group_key(item.feature, compression.group_by_exit_class), []).append(item)
    budgets = compression.tier_budgets.to_record()
    tiers = {"raw_retained": _build_tier("raw_retained", groups, None, compression)}
    for tier_name, budget in budgets.items():
        tiers[tier_name] = _build_tier(tier_name, groups, budget, compression)

    candidate_index: dict[str, PrimitiveCandidate] = {}
    if grammar_spec is not None:
        ids = {item.feature.primitive_id for item in items}
        candidate_index = {
            candidate.primitive_id: candidate
            for candidate in generate_primitives(grammar_spec)
            if candidate.primitive_id in ids
        }
    metadata = {
        "retained_candidate_count": len({item.feature.primitive_id for item in items}),
        "rollout_count": validation_report.rollout_count,
        "scenario_count": validation_report.scenario_count,
        "compression_method": compression.method,
    }
    return CompressedPrimitiveLibrary(tiers=tiers, candidate_index=candidate_index, metadata=metadata)
