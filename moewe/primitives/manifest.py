"""Serialisable primitive validation manifest helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _normalise(value: object) -> object:
    if isinstance(value, np.ndarray):
        raise TypeError("Validation manifests must not contain raw trajectory arrays.")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _normalise(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported manifest value type: {type(value).__name__}")


def _reason_counts(records: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if bool(record.get("retained")):
            continue
        reason = str(record.get("retention_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _class_coverage(records: list[dict[str, object]], key: str) -> list[str]:
    return sorted({str(record[key]) for record in records if record.get(key) is not None})


def _family_controller_summary(records: list[dict[str, object]]) -> list[dict[str, object]]:
    counts: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        key = (str(record["family"]), str(record["controller_type"]))
        item = counts.setdefault(
            key,
            {
                "family": key[0],
                "controller_type": key[1],
                "rollout_count": 0,
                "retained_count": 0,
            },
        )
        item["rollout_count"] = int(item["rollout_count"]) + 1
        if bool(record.get("retained")):
            item["retained_count"] = int(item["retained_count"]) + 1
    return [counts[key] for key in sorted(counts)]


@dataclass(frozen=True)
class PrimitiveValidationManifest:
    """Tiny public manifest for primitive validation smoke records."""

    records: tuple[dict[str, object], ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, object]],
        metadata: dict[str, object] | None = None,
    ) -> "PrimitiveValidationManifest":
        normalised = tuple(_normalise(record) for record in records)
        return cls(records=normalised, metadata={} if metadata is None else dict(metadata))

    def to_records(self) -> list[dict[str, object]]:
        return [dict(record) for record in self.records]

    def to_summary(self) -> dict[str, object]:
        records = self.to_records()
        return {
            "candidate_count": int(self.metadata.get("candidate_count", 0)),
            "scenario_count": int(self.metadata.get("scenario_count", 0)),
            "rollout_count": len(records),
            "retained_count": sum(1 for record in records if bool(record.get("retained"))),
            "rejection_reason_counts": _reason_counts(records),
            "entry_class_coverage": _class_coverage(records, "entry_class"),
            "exit_class_coverage": _class_coverage(records, "exit_class"),
            "by_family_controller": _family_controller_summary(records),
        }

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": _normalise(self.metadata),
            "summary": self.to_summary(),
            "records": self.to_records(),
        }
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def write_csv(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = self.to_records()
        keys = sorted({key for record in records for key in record})
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for record in records:
                row = {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in record.items()
                }
                writer.writerow(row)
