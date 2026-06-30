"""Serialisation helpers for returnability reports."""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np


def normalise_manifest_value(value: object) -> object:
    """Normalise public report payloads and reject raw arrays."""

    if isinstance(value, np.ndarray):
        raise TypeError("Returnability reports must not contain raw trajectory arrays.")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalise_manifest_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [normalise_manifest_value(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported returnability manifest value type: {type(value).__name__}")


def write_json_manifest(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(normalise_manifest_value(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv_records(path: str | Path, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for record in records for key in record})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            row = {
                key: json.dumps(normalise_manifest_value(value), sort_keys=True)
                if isinstance(value, (dict, list, tuple, set, frozenset))
                else normalise_manifest_value(value)
                for key, value in record.items()
            }
            writer.writerow(row)
