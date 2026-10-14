#!/usr/bin/env python3
"""Evaluate the impulse detector with a fixed noise-generation width.

This is a detector-only sensitivity study.  It reuses the archived noisy
arrays and synthetic assignment masks generated with ``b=4`` and applies the
detector at each ``tau`` in ``{0, 1, 2, 3, 4}``.  No restoration method is
called.

The archive contains the arrays needed for this experiment in the legacy
tolerance sweeps.  They are used rather than rendered PNG files so threshold
decisions are made on the original stored observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATION_WIDTH = 4
TAUS = (0, 1, 2, 3, 4)
LEVELS = ("low", "moderate", "medium", "high", "extreme")
SOURCES = {
    "set12": PROJECT_ROOT / "legacy/data/output/set12ImpulseTolerance",
    "set50": PROJECT_ROOT / "legacy/data/output/set50ImpulseToleranceH1",
}


def array_digest(array: np.ndarray) -> str:
    """Return an unambiguous digest for an archived array."""
    contiguous = np.ascontiguousarray(array)
    payload = f"{contiguous.dtype}|{contiguous.shape}|".encode() + contiguous.tobytes()
    return hashlib.sha256(payload).hexdigest()


def ratio(numerator: int, denominator: int) -> float | None:
    """Return None for an undefined rate; do not silently replace it by zero."""
    return None if denominator == 0 else numerator / denominator


def csv_value(value: Any) -> Any:
    return "" if value is None else value


def detector_record(
    *,
    dataset: str,
    level: str,
    file_name: str,
    noisy: np.ndarray,
    assignment_mask: np.ndarray,
    tau: int,
    source_path: Path,
) -> dict[str, Any]:
    """Measure one detector decision against the stored synthetic mask."""
    detected = (noisy <= tau) | (noisy >= 255 - tau)
    assigned = np.asarray(assignment_mask, dtype=bool)
    if detected.shape != assigned.shape:
        raise ValueError(f"Mask/image shape mismatch in {source_path}: {detected.shape} vs {assigned.shape}")

    true_positive = int(np.count_nonzero(detected & assigned))
    false_positive = int(np.count_nonzero(detected & ~assigned))
    false_negative = int(np.count_nonzero(~detected & assigned))
    true_negative = int(np.count_nonzero(~detected & ~assigned))
    detected_pixels = true_positive + false_positive
    total_pixels = detected_pixels + false_negative + true_negative

    return {
        "dataset": dataset,
        "level": level,
        "file_name": file_name,
        "generation_width_b": GENERATION_WIDTH,
        "detector_tolerance_tau": tau,
        "source_pickle": str(source_path.relative_to(PROJECT_ROOT)),
        "noisy_sha256": array_digest(noisy),
        "assignment_mask_sha256": array_digest(assigned),
        "total_pixels": total_pixels,
        "assigned_impulses": int(assigned.sum()),
        "detected_pixels": detected_pixels,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "precision": ratio(true_positive, detected_pixels),
        "recall": ratio(true_positive, true_positive + false_negative),
        "false_positive_rate": ratio(false_positive, false_positive + true_negative),
        "false_discovery_rate": ratio(false_positive, detected_pixels),
    }


def clean_control_record(
    *, dataset: str, file_name: str, reference: np.ndarray, tau: int
) -> dict[str, Any]:
    """Measure detector selections on a clean reference, whose true mask is empty."""
    detected = (reference <= tau) | (reference >= 255 - tau)
    false_positive = int(np.count_nonzero(detected))
    total_pixels = int(reference.size)
    return {
        "dataset": dataset,
        "level": "clean_control",
        "file_name": file_name,
        "generation_width_b": "none",
        "detector_tolerance_tau": tau,
        "source_pickle": "clean reference recovered from b=4 archive",
        "reference_sha256": array_digest(reference),
        "total_pixels": total_pixels,
        "assigned_impulses": 0,
        "detected_pixels": false_positive,
        "true_positives": 0,
        "false_positives": false_positive,
        "false_negatives": 0,
        "true_negatives": total_pixels - false_positive,
        # Precision and recall have a zero denominator for a truly clean image.
        "precision": None,
        "recall": None,
        "false_positive_rate": ratio(false_positive, total_pixels),
        "false_discovery_rate": 1.0 if false_positive else None,
    }


def load_archive_items(dataset: str) -> Iterable[tuple[str, Path, dict[str, Any]]]:
    """Yield each archived b=4 noisy case and validate the archive label."""
    root = SOURCES[dataset] / f"tolerance_{GENERATION_WIDTH}"
    for level in LEVELS:
        source = root / f"salt_pepper_{level}" / "full_512" / "results" / (
            f"array_nlm_impulse_tolerance_{level}.pkl"
        )
        if not source.is_file():
            raise FileNotFoundError(f"Missing archived b={GENERATION_WIDTH} source: {source}")
        with source.open("rb") as handle:
            items = pickle.load(handle)
        for item in items:
            if int(item["impulse_tolerance"]) != GENERATION_WIDTH:
                raise ValueError(f"Unexpected generation width in {source}: {item['impulse_tolerance']}")
            yield level, source, item


def mean_defined(records: list[dict[str, Any]], field: str) -> float | None:
    values = [record[field] for record in records if record[field] is not None]
    return None if not values else float(np.mean(values))


def summarize(records: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    """Report both unweighted image means and rates from pooled counts."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in group_fields)].append(record)

    rows = []
    for group, items in sorted(groups.items()):
        sums = {field: sum(int(item[field]) for item in items) for field in (
            "total_pixels", "assigned_impulses", "detected_pixels", "true_positives",
            "false_positives", "false_negatives", "true_negatives",
        )}
        row = dict(zip(group_fields, group))
        row.update({"n_images": len(items), **sums})
        for field in ("precision", "recall", "false_positive_rate", "false_discovery_rate"):
            row[f"mean_{field}"] = mean_defined(items, field)
        row.update({
            "pooled_precision": ratio(sums["true_positives"], sums["detected_pixels"]),
            "pooled_recall": ratio(sums["true_positives"], sums["true_positives"] + sums["false_negatives"]),
            "pooled_false_positive_rate": ratio(sums["false_positives"], sums["false_positives"] + sums["true_negatives"]),
            "pooled_false_discovery_rate": ratio(sums["false_positives"], sums["detected_pixels"]),
        })
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Inconsistent fields while writing {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: csv_value(row[field]) for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "data/output/detectorToleranceFixedB4",
        help="directory for tables and metadata",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    detector_records: list[dict[str, Any]] = []
    clean_references: dict[tuple[str, str], np.ndarray] = {}
    source_cases = 0
    for dataset in SOURCES:
        for level, source, item in load_archive_items(dataset):
            noisy = np.asarray(item["img_noisy_salt_pepper_np"])
            mask = np.asarray(item["corruption_mask"], dtype=bool)
            reference = np.asarray(item["img_reference_np"])
            source_cases += 1
            key = (dataset, str(item["file_name"]))
            if key in clean_references and not np.array_equal(clean_references[key], reference):
                raise ValueError(f"Reference changed between archived levels for {key}")
            clean_references[key] = reference
            for tau in TAUS:
                detector_records.append(detector_record(
                    dataset=dataset, level=level, file_name=key[1], noisy=noisy,
                    assignment_mask=mask, tau=tau, source_path=source,
                ))

    clean_records = [
        clean_control_record(dataset=dataset, file_name=file_name, reference=reference, tau=tau)
        for (dataset, file_name), reference in sorted(clean_references.items())
        for tau in TAUS
    ]

    by_dataset_level = summarize(
        detector_records, ("dataset", "level", "generation_width_b", "detector_tolerance_tau")
    )
    by_dataset = summarize(
        detector_records, ("dataset", "generation_width_b", "detector_tolerance_tau")
    )
    clean_by_dataset = summarize(clean_records, ("dataset", "detector_tolerance_tau"))

    write_csv(output / "detector_records.csv", detector_records)
    write_csv(output / "detector_summary_by_dataset_level.csv", by_dataset_level)
    write_csv(output / "detector_summary_by_dataset.csv", by_dataset)
    write_csv(output / "clean_control_records.csv", clean_records)
    write_csv(output / "clean_control_summary_by_dataset.csv", clean_by_dataset)
    metadata = {
        "study": "fixed_b_detector_tolerance_sensitivity",
        "generation_width_b": GENERATION_WIDTH,
        "detector_tolerances": list(TAUS),
        "noise_cases": source_cases,
        "detector_evaluations": len(detector_records),
        "clean_control_images": len(clean_references),
        "clean_control_evaluations": len(clean_records),
        "zero_denominator_policy": "undefined rates are encoded as null in JSON and blank in CSV",
        "rates": {
            "precision": "TP / (TP + FP)",
            "recall": "TP / (TP + FN)",
            "false_positive_rate": "FP / (FP + TN)",
            "false_discovery_rate": "FP / (TP + FP)",
        },
        "source_archives": {dataset: str(path.relative_to(PROJECT_ROOT)) for dataset, path in SOURCES.items()},
        "notes": [
            "No restoration algorithm is invoked.",
            "Detector summaries provide unweighted means of image-level rates and separately pooled-count rates.",
            "Clean controls use an empty ground-truth impulse mask; precision and recall are therefore undefined.",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
