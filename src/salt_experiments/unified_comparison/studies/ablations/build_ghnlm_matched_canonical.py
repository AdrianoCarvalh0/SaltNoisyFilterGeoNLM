#!/usr/bin/env python3
"""Build a complete derived archive with only GHNLM replaced by its matched rerun.

The historical canonical archive remains untouched.  Files for the six
unchanged methods and the archived inputs are hard-linked into a new archive;
the 610 corrected GHNLM arrays and records replace their historical
counterparts.  Aggregate CSVs are rebuilt from the resulting per-case records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[5]
PACKAGE_ROOT = ROOT / "src/salt_experiments/unified_comparison"
sys.path.insert(0, str(PACKAGE_ROOT))

from run import METHODS


BASE_ARCHIVE = ROOT / "data/output/unifiedComparisonCanonical"
CORRECTED_GHNLM = ROOT / "data/output/unifiedComparisonGHNLMMatchedSwitchingV1"
OUTPUT = ROOT / "data/output/unifiedComparisonCanonicalV5_GHNLMMatched"
METHOD = "ghnlm"
CORRECTED_METHOD = "ghnlm_matched_switching"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_records(output: Path) -> pd.DataFrame:
    records = []
    for case in sorted(output.glob("set*/tolerance_*/*/*/case.json")):
        for method in METHODS:
            record_path = case.parent / f"{method}.json"
            if not record_path.is_file():
                raise FileNotFoundError(f"Missing method record: {record_path}")
            records.append(json.loads(record_path.read_text()))
    frame = pd.DataFrame(records)
    expected = 610 * len(METHODS)
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} records, found {len(frame)}")
    if frame.duplicated(["dataset", "level", "file_name", "tolerance", "method"]).any():
        raise ValueError("Duplicate method record in derived archive")
    return frame.sort_values(["dataset", "tolerance", "level", "file_name", "method"], kind="stable")


def rebuild_aggregates(output: Path) -> None:
    frame = read_records(output)
    atomic_csv(output / "results_partial.csv", frame)
    summary = frame.groupby(["dataset", "tolerance", "level", "method"], sort=True).agg(
        n=("score", "size"),
        psnr=("psnr", "mean"),
        ssim=("ssim", "mean"),
        score=("score", "mean"),
        time_filter_call_s=("time_filter_call_s", "mean"),
    ).reset_index()
    atomic_csv(output / "summary_partial.csv", summary)


def replace_ghnlm(base: Path, corrected: Path, output: Path) -> int:
    corrected_records = sorted(corrected.glob(f"set*/tolerance_*/*/*/{CORRECTED_METHOD}.json"))
    if len(corrected_records) != 610:
        raise ValueError(f"Expected 610 corrected GHNLM records, found {len(corrected_records)}")
    replaced = 0
    for corrected_record_path in corrected_records:
        relative_case = corrected_record_path.relative_to(corrected).parent
        destination = output / relative_case
        historical_record_path = destination / f"{METHOD}.json"
        historical_image_path = destination / f"{METHOD}.npy"
        corrected_image_path = corrected_record_path.with_name(f"{CORRECTED_METHOD}.npy")
        if not historical_record_path.is_file() or not historical_image_path.is_file():
            raise FileNotFoundError(f"Historical GHNLM case missing: {destination}")
        if not corrected_image_path.is_file():
            raise FileNotFoundError(f"Corrected GHNLM image missing: {corrected_image_path}")

        historical = json.loads(historical_record_path.read_text())
        corrected_record = json.loads(corrected_record_path.read_text())
        case = json.loads((destination / "case.json").read_text())
        if corrected_record["source_noisy_sha256"] != case["noisy_sha256"]:
            raise ValueError(f"Corrected noisy identity differs: {relative_case}")
        if corrected_record["source_reference_sha256"] != case["reference_sha256"]:
            raise ValueError(f"Corrected reference identity differs: {relative_case}")

        # Preserve the canonical record schema and source identities while
        # replacing the measured GHNLM outcomes and documenting the correction.
        historical.update({
            "psnr": corrected_record["psnr"],
            "ssim": corrected_record["ssim"],
            "score": corrected_record["score"],
            "time_filter_call_s": corrected_record["time_filter_call_s"],
            "target_switching": corrected_record["target_switching"],
            "graph_candidate_policy": corrected_record["graph_candidate_policy"],
            "corrected_from_method": CORRECTED_METHOD,
            "corrected_source_case": corrected_record["source_case"],
            "corrected_source_archive": str(corrected.relative_to(ROOT)),
        })
        # Unlinking affects the derived hard link only, never the historical
        # archive.  The corrected array itself is hard-linked, not copied.
        historical_image_path.unlink()
        os.link(corrected_image_path, historical_image_path)
        atomic_json(historical_record_path, historical)
        replaced += 1
    return replaced


def build_protocol(base: Path, corrected: Path) -> dict:
    protocol = json.loads((base / "protocol.json").read_text())
    protocol["version"] = 5
    protocol["git_describe"] = "canonical-v5-ghnlm-matched-switching"
    protocol["derived_from"] = {
        "archive": str(base.relative_to(ROOT)),
        "protocol_sha256": sha256_file(base / "protocol.json"),
        "preserved_methods": [method for method in METHODS if method != METHOD],
        "replaced_method": METHOD,
        "corrected_archive": str(corrected.relative_to(ROOT)),
        "corrected_protocol_sha256": sha256_file(corrected / "protocol.json"),
    }
    protocol["ghnlm"] = {
        **protocol["ghnlm"],
        "target_switching": "input mask; reliable pixels copied unchanged",
        "candidate_graph_vertices": "admissible candidate centers plus target source",
        "shared_ianlm_operators": "robust replacement, 1-1-10 spatial weights, and local fallback",
    }
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE_ARCHIVE)
    parser.add_argument("--corrected", type=Path, default=CORRECTED_GHNLM)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    base, corrected, output = args.base.resolve(), args.corrected.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to alter existing derived archive: {output}")
    if not (base / "protocol.json").is_file() or not (corrected / "protocol.json").is_file():
        raise FileNotFoundError("Base or corrected archive lacks protocol.json")

    shutil.copytree(base, output, copy_function=os.link)
    # Historical charts contain GHNLM values from version 4.  Do not retain
    # them under a corrected archive; regenerate reports from its CSVs later.
    reports = output / "reports"
    if reports.exists():
        shutil.rmtree(reports)
    replaced = replace_ghnlm(base, corrected, output)
    rebuild_aggregates(output)
    atomic_json(output / "protocol.json", build_protocol(base, corrected))
    runtime = json.loads((base / "runtime.json").read_text())
    runtime["derived_archive"] = True
    runtime["ghnlm_timing_source"] = str(corrected.relative_to(ROOT))
    atomic_json(output / "runtime.json", runtime)
    atomic_json(output / "run_requests.json", [{
        "operation": "derived_archive_ghnlm_replacement",
        "base_archive": str(base.relative_to(ROOT)),
        "corrected_ghnlm_archive": str(corrected.relative_to(ROOT)),
        "replaced_ghnlm_cases": replaced,
        "preserved_methods": [method for method in METHODS if method != METHOD],
    }])
    atomic_json(output / "derived_archive_manifest.json", {
        "base_archive": str(base.relative_to(ROOT)),
        "corrected_ghnlm_archive": str(corrected.relative_to(ROOT)),
        "output_archive": str(output.relative_to(ROOT)),
        "hard_linked_content": "all unchanged archived files and six non-GHNLM method outputs",
        "replaced_ghnlm_cases": replaced,
        "stale_reports_removed": True,
        "aggregate_csvs_rebuilt": ["results_partial.csv", "summary_partial.csv"],
    })
    print(f"Built {output} with {replaced} corrected GHNLM cases.")


if __name__ == "__main__":
    main()
