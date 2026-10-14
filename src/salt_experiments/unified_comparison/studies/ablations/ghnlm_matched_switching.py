#!/usr/bin/env python3
"""Re-evaluate only GHNLM under matched IANLM switching and graph candidates.

The canonical archive is read-only input.  This study reuses each archived
noisy array and assignment mask, reruns no other method, and writes a separate
versioned archive.  GHNLM differs from IANLM only in the graph/geodesic
similarity operation: both restore detected targets only, reject impulse
candidate centers, use the same robust replacement, spatial weights, fallback,
and ``f=1, t=3, h=1`` settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ROOT = PACKAGE_ROOT.parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

from lib.impulse_tolerance_filters import run_ghnlm_matched_switching_pipeline


DEFAULT_SOURCE = ROOT / "data/output/unifiedComparisonCanonical"
DEFAULT_OUTPUT = ROOT / "data/output/unifiedComparisonGHNLMMatchedSwitchingV1"
METHOD = "ghnlm_matched_switching"


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_reference(dataset: str, file_name: str) -> np.ndarray:
    matches = sorted((ROOT / "data/input" / dataset).glob(f"{file_name}.*"))
    if not matches:
        raise FileNotFoundError(f"Missing reference for {dataset}/{file_name}")
    with Image.open(matches[0]) as handle:
        return np.asarray(handle.convert("L"), dtype=np.float32)


def metric_record(reference: np.ndarray, restored: np.ndarray) -> dict[str, float]:
    reference_u8 = np.clip(reference, 0, 255).astype(np.uint8)
    return {
        "psnr": float(peak_signal_noise_ratio(reference_u8, restored, data_range=255)),
        "ssim": float(structural_similarity(reference_u8, restored, data_range=255)),
    }


def study_protocol(source: Path) -> dict:
    base_protocol = source / "protocol.json"
    return {
        "study": "GHNLM matched switching and graph-candidate re-evaluation",
        "study_version": 1,
        "source_archive": str(source.relative_to(ROOT)),
        "source_protocol_sha256": sha256_file(base_protocol),
        "source_cases": "archived noisy.npy and case.json; no noise is regenerated",
        "rerun_method": METHOD,
        "unchanged_methods": ["nlm", "ianlm", "median", "aswmf", "nlmedians", "gnlm"],
        "target_switching": "detected targets only; reliable input pixels are copied unchanged",
        "candidate_policy": "detected impulse centers are excluded from aggregation and graph vertices; the detected target remains the graph source",
        "shared_ianlm_operators": {
            "f": 1,
            "t": 3,
            "h": 1.0,
            "k": 7,
            "robust_replacement": "strict m +/- 1.96 population-standard-deviations; outlier alpha 0",
            "spatial_weights": [1.0, 1.0, 10.0],
            "fallback": "same ASWMF-style local fallback",
        },
        "difference_from_ianlm": "geodesic shortest-path similarity on the matched candidate graph",
    }


def selected(values: list[str], requested: list[str] | None) -> list[str]:
    if requested is None:
        return values
    unknown = set(requested).difference(values)
    if unknown:
        raise ValueError(f"Unknown selection: {sorted(unknown)}")
    return [value for value in values if value in requested]


def rebuild_tables(output: Path) -> None:
    rows = []
    for path in sorted(output.glob("set*/tolerance_*/*/*/ghnlm_matched_switching.json")):
        rows.append(json.loads(path.read_text()))
    if not rows:
        return
    fields = list(rows[0])
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["tolerance"], row["level"]), []).append(row)
    summary = []
    for (dataset, tolerance, level), records in sorted(groups.items()):
        summary.append({
            "dataset": dataset,
            "tolerance": tolerance,
            "level": level,
            "method": METHOD,
            "n_images": len(records),
            "mean_psnr": float(np.mean([record["psnr"] for record in records])),
            "mean_ssim": float(np.mean([record["ssim"] for record in records])),
            "mean_score": float(np.mean([record["score"] for record in records])),
            "mean_time_filter_call_s": float(np.mean([record["time_filter_call_s"] for record in records])),
        })
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", choices=("set12", "set50"))
    parser.add_argument("--tolerances", nargs="+", type=int, choices=(0, 4))
    parser.add_argument("--levels", nargs="+", choices=("low", "moderate", "medium", "high", "extreme"))
    parser.add_argument("--max-images", type=int)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    base = json.loads((source / "protocol.json").read_text())
    protocol = study_protocol(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        raise ValueError("Different matched-switching protocol in output directory")
    atomic_json(manifest, protocol)

    datasets = selected(list(base["dataset_images"]), args.datasets)
    tolerances = selected([int(value) for value in base["tolerances"]], args.tolerances)
    levels = selected(list(base["densities"]), args.levels)
    completed = 0
    for dataset in datasets:
        image_names = base["dataset_images"][dataset]
        if args.max_images is not None:
            image_names = image_names[:args.max_images]
        for tolerance in tolerances:
            for level in levels:
                for file_name in image_names:
                    source_case = source / dataset / f"tolerance_{tolerance}" / level / str(file_name)
                    case = json.loads((source_case / "case.json").read_text())
                    noisy = np.load(source_case / "noisy.npy")
                    reference = load_reference(dataset, str(file_name))
                    if digest(reference) != case["reference_sha256"] or digest(noisy) != case["noisy_sha256"]:
                        raise ValueError(f"Source identity mismatch: {source_case}")
                    destination = output / dataset / f"tolerance_{tolerance}" / level / str(file_name)
                    record_path = destination / f"{METHOD}.json"
                    image_path = destination / f"{METHOD}.npy"
                    if record_path.exists() and image_path.exists():
                        prior = json.loads(record_path.read_text())
                        if prior["source_noisy_sha256"] == case["noisy_sha256"]:
                            completed += 1
                            continue
                        raise ValueError(f"Incompatible checkpoint: {record_path}")
                    start = time.perf_counter()
                    restored, h, _, _, _ = run_ghnlm_matched_switching_pipeline(
                        img_original=reference, img_noisy=noisy, h_base=1.0, mult=1.0,
                        f=1, t=3, nn=7, impulse_tolerance=tolerance, n_jobs=8,
                        reject_impulse_candidates=True, use_aswmf_spatial_weights=True,
                    )
                    elapsed = time.perf_counter() - start
                    if restored.dtype != np.uint8 or restored.shape != reference.shape or not np.isfinite(restored).all():
                        raise ValueError(f"Invalid matched GHNLM output: {source_case}")
                    measures = metric_record(reference, restored)
                    score = 0.5 * measures["psnr"] + 50.0 * measures["ssim"]
                    destination.mkdir(parents=True, exist_ok=True)
                    np.save(image_path, restored)
                    record = {
                        "dataset": dataset, "tolerance": tolerance, "level": level,
                        "file_name": str(file_name), "method": METHOD,
                        "source_case": str(source_case.relative_to(ROOT)),
                        "source_reference_sha256": case["reference_sha256"],
                        "source_noisy_sha256": case["noisy_sha256"],
                        "f": 1, "t": 3, "k": 7, "h": h,
                        "target_switching": "detected_only",
                        "graph_candidate_policy": "admissible_centers_plus_target_source",
                        **measures, "score": score, "time_filter_call_s": elapsed,
                    }
                    atomic_json(record_path, record)
                    completed += 1
                    print(f"{completed}: {dataset} tau={tolerance} {level} image={file_name} {elapsed:.2f}s", flush=True)
                    rebuild_tables(output)
    rebuild_tables(output)
    print(f"Completed {completed} GHNLM matched-switching cases in {output}")


if __name__ == "__main__":
    main()
