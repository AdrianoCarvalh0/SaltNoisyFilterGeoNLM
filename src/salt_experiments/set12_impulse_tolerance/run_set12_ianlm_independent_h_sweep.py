"""Independent absolute-h sweep for IANLM on Set12-NoLena tolerance data."""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx
from functions.anlm_functions import Parallel_Switch_ANLM
from functions.nlm_functions import mirror_cpu


SOURCE_BASE = Path("/workspace/data/output/set12ImpulseTolerance")
OUTPUT_BASE = SOURCE_BASE / "ianlm_independent_h"
LEVELS = ("low", "moderate", "medium", "high", "extreme")
DEFAULT_TOLERANCES = (0, 1, 2, 3, 4)
DEFAULT_H_VALUES = (0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
FIXED_CONFIG = {
    "f": 1,
    "t": 3,
    "z_alpha": 1.96,
    "outlier_pixel_alpha": 0.0,
    "reject_impulse_candidates": True,
    "use_aswmf_spatial_weights": True,
    "aswmf_weight_diag_1": 1.0,
    "aswmf_weight_diag_2": 1.0,
    "aswmf_weight_other": 10.0,
}


def _csv_numbers(env_name, defaults, cast):
    raw = os.environ.get(env_name)
    return defaults if not raw else tuple(
        cast(value.strip()) for value in raw.split(",") if value.strip()
    )


def selected_tolerances():
    values = _csv_numbers("SET12_IANLM_H_TOLERANCES", DEFAULT_TOLERANCES, int)
    if any(value < 0 or value > 127 for value in values):
        raise ValueError("Tolerances must be between 0 and 127.")
    return tuple(dict.fromkeys(values))


def selected_h_values():
    values = _csv_numbers("SET12_IANLM_H_VALUES", DEFAULT_H_VALUES, float)
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("All h values must be positive and finite.")
    return tuple(dict.fromkeys(values))


def selected_levels():
    raw = os.environ.get("SET12_IANLM_H_LEVELS")
    if not raw:
        return LEVELS
    requested = {value.strip() for value in raw.split(",") if value.strip()}
    unknown = requested.difference(LEVELS)
    if unknown:
        raise ValueError(f"Unknown levels: {sorted(unknown)}")
    return tuple(level for level in LEVELS if level in requested)


def max_images():
    raw = os.environ.get("SET12_IANLM_H_MAX_IMAGES")
    return None if not raw else int(raw)


def output_dir():
    tag = os.environ.get("SET12_IANLM_H_RUN_TAG", "").strip()
    return OUTPUT_BASE if not tag else OUTPUT_BASE / tag


def source_items(tolerance, level):
    root = (
        SOURCE_BASE / f"tolerance_{tolerance}" / f"salt_pepper_{level}"
        / "full_512" / "results"
    )
    name = f"array_nlm_impulse_tolerance_{level}.pkl"
    items = load_pickle(root, name)
    limit = max_images()
    return items if limit is None else items[:limit]


def metrics(reference, filtered):
    reference = np.clip(reference, 0, 255).astype(np.uint8)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(reference, filtered, data_range=255)
    ssim = structural_similarity(reference, filtered, data_range=255)
    return float(psnr), float(ssim), float(0.5 * psnr + 50.0 * ssim)


def run_one(item, tolerance, level, h):
    reference = np.asarray(item["img_reference_np"], dtype=np.float32)
    noisy = np.asarray(item["img_noisy_salt_pepper_np"], dtype=np.float32)
    padded = mirror_cpu(noisy, FIXED_CONFIG["f"])
    start = time.time()
    filtered, stats = Parallel_Switch_ANLM(
        padded,
        h=float(h),
        impulse_tolerance=int(tolerance),
        return_stats=True,
        **FIXED_CONFIG,
    )
    elapsed = time.time() - start
    filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
    psnr, ssim, method_score = metrics(reference, filtered_uint8)
    return {
        "impulse_tolerance": int(tolerance),
        "level": level,
        "file_name": item["file_name"],
        "h_source": "independent_absolute",
        "h_ianlm": float(h),
        "nlm_h_reference": float(item["nlm_h"]),
        "previous_nlm_scaled_h": float(item["nlm_h"]) * 0.001,
        "psnr_ianlm": psnr,
        "ssim_ianlm": ssim,
        "score_ianlm": method_score,
        "time_ianlm": elapsed,
        **stats,
    }


def create_summaries(records):
    df = pd.DataFrame(records)
    summary = (
        df.groupby(["impulse_tolerance", "level", "h_ianlm"], sort=False)
        .agg(
            mean_psnr_ianlm=("psnr_ianlm", "mean"),
            mean_ssim_ianlm=("ssim_ianlm", "mean"),
            mean_score_ianlm=("score_ianlm", "mean"),
            std_score_ianlm=("score_ianlm", "std"),
            mean_time_ianlm=("time_ianlm", "mean"),
            mean_fallback_fraction=("fallback_fraction", "mean"),
            mean_fallback_pixels=("fallback_pixels", "mean"),
            mean_nonlocal_pixels=("nonlocal_pixels", "mean"),
        )
        .reset_index()
    )
    overall = (
        df.groupby(["impulse_tolerance", "h_ianlm"], sort=False)
        .agg(
            mean_score_ianlm=("score_ianlm", "mean"),
            mean_psnr_ianlm=("psnr_ianlm", "mean"),
            mean_ssim_ianlm=("ssim_ianlm", "mean"),
            mean_fallback_fraction=("fallback_fraction", "mean"),
            mean_nonlocal_pixels=("nonlocal_pixels", "mean"),
        )
        .reset_index()
    )
    best_score = summary.loc[
        summary.groupby(["impulse_tolerance", "level"])["mean_score_ianlm"].idxmax()
    ].sort_values(["impulse_tolerance", "level"])
    active = summary[summary["mean_fallback_fraction"] < 0.95]
    best_active = (
        active.loc[
            active.groupby(["impulse_tolerance", "level"])["mean_score_ianlm"].idxmax()
        ].sort_values(["impulse_tolerance", "level"])
        if not active.empty else active
    )
    return summary, overall, best_score, best_active


def save_outputs(records):
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    summary, overall, best_score, best_active = create_summaries(records)
    save_pickle(records, destination, "results_set12_ianlm_independent_h_all.pkl")
    save_results_to_xlsx(records, destination, "results_set12_ianlm_independent_h_all.xlsx")
    summary.to_excel(destination / "results_set12_ianlm_independent_h_summary.xlsx", index=False)
    overall.to_excel(destination / "results_set12_ianlm_independent_h_overall.xlsx", index=False)
    best_score.to_excel(destination / "results_set12_ianlm_independent_h_best_score.xlsx", index=False)
    best_active.to_excel(destination / "results_set12_ianlm_independent_h_best_active.xlsx", index=False)
    return overall, best_score, best_active


if __name__ == "__main__":
    # Compile the instrumented Numba path before measuring runtimes.
    dummy = np.full((16, 16), 128, dtype=np.float32)
    dummy[8, 8] = 0
    Parallel_Switch_ANLM(
        mirror_cpu(dummy, FIXED_CONFIG["f"]),
        h=1.0,
        impulse_tolerance=0,
        return_stats=True,
        **FIXED_CONFIG,
    )

    records = []
    h_values = selected_h_values()
    for tolerance in selected_tolerances():
        for level in selected_levels():
            for item in source_items(tolerance, level):
                for h in h_values:
                    record = run_one(item, tolerance, level, h)
                    records.append(record)
                print(
                    f"tolerance={tolerance} level={level} image={item['file_name']} "
                    f"completed {len(h_values)} h values",
                    flush=True,
                )
    overall, best_score, best_active = save_outputs(records)
    print("\nBest score by tolerance and level:")
    print(best_score.to_string(index=False))
    print("\nBest score with fallback fraction below 95%:")
    print(best_active.to_string(index=False))
