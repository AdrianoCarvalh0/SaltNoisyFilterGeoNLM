"""Build the definitive Set12-NoLena tolerance experiment with fixed IANLM h=1."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import skimage.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx
from functions.anlm_functions import Parallel_Switch_ANLM
from functions.nlm_functions import mirror_cpu


SOURCE_BASE = Path("/workspace/data/output/set12ImpulseTolerance")
OUTPUT_BASE = Path("/workspace/data/output/set12ImpulseToleranceH1")
TOLERANCES = (0, 1, 2, 3, 4)
LEVELS = ("low", "moderate", "medium", "high", "extreme")
H_IANLM = 1.0
CONFIG = {
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


def metrics(reference, filtered):
    reference = np.clip(reference, 0, 255).astype(np.uint8)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(reference, filtered, data_range=255)
    ssim = structural_similarity(reference, filtered, data_range=255)
    return float(psnr), float(ssim), float(0.5 * psnr + 50.0 * ssim)


def source_root(tolerance, level):
    return (
        SOURCE_BASE / f"tolerance_{tolerance}" / f"salt_pepper_{level}"
        / "full_512"
    )


def load_sources(tolerance, level):
    root = source_root(tolerance, level)
    items = load_pickle(root / "results", f"array_nlm_impulse_tolerance_{level}.pkl")
    records = load_pickle(root / "results", f"results_set12_impulse_tolerance_{level}.pkl")
    return items, {record["file_name"]: record for record in records}


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(str(path), np.clip(image, 0, 255).astype(np.uint8))


def run_level(tolerance, level):
    items, baseline_by_file = load_sources(tolerance, level)
    destination = (
        OUTPUT_BASE / f"tolerance_{tolerance}" / f"salt_pepper_{level}"
        / "full_512"
    )
    records = []
    for item in items:
        reference = np.asarray(item["img_reference_np"], dtype=np.float32)
        noisy = np.asarray(item["img_noisy_salt_pepper_np"], dtype=np.float32)
        start = time.time()
        filtered, stats = Parallel_Switch_ANLM(
            mirror_cpu(noisy, CONFIG["f"]),
            h=H_IANLM,
            impulse_tolerance=tolerance,
            return_stats=True,
            **CONFIG,
        )
        elapsed = time.time() - start
        filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
        psnr, ssim, method_score = metrics(reference, filtered_uint8)
        save_image(destination / "IANLM" / item["file_name"], filtered_uint8)

        baseline = baseline_by_file[item["file_name"]]
        excluded = {
            "h_ianlm", "psnr_ianlm", "ssim_ianlm", "score_ianlm", "time_ianlm",
            "h_ghnlm", "psnr_ghnlm", "ssim_ghnlm", "score_ghnlm", "time_ghnlm",
        }
        record = {key: value for key, value in baseline.items() if key not in excluded}
        record.update({
            "h_ianlm": H_IANLM,
            "h_ianlm_source": "fixed_independent",
            "psnr_ianlm": psnr,
            "ssim_ianlm": ssim,
            "score_ianlm": method_score,
            "time_ianlm": elapsed,
            **stats,
        })
        records.append(record)
        print(
            f"tolerance={tolerance} level={level} {item['file_name']}: "
            f"IANLM(h=1)={method_score:.4f} fallback={stats['fallback_fraction']:.2%}",
            flush=True,
        )

    results_dir = destination / "results"
    save_pickle(records, results_dir, f"results_set12_ianlm_h1_{level}.pkl")
    save_results_to_xlsx(records, results_dir, f"results_set12_ianlm_h1_{level}.xlsx")
    return records


def save_summaries(records):
    summary_dir = OUTPUT_BASE / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    methods = ("noisy", "nlm", "median", "nlmedians", "aswmf", "ianlm")
    aggregations = {}
    for method in methods:
        for metric in ("psnr", "ssim", "score"):
            column = f"{metric}_{method}"
            aggregations[f"mean_{column}"] = (column, "mean")
    aggregations.update({
        "mean_time_ianlm": ("time_ianlm", "mean"),
        "mean_fallback_fraction": ("fallback_fraction", "mean"),
        "mean_nonlocal_pixels": ("nonlocal_pixels", "mean"),
        "mean_detector_precision": ("detector_precision", "mean"),
        "mean_detector_recall": ("detector_recall", "mean"),
    })
    summary = (
        df.groupby(["impulse_tolerance", "level"], sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    overall = (
        df.groupby("impulse_tolerance", sort=False)
        .agg(**aggregations)
        .reset_index()
    )
    save_pickle(records, summary_dir, "results_set12_ianlm_h1_all.pkl")
    save_results_to_xlsx(records, summary_dir, "results_set12_ianlm_h1_all.xlsx")
    summary.to_excel(summary_dir / "results_set12_ianlm_h1_summary.xlsx", index=False)
    overall.to_excel(summary_dir / "results_set12_ianlm_h1_overall.xlsx", index=False)
    return overall


if __name__ == "__main__":
    # Warm up Numba outside runtime measurements.
    dummy = np.full((16, 16), 128, dtype=np.float32)
    dummy[8, 8] = 0
    Parallel_Switch_ANLM(
        mirror_cpu(dummy, CONFIG["f"]),
        h=H_IANLM,
        impulse_tolerance=0,
        return_stats=True,
        **CONFIG,
    )

    all_records = []
    for selected_tolerance in TOLERANCES:
        for selected_level in LEVELS:
            all_records.extend(run_level(selected_tolerance, selected_level))
    print(save_summaries(all_records).to_string(index=False))
