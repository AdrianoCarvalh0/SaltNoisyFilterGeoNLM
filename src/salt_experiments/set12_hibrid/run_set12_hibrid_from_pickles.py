import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "set12"))

import numpy as np
import pandas as pd
import skimage.io
from scipy.ndimage import median_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx
from functions.geonlm_medians_functions import run_geonlm_medians_pipeline
from NLMedians import run_nlmedians


SOURCE_BASE = Path("/workspace/data/output/set12")
OUTPUT_BASE = Path("/workspace/data/output/set12Hibrid")
LEVELS = ["low", "medium", "moderate", "high", "extreme"]
RESOLUTION = "full_512"

HYBRID_CONFIG = {
    "f": 2,
    "t": 3,
    "nn": 7,
    "h_multiplier": 0.005,
    "switch_impulse_only": True,
    "reject_impulse_candidates": True,
    "use_aswmf_spatial_weights": True,
    "aswmf_weight_diag_1": 1.0,
    "aswmf_weight_diag_2": 1.0,
    "aswmf_weight_other": 10.0,
}

NLMEDIANS_CONFIG = {
    "f": 2,
    "t": 3,
    "h_multiplier": 0.005,
}

MEDIAN_CONFIG = {
    "size": 3,
    "mode": "reflect",
}


def selected_levels():
    raw = os.environ.get("SET12_HIBRID_LEVELS")
    if not raw:
        return LEVELS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return [level for level in LEVELS if level in wanted]


def max_images():
    raw = os.environ.get("SET12_HIBRID_MAX_IMAGES")
    if not raw:
        return None
    return int(raw)


def force_run():
    return os.environ.get("SET12_HIBRID_FORCE", "0") == "1"


def score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


def metrics(reference, image):
    reference_uint8 = np.clip(reference, 0, 255).astype(np.uint8)
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(reference_uint8, image_uint8, data_range=255)
    ssim = structural_similarity(reference_uint8, image_uint8, data_range=255)
    return psnr, ssim, score(psnr, ssim)


def ensure_dirs(root):
    for subdir in ["NLM", "MEDIAN", "NLMedians", "GEONLMHibrid", "results"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def load_original_results_by_file(level):
    path = SOURCE_BASE / f"salt_pepper_{level}" / RESOLUTION / "results" / (
        f"results_salt_pepper_gnlm_median_aswmf_{level}.pkl"
    )
    if not path.exists():
        return {}
    return {row["file_name"]: row for row in load_pickle(path.parent, path.name)}


def save_uint8(path, image):
    skimage.io.imsave(str(path), np.clip(image, 0, 255).astype(np.uint8))


def read_or_run_nlmedians(item, out_path):
    reference = item["img_reference_np"]
    if out_path.exists() and not force_run():
        image = skimage.io.imread(str(out_path))
        psnr, ssim, method_score = metrics(reference, image)
        return image, psnr, ssim, method_score, np.nan

    start = time.time()
    result = run_nlmedians(
        reference=reference,
        noisy=item["img_noisy_salt_pepper_np"],
        h=float(item["nlm_h"]) * NLMEDIANS_CONFIG["h_multiplier"],
        f=NLMEDIANS_CONFIG["f"],
        t=NLMEDIANS_CONFIG["t"],
    )
    elapsed = time.time() - start
    save_uint8(out_path, result["filtered"])
    return result["filtered"], result["psnr"], result["ssim"], result["score"], elapsed


def read_or_run_median(item, out_path):
    reference = item["img_reference_np"]
    if out_path.exists() and not force_run():
        image = skimage.io.imread(str(out_path))
        psnr, ssim, method_score = metrics(reference, image)
        return image, psnr, ssim, method_score, np.nan

    start = time.time()
    filtered = median_filter(
        item["img_noisy_salt_pepper_np"],
        size=MEDIAN_CONFIG["size"],
        mode=MEDIAN_CONFIG["mode"],
    )
    elapsed = time.time() - start
    filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
    save_uint8(out_path, filtered_uint8)
    psnr, ssim, method_score = metrics(reference, filtered_uint8)
    return filtered_uint8, psnr, ssim, method_score, elapsed


def read_or_run_hybrid(item, out_path):
    reference = item["img_reference_np"]
    if out_path.exists() and not force_run():
        image = skimage.io.imread(str(out_path))
        psnr, ssim, method_score = metrics(reference, image)
        return image, psnr, ssim, method_score, np.nan, (
            float(item["nlm_h"]) * HYBRID_CONFIG["h_multiplier"]
        )

    start = time.time()
    filtered, h_used, psnr, ssim, method_score = run_geonlm_medians_pipeline(
        img_original=reference,
        h_base=float(item["nlm_h"]),
        img_noisy=item["img_noisy_salt_pepper_np"],
        f=HYBRID_CONFIG["f"],
        t=HYBRID_CONFIG["t"],
        mult=HYBRID_CONFIG["h_multiplier"],
        nn=HYBRID_CONFIG["nn"],
        switch_impulse_only=HYBRID_CONFIG["switch_impulse_only"],
        reject_impulse_candidates=HYBRID_CONFIG["reject_impulse_candidates"],
        use_aswmf_spatial_weights=HYBRID_CONFIG["use_aswmf_spatial_weights"],
        aswmf_weight_diag_1=HYBRID_CONFIG["aswmf_weight_diag_1"],
        aswmf_weight_diag_2=HYBRID_CONFIG["aswmf_weight_diag_2"],
        aswmf_weight_other=HYBRID_CONFIG["aswmf_weight_other"],
    )
    elapsed = time.time() - start
    save_uint8(out_path, filtered)
    return filtered, psnr, ssim, method_score, elapsed, h_used


def run_level(level):
    source_results = SOURCE_BASE / f"salt_pepper_{level}" / RESOLUTION / "results"
    output_root = OUTPUT_BASE / f"salt_pepper_{level}" / RESOLUTION
    output_results = output_root / "results"
    ensure_dirs(output_root)

    nlm_pickle = f"array_nlm_salt_pepper_{level}_filtereds.pkl"
    vector = load_pickle(source_results, nlm_pickle)
    limit = max_images()
    if limit is not None:
        vector = vector[:limit]

    save_pickle(vector, output_results, nlm_pickle)
    original_by_file = load_original_results_by_file(level)
    records = []

    for item in vector:
        file_name = item["file_name"]
        reference = item["img_reference_np"]
        noisy = item["img_noisy_salt_pepper_np"]
        nlm = item["img_filtered_nlm"]

        save_uint8(output_root / "NLM" / file_name, nlm)

        psnr_noisy, ssim_noisy, score_noisy = metrics(reference, noisy)
        psnr_nlm, ssim_nlm, score_nlm = metrics(reference, nlm)

        _, psnr_nlmed, ssim_nlmed, score_nlmed, time_nlmed = read_or_run_nlmedians(
            item,
            output_root / "NLMedians" / file_name,
        )
        _, psnr_median, ssim_median, score_median, time_median = read_or_run_median(
            item,
            output_root / "MEDIAN" / file_name,
        )
        _, psnr_hybrid, ssim_hybrid, score_hybrid, time_hybrid, h_hybrid = read_or_run_hybrid(
            item,
            output_root / "GEONLMHibrid" / file_name,
        )

        original = original_by_file.get(file_name, {})
        record = {
            "level": level,
            "resolution": RESOLUTION,
            "file_name": file_name,
            "source_nlm_pickle": str(source_results / nlm_pickle),
            "estimated_sigma_salt_pepper": float(item["estimated_sigma_salt_pepper"]),
            "salt_pepper_density": float(item["salt_pepper_density"]),
            "salt_prob": float(item["salt_prob"]),
            "pepper_prob": float(item["pepper_prob"]),
            "psnr_noisy": psnr_noisy,
            "ssim_noisy": ssim_noisy,
            "score_noisy": score_noisy,
            "nlm_h": float(item["nlm_h"]),
            "psnr_nlm": psnr_nlm,
            "ssim_nlm": ssim_nlm,
            "score_nlm": score_nlm,
            "nlmedians_f": NLMEDIANS_CONFIG["f"],
            "nlmedians_t": NLMEDIANS_CONFIG["t"],
            "nlmedians_h_multiplier": NLMEDIANS_CONFIG["h_multiplier"],
            "nlmedians_h": float(item["nlm_h"]) * NLMEDIANS_CONFIG["h_multiplier"],
            "psnr_nlmedians": psnr_nlmed,
            "ssim_nlmedians": ssim_nlmed,
            "score_nlmedians": score_nlmed,
            "time_nlmedians": time_nlmed,
            "median_size": MEDIAN_CONFIG["size"],
            "median_mode": MEDIAN_CONFIG["mode"],
            "psnr_median": psnr_median,
            "ssim_median": ssim_median,
            "score_median": score_median,
            "time_median": time_median,
            "hybrid_f": HYBRID_CONFIG["f"],
            "hybrid_t": HYBRID_CONFIG["t"],
            "hybrid_nn": HYBRID_CONFIG["nn"],
            "hybrid_h_multiplier": HYBRID_CONFIG["h_multiplier"],
            "hybrid_h": h_hybrid,
            "hybrid_switch_impulse_only": HYBRID_CONFIG["switch_impulse_only"],
            "hybrid_reject_impulse_candidates": HYBRID_CONFIG["reject_impulse_candidates"],
            "hybrid_use_aswmf_spatial_weights": HYBRID_CONFIG["use_aswmf_spatial_weights"],
            "hybrid_aswmf_weight_diag_1": HYBRID_CONFIG["aswmf_weight_diag_1"],
            "hybrid_aswmf_weight_diag_2": HYBRID_CONFIG["aswmf_weight_diag_2"],
            "hybrid_aswmf_weight_other": HYBRID_CONFIG["aswmf_weight_other"],
            "psnr_geonlm_hibrid": psnr_hybrid,
            "ssim_geonlm_hibrid": ssim_hybrid,
            "score_geonlm_hibrid": score_hybrid,
            "time_geonlm_hibrid": time_hybrid,
            "delta_score_hibrid_vs_nlm": score_hybrid - score_nlm,
            "delta_score_hibrid_vs_nlmedians": score_hybrid - score_nlmed,
            "psnr_gnlm_original": np.nan if "psnr_gnlm" not in original else float(original["psnr_gnlm"]),
            "ssim_gnlm_original": np.nan if "ssim_gnlm" not in original else float(original["ssim_gnlm"]),
            "score_gnlm_original": np.nan if "score_gnlm" not in original else float(original["score_gnlm"]),
            "psnr_median_original": np.nan if "psnr_median" not in original else float(original["psnr_median"]),
            "ssim_median_original": np.nan if "ssim_median" not in original else float(original["ssim_median"]),
            "score_median_original": np.nan if "score_median" not in original else float(original["score_median"]),
            "psnr_aswmf_original": np.nan if "psnr_aswmf" not in original else float(original["psnr_aswmf"]),
            "ssim_aswmf_original": np.nan if "ssim_aswmf" not in original else float(original["ssim_aswmf"]),
            "score_aswmf_original": np.nan if "score_aswmf" not in original else float(original["score_aswmf"]),
        }
        records.append(record)
        print(
            f"{level} {file_name}: "
            f"Hibrid={score_hybrid:.4f} "
            f"NLMedians={score_nlmed:.4f} "
            f"dScore={record['delta_score_hibrid_vs_nlmedians']:+.4f} "
            f"time_hibrid={time_hybrid if not np.isnan(time_hybrid) else 0:.2f}s",
            flush=True,
        )

    save_pickle(records, output_results, f"results_set12_hibrid_{level}.pkl")
    save_results_to_xlsx(records, output_results, f"results_set12_hibrid_{level}.xlsx")
    return records


if __name__ == "__main__":
    all_records = []
    for level in selected_levels():
        all_records.extend(run_level(level))

    summary_dir = OUTPUT_BASE / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(all_records, summary_dir, "results_set12_hibrid_all.pkl")
    save_results_to_xlsx(all_records, summary_dir, "results_set12_hibrid_all.xlsx")

    df = pd.DataFrame(all_records)
    summary = (
        df.groupby("level")
        .agg(
            mean_score_nlm=("score_nlm", "mean"),
            mean_score_median=("score_median", "mean"),
            mean_score_nlmedians=("score_nlmedians", "mean"),
            mean_score_geonlm_hibrid=("score_geonlm_hibrid", "mean"),
            mean_score_gnlm_original=("score_gnlm_original", "mean"),
            mean_score_median_original=("score_median_original", "mean"),
            mean_score_aswmf_original=("score_aswmf_original", "mean"),
            mean_delta_hibrid_vs_nlm=("delta_score_hibrid_vs_nlm", "mean"),
            mean_delta_hibrid_vs_nlmedians=("delta_score_hibrid_vs_nlmedians", "mean"),
            wins_hibrid_vs_nlmedians=(
                "delta_score_hibrid_vs_nlmedians",
                lambda s: int((s > 0).sum()),
            ),
            mean_time_median=("time_median", "mean"),
            mean_time_nlmedians=("time_nlmedians", "mean"),
            mean_time_geonlm_hibrid=("time_geonlm_hibrid", "mean"),
        )
        .reset_index()
    )
    summary.to_excel(summary_dir / "results_set12_hibrid_summary.xlsx", index=False)
    print(summary.to_string(index=False))
