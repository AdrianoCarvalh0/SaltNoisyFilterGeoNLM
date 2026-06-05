import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import time
import numpy as np
import pandas as pd
import skimage.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx
from functions.adaptive_region_nlm_functions import run_adaptive_region_nlm


BASE = Path("/workspace/data/output/set12")
LEVELS = ["low", "medium", "moderate"]

CONFIGS = [
    {
        "name": "smooth_t9_no_class",
        "f_small": 4,
        "t_small": 7,
        "f_large": 4,
        "t_large": 9,
        "n_clusters": 3,
        "h_ifd_strength": 0.0,
        "require_same_label": False,
        "h_multiplier": 1.0,
    },
    {
        "name": "edge_small_smooth_t9_no_class",
        "f_small": 2,
        "t_small": 5,
        "f_large": 4,
        "t_large": 9,
        "n_clusters": 3,
        "h_ifd_strength": 0.0,
        "require_same_label": False,
        "h_multiplier": 1.0,
    },
    {
        "name": "edge_small_smooth_t9_class",
        "f_small": 2,
        "t_small": 5,
        "f_large": 4,
        "t_large": 9,
        "n_clusters": 3,
        "h_ifd_strength": 0.0,
        "require_same_label": True,
        "h_multiplier": 1.0,
    },
    {
        "name": "smooth_t11_no_class",
        "f_small": 4,
        "t_small": 7,
        "f_large": 4,
        "t_large": 11,
        "n_clusters": 3,
        "h_ifd_strength": 0.0,
        "require_same_label": False,
        "h_multiplier": 1.0,
    },
    {
        "name": "smooth_t9_no_class_h110",
        "f_small": 4,
        "t_small": 7,
        "f_large": 4,
        "t_large": 9,
        "n_clusters": 3,
        "h_ifd_strength": 0.0,
        "require_same_label": False,
        "h_multiplier": 1.1,
    },
    {
        "name": "smooth_t9_no_class_ifd005",
        "f_small": 4,
        "t_small": 7,
        "f_large": 4,
        "t_large": 9,
        "n_clusters": 3,
        "h_ifd_strength": 0.05,
        "require_same_label": False,
        "h_multiplier": 1.0,
    },
]


def score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


def metrics(reference, image):
    reference_uint8 = np.clip(reference, 0, 255).astype(np.uint8)
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(reference_uint8, image_uint8, data_range=255)
    ssim = structural_similarity(reference_uint8, image_uint8, data_range=255)
    return psnr, ssim, score(psnr, ssim)


def load_geonlm_by_file(level):
    path = BASE / f"salt_pepper_{level}" / "test_256" / "results" / (
        f"results_salt_pepper_gnlm_median_aswmf_{level}.pkl"
    )
    if not path.exists():
        return {}
    return {row["file_name"]: row for row in load_pickle(path.parent, path.name)}


def run_level(level):
    base_output = BASE / f"salt_pepper_{level}" / "test_256"
    results_dir = base_output / "results"
    adaptive_base = base_output / "AdaptiveRegionNLM_sweep"
    adaptive_base.mkdir(parents=True, exist_ok=True)

    vector = load_pickle(results_dir, f"array_nlm_salt_pepper_{level}_filtereds.pkl")
    geonlm_by_file = load_geonlm_by_file(level)
    records = []

    for cfg in CONFIGS:
        out_dir = adaptive_base / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for item in vector:
            file_name = item["file_name"]
            reference = item["img_reference_np"]
            noisy = item["img_noisy_salt_pepper_np"]
            nlm = item["img_filtered_nlm"]
            h_base = float(item["nlm_h"]) * cfg["h_multiplier"]

            start = time.time()
            result = run_adaptive_region_nlm(
                reference=reference,
                noisy=noisy,
                predenoised=nlm,
                h_base=h_base,
                f_small=cfg["f_small"],
                t_small=cfg["t_small"],
                f_large=cfg["f_large"],
                t_large=cfg["t_large"],
                n_clusters=cfg["n_clusters"],
                h_ifd_strength=cfg["h_ifd_strength"],
                require_same_label=cfg["require_same_label"],
                random_state=0,
            )
            elapsed = time.time() - start

            psnr_noisy, ssim_noisy, score_noisy = metrics(reference, noisy)
            psnr_nlm, ssim_nlm, score_nlm = metrics(reference, nlm)
            geo = geonlm_by_file.get(file_name)

            skimage.io.imsave(str(out_dir / file_name), result["filtered"])

            record = {
                "level": level,
                "file_name": file_name,
                "config": cfg["name"],
                "nlm_h_original": float(item["nlm_h"]),
                "h_base_used": h_base,
                "estimated_sigma_salt_pepper": float(item["estimated_sigma_salt_pepper"]),
                "salt_pepper_density": float(item["salt_pepper_density"]),
                "psnr_noisy": psnr_noisy,
                "ssim_noisy": ssim_noisy,
                "score_noisy": score_noisy,
                "psnr_nlm": psnr_nlm,
                "ssim_nlm": ssim_nlm,
                "score_nlm": score_nlm,
                "psnr_gnlm": np.nan if geo is None else float(geo["psnr_gnlm"]),
                "ssim_gnlm": np.nan if geo is None else float(geo["ssim_gnlm"]),
                "score_gnlm": np.nan if geo is None else float(geo["score_gnlm"]),
                "psnr_adaptive_region_nlm": result["psnr"],
                "ssim_adaptive_region_nlm": result["ssim"],
                "score_adaptive_region_nlm": result["score"],
                "delta_psnr_vs_nlm": result["psnr"] - psnr_nlm,
                "delta_ssim_vs_nlm": result["ssim"] - ssim_nlm,
                "delta_score_vs_nlm": result["score"] - score_nlm,
                "delta_score_vs_gnlm": (
                    np.nan if geo is None else result["score"] - float(geo["score_gnlm"])
                ),
                "edge_ratio": result["edge_ratio"],
                "h_min": result["h_min"],
                "h_max": result["h_max"],
                "h_mean": result["h_mean"],
                "time_adaptive_region_nlm": elapsed,
            }
            records.append(record)
            print(
                f"{level} {cfg['name']} {file_name}: "
                f"dPSNR={record['delta_psnr_vs_nlm']:+.4f} "
                f"dScore={record['delta_score_vs_nlm']:+.4f} "
                f"time={elapsed:.2f}s"
            )

    save_pickle(records, results_dir, f"adaptive_region_nlm_sweep_{level}.pkl")
    save_results_to_xlsx(records, results_dir, f"adaptive_region_nlm_sweep_{level}.xlsx")
    return records


if __name__ == "__main__":
    all_records = []
    for level in LEVELS:
        all_records.extend(run_level(level))

    summary_dir = BASE / "adaptive_region_nlm_sweep_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(all_records, summary_dir, "adaptive_region_nlm_sweep_all.pkl")
    save_results_to_xlsx(all_records, summary_dir, "adaptive_region_nlm_sweep_all.xlsx")

    df = pd.DataFrame(all_records)
    summary = (
        df.groupby(["level", "config"])
        .agg(
            mean_delta_psnr_vs_nlm=("delta_psnr_vs_nlm", "mean"),
            mean_delta_ssim_vs_nlm=("delta_ssim_vs_nlm", "mean"),
            mean_delta_score_vs_nlm=("delta_score_vs_nlm", "mean"),
            wins_psnr_vs_nlm=("delta_psnr_vs_nlm", lambda s: int((s > 0).sum())),
            wins_score_vs_nlm=("delta_score_vs_nlm", lambda s: int((s > 0).sum())),
            mean_score_adaptive=("score_adaptive_region_nlm", "mean"),
            mean_score_nlm=("score_nlm", "mean"),
            mean_score_gnlm=("score_gnlm", "mean"),
            mean_delta_score_vs_gnlm=("delta_score_vs_gnlm", "mean"),
            mean_time=("time_adaptive_region_nlm", "mean"),
        )
        .reset_index()
        .sort_values(["level", "mean_delta_score_vs_nlm"], ascending=[True, False])
    )
    summary.to_excel(summary_dir / "adaptive_region_nlm_sweep_summary.xlsx", index=False)
    print(summary.to_string(index=False))
