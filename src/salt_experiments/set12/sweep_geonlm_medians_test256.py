import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import skimage.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.Utils import get_multiplier, load_pickle, save_pickle, save_results_to_xlsx
from functions.geonlm_medians_functions import run_geonlm_medians_pipeline


BASE = Path("/workspace/data/output/set12")
LEVELS = ["low", "medium", "moderate"]

CONFIGS = [
    {"name": "geo_med_f2_t3_nn5_h0005", "f": 2, "t": 3, "nn": 5, "h_multiplier": 0.005},
    {"name": "geo_med_f2_t3_nn5_h0010", "f": 2, "t": 3, "nn": 5, "h_multiplier": 0.010},
    {"name": "geo_med_f2_t3_nn7_h0005", "f": 2, "t": 3, "nn": 7, "h_multiplier": 0.005},
    {"name": "geo_med_f2_t3_nn7_h0010", "f": 2, "t": 3, "nn": 7, "h_multiplier": 0.010},
    {
        "name": "geo_med_f2_t3_nn7_h0005_alpha025",
        "f": 2,
        "t": 3,
        "nn": 7,
        "h_multiplier": 0.005,
        "outlier_pixel_alpha": 0.25,
    },
    {
        "name": "geo_med_f2_t3_nn7_h0005_alpha050",
        "f": 2,
        "t": 3,
        "nn": 7,
        "h_multiplier": 0.005,
        "outlier_pixel_alpha": 0.50,
    },
    {
        "name": "geo_med_f2_t3_nn7_h0005_alpha075",
        "f": 2,
        "t": 3,
        "nn": 7,
        "h_multiplier": 0.005,
        "outlier_pixel_alpha": 0.75,
    },
    {
        "name": "switch_aswmf_geo_med_f2_t3_nn7_h0005",
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
    },
    {"name": "geo_med_f2_t3_nn10_h0005", "f": 2, "t": 3, "nn": 10, "h_multiplier": 0.005},
    {"name": "geo_med_f2_t3_nn10_h0010", "f": 2, "t": 3, "nn": 10, "h_multiplier": 0.010},
    {"name": "geo_med_f3_t5_nn10_h0005", "f": 3, "t": 5, "nn": 10, "h_multiplier": 0.005},
    {"name": "geo_med_f4_t7_nn10_h0005", "f": 4, "t": 7, "nn": 10, "h_multiplier": 0.005},
    {"name": "geo_med_f4_t7_nn10_h0010", "f": 4, "t": 7, "nn": 10, "h_multiplier": 0.010},
    {"name": "geo_med_f4_t7_nn10_adaptive", "f": 4, "t": 7, "nn": 10, "h_multiplier": "adaptive"},
]


def selected_levels():
    raw = os.environ.get("GEONLM_MEDIANS_LEVELS")
    if not raw:
        return LEVELS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return [level for level in LEVELS if level in wanted]


def selected_configs():
    raw = os.environ.get("GEONLM_MEDIANS_CONFIGS")
    if not raw:
        return CONFIGS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return [cfg for cfg in CONFIGS if cfg["name"] in wanted]


def max_images():
    raw = os.environ.get("GEONLM_MEDIANS_MAX_IMAGES")
    if not raw:
        return None
    return int(raw)


def n_jobs():
    raw = os.environ.get("GEONLM_MEDIANS_N_JOBS")
    if not raw:
        return -1
    return int(raw)


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


def load_nlmedians_article_by_file(level):
    path = BASE / "nlmedians_article_sweep_summary" / "nlmedians_article_sweep_all.pkl"
    if not path.exists():
        return {}
    rows = load_pickle(path.parent, path.name)
    best_config = "article_f2_t3_h0005"
    return {
        row["file_name"]: row
        for row in rows
        if row["level"] == level and row["config"] == best_config
    }


def run_level(level):
    base_output = BASE / f"salt_pepper_{level}" / "test_256"
    results_dir = base_output / "results"
    geomed_base = base_output / "GEONLMedians_sweep"
    geomed_base.mkdir(parents=True, exist_ok=True)

    nlm_pickle = f"array_nlm_salt_pepper_{level}_filtereds.pkl"
    vector = load_pickle(results_dir, nlm_pickle)
    limit = max_images()
    if limit is not None:
        vector = vector[:limit]

    geonlm_by_file = load_geonlm_by_file(level)
    nlmedians_by_file = load_nlmedians_article_by_file(level)
    records = []

    for cfg in selected_configs():
        out_dir = geomed_base / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for item in vector:
            file_name = item["file_name"]
            reference = item["img_reference_np"]
            noisy = item["img_noisy_salt_pepper_np"]
            nlm = item["img_filtered_nlm"]
            h_multiplier = cfg["h_multiplier"]
            if h_multiplier == "adaptive":
                h_multiplier = get_multiplier(
                    float(item["nlm_h"]),
                    float(item["estimated_sigma_salt_pepper"]),
                )

            start = time.time()
            filtered, h_used, psnr_geo_med, ssim_geo_med, score_geo_med = (
                run_geonlm_medians_pipeline(
                    img_original=reference,
                    h_base=float(item["nlm_h"]),
                    img_noisy=noisy,
                    f=cfg["f"],
                    t=cfg["t"],
                    mult=h_multiplier,
                    nn=cfg["nn"],
                    outlier_pixel_alpha=cfg.get("outlier_pixel_alpha", 0.0),
                    switch_impulse_only=cfg.get("switch_impulse_only", False),
                    reject_impulse_candidates=cfg.get("reject_impulse_candidates", False),
                    use_aswmf_spatial_weights=cfg.get("use_aswmf_spatial_weights", False),
                    aswmf_weight_diag_1=cfg.get("aswmf_weight_diag_1", 1.0),
                    aswmf_weight_diag_2=cfg.get("aswmf_weight_diag_2", 1.0),
                    aswmf_weight_other=cfg.get("aswmf_weight_other", 10.0),
                    n_jobs=n_jobs(),
                )
            )
            elapsed = time.time() - start

            psnr_noisy, ssim_noisy, score_noisy = metrics(reference, noisy)
            psnr_nlm, ssim_nlm, score_nlm = metrics(reference, nlm)
            geo = geonlm_by_file.get(file_name)
            nlmed = nlmedians_by_file.get(file_name)

            skimage.io.imsave(str(out_dir / file_name), filtered)

            record = {
                "level": level,
                "file_name": file_name,
                "config": cfg["name"],
                "method": "GEONLMedians",
                "source_nlm_pickle": str(results_dir / nlm_pickle),
                "f": cfg["f"],
                "t": cfg["t"],
                "nn": cfg["nn"],
                "nlm_h_original": float(item["nlm_h"]),
                "h_used": h_used,
                "h_multiplier": h_multiplier,
                "outlier_pixel_alpha": cfg.get("outlier_pixel_alpha", 0.0),
                "switch_impulse_only": cfg.get("switch_impulse_only", False),
                "reject_impulse_candidates": cfg.get("reject_impulse_candidates", False),
                "use_aswmf_spatial_weights": cfg.get("use_aswmf_spatial_weights", False),
                "aswmf_weight_diag_1": cfg.get("aswmf_weight_diag_1", 1.0),
                "aswmf_weight_diag_2": cfg.get("aswmf_weight_diag_2", 1.0),
                "aswmf_weight_other": cfg.get("aswmf_weight_other", 10.0),
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
                "psnr_nlmedians_article": (
                    np.nan if nlmed is None else float(nlmed["psnr_nlmedians"])
                ),
                "ssim_nlmedians_article": (
                    np.nan if nlmed is None else float(nlmed["ssim_nlmedians"])
                ),
                "score_nlmedians_article": (
                    np.nan if nlmed is None else float(nlmed["score_nlmedians"])
                ),
                "psnr_geonlm_medians": psnr_geo_med,
                "ssim_geonlm_medians": ssim_geo_med,
                "score_geonlm_medians": score_geo_med,
                "delta_psnr_vs_nlm": psnr_geo_med - psnr_nlm,
                "delta_ssim_vs_nlm": ssim_geo_med - ssim_nlm,
                "delta_score_vs_nlm": score_geo_med - score_nlm,
                "delta_score_vs_gnlm": (
                    np.nan if geo is None else score_geo_med - float(geo["score_gnlm"])
                ),
                "delta_score_vs_nlmedians_article": (
                    np.nan
                    if nlmed is None
                    else score_geo_med - float(nlmed["score_nlmedians"])
                ),
                "time_geonlm_medians": elapsed,
            }
            records.append(record)
            print(
                f"{level} {cfg['name']} {file_name}: "
                f"dPSNR_NLM={record['delta_psnr_vs_nlm']:+.4f} "
                f"dScore_NLM={record['delta_score_vs_nlm']:+.4f} "
                f"dScore_NLMedians={record['delta_score_vs_nlmedians_article']:+.4f} "
                f"time={elapsed:.2f}s",
                flush=True,
            )

    save_pickle(records, results_dir, f"geonlm_medians_sweep_{level}.pkl")
    save_results_to_xlsx(records, results_dir, f"geonlm_medians_sweep_{level}.xlsx")
    return records


if __name__ == "__main__":
    all_records = []
    for level in selected_levels():
        all_records.extend(run_level(level))

    summary_dir = BASE / "geonlm_medians_sweep_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(all_records, summary_dir, "geonlm_medians_sweep_all.pkl")
    save_results_to_xlsx(all_records, summary_dir, "geonlm_medians_sweep_all.xlsx")

    df = pd.DataFrame(all_records)
    summary = (
        df.groupby(["level", "config"])
        .agg(
            mean_delta_psnr_vs_nlm=("delta_psnr_vs_nlm", "mean"),
            mean_delta_score_vs_nlm=("delta_score_vs_nlm", "mean"),
            mean_delta_score_vs_gnlm=("delta_score_vs_gnlm", "mean"),
            mean_delta_score_vs_nlmedians_article=(
                "delta_score_vs_nlmedians_article",
                "mean",
            ),
            wins_score_vs_nlm=("delta_score_vs_nlm", lambda s: int((s > 0).sum())),
            wins_score_vs_nlmedians_article=(
                "delta_score_vs_nlmedians_article",
                lambda s: int((s > 0).sum()),
            ),
            mean_score_geonlm_medians=("score_geonlm_medians", "mean"),
            mean_score_nlm=("score_nlm", "mean"),
            mean_score_gnlm=("score_gnlm", "mean"),
            mean_score_nlmedians_article=("score_nlmedians_article", "mean"),
            mean_time=("time_geonlm_medians", "mean"),
        )
        .reset_index()
        .sort_values(["level", "mean_delta_score_vs_nlmedians_article"], ascending=[True, False])
    )
    summary.to_excel(summary_dir / "geonlm_medians_sweep_summary.xlsx", index=False)
    print(summary.to_string(index=False))
