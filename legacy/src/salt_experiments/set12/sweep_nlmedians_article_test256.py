import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import skimage.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx
from NLMedians import run_nlmedians


BASE = Path("/workspace/data/output/set12")
LEVELS = ["low", "medium", "moderate"]

CONFIGS = [
    {"name": "article_f2_t3_h0005", "f": 2, "t": 3, "h_multiplier": 0.005},
    {"name": "article_f2_t3_h0010", "f": 2, "t": 3, "h_multiplier": 0.010},
    {"name": "article_f2_t3_h0020", "f": 2, "t": 3, "h_multiplier": 0.020},
    {"name": "article_f2_t3_h0040", "f": 2, "t": 3, "h_multiplier": 0.040},
    {"name": "article_f3_t5_h0005", "f": 3, "t": 5, "h_multiplier": 0.005},
    {"name": "article_f3_t5_h0010", "f": 3, "t": 5, "h_multiplier": 0.010},
]


def selected_levels():
    raw = os.environ.get("NLMEDIANS_ARTICLE_LEVELS")
    if not raw:
        return LEVELS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return [level for level in LEVELS if level in wanted]


def selected_configs():
    raw = os.environ.get("NLMEDIANS_ARTICLE_CONFIGS")
    if not raw:
        return CONFIGS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return [cfg for cfg in CONFIGS if cfg["name"] in wanted]


def max_images():
    raw = os.environ.get("NLMEDIANS_ARTICLE_MAX_IMAGES")
    if not raw:
        return None
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


def run_level(level):
    base_output = BASE / f"salt_pepper_{level}" / "test_256"
    results_dir = base_output / "results"
    nlmedians_base = base_output / "NLMedians_article_sweep"
    nlmedians_base.mkdir(parents=True, exist_ok=True)

    nlm_pickle = f"array_nlm_salt_pepper_{level}_filtereds.pkl"
    vector = load_pickle(results_dir, nlm_pickle)
    limit = max_images()
    if limit is not None:
        vector = vector[:limit]

    geonlm_by_file = load_geonlm_by_file(level)
    records = []

    for cfg in selected_configs():
        out_dir = nlmedians_base / cfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for item in vector:
            file_name = item["file_name"]
            reference = item["img_reference_np"]
            noisy = item["img_noisy_salt_pepper_np"]
            nlm = item["img_filtered_nlm"]
            h_used = float(item["nlm_h"]) * cfg["h_multiplier"]

            start = time.time()
            result = run_nlmedians(
                reference=reference,
                noisy=noisy,
                h=h_used,
                f=cfg["f"],
                t=cfg["t"],
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
                "method": "NLMedians_Levada2021",
                "source_nlm_pickle": str(results_dir / nlm_pickle),
                "f": cfg["f"],
                "t": cfg["t"],
                "nlm_h_original": float(item["nlm_h"]),
                "h_used": h_used,
                "h_multiplier": cfg["h_multiplier"],
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
                "psnr_nlmedians": result["psnr"],
                "ssim_nlmedians": result["ssim"],
                "score_nlmedians": result["score"],
                "delta_psnr_vs_nlm": result["psnr"] - psnr_nlm,
                "delta_ssim_vs_nlm": result["ssim"] - ssim_nlm,
                "delta_score_vs_nlm": result["score"] - score_nlm,
                "delta_score_vs_gnlm": (
                    np.nan if geo is None else result["score"] - float(geo["score_gnlm"])
                ),
                "time_nlmedians": elapsed,
            }
            records.append(record)
            print(
                f"{level} {cfg['name']} {file_name}: "
                f"dPSNR={record['delta_psnr_vs_nlm']:+.4f} "
                f"dScore={record['delta_score_vs_nlm']:+.4f} "
                f"time={elapsed:.2f}s",
                flush=True,
            )

    save_pickle(records, results_dir, f"nlmedians_article_sweep_{level}.pkl")
    save_results_to_xlsx(records, results_dir, f"nlmedians_article_sweep_{level}.xlsx")
    return records


if __name__ == "__main__":
    all_records = []
    for level in selected_levels():
        all_records.extend(run_level(level))

    summary_dir = BASE / "nlmedians_article_sweep_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(all_records, summary_dir, "nlmedians_article_sweep_all.pkl")
    save_results_to_xlsx(all_records, summary_dir, "nlmedians_article_sweep_all.xlsx")

    df = pd.DataFrame(all_records)
    summary = (
        df.groupby(["level", "config"])
        .agg(
            mean_delta_psnr_vs_nlm=("delta_psnr_vs_nlm", "mean"),
            mean_delta_ssim_vs_nlm=("delta_ssim_vs_nlm", "mean"),
            mean_delta_score_vs_nlm=("delta_score_vs_nlm", "mean"),
            wins_psnr_vs_nlm=("delta_psnr_vs_nlm", lambda s: int((s > 0).sum())),
            wins_score_vs_nlm=("delta_score_vs_nlm", lambda s: int((s > 0).sum())),
            mean_score_nlmedians=("score_nlmedians", "mean"),
            mean_score_nlm=("score_nlm", "mean"),
            mean_score_gnlm=("score_gnlm", "mean"),
            mean_delta_score_vs_gnlm=("delta_score_vs_gnlm", "mean"),
            mean_time=("time_nlmedians", "mean"),
        )
        .reset_index()
        .sort_values(["level", "mean_delta_score_vs_nlm"], ascending=[True, False])
    )
    summary.to_excel(summary_dir / "nlmedians_article_sweep_summary.xlsx", index=False)
    print(summary.to_string(index=False))
