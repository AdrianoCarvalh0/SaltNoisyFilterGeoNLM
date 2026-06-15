import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx


LEVELS = ["low", "medium", "moderate", "high", "extreme"]
RESOLUTION = "full_512"


def score(psnr, ssim):
    if pd.isna(psnr) or pd.isna(ssim):
        return np.nan
    return 0.5 * float(psnr) + 0.5 * (float(ssim) * 100)


def load_original_by_file(source_base, level):
    path = (
        source_base
        / f"salt_pepper_{level}"
        / RESOLUTION
        / "results"
        / f"results_salt_pepper_gnlm_median_aswmf_{level}.pkl"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    rows = load_pickle(path.parent, path.name)
    return {row["file_name"]: row for row in rows}


def merge_level(dataset_name, source_base, hibrid_base, level):
    hibrid_results_dir = hibrid_base / f"salt_pepper_{level}" / RESOLUTION / "results"
    hibrid_path = hibrid_results_dir / f"results_{dataset_name}_hibrid_{level}.pkl"
    if not hibrid_path.exists():
        raise FileNotFoundError(hibrid_path)

    originals = load_original_by_file(source_base, level)
    hibrid_rows = load_pickle(hibrid_path.parent, hibrid_path.name)

    merged = 0
    missing = []
    for row in hibrid_rows:
        original = originals.get(row["file_name"])
        if original is None:
            missing.append(row["file_name"])
            continue

        row["ssim_gnlm"] = float(original["ssim_gnlm"])
        row["psnr_gnlm"] = float(original["psnr_gnlm"])
        row["time_geonlm"] = float(original["time_geonlm"])
        row["score_gnlm"] = score(row["psnr_gnlm"], row["ssim_gnlm"])
        merged += 1

    save_pickle(hibrid_rows, hibrid_results_dir, hibrid_path.name)
    save_results_to_xlsx(
        hibrid_rows,
        hibrid_results_dir,
        f"results_{dataset_name}_hibrid_{level}.xlsx",
    )
    return hibrid_rows, merged, missing


def rebuild_all_and_summary(dataset_name, hibrid_base):
    all_rows = []
    for level in LEVELS:
        path = (
            hibrid_base
            / f"salt_pepper_{level}"
            / RESOLUTION
            / "results"
            / f"results_{dataset_name}_hibrid_{level}.pkl"
        )
        if path.exists():
            all_rows.extend(load_pickle(path.parent, path.name))

    summary_dir = hibrid_base / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(all_rows, summary_dir, f"results_{dataset_name}_hibrid_all.pkl")
    save_results_to_xlsx(all_rows, summary_dir, f"results_{dataset_name}_hibrid_all.xlsx")

    df = pd.DataFrame(all_rows)
    aggregations = {
        "mean_ssim_nlm": ("ssim_nlm", "mean"),
        "mean_ssim_geonlm_hibrid": ("ssim_geonlm_hibrid", "mean"),
        "mean_ssim_gnlm": ("ssim_gnlm", "mean"),
        "mean_psnr_nlm": ("psnr_nlm", "mean"),
        "mean_psnr_geonlm_hibrid": ("psnr_geonlm_hibrid", "mean"),
        "mean_psnr_gnlm": ("psnr_gnlm", "mean"),
        "mean_score_nlm": ("score_nlm", "mean"),
        "mean_score_geonlm_hibrid": ("score_geonlm_hibrid", "mean"),
        "mean_score_gnlm": ("score_gnlm", "mean"),
        "mean_time_geonlm_hibrid": ("time_geonlm_hibrid", "mean"),
        "mean_time_geonlm": ("time_geonlm", "mean"),
    }

    optional_columns = {
        "median": ["ssim_median", "psnr_median", "score_median", "time_median"],
        "aswmf": ["ssim_aswmf_original", "psnr_aswmf_original", "score_aswmf_original"],
        "nlmedians": ["ssim_nlmedians", "psnr_nlmedians", "score_nlmedians", "time_nlmedians"],
        "aswmf_nlm_h001": [
            "ssim_aswmf_nlm_h001",
            "psnr_aswmf_nlm_h001",
            "score_aswmf_nlm_h001",
            "time_aswmf_nlm_h001",
        ],
    }
    for method, columns in optional_columns.items():
        for column in columns:
            if column in df.columns:
                metric = column.split("_", 1)[0]
                aggregations[f"mean_{metric}_{method}"] = (column, "mean")

    summary = df.groupby("level").agg(**aggregations).reset_index()
    summary.to_excel(summary_dir / f"results_{dataset_name}_hibrid_summary.xlsx", index=False)
    return all_rows, summary


def run_dataset(dataset_name):
    source_base = Path(f"/workspace/data/output/{dataset_name}")
    hibrid_base = Path(f"/workspace/data/output/{dataset_name}Hibrid")

    total = 0
    all_missing = {}
    for level in LEVELS:
        _, merged, missing = merge_level(dataset_name, source_base, hibrid_base, level)
        total += merged
        if missing:
            all_missing[level] = missing

    all_rows, summary = rebuild_all_and_summary(dataset_name, hibrid_base)
    print(f"{dataset_name}: merged {total} rows; all rows {len(all_rows)}")
    if all_missing:
        print(f"{dataset_name}: missing {all_missing}")
    print(summary[["level", "mean_psnr_gnlm", "mean_ssim_gnlm", "mean_time_geonlm"]].to_string(index=False))


if __name__ == "__main__":
    run_dataset("set12")
    run_dataset("set50")
