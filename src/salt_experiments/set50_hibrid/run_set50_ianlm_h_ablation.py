import os
import re
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
from functions.anlm_functions import run_anlm_pipeline


SOURCE_BASE = Path("/workspace/data/output/set50")
BASELINE_RESULTS = Path("/workspace/data/output/set50Hibrid/summary/results_set50_hibrid_all.xlsx")
OUTPUT_BASE = Path("/workspace/data/output/set50Hibrid/ablation_h_ianlm_f1_t3")
LEVELS = ["low", "moderate", "medium", "high", "extreme"]
LEVEL_ORDER = {level: index for index, level in enumerate(LEVELS)}
RESOLUTION = "full_512"

FIXED_CONFIG = {
    "f": 1,
    "t": 3,
    "z_alpha": 1.96,
    "outlier_pixel_alpha": 0.0,
    "switch_impulse_only": True,
    "reject_impulse_candidates": True,
    "use_aswmf_spatial_weights": True,
    "aswmf_weight_diag_1": 1.0,
    "aswmf_weight_diag_2": 1.0,
    "aswmf_weight_other": 10.0,
}

DEFAULT_H_MULTIPLIERS = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.02]


def selected_levels():
    raw = os.environ.get("SET50_IANLM_ABLATION_LEVELS")
    if not raw:
        return LEVELS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return [level for level in LEVELS if level in wanted]


def h_multipliers():
    raw = os.environ.get("SET50_IANLM_ABLATION_H_MULTIPLIERS")
    if not raw:
        return DEFAULT_H_MULTIPLIERS
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def max_images():
    raw = os.environ.get("SET50_IANLM_ABLATION_MAX_IMAGES")
    if not raw:
        return None
    return int(raw)


def images_per_level():
    raw = os.environ.get("SET50_IANLM_ABLATION_IMAGES_PER_LEVEL")
    if raw is None:
        return 2
    raw = raw.strip().lower()
    if raw in {"", "all", "none"}:
        return None
    return int(raw)


def sample_seed():
    return int(os.environ.get("SET50_IANLM_ABLATION_SAMPLE_SEED", "42"))


def force_run():
    return os.environ.get("SET50_IANLM_ABLATION_FORCE", "0") == "1"


def n_jobs():
    raw = os.environ.get("SET50_IANLM_ABLATION_N_JOBS")
    if not raw:
        return -1
    return int(raw)


def score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


def natural_sort_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def natural_sort_text(value):
    return "".join(
        f"{int(part):08d}" if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    )


def metrics(reference, image):
    reference_uint8 = np.clip(reference, 0, 255).astype(np.uint8)
    image_uint8 = to_grayscale_uint8(image)
    psnr = peak_signal_noise_ratio(reference_uint8, image_uint8, data_range=255)
    ssim = structural_similarity(reference_uint8, image_uint8, data_range=255)
    return psnr, ssim, score(psnr, ssim)


def to_grayscale_uint8(image):
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    if image.ndim == 3 and image.shape[-1] == 3:
        image = (
            0.2125 * image[..., 0]
            + 0.7154 * image[..., 1]
            + 0.0721 * image[..., 2]
        )
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    return np.clip(image, 0, 255).astype(np.uint8)


def h_tag(multiplier):
    return f"h{int(round(multiplier * 10000)):04d}"


def load_baselines():
    if not BASELINE_RESULTS.exists():
        return {}

    df = pd.read_excel(BASELINE_RESULTS)
    baselines = {}
    for row in df.to_dict("records"):
        baselines[(row["level"], row["file_name"])] = row
    return baselines


def source_pickle(level):
    source_results = SOURCE_BASE / f"salt_pepper_{level}" / RESOLUTION / "results"
    return source_results / f"array_nlm_salt_pepper_{level}_filtereds.pkl"


def select_items(vector):
    vector = sorted(vector, key=lambda item: natural_sort_key(item["file_name"]))

    limit = max_images()
    if limit is not None:
        return vector[:limit]

    count = images_per_level()
    if count is None or count >= len(vector):
        return vector

    rng = np.random.default_rng(sample_seed())
    selected_indices = sorted(rng.choice(len(vector), size=count, replace=False))
    return [vector[index] for index in selected_indices]


def save_uint8(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(str(path), np.clip(image, 0, 255).astype(np.uint8))


def read_or_run_ianlm(item, out_path, h_multiplier):
    reference = item["img_reference_np"]
    h_used = float(item["nlm_h"]) * float(h_multiplier)

    if out_path.exists() and not force_run():
        image = skimage.io.imread(str(out_path))
        psnr, ssim, method_score = metrics(reference, image)
        return psnr, ssim, method_score, np.nan, h_used

    start = time.perf_counter()
    filtered, h_used, psnr, ssim, method_score = run_anlm_pipeline(
        img_original=reference,
        h_base=float(item["nlm_h"]),
        img_noisy=item["img_noisy_salt_pepper_np"],
        f=FIXED_CONFIG["f"],
        t=FIXED_CONFIG["t"],
        mult=h_multiplier,
        z_alpha=FIXED_CONFIG["z_alpha"],
        outlier_pixel_alpha=FIXED_CONFIG["outlier_pixel_alpha"],
        switch_impulse_only=FIXED_CONFIG["switch_impulse_only"],
        reject_impulse_candidates=FIXED_CONFIG["reject_impulse_candidates"],
        use_aswmf_spatial_weights=FIXED_CONFIG["use_aswmf_spatial_weights"],
        aswmf_weight_diag_1=FIXED_CONFIG["aswmf_weight_diag_1"],
        aswmf_weight_diag_2=FIXED_CONFIG["aswmf_weight_diag_2"],
        aswmf_weight_other=FIXED_CONFIG["aswmf_weight_other"],
        n_jobs=n_jobs(),
    )
    elapsed = time.perf_counter() - start
    save_uint8(out_path, filtered)
    return psnr, ssim, method_score, elapsed, h_used


def warmup_ianlm():
    dummy = np.zeros((16, 16), dtype=np.float32)
    dummy[8, 8] = 255.0
    run_anlm_pipeline(
        img_original=dummy,
        h_base=100.0,
        img_noisy=dummy,
        f=FIXED_CONFIG["f"],
        t=FIXED_CONFIG["t"],
        mult=0.001,
        z_alpha=FIXED_CONFIG["z_alpha"],
        outlier_pixel_alpha=FIXED_CONFIG["outlier_pixel_alpha"],
        switch_impulse_only=FIXED_CONFIG["switch_impulse_only"],
        reject_impulse_candidates=FIXED_CONFIG["reject_impulse_candidates"],
        use_aswmf_spatial_weights=FIXED_CONFIG["use_aswmf_spatial_weights"],
        aswmf_weight_diag_1=FIXED_CONFIG["aswmf_weight_diag_1"],
        aswmf_weight_diag_2=FIXED_CONFIG["aswmf_weight_diag_2"],
        aswmf_weight_other=FIXED_CONFIG["aswmf_weight_other"],
        n_jobs=n_jobs(),
    )


def baseline_value(baseline, key):
    if not baseline:
        return np.nan
    value = baseline.get(key, np.nan)
    return np.nan if pd.isna(value) else float(value)


def run_level(level, baselines):
    path = source_pickle(level)
    vector = load_pickle(path.parent, path.name)
    vector = select_items(vector)

    records = []
    for h_multiplier in h_multipliers():
        tag = h_tag(h_multiplier)
        output_root = OUTPUT_BASE / tag / f"salt_pepper_{level}" / RESOLUTION
        image_dir = output_root / "IANLM"
        results_dir = output_root / "results"
        image_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        for item in vector:
            file_name = item["file_name"]
            baseline = baselines.get((level, file_name), {})
            psnr_ianlm, ssim_ianlm, score_ianlm, elapsed, h_used = read_or_run_ianlm(
                item,
                image_dir / file_name,
                h_multiplier,
            )
            score_nlm = baseline_value(baseline, "score_nlm")
            score_median = baseline_value(baseline, "score_median")
            score_nlmedians = baseline_value(baseline, "score_nlmedians")
            score_ghnlm = baseline_value(baseline, "score_geonlm_hibrid")

            record = {
                "level": level,
                "resolution": RESOLUTION,
                "file_name": file_name,
                "source_nlm_pickle": str(path),
                "ablation_name": tag,
                "h_multiplier": h_multiplier,
                "h_used": h_used,
                "nlm_h": float(item["nlm_h"]),
                "f": FIXED_CONFIG["f"],
                "t": FIXED_CONFIG["t"],
                "z_alpha": FIXED_CONFIG["z_alpha"],
                "outlier_pixel_alpha": FIXED_CONFIG["outlier_pixel_alpha"],
                "switch_impulse_only": FIXED_CONFIG["switch_impulse_only"],
                "reject_impulse_candidates": FIXED_CONFIG["reject_impulse_candidates"],
                "use_aswmf_spatial_weights": FIXED_CONFIG["use_aswmf_spatial_weights"],
                "aswmf_weight_diag_1": FIXED_CONFIG["aswmf_weight_diag_1"],
                "aswmf_weight_diag_2": FIXED_CONFIG["aswmf_weight_diag_2"],
                "aswmf_weight_other": FIXED_CONFIG["aswmf_weight_other"],
                "sample_seed": sample_seed(),
                "sample_size_per_level": len(vector),
                "estimated_sigma_salt_pepper": float(item["estimated_sigma_salt_pepper"]),
                "salt_pepper_density": float(item["salt_pepper_density"]),
                "psnr_ianlm": psnr_ianlm,
                "ssim_ianlm": ssim_ianlm,
                "score_ianlm": score_ianlm,
                "time_ianlm": elapsed,
                "psnr_nlm_baseline": baseline_value(baseline, "psnr_nlm"),
                "ssim_nlm_baseline": baseline_value(baseline, "ssim_nlm"),
                "score_nlm_baseline": score_nlm,
                "score_median_baseline": score_median,
                "score_nlmedians_baseline": score_nlmedians,
                "score_ghnlm_baseline": score_ghnlm,
                "delta_score_vs_nlm": score_ianlm - score_nlm,
                "delta_score_vs_median": score_ianlm - score_median,
                "delta_score_vs_nlmedians": score_ianlm - score_nlmedians,
                "delta_score_vs_ghnlm": score_ianlm - score_ghnlm,
            }
            records.append(record)
            print(
                f"{level} {tag} {file_name}: "
                f"score={score_ianlm:.4f} "
                f"dGHNLM={record['delta_score_vs_ghnlm']:+.4f} "
                f"dNLM={record['delta_score_vs_nlm']:+.4f} "
                f"time={elapsed if not np.isnan(elapsed) else 0:.4f}s",
                flush=True,
            )

        level_variant_records = [
            row for row in records
            if row["level"] == level and row["h_multiplier"] == h_multiplier
        ]
        save_pickle(level_variant_records, results_dir, f"results_set50_ianlm_h_ablation_{level}_{tag}.pkl")
        save_results_to_xlsx(
            level_variant_records,
            results_dir,
            f"results_set50_ianlm_h_ablation_{level}_{tag}.xlsx",
        )

    return records


def make_summary(records, output_dir):
    df = pd.DataFrame(records)
    if df.empty:
        summary = pd.DataFrame()
        best = pd.DataFrame()
        overall = pd.DataFrame()
    else:
        summary = (
            df.groupby(["level", "h_multiplier", "ablation_name"])
            .agg(
                mean_psnr_ianlm=("psnr_ianlm", "mean"),
                std_psnr_ianlm=("psnr_ianlm", "std"),
                sem_psnr_ianlm=("psnr_ianlm", "sem"),
                mean_ssim_ianlm=("ssim_ianlm", "mean"),
                std_ssim_ianlm=("ssim_ianlm", "std"),
                sem_ssim_ianlm=("ssim_ianlm", "sem"),
                mean_score_ianlm=("score_ianlm", "mean"),
                std_score_ianlm=("score_ianlm", "std"),
                sem_score_ianlm=("score_ianlm", "sem"),
                mean_time_ianlm=("time_ianlm", "mean"),
                wins_vs_nlm=("delta_score_vs_nlm", lambda s: int((s > 0).sum())),
                wins_vs_median=("delta_score_vs_median", lambda s: int((s > 0).sum())),
                wins_vs_nlmedians=("delta_score_vs_nlmedians", lambda s: int((s > 0).sum())),
                wins_vs_ghnlm=("delta_score_vs_ghnlm", lambda s: int((s > 0).sum())),
                n_images=("file_name", "nunique"),
            )
            .reset_index()
            .sort_values(["level", "h_multiplier"])
        )
        best = (
            summary.sort_values(["level", "mean_score_ianlm"], ascending=[True, False])
            .groupby("level", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        overall = (
            df.groupby(["h_multiplier", "ablation_name", "f", "t"])
            .agg(
                mean_psnr_ianlm=("psnr_ianlm", "mean"),
                std_psnr_ianlm=("psnr_ianlm", "std"),
                sem_psnr_ianlm=("psnr_ianlm", "sem"),
                mean_ssim_ianlm=("ssim_ianlm", "mean"),
                std_ssim_ianlm=("ssim_ianlm", "std"),
                sem_ssim_ianlm=("ssim_ianlm", "sem"),
                mean_score_ianlm=("score_ianlm", "mean"),
                std_score_ianlm=("score_ianlm", "std"),
                sem_score_ianlm=("score_ianlm", "sem"),
                mean_time_ianlm=("time_ianlm", "mean"),
                wins_vs_nlm=("delta_score_vs_nlm", lambda s: int((s > 0).sum())),
                wins_vs_median=("delta_score_vs_median", lambda s: int((s > 0).sum())),
                wins_vs_nlmedians=("delta_score_vs_nlmedians", lambda s: int((s > 0).sum())),
                wins_vs_ghnlm=("delta_score_vs_ghnlm", lambda s: int((s > 0).sum())),
                n_images=("file_name", "count"),
                n_levels=("level", "nunique"),
            )
            .reset_index()
            .sort_values("mean_score_ianlm", ascending=False)
        )

    save_results_to_xlsx(summary.to_dict("records"), output_dir, "results_set50_ianlm_h_ablation_summary.xlsx")
    save_results_to_xlsx(best.to_dict("records"), output_dir, "results_set50_ianlm_h_ablation_best_by_level.xlsx")
    save_results_to_xlsx(overall.to_dict("records"), output_dir, "results_set50_ianlm_h_ablation_overall.xlsx")
    save_pickle(summary.to_dict("records"), output_dir, "results_set50_ianlm_h_ablation_summary.pkl")
    save_pickle(best.to_dict("records"), output_dir, "results_set50_ianlm_h_ablation_best_by_level.pkl")
    save_pickle(overall.to_dict("records"), output_dir, "results_set50_ianlm_h_ablation_overall.pkl")

    if not df.empty:
        selected = (
            df[["level", "file_name", "sample_seed", "sample_size_per_level"]]
            .drop_duplicates()
        )
        selected["_level_order"] = selected["level"].map(LEVEL_ORDER).fillna(999)
        selected["_file_order"] = selected["file_name"].map(natural_sort_text)
        selected = selected.sort_values(["_level_order", "_file_order"]).drop(
            columns=["_level_order", "_file_order"]
        )
        save_results_to_xlsx(
            selected.to_dict("records"),
            output_dir,
            "results_set50_ianlm_h_ablation_selected_images.xlsx",
        )
    return summary, best


if __name__ == "__main__":
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    warmup_ianlm()
    all_records = []
    baselines = load_baselines()

    for level in selected_levels():
        all_records.extend(run_level(level, baselines))

    summary_dir = OUTPUT_BASE / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(all_records, summary_dir, "results_set50_ianlm_h_ablation_all.pkl")
    save_results_to_xlsx(
        all_records,
        summary_dir,
        "results_set50_ianlm_h_ablation_all.xlsx",
    )
    summary, best = make_summary(all_records, summary_dir)
    print(summary.to_string(index=False))
    print("\nBest h per level:")
    print(best.to_string(index=False))
