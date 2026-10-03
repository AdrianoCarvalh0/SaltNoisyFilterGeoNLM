import itertools
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
from functions.geonlm_medians_functions import run_geonlm_medians_pipeline


SOURCE_BASE = Path("/workspace/data/output/set50")
BASELINE_RESULTS = Path("/workspace/data/output/set50Hibrid/summary/results_set50_hibrid_all.xlsx")
OUTPUT_BASE = Path("/workspace/data/output/set50Hibrid/ablation_params")
LEVELS = ["low", "moderate", "medium", "high", "extreme"]
LEVEL_ORDER = {level: index for index, level in enumerate(LEVELS)}
RESOLUTION = "full_512"

FIXED_CONFIG = {
    "h_multiplier": 0.005,
    "z_alpha": 1.96,
    "outlier_pixel_alpha": 0.0,
    "switch_impulse_only": True,
    "reject_impulse_candidates": True,
    "use_aswmf_spatial_weights": True,
    "aswmf_weight_diag_1": 1.0,
    "aswmf_weight_diag_2": 1.0,
    "aswmf_weight_other": 10.0,
}

# Local, reviewer-friendly ablation centered on the chosen configuration.
# It changes one structural parameter at a time around f=2, t=3, nn=7, plus
# a few nearby joint alternatives, without turning the experiment into a huge grid.
DEFAULT_VARIANTS = [
    {"name": "C0_reference_f2_t3_nn7", "label": "C0", "description": "Reference", "f": 2, "t": 3, "nn": 7},
    {"name": "C1_reduced_connectivity_f2_t3_nn5", "label": "C1", "description": "Reduced graph connectivity", "f": 2, "t": 3, "nn": 5},
    {"name": "C2_increased_connectivity_f2_t3_nn10", "label": "C2", "description": "Increased graph connectivity", "f": 2, "t": 3, "nn": 10},
    {"name": "C3_smaller_patch_f1_t3_nn7", "label": "C3", "description": "Smaller patch", "f": 1, "t": 3, "nn": 7},
    {"name": "C4_smaller_search_f2_t2_nn7", "label": "C4", "description": "Smaller search window", "f": 2, "t": 2, "nn": 7},
    {"name": "C5_larger_search_f2_t5_nn7", "label": "C5", "description": "Larger search window", "f": 2, "t": 5, "nn": 7},
]


def selected_levels():
    raw = os.environ.get("SET50_HIBRID_PARAM_ABLATION_LEVELS")
    if not raw:
        return LEVELS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return [level for level in LEVELS if level in wanted]


def parse_int_list(env_name, default):
    raw = os.environ.get(env_name)
    if not raw:
        return default
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def variants():
    if os.environ.get("SET50_HIBRID_PARAM_ABLATION_FULL_GRID", "0") == "1":
        f_values = parse_int_list("SET50_HIBRID_PARAM_ABLATION_F", [1, 2, 3])
        t_values = parse_int_list("SET50_HIBRID_PARAM_ABLATION_T", [2, 3, 5])
        nn_values = parse_int_list("SET50_HIBRID_PARAM_ABLATION_NN", [5, 7, 10])
        return [
            {
                "name": f"f{f}_t{t}_nn{nn}",
                "label": f"f{f}_t{t}_nn{nn}",
                "description": "Full grid variant",
                "f": f,
                "t": t,
                "nn": nn,
            }
            for f, t, nn in itertools.product(f_values, t_values, nn_values)
        ]
    return DEFAULT_VARIANTS


def images_per_level():
    raw = os.environ.get("SET50_HIBRID_PARAM_ABLATION_IMAGES_PER_LEVEL")
    if raw is None:
        return 2
    raw = raw.strip().lower()
    if raw in {"", "all", "none"}:
        return None
    return int(raw)


def sample_seed():
    return int(os.environ.get("SET50_HIBRID_PARAM_ABLATION_SAMPLE_SEED", "42"))


def max_images():
    raw = os.environ.get("SET50_HIBRID_PARAM_ABLATION_MAX_IMAGES")
    if not raw:
        return None
    return int(raw)


def force_run():
    return os.environ.get("SET50_HIBRID_PARAM_ABLATION_FORCE", "0") == "1"


def n_jobs():
    raw = os.environ.get("SET50_HIBRID_PARAM_ABLATION_N_JOBS")
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


def load_baselines():
    if not BASELINE_RESULTS.exists():
        return {}

    df = pd.read_excel(BASELINE_RESULTS)
    return {
        (row["level"], row["file_name"]): row
        for row in df.to_dict("records")
    }


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


def baseline_value(baseline, key):
    if not baseline:
        return np.nan
    value = baseline.get(key, np.nan)
    return np.nan if pd.isna(value) else float(value)


def read_or_run_hybrid(item, out_path, variant):
    reference = item["img_reference_np"]
    h_used = float(item["nlm_h"]) * FIXED_CONFIG["h_multiplier"]

    if out_path.exists() and not force_run():
        image = skimage.io.imread(str(out_path))
        psnr, ssim, method_score = metrics(reference, image)
        return psnr, ssim, method_score, np.nan, h_used

    start = time.time()
    filtered, h_used, psnr, ssim, method_score = run_geonlm_medians_pipeline(
        img_original=reference,
        h_base=float(item["nlm_h"]),
        img_noisy=item["img_noisy_salt_pepper_np"],
        f=variant["f"],
        t=variant["t"],
        mult=FIXED_CONFIG["h_multiplier"],
        nn=variant["nn"],
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
    elapsed = time.time() - start
    save_uint8(out_path, filtered)
    return psnr, ssim, method_score, elapsed, h_used


def run_level(level, baselines):
    path = source_pickle(level)
    vector = load_pickle(path.parent, path.name)
    vector = select_items(vector)

    records = []
    for variant in variants():
        output_root = OUTPUT_BASE / variant["name"] / f"salt_pepper_{level}" / RESOLUTION
        image_dir = output_root / "GeoHibNLM"
        results_dir = output_root / "results"
        image_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

        variant_records = []
        for item in vector:
            file_name = item["file_name"]
            baseline = baselines.get((level, file_name), {})
            psnr_hybrid, ssim_hybrid, score_hybrid, elapsed, h_used = read_or_run_hybrid(
                item,
                image_dir / file_name,
                variant,
            )
            score_nlm = baseline_value(baseline, "score_nlm")
            score_median = baseline_value(baseline, "score_median")
            score_nlmedians = baseline_value(baseline, "score_nlmedians")

            record = {
                "level": level,
                "resolution": RESOLUTION,
                "file_name": file_name,
                "source_nlm_pickle": str(path),
                "ablation_name": variant["name"],
                "configuration_label": variant["label"],
                "configuration_description": variant["description"],
                "f": variant["f"],
                "t": variant["t"],
                "nn": variant["nn"],
                "h_multiplier": FIXED_CONFIG["h_multiplier"],
                "h_used": h_used,
                "nlm_h": float(item["nlm_h"]),
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
                "psnr_geohibnlm": psnr_hybrid,
                "ssim_geohibnlm": ssim_hybrid,
                "score_geohibnlm": score_hybrid,
                "time_geohibnlm": elapsed,
                "score_nlm_baseline": score_nlm,
                "score_median_baseline": score_median,
                "score_nlmedians_baseline": score_nlmedians,
                "delta_score_vs_nlm": score_hybrid - score_nlm,
                "delta_score_vs_median": score_hybrid - score_median,
                "delta_score_vs_nlmedians": score_hybrid - score_nlmedians,
            }
            variant_records.append(record)
            records.append(record)
            print(
                f"{level} {variant['name']} {file_name}: "
                f"score={score_hybrid:.4f} "
                f"dNLM={record['delta_score_vs_nlm']:+.4f} "
                f"dNLMedians={record['delta_score_vs_nlmedians']:+.4f} "
                f"time={elapsed if not np.isnan(elapsed) else 0:.2f}s",
                flush=True,
            )

        save_pickle(variant_records, results_dir, f"results_set50_hibrid_param_ablation_{level}_{variant['name']}.pkl")
        save_results_to_xlsx(
            variant_records,
            results_dir,
            f"results_set50_hibrid_param_ablation_{level}_{variant['name']}.xlsx",
        )

    return records


def empty_summary():
    return pd.DataFrame(
        columns=[
            "level",
            "ablation_name",
            "configuration_label",
            "configuration_description",
            "f",
            "t",
            "nn",
            "h_multiplier",
            "mean_psnr_geohibnlm",
            "mean_ssim_geohibnlm",
            "mean_score_geohibnlm",
            "mean_time_geohibnlm",
            "wins_vs_nlm",
            "wins_vs_median",
            "wins_vs_nlmedians",
        ]
    )


def make_summary(records, output_dir):
    df = pd.DataFrame(records)
    if df.empty:
        summary = empty_summary()
        best = summary.copy()
        overall = summary.copy()
    else:
        summary = (
            df.groupby([
                "level",
                "ablation_name",
                "configuration_label",
                "configuration_description",
                "f",
                "t",
                "nn",
                "h_multiplier",
            ])
            .agg(
                mean_psnr_geohibnlm=("psnr_geohibnlm", "mean"),
                std_psnr_geohibnlm=("psnr_geohibnlm", "std"),
                sem_psnr_geohibnlm=("psnr_geohibnlm", "sem"),
                mean_ssim_geohibnlm=("ssim_geohibnlm", "mean"),
                std_ssim_geohibnlm=("ssim_geohibnlm", "std"),
                sem_ssim_geohibnlm=("ssim_geohibnlm", "sem"),
                mean_score_geohibnlm=("score_geohibnlm", "mean"),
                std_score_geohibnlm=("score_geohibnlm", "std"),
                sem_score_geohibnlm=("score_geohibnlm", "sem"),
                mean_time_geohibnlm=("time_geohibnlm", "mean"),
                wins_vs_nlm=("delta_score_vs_nlm", lambda s: int((s > 0).sum())),
                wins_vs_median=("delta_score_vs_median", lambda s: int((s > 0).sum())),
                wins_vs_nlmedians=("delta_score_vs_nlmedians", lambda s: int((s > 0).sum())),
                n_images=("file_name", "nunique"),
            )
            .reset_index()
            .sort_values(["level", "mean_score_geohibnlm"], ascending=[True, False])
        )
        best = (
            summary.groupby("level", as_index=False)
            .head(1)
            .reset_index(drop=True)
        )
        overall = (
            df.groupby([
                "ablation_name",
                "configuration_label",
                "configuration_description",
                "f",
                "t",
                "nn",
                "h_multiplier",
            ])
            .agg(
                mean_psnr_geohibnlm=("psnr_geohibnlm", "mean"),
                std_psnr_geohibnlm=("psnr_geohibnlm", "std"),
                sem_psnr_geohibnlm=("psnr_geohibnlm", "sem"),
                mean_ssim_geohibnlm=("ssim_geohibnlm", "mean"),
                std_ssim_geohibnlm=("ssim_geohibnlm", "std"),
                sem_ssim_geohibnlm=("ssim_geohibnlm", "sem"),
                mean_score_geohibnlm=("score_geohibnlm", "mean"),
                std_score_geohibnlm=("score_geohibnlm", "std"),
                sem_score_geohibnlm=("score_geohibnlm", "sem"),
                mean_time_geohibnlm=("time_geohibnlm", "mean"),
                wins_vs_nlm=("delta_score_vs_nlm", lambda s: int((s > 0).sum())),
                wins_vs_median=("delta_score_vs_median", lambda s: int((s > 0).sum())),
                wins_vs_nlmedians=("delta_score_vs_nlmedians", lambda s: int((s > 0).sum())),
                n_images=("file_name", "count"),
                n_levels=("level", "nunique"),
            )
            .reset_index()
            .sort_values("mean_score_geohibnlm", ascending=False)
        )

    save_results_to_xlsx(summary.to_dict("records"), output_dir, "results_set50_hibrid_param_ablation_summary.xlsx")
    save_results_to_xlsx(best.to_dict("records"), output_dir, "results_set50_hibrid_param_ablation_best_by_level.xlsx")
    save_results_to_xlsx(overall.to_dict("records"), output_dir, "results_set50_hibrid_param_ablation_overall.xlsx")
    save_pickle(summary.to_dict("records"), output_dir, "results_set50_hibrid_param_ablation_summary.pkl")
    save_pickle(best.to_dict("records"), output_dir, "results_set50_hibrid_param_ablation_best_by_level.pkl")
    save_pickle(overall.to_dict("records"), output_dir, "results_set50_hibrid_param_ablation_overall.pkl")

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
            "results_set50_hibrid_param_ablation_selected_images.xlsx",
        )
    return summary, best


if __name__ == "__main__":
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    all_records = []
    baselines = load_baselines()

    for level in selected_levels():
        all_records.extend(run_level(level, baselines))

    summary_dir = OUTPUT_BASE / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(all_records, summary_dir, "results_set50_hibrid_param_ablation_all.pkl")
    save_results_to_xlsx(
        all_records,
        summary_dir,
        "results_set50_hibrid_param_ablation_all.xlsx",
    )
    summary, best = make_summary(all_records, summary_dir)
    print(summary.to_string(index=False))
    print("\nBest structural parameters per level:")
    print(best.to_string(index=False))
