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

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx
from functions.nlm_functions import NLM_fast_cpu
from functions.salt_filters import aswmf_filter


NLM_SOURCE_BASE = Path("/workspace/data/output/set12")
FALLBACK_NLM_SOURCE_BASE = Path("/workspace/data/output/set12Hibrid")
HYBRID_RESULTS_BASE = Path("/workspace/data/output/set12Hibrid")
OUTPUT_BASE = Path("/workspace/data/output/set12Hibrid/aswmf_nlm")
LEVELS = ["low", "medium", "moderate", "high", "extreme"]
RESOLUTION = "full_512"

NLM_CONFIG = {
    "f": 4,
    "t": 7,
}

ASWMF_CONFIG = {
    "radius": 3,
    "weight_diag_1": 1.0,
    "weight_diag_2": 1.0,
    "weight_other": 10.0,
}


def selected_levels():
    raw = os.environ.get("SET12_ASWMF_NLM_LEVELS")
    if not raw:
        return LEVELS
    wanted = {part.strip() for part in raw.split(",") if part.strip()}
    return [level for level in LEVELS if level in wanted]


def max_images():
    raw = os.environ.get("SET12_ASWMF_NLM_MAX_IMAGES")
    if not raw:
        return None
    return int(raw)


def force_run():
    return os.environ.get("SET12_ASWMF_NLM_FORCE", "0") == "1"


def score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


def metrics(reference, image):
    reference_uint8 = np.clip(reference, 0, 255).astype(np.uint8)
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(reference_uint8, image_uint8, data_range=255)
    ssim = structural_similarity(reference_uint8, image_uint8, data_range=255)
    return psnr, ssim, score(psnr, ssim)


def save_uint8(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(str(path), np.clip(image, 0, 255).astype(np.uint8))


def ensure_dirs(root):
    for subdir in ["ASWMF", "ASWMF_NLM", "results"]:
        (root / subdir).mkdir(parents=True, exist_ok=True)


def source_results_dir(level):
    primary = NLM_SOURCE_BASE / f"salt_pepper_{level}" / RESOLUTION / "results"
    if primary.exists():
        return primary
    return FALLBACK_NLM_SOURCE_BASE / f"salt_pepper_{level}" / RESOLUTION / "results"


def load_hybrid_results_by_file(level):
    path = (
        HYBRID_RESULTS_BASE
        / f"salt_pepper_{level}"
        / RESOLUTION
        / "results"
        / f"results_set12_hibrid_{level}.pkl"
    )
    if not path.exists():
        return {}
    return {row["file_name"]: row for row in load_pickle(path.parent, path.name)}


def read_or_run_aswmf(item, out_path):
    reference = item["img_reference_np"]
    if out_path.exists() and not force_run():
        image = skimage.io.imread(str(out_path))
        psnr, ssim, method_score = metrics(reference, image)
        return image, psnr, ssim, method_score, np.nan

    start = time.time()
    filtered = aswmf_filter(
        item["img_noisy_salt_pepper_np"].astype(np.float32),
        radius=ASWMF_CONFIG["radius"],
        weight_diag_1=ASWMF_CONFIG["weight_diag_1"],
        weight_diag_2=ASWMF_CONFIG["weight_diag_2"],
        weight_other=ASWMF_CONFIG["weight_other"],
    )
    elapsed = time.time() - start
    filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
    save_uint8(out_path, filtered_uint8)
    psnr, ssim, method_score = metrics(reference, filtered_uint8)
    return filtered_uint8, psnr, ssim, method_score, elapsed


def read_or_run_aswmf_nlm(item, aswmf_image, out_path):
    reference = item["img_reference_np"]
    h = float(item["nlm_h"])
    if out_path.exists() and not force_run():
        image = skimage.io.imread(str(out_path))
        psnr, ssim, method_score = metrics(reference, image)
        return image, psnr, ssim, method_score, np.nan, h

    start = time.time()
    filtered = NLM_fast_cpu(
        aswmf_image.astype(np.float32),
        h=h,
        f=NLM_CONFIG["f"],
        t=NLM_CONFIG["t"],
    )
    elapsed = time.time() - start
    filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
    save_uint8(out_path, filtered_uint8)
    psnr, ssim, method_score = metrics(reference, filtered_uint8)
    return filtered_uint8, psnr, ssim, method_score, elapsed, h


def run_level(level):
    source_results = source_results_dir(level)
    nlm_pickle = f"array_nlm_salt_pepper_{level}_filtereds.pkl"
    vector = load_pickle(source_results, nlm_pickle)
    limit = max_images()
    if limit is not None:
        vector = vector[:limit]

    output_root = OUTPUT_BASE / f"salt_pepper_{level}" / RESOLUTION
    output_results = output_root / "results"
    ensure_dirs(output_root)
    save_pickle(vector, output_results, nlm_pickle)

    hybrid_by_file = load_hybrid_results_by_file(level)
    records = []

    for item in vector:
        file_name = item["file_name"]
        reference = item["img_reference_np"]
        noisy = item["img_noisy_salt_pepper_np"]
        nlm = item["img_filtered_nlm"]

        psnr_noisy, ssim_noisy, score_noisy = metrics(reference, noisy)
        psnr_nlm, ssim_nlm, score_nlm = metrics(reference, nlm)

        aswmf_image, psnr_aswmf, ssim_aswmf, score_aswmf, time_aswmf = read_or_run_aswmf(
            item,
            output_root / "ASWMF" / file_name,
        )
        (
            _,
            psnr_aswmf_nlm,
            ssim_aswmf_nlm,
            score_aswmf_nlm,
            time_aswmf_nlm,
            h_aswmf_nlm,
        ) = read_or_run_aswmf_nlm(
            item,
            aswmf_image,
            output_root / "ASWMF_NLM" / file_name,
        )

        hybrid = hybrid_by_file.get(file_name, {})
        score_hybrid = (
            np.nan
            if "score_geonlm_hibrid" not in hybrid
            else float(hybrid["score_geonlm_hibrid"])
        )

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
            "nlm_f": NLM_CONFIG["f"],
            "nlm_t": NLM_CONFIG["t"],
            "nlm_h": float(item["nlm_h"]),
            "psnr_nlm": psnr_nlm,
            "ssim_nlm": ssim_nlm,
            "score_nlm": score_nlm,
            "aswmf_radius": ASWMF_CONFIG["radius"],
            "aswmf_weight_diag_1": ASWMF_CONFIG["weight_diag_1"],
            "aswmf_weight_diag_2": ASWMF_CONFIG["weight_diag_2"],
            "aswmf_weight_other": ASWMF_CONFIG["weight_other"],
            "psnr_aswmf": psnr_aswmf,
            "ssim_aswmf": ssim_aswmf,
            "score_aswmf": score_aswmf,
            "time_aswmf": time_aswmf,
            "aswmf_nlm_f": NLM_CONFIG["f"],
            "aswmf_nlm_t": NLM_CONFIG["t"],
            "aswmf_nlm_h": h_aswmf_nlm,
            "psnr_aswmf_nlm": psnr_aswmf_nlm,
            "ssim_aswmf_nlm": ssim_aswmf_nlm,
            "score_aswmf_nlm": score_aswmf_nlm,
            "time_aswmf_nlm": time_aswmf_nlm,
            "psnr_geonlm_hibrid": (
                np.nan
                if "psnr_geonlm_hibrid" not in hybrid
                else float(hybrid["psnr_geonlm_hibrid"])
            ),
            "ssim_geonlm_hibrid": (
                np.nan
                if "ssim_geonlm_hibrid" not in hybrid
                else float(hybrid["ssim_geonlm_hibrid"])
            ),
            "score_geonlm_hibrid": score_hybrid,
            "delta_score_aswmf_nlm_vs_nlm": score_aswmf_nlm - score_nlm,
            "delta_score_aswmf_nlm_vs_aswmf": score_aswmf_nlm - score_aswmf,
            "delta_score_aswmf_nlm_vs_geonlm_hibrid": score_aswmf_nlm - score_hybrid,
        }
        records.append(record)
        print(
            f"{level} {file_name}: "
            f"ASWMF+NLM={score_aswmf_nlm:.4f} "
            f"GHNLM={score_hybrid:.4f} "
            f"dScore={record['delta_score_aswmf_nlm_vs_geonlm_hibrid']:+.4f} "
            f"time_aswmf_nlm={time_aswmf_nlm if not np.isnan(time_aswmf_nlm) else 0:.2f}s",
            flush=True,
        )

    save_pickle(records, output_results, f"results_set12_aswmf_nlm_{level}.pkl")
    save_results_to_xlsx(records, output_results, f"results_set12_aswmf_nlm_{level}.xlsx")
    return records


def make_summary(all_records, summary_dir):
    df = pd.DataFrame(all_records)
    summary = (
        df.groupby("level")
        .agg(
            n_images=("file_name", "count"),
            mean_psnr_nlm=("psnr_nlm", "mean"),
            mean_psnr_aswmf=("psnr_aswmf", "mean"),
            mean_psnr_aswmf_nlm=("psnr_aswmf_nlm", "mean"),
            mean_psnr_geonlm_hibrid=("psnr_geonlm_hibrid", "mean"),
            mean_ssim_nlm=("ssim_nlm", "mean"),
            mean_ssim_aswmf=("ssim_aswmf", "mean"),
            mean_ssim_aswmf_nlm=("ssim_aswmf_nlm", "mean"),
            mean_ssim_geonlm_hibrid=("ssim_geonlm_hibrid", "mean"),
            mean_score_nlm=("score_nlm", "mean"),
            mean_score_aswmf=("score_aswmf", "mean"),
            mean_score_aswmf_nlm=("score_aswmf_nlm", "mean"),
            mean_score_geonlm_hibrid=("score_geonlm_hibrid", "mean"),
            mean_time_aswmf=("time_aswmf", "mean"),
            mean_time_aswmf_nlm=("time_aswmf_nlm", "mean"),
            wins_aswmf_nlm_vs_nlm=(
                "delta_score_aswmf_nlm_vs_nlm",
                lambda s: int((s > 0).sum()),
            ),
            wins_aswmf_nlm_vs_aswmf=(
                "delta_score_aswmf_nlm_vs_aswmf",
                lambda s: int((s > 0).sum()),
            ),
            wins_aswmf_nlm_vs_geonlm_hibrid=(
                "delta_score_aswmf_nlm_vs_geonlm_hibrid",
                lambda s: int((s > 0).sum()),
            ),
        )
        .reset_index()
    )
    summary.to_excel(summary_dir / "results_set12_aswmf_nlm_summary.xlsx", index=False)
    return summary


if __name__ == "__main__":
    all_records = []
    for level in selected_levels():
        all_records.extend(run_level(level))

    summary_dir = OUTPUT_BASE / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(all_records, summary_dir, "results_set12_aswmf_nlm_all.pkl")
    save_results_to_xlsx(all_records, summary_dir, "results_set12_aswmf_nlm_all.xlsx")

    summary = make_summary(all_records, summary_dir)
    print(summary.to_string(index=False))
