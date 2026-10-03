"""Set12 near-extreme impulse experiment (NLM h shared by IANLM/GHNLM).

This experiment deliberately does not run standard GEO-NLM.  It reads only
the clean references and experiment metadata from the legacy Set12 pickles;
all noisy and filtered images are generated again in an isolated output tree.
"""

import contextlib
import io
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "set12"))

import cupy as cp
import numpy as np
import pandas as pd
import skimage.io
from scipy.ndimage import median_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.restoration import estimate_sigma

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx
from functions.impulse_tolerance_filters import (
    aswmf_impulse_tolerance_filter,
    run_ghnlm_impulse_tolerance_pipeline,
    run_ianlm_impulse_tolerance_pipeline,
)
from functions.nlm_functions import (
    NLM_fast_cuda_global,
    compute_adaptive_q,
    select_best_h_using_adaptive_q,
)
from functions.noisy_functions import add_near_extreme_impulse_noise
from NLMedians import run_nlmedians


SOURCE_BASE = Path("/workspace/data/output/set12")
OUTPUT_BASE = Path("/workspace/data/output/set12ImpulseTolerance")
DATASET_NAME = "set12"
RUN_GHNLM = True
IANLM_FIXED_H = None
REUSE_SOURCE_NLM_H = False
RESOLUTION = "full_512"
LEVEL_DENSITIES = {
    "low": 0.01,
    "moderate": 0.03,
    "medium": 0.05,
    "high": 0.10,
    "extreme": 0.20,
}
DEFAULT_TOLERANCES = (0, 1, 2, 3, 4)
RANDOM_SEED = 42

NLM_CONFIG = {"f": 4, "t": 7, "alpha": 0.5}
NLMEDIANS_CONFIG = {"f": 2, "t": 3, "h_multiplier": 0.005}
HYBRID_CONFIG = {
    "f": 1,
    "t": 3,
    "nn": 7,
    "h_multiplier": 0.001,
    "switch_impulse_only": True,
    "reject_impulse_candidates": True,
    "use_aswmf_spatial_weights": True,
    "aswmf_weight_diag_1": 1.0,
    "aswmf_weight_diag_2": 1.0,
    "aswmf_weight_other": 10.0,
}
ASWMF_CONFIG = {
    "radius": 3,
    "weight_diag_1": 1.0,
    "weight_diag_2": 1.0,
    "weight_other": 10.0,
}


def selected_levels():
    raw = os.environ.get("SET12_TOLERANCE_LEVELS")
    if not raw:
        return list(LEVEL_DENSITIES)
    requested = {value.strip() for value in raw.split(",") if value.strip()}
    unknown = requested.difference(LEVEL_DENSITIES)
    if unknown:
        raise ValueError(f"Unknown levels: {sorted(unknown)}")
    return [level for level in LEVEL_DENSITIES if level in requested]


def selected_tolerances():
    raw = os.environ.get("SET12_TOLERANCES")
    values = DEFAULT_TOLERANCES if not raw else tuple(
        int(value.strip()) for value in raw.split(",") if value.strip()
    )
    if not values or any(value < 0 or value > 127 for value in values):
        raise ValueError("SET12_TOLERANCES must contain integers from 0 to 127.")
    return tuple(dict.fromkeys(values))


def max_images():
    value = os.environ.get("SET12_TOLERANCE_MAX_IMAGES")
    return None if not value else int(value)


def force_run():
    return os.environ.get("SET12_TOLERANCE_FORCE", "0") == "1"


def verbose_nlm():
    return os.environ.get("SET12_TOLERANCE_VERBOSE_NLM", "0") == "1"


def metrics(reference, image):
    reference = np.clip(reference, 0, 255).astype(np.uint8)
    image = np.clip(image, 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(reference, image, data_range=255)
    ssim = structural_similarity(reference, image, data_range=255)
    return psnr, ssim, 0.5 * psnr + 0.5 * (ssim * 100)


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    skimage.io.imsave(str(path), np.clip(image, 0, 255).astype(np.uint8))


def source_vector(level):
    results = SOURCE_BASE / f"salt_pepper_{level}" / RESOLUTION / "results"
    name = f"array_nlm_salt_pepper_{level}_filtereds.pkl"
    vector = load_pickle(results, name)
    limit = max_images()
    return vector if limit is None else vector[:limit]


def run_nlm(reference, noisy, h_override=None):
    sigma = float(estimate_sigma(noisy))
    if h_override is not None:
        h = float(h_override)
        filtered_gpu = NLM_fast_cuda_global(
            cp.asarray(noisy, dtype=cp.float32),
            h=h,
            f=NLM_CONFIG["f"],
            t=NLM_CONFIG["t"],
        )
        cp.cuda.Stream.null.synchronize()
        filtered = cp.asnumpy(filtered_gpu)
        filtered = np.nan_to_num(filtered, nan=0.0, posinf=255.0, neginf=0.0)
        psnr, ssim, method_score = metrics(reference, filtered)
        return filtered, h, psnr, ssim, method_score, sigma

    base = compute_adaptive_q(sigma)
    candidates = np.array([base + delta for delta in range(25, 125)])
    call = lambda: select_best_h_using_adaptive_q(
        image=reference,
        image_gpu=cp.asarray(noisy, dtype=cp.float32),
        q_nlm_candidates=candidates,
        f=NLM_CONFIG["f"],
        t=NLM_CONFIG["t"],
        alpha=NLM_CONFIG["alpha"],
    )
    if verbose_nlm():
        filtered, h, psnr, ssim, method_score = call()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            filtered, h, psnr, ssim, method_score = call()
    return filtered, float(h), float(psnr), float(ssim), float(method_score), sigma


def timed(function, *args, **kwargs):
    start = time.time()
    result = function(*args, **kwargs)
    return result, time.time() - start


def run_level(level, tolerance):
    density = LEVEL_DENSITIES[level]
    output_root = (
        OUTPUT_BASE / f"tolerance_{tolerance}" / f"salt_pepper_{level}" / RESOLUTION
    )
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_pickle = results_dir / f"results_{DATASET_NAME}_impulse_tolerance_{level}.pkl"
    vector = source_vector(level)
    if result_pickle.exists() and not force_run():
        cached = load_pickle(result_pickle.parent, result_pickle.name)
        if len(cached) == len(vector):
            return cached
        print(
            f"Ignoring incomplete cache for {level}: "
            f"{len(cached)} of {len(vector)} images.",
            flush=True,
        )

    records = []
    nlm_items = []
    for old_item in vector:
        file_name = old_item["file_name"]
        reference = np.asarray(old_item["img_reference_np"], dtype=np.float32)
        noisy, corruption_mask = add_near_extreme_impulse_noise(
            reference,
            salt_prob=density / 2,
            pepper_prob=density / 2,
            impulse_tolerance=tolerance,
            seed=RANDOM_SEED,
            return_mask=True,
        )
        noisy = np.asarray(noisy, dtype=np.float32)
        detected_mask = (noisy <= tolerance) | (noisy >= 255 - tolerance)
        true_positives = int(np.count_nonzero(detected_mask & corruption_mask))
        false_positives = int(np.count_nonzero(detected_mask & ~corruption_mask))
        false_negatives = int(np.count_nonzero(~detected_mask & corruption_mask))
        save_image(output_root / "NOISY" / file_name, noisy)
        mask_path = output_root / "MASK" / f"{Path(file_name).stem}.npy"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_path, corruption_mask)

        source_nlm_h = float(old_item["nlm_h"]) if REUSE_SOURCE_NLM_H else None
        (nlm, nlm_h, psnr_nlm, ssim_nlm, score_nlm, sigma), time_nlm = timed(
            run_nlm, reference, noisy, source_nlm_h
        )
        save_image(output_root / "NLM" / file_name, nlm)

        (median_result, time_median) = timed(
            median_filter, noisy, size=3, mode="reflect"
        )
        save_image(output_root / "MEDIAN" / file_name, median_result)
        psnr_median, ssim_median, score_median = metrics(reference, median_result)

        (nlmed_result, time_nlmedians) = timed(
            run_nlmedians,
            reference=reference,
            noisy=noisy,
            h=nlm_h * NLMEDIANS_CONFIG["h_multiplier"],
            f=NLMEDIANS_CONFIG["f"],
            t=NLMEDIANS_CONFIG["t"],
        )
        save_image(output_root / "NLMedians" / file_name, nlmed_result["filtered"])

        (aswmf, time_aswmf) = timed(
            aswmf_impulse_tolerance_filter,
            noisy,
            impulse_tolerance=tolerance,
            **ASWMF_CONFIG,
        )
        save_image(output_root / "ASWMF" / file_name, aswmf)
        psnr_aswmf, ssim_aswmf, score_aswmf = metrics(reference, aswmf)

        common = dict(
            img_original=reference,
            h_base=nlm_h,
            img_noisy=noisy,
            f=HYBRID_CONFIG["f"],
            t=HYBRID_CONFIG["t"],
            mult=HYBRID_CONFIG["h_multiplier"],
            impulse_tolerance=tolerance,
            switch_impulse_only=HYBRID_CONFIG["switch_impulse_only"],
            reject_impulse_candidates=HYBRID_CONFIG["reject_impulse_candidates"],
            use_aswmf_spatial_weights=HYBRID_CONFIG["use_aswmf_spatial_weights"],
            aswmf_weight_diag_1=HYBRID_CONFIG["aswmf_weight_diag_1"],
            aswmf_weight_diag_2=HYBRID_CONFIG["aswmf_weight_diag_2"],
            aswmf_weight_other=HYBRID_CONFIG["aswmf_weight_other"],
        )
        ianlm_common = common.copy()
        if IANLM_FIXED_H is not None:
            ianlm_common["h_base"] = float(IANLM_FIXED_H)
            ianlm_common["mult"] = 1.0
        (ianlm_result, time_ianlm) = timed(
            run_ianlm_impulse_tolerance_pipeline, **ianlm_common
        )
        ianlm, h_ianlm, psnr_ianlm, ssim_ianlm, score_ianlm = ianlm_result
        save_image(output_root / "IANLM" / file_name, ianlm)

        if RUN_GHNLM:
            (ghnlm_result, time_ghnlm) = timed(
                run_ghnlm_impulse_tolerance_pipeline,
                nn=HYBRID_CONFIG["nn"],
                **common,
            )
            ghnlm, h_ghnlm, psnr_ghnlm, ssim_ghnlm, score_ghnlm = ghnlm_result
            save_image(output_root / "GEONLMHibrid" / file_name, ghnlm)

        psnr_noisy, ssim_noisy, score_noisy = metrics(reference, noisy)
        record = {
            "level": level,
            "file_name": file_name,
            "impulse_tolerance": tolerance,
            "pepper_min": 0,
            "pepper_max": tolerance,
            "salt_min": 255 - tolerance,
            "salt_max": 255,
            "random_seed": RANDOM_SEED,
            "noise_sampling": "paired_scaled_uniform_v1",
            "salt_pepper_density": density,
            "corrupted_pixels": int(corruption_mask.sum()),
            "detected_impulse_pixels": int(detected_mask.sum()),
            "detector_true_positives": true_positives,
            "detector_false_positives": false_positives,
            "detector_false_negatives": false_negatives,
            "detector_precision": true_positives / max(1, true_positives + false_positives),
            "detector_recall": true_positives / max(1, true_positives + false_negatives),
            "estimated_sigma_salt_pepper": sigma,
            "nlm_h": nlm_h,
            "nlm_h_source": (
                "reused_source_pickle" if REUSE_SOURCE_NLM_H else "current_noise_sweep"
            ),
            "h_ianlm": h_ianlm,
            "h_ianlm_source": "fixed_independent" if IANLM_FIXED_H is not None else "nlm_scaled",
            "psnr_noisy": psnr_noisy, "ssim_noisy": ssim_noisy, "score_noisy": score_noisy,
            "psnr_nlm": psnr_nlm, "ssim_nlm": ssim_nlm, "score_nlm": score_nlm,
            "psnr_median": psnr_median, "ssim_median": ssim_median, "score_median": score_median,
            "psnr_nlmedians": nlmed_result["psnr"], "ssim_nlmedians": nlmed_result["ssim"], "score_nlmedians": nlmed_result["score"],
            "psnr_aswmf": psnr_aswmf, "ssim_aswmf": ssim_aswmf, "score_aswmf": score_aswmf,
            "psnr_ianlm": psnr_ianlm, "ssim_ianlm": ssim_ianlm, "score_ianlm": score_ianlm,
            "time_nlm": time_nlm, "time_median": time_median,
            "time_nlmedians": time_nlmedians, "time_aswmf": time_aswmf,
            "time_ianlm": time_ianlm,
        }
        if RUN_GHNLM:
            record.update({
                "h_ghnlm": h_ghnlm,
                "psnr_ghnlm": psnr_ghnlm,
                "ssim_ghnlm": ssim_ghnlm,
                "score_ghnlm": score_ghnlm,
                "time_ghnlm": time_ghnlm,
            })
        records.append(record)
        nlm_items.append({
            "file_name": file_name,
            "img_reference_np": reference,
            "img_noisy_salt_pepper_np": noisy,
            "corruption_mask": corruption_mask,
            "img_filtered_nlm": nlm,
            "nlm_h": nlm_h,
            "estimated_sigma_salt_pepper": sigma,
            "salt_pepper_density": density,
            "salt_prob": density / 2,
            "pepper_prob": density / 2,
            "impulse_tolerance": tolerance,
            "random_seed": RANDOM_SEED,
        })
        status = (
            f"{level} {file_name}: NLM={score_nlm:.4f} ASWMF={score_aswmf:.4f} "
            f"IANLM={score_ianlm:.4f}"
        )
        if RUN_GHNLM:
            status += f" GHNLM={score_ghnlm:.4f}"
        print(status, flush=True)

    save_pickle(nlm_items, results_dir, f"array_nlm_impulse_tolerance_{level}.pkl")
    save_pickle(records, results_dir, result_pickle.name)
    save_results_to_xlsx(records, results_dir, result_pickle.with_suffix(".xlsx").name)
    return records


def save_summary(records, tolerance):
    summary_dir = OUTPUT_BASE / f"tolerance_{tolerance}" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    metric_columns = [
        column for column in df.columns
        if column.startswith(("psnr_", "ssim_", "score_", "time_"))
    ]
    summary = df.groupby("level", sort=False)[metric_columns].mean().reset_index()
    save_pickle(records, summary_dir, f"results_{DATASET_NAME}_impulse_tolerance_all.pkl")
    save_results_to_xlsx(records, summary_dir, f"results_{DATASET_NAME}_impulse_tolerance_all.xlsx")
    summary.to_excel(summary_dir / f"results_{DATASET_NAME}_impulse_tolerance_summary.xlsx", index=False)
    return summary


def save_combined_summary(records):
    summary_dir = OUTPUT_BASE / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    metric_columns = [
        column for column in df.columns
        if column.startswith(("psnr_", "ssim_", "score_", "time_"))
    ]
    detector_columns = [
        "detector_precision", "detector_recall", "detector_false_positives",
        "detector_false_negatives",
    ]
    summary = (
        df.groupby(["impulse_tolerance", "level"], sort=False)[
            metric_columns + detector_columns
        ]
        .mean()
        .reset_index()
    )
    save_pickle(records, summary_dir, f"results_{DATASET_NAME}_tolerance_sweep_all.pkl")
    save_results_to_xlsx(records, summary_dir, f"results_{DATASET_NAME}_tolerance_sweep_all.xlsx")
    summary.to_excel(summary_dir / f"results_{DATASET_NAME}_tolerance_sweep_summary.xlsx", index=False)
    return summary


def run_experiment():
    all_records = []
    for tolerance in selected_tolerances():
        tolerance_records = []
        for selected_level in selected_levels():
            tolerance_records.extend(run_level(selected_level, tolerance))
        save_summary(tolerance_records, tolerance)
        all_records.extend(tolerance_records)
    print(save_combined_summary(all_records).to_string(index=False))


if __name__ == "__main__":
    run_experiment()
