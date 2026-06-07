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


SOURCE_NLM = Path(
    "/workspace/data/output/set12/salt_pepper_low/full_512/results/"
    "array_nlm_salt_pepper_low_filtereds.pkl"
)
OLD_RESULTS = Path(
    "/workspace/data/output/set12Hibrid/salt_pepper_low/full_512/results/"
    "results_set12_hibrid_low.xlsx"
)
OUTPUT_ROOT = Path("/workspace/data/output/set12Hibrid/compare_low_f1_t3_nn7_h001")
IMAGE_DIR = OUTPUT_ROOT / "GeoHibNLM"
RESULTS_DIR = OUTPUT_ROOT / "results"

NEW_CONFIG = {
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


def run_new_config(item, out_path):
    reference = item["img_reference_np"]
    if out_path.exists():
        image = skimage.io.imread(str(out_path))
        psnr, ssim, method_score = metrics(reference, image)
        return psnr, ssim, method_score, np.nan

    start = time.time()
    filtered, _, psnr, ssim, method_score = run_geonlm_medians_pipeline(
        img_original=reference,
        h_base=float(item["nlm_h"]),
        img_noisy=item["img_noisy_salt_pepper_np"],
        f=NEW_CONFIG["f"],
        t=NEW_CONFIG["t"],
        mult=NEW_CONFIG["h_multiplier"],
        nn=NEW_CONFIG["nn"],
        switch_impulse_only=NEW_CONFIG["switch_impulse_only"],
        reject_impulse_candidates=NEW_CONFIG["reject_impulse_candidates"],
        use_aswmf_spatial_weights=NEW_CONFIG["use_aswmf_spatial_weights"],
        aswmf_weight_diag_1=NEW_CONFIG["aswmf_weight_diag_1"],
        aswmf_weight_diag_2=NEW_CONFIG["aswmf_weight_diag_2"],
        aswmf_weight_other=NEW_CONFIG["aswmf_weight_other"],
    )
    elapsed = time.time() - start
    save_uint8(out_path, filtered)
    return psnr, ssim, method_score, elapsed


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    vector = load_pickle(SOURCE_NLM.parent, SOURCE_NLM.name)
    old_df = pd.read_excel(OLD_RESULTS).set_index("file_name")
    records = []

    for item in vector:
        file_name = item["file_name"]
        old = old_df.loc[file_name]
        psnr_new, ssim_new, score_new, elapsed = run_new_config(
            item,
            IMAGE_DIR / file_name,
        )
        record = {
            "file_name": file_name,
            "old_f": int(old["hybrid_f"]),
            "old_t": int(old["hybrid_t"]),
            "old_nn": int(old["hybrid_nn"]),
            "old_h_multiplier": float(old["hybrid_h_multiplier"]),
            "old_psnr_geohibnlm": float(old["psnr_geonlm_hibrid"]),
            "old_ssim_geohibnlm": float(old["ssim_geonlm_hibrid"]),
            "old_score_geohibnlm": float(old["score_geonlm_hibrid"]),
            "new_f": NEW_CONFIG["f"],
            "new_t": NEW_CONFIG["t"],
            "new_nn": NEW_CONFIG["nn"],
            "new_h_multiplier": NEW_CONFIG["h_multiplier"],
            "new_h": float(item["nlm_h"]) * NEW_CONFIG["h_multiplier"],
            "new_psnr_geohibnlm": psnr_new,
            "new_ssim_geohibnlm": ssim_new,
            "new_score_geohibnlm": score_new,
            "time_new_geohibnlm": elapsed,
            "delta_psnr_new_minus_old": psnr_new - float(old["psnr_geonlm_hibrid"]),
            "delta_ssim_new_minus_old": ssim_new - float(old["ssim_geonlm_hibrid"]),
            "delta_score_new_minus_old": score_new - float(old["score_geonlm_hibrid"]),
            "score_nlm": float(old["score_nlm"]),
            "score_median": float(old["score_median"]),
            "score_nlmedians": float(old["score_nlmedians"]),
            "score_aswmf_original": float(old["score_aswmf_original"]),
        }
        records.append(record)
        print(
            f"{file_name}: old={record['old_score_geohibnlm']:.4f} "
            f"new={score_new:.4f} "
            f"delta={record['delta_score_new_minus_old']:+.4f} "
            f"time={elapsed if not np.isnan(elapsed) else 0:.2f}s",
            flush=True,
        )

    df = pd.DataFrame(records)
    summary = pd.DataFrame([
        {
            "n_images": len(df),
            "old_mean_psnr": df["old_psnr_geohibnlm"].mean(),
            "new_mean_psnr": df["new_psnr_geohibnlm"].mean(),
            "delta_mean_psnr": df["delta_psnr_new_minus_old"].mean(),
            "old_mean_ssim": df["old_ssim_geohibnlm"].mean(),
            "new_mean_ssim": df["new_ssim_geohibnlm"].mean(),
            "delta_mean_ssim": df["delta_ssim_new_minus_old"].mean(),
            "old_mean_score": df["old_score_geohibnlm"].mean(),
            "new_mean_score": df["new_score_geohibnlm"].mean(),
            "delta_mean_score": df["delta_score_new_minus_old"].mean(),
            "new_wins_vs_old": int((df["delta_score_new_minus_old"] > 0).sum()),
            "new_losses_vs_old": int((df["delta_score_new_minus_old"] < 0).sum()),
            "new_ties_vs_old": int((df["delta_score_new_minus_old"] == 0).sum()),
            "mean_time_new_geohibnlm": df["time_new_geohibnlm"].mean(),
        }
    ])

    save_pickle(records, RESULTS_DIR, "compare_set12_low_f1_t3_nn7_h001.pkl")
    save_results_to_xlsx(records, RESULTS_DIR, "compare_set12_low_f1_t3_nn7_h001.xlsx")
    save_results_to_xlsx(summary.to_dict("records"), RESULTS_DIR, "compare_set12_low_f1_t3_nn7_h001_summary.xlsx")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
