import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import time
import numpy as np
import skimage.io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.Utils import load_pickle, save_pickle, save_results_to_xlsx
from functions.adaptive_region_nlm_functions import run_adaptive_region_nlm


def _score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


def _metrics(reference, image):
    reference_uint8 = np.clip(reference, 0, 255).astype(np.uint8)
    image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    psnr = peak_signal_noise_ratio(reference_uint8, image_uint8, data_range=255)
    ssim = structural_similarity(reference_uint8, image_uint8, data_range=255)
    return psnr, ssim, _score(psnr, ssim)


if __name__ == "__main__":
    base_output = Path("/workspace/data/output/set12/salt_pepper_low/test_256")
    results_dir = base_output / "results"
    adaptive_dir = base_output / "AdaptiveRegionNLM"
    adaptive_dir.mkdir(parents=True, exist_ok=True)

    vector = load_pickle(results_dir, "array_nlm_salt_pepper_low_filtereds.pkl")

    records = []
    for item in vector:
        file_name = item["file_name"]
        reference = item["img_reference_np"]
        noisy = item["img_noisy_salt_pepper_np"]
        nlm = item["img_filtered_nlm"]
        h_base = float(item["nlm_h"])

        start = time.time()
        result = run_adaptive_region_nlm(
            reference=reference,
            noisy=noisy,
            predenoised=nlm,
            h_base=h_base,
            f_small=4,
            t_small=7,
            f_large=4,
            t_large=9,
            n_clusters=3,
            h_ifd_strength=0.0,
            require_same_label=False,
            random_state=0,
        )
        elapsed = time.time() - start

        psnr_noisy, ssim_noisy, score_noisy = _metrics(reference, noisy)
        psnr_nlm, ssim_nlm, score_nlm = _metrics(reference, nlm)

        out_path = adaptive_dir / file_name
        skimage.io.imsave(str(out_path), result["filtered"])

        record = {
            "file_name": file_name,
            "method": "AdaptiveRegionNLM",
            "source_pickle": str(results_dir / "array_nlm_salt_pepper_low_filtereds.pkl"),
            "params": (
                "test_256_calibrated:f_small=4,t_small=7,"
                "f_large=4,t_large=9,K=3,h_ifd_strength=0.0,"
                "require_same_label=False"
            ),
            "nlm_h": h_base,
            "estimated_sigma_salt_pepper": float(item["estimated_sigma_salt_pepper"]),
            "salt_pepper_density": float(item["salt_pepper_density"]),
            "psnr_noisy": psnr_noisy,
            "ssim_noisy": ssim_noisy,
            "score_noisy": score_noisy,
            "psnr_nlm": psnr_nlm,
            "ssim_nlm": ssim_nlm,
            "score_nlm": score_nlm,
            "psnr_adaptive_region_nlm": result["psnr"],
            "ssim_adaptive_region_nlm": result["ssim"],
            "score_adaptive_region_nlm": result["score"],
            "edge_ratio": result["edge_ratio"],
            "h_min": result["h_min"],
            "h_max": result["h_max"],
            "h_mean": result["h_mean"],
            "time_adaptive_region_nlm": elapsed,
        }
        records.append(record)

        print(
            f"{file_name}: "
            f"NLM PSNR={psnr_nlm:.4f} SSIM={ssim_nlm:.4f} | "
            f"AdaptiveRegionNLM PSNR={result['psnr']:.4f} SSIM={result['ssim']:.4f} | "
            f"time={elapsed:.2f}s"
        )

    save_pickle(records, results_dir, "adaptive_region_nlm_low_test_results.pkl")
    save_results_to_xlsx(records, results_dir, "adaptive_region_nlm_low_test_results.xlsx")
