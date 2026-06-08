import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import contextlib
import io
from skimage.restoration import estimate_sigma
import numpy as np
import cupy as cp
import skimage
from functions.Utils import (read_directories, save_pickle, save_results_to_xlsx, load_pickle, get_multiplier)
from functions.noisy_functions import add_salt_pepper_noise
from functions.nlm_functions import (compute_adaptive_q, select_best_h_using_adaptive_q)
from functions.geonlm_functions import run_geonlm_pipeline
from functions.salt_filters import aswmf_filter
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.transform import resize
from scipy.ndimage import median_filter


def _read_grayscale_float255(path, resize_shape=None):
    img = skimage.io.imread(path)

    if img.ndim == 4:
        img = img[0]

    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]

    if img.ndim == 3 and img.shape[-1] == 3:
        img = skimage.color.rgb2gray(img)

    if img.dtype.kind == 'f':
        img = (np.clip(img, 0, 1) * 255).astype(np.float32)
    else:
        img = img.astype(np.float32)

    if resize_shape is not None:
        img = resize(
            img,
            resize_shape,
            preserve_range=True,
            anti_aliasing=True
        ).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.float32)


def _print_image_metrics(results):
    print(f"\nImage: {results['file_name']}")
    print(f"  Noisy  PSNR={results['psnr_noisy']:.4f} SSIM={results['ssim_noisy']:.4f}")
    print(f"  NLM    PSNR={results['psnr_nlm']:.4f} SSIM={results['ssim_nlm']:.4f}")
    print(f"  GEONLM PSNR={results['psnr_gnlm']:.4f} SSIM={results['ssim_gnlm']:.4f}")
    print(f"  Median PSNR={results['psnr_median']:.4f} SSIM={results['ssim_median']:.4f}")
    print(f"  ASWMF  PSNR={results['psnr_aswmf']:.4f} SSIM={results['ssim_aswmf']:.4f}")


def _call_with_optional_stdout(func, verbose, *args, **kwargs):
    if verbose:
        return func(*args, **kwargs)

    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def generate_salt_experiment_low(parameters):
    """
    Run the low-noise salt-and-pepper denoising experiment using NLM, GEO-NLM,
    median filtering, and ASWMF.
    
    The function:
      1. Reads all images in dir_images_general.
      2. Adds low-level salt-and-pepper noise.
      3. Estimates the noise level and computes an adaptive NLM parameter h.
      4. Runs NLM and stores intermediate results in a pickle file.
      5. Reloads those results to run GEO-NLM, median filtering, and ASWMF.
      6. Computes PSNR, SSIM, and a custom score for each method.
      7. Saves filtered images and metrics (pickle + XLSX).
    """

    # Unpack configuration parameters
    root_dir_output_low = parameters['root_dir_output_low']
    dir_images_general = parameters['dir_images_set12']
    dir_out_nlm = parameters['dir_out_nlm']
    dir_out_geonlm = parameters['dir_out_geonlm']
    dir_out_median = parameters['dir_out_median']
    dir_out_aswmf = parameters['dir_out_aswmf']
    dir_out_results = parameters['dir_out_results']
    name_pickle_nlm_output_low = parameters['name_pickle_nlm_output_low']
    name_pickle_results_gnlm_median_aswmf_output_low = parameters['name_pickle_results_gnlm_median_aswmf_output_low']
    name_results_xlsx_nlm_gnlm_median_aswmf_output_low = parameters['name_results_xlsx_nlm_gnlm_median_aswmf_output_low']

    f = parameters['f']          # Patch radius for NLM / GEO-NLM
    t = parameters['t']          # Search window radius for NLM / GEO-NLM
    alpha = parameters['alpha']  # Weight parameter for adaptive score / geometry

    # Salt-and-pepper total noise density.
    # Example: 0.01 means 1% of corrupted pixels in total:
    # 0.5% salt and 0.5% pepper.
    salt_pepper_density = parameters.get('salt_pepper_density', 0.01)

    salt_prob = salt_pepper_density / 2
    pepper_prob = salt_pepper_density / 2
    median_size = parameters.get('median_size', 3)
    aswmf_radius = parameters.get('aswmf_radius', 3)
    aswmf_weight_diag_1 = parameters.get('aswmf_weight_diag_1', 1.0)
    aswmf_weight_diag_2 = parameters.get('aswmf_weight_diag_2', 1.0)
    aswmf_weight_other = parameters.get('aswmf_weight_other', 10.0)
    resize_shape = parameters.get('resize_shape')
    print_metrics = parameters.get('print_metrics', False)
    verbose_internal = parameters.get('verbose_internal', False)

    # List all input image filenames in the general image directory
    array_dir = sorted(read_directories(dir_images_general))

    # Will store intermediate NLM results for all images
    array_nlm_low_filtereds = []

    # ---------------------------------------
    # 1) NLM PHASE — LOW SALT-AND-PEPPER NOISE
    # ---------------------------------------
    for file in array_dir:

        file_name = file

        # Read image from disk, convert to grayscale [0, 255], and resize if requested.
        img = _read_grayscale_float255(
            f'{dir_images_general}/{file_name}',
            resize_shape=resize_shape
        )

        m, n = img.shape  # Image dimensions

        # Add low salt-and-pepper noise to create the noisy observation
        noised = add_salt_pepper_noise(
            img,
            salt_prob=salt_prob,
            pepper_prob=pepper_prob,
            seed=42
        )

        # Clip noisy image to valid intensity range [0, 255]
        noised = np.clip(noised, 0, 255)

        # --- CPU/GPU data preparation ---
        img_noisy_salt_pepper_np = noised.astype(np.float32)

        # Estimate noise level from the salt-and-pepper noisy image in [0, 255]
        estimated_sigma_salt_pepper_np = estimate_sigma(img_noisy_salt_pepper_np)

        # Copy noisy image to GPU for NLM / GEO-NLM computations
        img_gpu_noisy_salt_pepper = cp.array(img_noisy_salt_pepper_np)

        # Compute base NLM parameter h using the adaptive function
        h_nlm = compute_adaptive_q(estimated_sigma_salt_pepper_np)

        # Define positive candidate h values around h_nlm.
        # h <= 0 makes the NLM exponential ill-defined and can produce NaNs.
        q_nlm_candidates = np.array([h_nlm + delta for delta in range(25, 125, 1)])

        # IMPORTANT: reference image must also be in [0, 255]
        img_filtered_nlm, nlm_h, psnr_nlm, ssim_nlm, score_nlm = _call_with_optional_stdout(
            select_best_h_using_adaptive_q,
            verbose_internal,
            image=img,                              # Reference image
            image_gpu=img_gpu_noisy_salt_pepper,    # Salt-and-pepper noisy image on GPU
            q_nlm_candidates=q_nlm_candidates,      # Candidate h values
            f=f,
            t=t,
            alpha=alpha,
        )

        # Store all relevant intermediate results for this image
        dct = {
            'img_noisy_salt_pepper_np': img_noisy_salt_pepper_np,
            'img_filtered_nlm': img_filtered_nlm,
            'img_reference_np': img,
            'estimated_sigma_salt_pepper': estimated_sigma_salt_pepper_np,
            'salt_pepper_density': salt_pepper_density,
            'salt_prob': salt_prob,
            'pepper_prob': pepper_prob,
            'nlm_h': nlm_h,
            'psnr_nlm': psnr_nlm,
            'ssim_nlm': ssim_nlm,
            'score_nlm': score_nlm,
            'file_name': file_name,
        }

        array_nlm_low_filtereds.append(dct)

        # Save NLM-filtered image to disk
        skimage.io.imsave(
            f'{dir_out_nlm}/{file_name}',
            np.clip(img_filtered_nlm, 0, 255).astype(np.uint8)
        )

    # Save all NLM results to a pickle file
    save_pickle(array_nlm_low_filtereds, dir_out_results, name_pickle_nlm_output_low)

    # --------------------------------------------------
    # 2) MEDIAN, ASWMF, AND GEO-NLM PHASE — LOW SALT-AND-PEPPER NOISE
    # --------------------------------------------------

    array_gnlm_median_aswmf_low_filtereds = []

    # Reload the NLM results to use them as input for GEO-NLM, median, and ASWMF
    vector = load_pickle(dir_out_results, name_pickle_nlm_output_low)

    for array in vector:

        # Retrieve data saved during the NLM phase
        img = array['img_reference_np']
        img_noisy_salt_pepper_np = array['img_noisy_salt_pepper_np']
        nlm_h = array['nlm_h']
        file_name = array['file_name']
        psnr_nlm = array['psnr_nlm']
        ssim_nlm = array['ssim_nlm']
        score_nlm = array['score_nlm']
        estimated_sigma_salt_pepper = array['estimated_sigma_salt_pepper']
        salt_pepper_density = array['salt_pepper_density']
        salt_prob = array['salt_prob']
        pepper_prob = array['pepper_prob']

        # Reuse parameters for GEO-NLM
        f = parameters['f']
        t = parameters['t']
        nn = parameters['nn']  # Number of nearest neighbors / similar patches in GEO-NLM

        # ------------------
        # GEO-NLM DENOISING
        # ------------------

        ini = time.time()

        # Select multiplier based on NLM h and estimated salt-and-pepper noise level
        mult = get_multiplier(nlm_h, estimated_sigma_salt_pepper)

        # Run the GEO-NLM pipeline on the salt-and-pepper noisy image
        img_filtered_gnlm, h_gnlm, psnr_gnlm, ssim_gnlm, score_gnlm = _call_with_optional_stdout(
            run_geonlm_pipeline,
            verbose_internal,
            img,
            nlm_h,
            img_noisy_salt_pepper_np,
            f,
            t,
            mult,
            nn
        )

        # Save GEO-NLM-filtered image
        skimage.io.imsave(
            f'{dir_out_geonlm}/{file_name}',
            img_filtered_gnlm.astype(np.uint8)
        )

        end = time.time()
        time_geonlm = end - ini

        # ------------
        # MEDIAN STEP
        # ------------

        ini = time.time()

        img_filtered_median = median_filter(
            img_noisy_salt_pepper_np,
            size=median_size,
            mode='reflect'
        )
        img_filtered_median_uint8 = np.clip(
            img_filtered_median, 0, 255
        ).astype(np.uint8)

        skimage.io.imsave(
            f'{dir_out_median}/{file_name}',
            img_filtered_median_uint8
        )

        img_reference_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        img_noisy_uint8 = np.clip(img_noisy_salt_pepper_np, 0, 255).astype(np.uint8)

        psnr_noisy = peak_signal_noise_ratio(
            img_reference_uint8,
            img_noisy_uint8
        )

        ssim_noisy = structural_similarity(
            img_reference_uint8,
            img_noisy_uint8
        )

        psnr_median = peak_signal_noise_ratio(
            img_reference_uint8,
            img_filtered_median_uint8
        )

        ssim_median = structural_similarity(
            img_reference_uint8,
            img_filtered_median_uint8
        )

        score_median = 0.5 * psnr_median + 0.5 * (ssim_median * 100)
        time_median = time.time() - ini

        # ----------
        # ASWMF STEP
        # ----------

        ini = time.time()

        img_filtered_aswmf = aswmf_filter(
            img_noisy_salt_pepper_np.astype(np.float32),
            radius=aswmf_radius,
            weight_diag_1=aswmf_weight_diag_1,
            weight_diag_2=aswmf_weight_diag_2,
            weight_other=aswmf_weight_other
        )
        img_filtered_aswmf_uint8 = np.clip(
            img_filtered_aswmf, 0, 255
        ).astype(np.uint8)

        skimage.io.imsave(
            f'{dir_out_aswmf}/{file_name}',
            img_filtered_aswmf_uint8
        )

        psnr_aswmf = peak_signal_noise_ratio(
            img_reference_uint8,
            img_filtered_aswmf_uint8
        )

        ssim_aswmf = structural_similarity(
            img_reference_uint8,
            img_filtered_aswmf_uint8
        )

        score_aswmf = 0.5 * psnr_aswmf + 0.5 * (ssim_aswmf * 100)
        time_aswmf = time.time() - ini

        # Collect all metrics for this image
        dict_results = {
            'nlm_h': nlm_h,
            'h_gnlm': h_gnlm,
            'estimated_sigma_salt_pepper': estimated_sigma_salt_pepper,
            'salt_pepper_density': salt_pepper_density,
            'salt_prob': salt_prob,
            'pepper_prob': pepper_prob,
            'median_size': median_size,
            'aswmf_radius': aswmf_radius,
            'aswmf_weight_diag_1': aswmf_weight_diag_1,
            'aswmf_weight_diag_2': aswmf_weight_diag_2,
            'aswmf_weight_other': aswmf_weight_other,
            'resize_shape': resize_shape,

            'ssim_noisy': ssim_noisy,
            'ssim_nlm': ssim_nlm,
            'ssim_gnlm': ssim_gnlm,
            'ssim_median': ssim_median,
            'ssim_aswmf': ssim_aswmf,

            'psnr_noisy': psnr_noisy,
            'psnr_nlm': psnr_nlm,
            'psnr_gnlm': psnr_gnlm,
            'psnr_median': psnr_median,
            'psnr_aswmf': psnr_aswmf,

            'score_nlm': score_nlm,
            'score_gnlm': score_gnlm,
            'score_median': score_median,
            'score_aswmf': score_aswmf,
            'time_geonlm': time_geonlm,
            'time_median': time_median,
            'time_aswmf': time_aswmf,

            'file_name': file_name,
        }

        if print_metrics:
            _print_image_metrics(dict_results)

        array_gnlm_median_aswmf_low_filtereds.append(dict_results)

    # Save the combined GEO-NLM, median, and ASWMF results to pickle
    save_pickle(
        array_gnlm_median_aswmf_low_filtereds,
        dir_out_results,
        name_pickle_results_gnlm_median_aswmf_output_low
    )

    # Export all results to an XLSX spreadsheet
    save_results_to_xlsx(
        array_gnlm_median_aswmf_low_filtereds,
        dir_out_results,
        name_results_xlsx_nlm_gnlm_median_aswmf_output_low
    )
