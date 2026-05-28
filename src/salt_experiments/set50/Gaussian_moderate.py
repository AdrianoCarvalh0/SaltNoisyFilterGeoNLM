import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from skimage.restoration import estimate_sigma
import numpy as np
import cupy as cp
import skimage
from functions.Utils import (read_directories, save_pickle, save_results_to_xlsx, load_pickle, get_multiplier)
from functions.noisy_functions import add_moderate_noise_gaussian
from functions.nlm_functions import (compute_adaptive_q, select_best_h_using_adaptive_q)
from functions.geonlm_functions import run_geonlm_pipeline
import time
from bm3d import bm3d, BM3DProfile
from skimage.restoration import estimate_sigma
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def generate_gaussian_experiment_moderate(parameters):
    """
    Run the moderate-noise Gaussian denoising experiment using NLM, GEO-NLM, and BM3D.
    
    The function:
      1. Reads all images in dir_images_general.
      2. Adds moderate-level Gaussian noise.
      3. Estimates sigma and computes an adaptive NLM parameter h.
      4. Runs NLM and stores intermediate results in a pickle file.
      5. Reloads those results to run GEO-NLM and BM3D.
      6. Computes PSNR, SSIM, and a custom score for each method.
      7. Saves filtered images and metrics (pickle + XLSX).
    """

    # Unpack configuration parameters
    root_dir_output_moderate = parameters['root_dir_output_moderate']
    dir_images_general = parameters['dir_images_general']
    dir_out_nlm = parameters['dir_out_nlm']
    dir_out_geonlm = parameters['dir_out_geonlm']
    dir_out_bm3d = parameters['dir_out_bm3d']
    dir_out_results = parameters['dir_out_results']
    name_pickle_nlm_output_moderate = parameters['name_pickle_nlm_output_moderate']
    name_pickle_results_gnlm_bm3d_output_moderate = parameters['name_pickle_results_gnlm_bm3d_output_moderate']
    name_results_xlsx_nlm_gnlm_bm3d_output_moderate = parameters['name_results_xlsx_nlm_gnlm_bm3d_output_moderate']
    f = parameters['f']        # Patch radius (NLM / GEO-NLM)
    t = parameters['t']        # Search window radius (NLM / GEO-NLM)
    alpha = parameters['alpha']  # Weight parameter for adaptive score / geometry

    # List all input image filenames in the general image directory
    array_dir = read_directories(dir_images_general)

    # Will store intermediate NLM results for all images
    array_nln_moderate_filtereds = []

    # --------------------------
    # 1) NLM PHASE (MODERATE NOISE)
    # --------------------------
    for file in array_dir:

        file_name = file

        # Read image from disk
        img = skimage.io.imread(f'{dir_images_general}/{file_name}')

        # If the image has 4 dimensions (e.g. multi-page TIFF), use only the first slice
        if img.ndim == 4:
            img = img[0]

        # If the image has an alpha channel (RGBA), discard the alpha and keep RGB
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img[..., :3]

        # Convert RGB to grayscale if the image is color
        if img.ndim == 3 and img.shape[-1] == 3:
            img = skimage.color.rgb2gray(img)

        # Ensure the image is in [0, 255] as float32
        if img.dtype.kind == 'f':
            # If already float, clip to [0,1] then scale to [0,255]
            img = (np.clip(img, 0, 1) * 255).astype(np.float32)
        else:
            # If integer, just cast to float32
            img = img.astype(np.float32)

        m, n = img.shape  # Image dimensions (not used later, but kept for clarity)

        # Add moderate Gaussian noise to create the noisy observation
        noised = add_moderate_noise_gaussian(img)

        # Clip noisy image to valid intensity range [0, 255]
        noised = np.clip(noised, 0, 255)

        # --- CPU/GPU data preparation ---
        img_noisse_gaussian_np = noised.astype(np.float32)

        # Estimate noise level (sigma) from the noisy image in [0,255]
        estimated_sigma_gaussian_np = estimate_sigma(img_noisse_gaussian_np)

        # Copy noisy image to GPU (CuPy) for NLM / GEO-NLM computations
        img_gpu_noisse_gaussian = cp.array(img_noisse_gaussian_np)

        # Compute base NLM parameter h using the adaptive function
        h_nlm = compute_adaptive_q(estimated_sigma_gaussian_np)

        # Define a range of candidate h values around h_nlm
        q_nlm_candidates = np.array([h_nlm + delta for delta in range(-50, 50, 1)])

        # IMPORTANT: reference image must also be in [0,255]
        img_filtered_nlm, nlm_h, psnr_nlm, ssim_nlm, score_nlm = select_best_h_using_adaptive_q(
            image=img,                          # Reference clean image
            image_gpu=img_gpu_noisse_gaussian,  # Noisy image on GPU
            q_nlm_candidates=q_nlm_candidates,  # Candidate h values
            f=f,
            t=t,
            alpha=alpha,
        )

        # Store all relevant intermediate results for this image
        dct = {
            'img_noisse_gaussian_np': img_noisse_gaussian_np,
            'img_filtered_nlm': img_filtered_nlm,
            'estimated_sigma_gaussian': estimated_sigma_gaussian_np,
            'nlm_h': nlm_h,
            'psnr_nlm': psnr_nlm,
            'ssim_nlm': ssim_nlm,
            'score_nlm': score_nlm,
            'file_name': file_name,
        }
        array_nln_moderate_filtereds.append(dct)

        # Save NLM-filtered image to disk
        skimage.io.imsave(
            f'{dir_out_nlm}/{file_name}',
            np.clip(img_filtered_nlm, 0, 255).astype(np.uint8)
        )

    # Save all NLM results (for all images) to a pickle file
    save_pickle(array_nln_moderate_filtereds, dir_out_results, name_pickle_nlm_output_moderate)

    # ---------------------------------------
    # 2) BM3D AND GEO-NLM PHASE (MODERATE NOISE)
    # ---------------------------------------

    array_gnlm_bm3d_moderate_filtereds = []

    # Reload the NLM results to use them as input for GEO-NLM and BM3D
    vector = load_pickle(dir_out_results, name_pickle_nlm_output_moderate)

    for array in vector:
        # Retrieve data saved during the NLM phase
        img_noisse_gaussian_np = array['img_noisse_gaussian_np']
        nlm_h = array['nlm_h']
        file_name = array['file_name']
        psnr_nlm = array['psnr_nlm']
        ssim_nlm = array['ssim_nlm']
        score_nlm = array['score_nlm']
        estimated_sigma_gaussian = array['estimated_sigma_gaussian']

        # Reload the original (clean) image from disk
        img = skimage.io.imread(f'{dir_images_general}/{file_name}')
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

        # Reuse parameters for GEO-NLM
        f = parameters['f']
        t = parameters['t']
        nn = parameters['nn']  # Number of nearest neighbors / similar patches in GEO-NLM

        # ------------------
        # GEO-NLM DENOISING
        # ------------------

        ini = time.time()
        # Select multiplier (h scaling) based on NLM h and estimated sigma
        mult = get_multiplier(nlm_h, estimated_sigma_gaussian)

        # Run the GEO-NLM pipeline on the image
        img_filtered_gnlm, h_gnlm, psnr_gnlm, ssim_gnlm, score_gnlm = run_geonlm_pipeline(
            img,
            nlm_h,
            img_noisse_gaussian_np,
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
        time_geonlm = end - ini  # GEO-NLM execution time (not stored, but kept here if needed)

        # -----------
        # BM3D STEP
        # -----------

        img_bm3d = img.copy()

        # If image has more than 2 dimensions, convert to grayscale and scale to [0,255]
        if len(img_bm3d.shape) > 2:
            img_bm3d = skimage.color.rgb2gray(img_bm3d)
            img_bm3d = 255 * img_bm3d

        # Handle potential extra dimensions (e.g., multi-page or singleton channels)
        if img_bm3d.ndim == 4:
            img_bm3d = img_bm3d[0]
        elif img_bm3d.ndim == 3 and img_bm3d.shape[2] != 3:
            img_bm3d = np.squeeze(img_bm3d)

        # Ensure we have a grayscale image in [0,1] before BM3D
        if img_bm3d.ndim == 3 and img_bm3d.shape[2] == 3:
            img_bm3d_gray = rgb2gray(img_bm3d)  # Returns float64 in [0,1]
        else:
            img_bm3d_gray = img_bm3d.astype(np.float32) / 255.0

        # Convert back to uint8 [0,255] for metric computation consistency
        img_bm3d_gray = np.clip(img_bm3d_gray * 255, 0, 255).astype(np.uint8)

        # Normalize the noisy image to [0,1] for BM3D input
        normalized_noizy = img_noisse_gaussian_np.astype(np.float32) / 255.0

        # Estimate noise level sigma for BM3D
        sigma_est = estimate_sigma(normalized_noizy, channel_axis=None)

        # BM3D profile and execution       
        perfil_bm3d = BM3DProfile()    
        

        # Apply BM3D denoising
        denoised = bm3d(
            normalized_noizy,
            sigma_psd=sigma_est,
            profile=perfil_bm3d
        )

        # Remove singleton dimensions if needed
        denoised_sq = np.squeeze(denoised)

        # Save BM3D-filtered image to disk (convert back to [0,255] uint8)
        skimage.io.imsave(
            f'{dir_out_bm3d}/{file_name}',
            np.clip(denoised_sq * 255, 0, 255).astype(np.uint8)
        )

        # Compute PSNR and SSIM between clean image and BM3D output
        psnr_bm3d = peak_signal_noise_ratio(
            img_bm3d_gray,
            (denoised * 255).astype(np.uint8)
        )
        ssim_bm3d = structural_similarity(
            img_bm3d_gray,
            (denoised * 255).astype(np.uint8)
        )

        # Custom score combining PSNR and SSIM (SSIM scaled to [0,100])
        score_bm3d = 0.5 * psnr_bm3d + 0.5 * (ssim_bm3d * 100)

        # Collect all metrics for this image
        dict = {
            'nlm_h': nlm_h,
            'h_gnlm': h_gnlm,
            'estimated_sigma_gaussian': estimated_sigma_gaussian,

            'ssim_nlm': ssim_nlm,
            'ssim_gnlm': ssim_gnlm,
            'ssim_bm3d': ssim_bm3d,

            'psnr_nlm': psnr_nlm,
            'psnr_gnlm': psnr_gnlm,
            'psnr_bm3d': psnr_bm3d,

            'score_nlm': score_nlm,
            'score_gnlm': score_gnlm,
            'score_bm3d': score_bm3d,
            'time_geonlm': time_geonlm,

            'file_name': file_name,
        }

        array_gnlm_bm3d_moderate_filtereds.append(dict)

    # Save the combined GEO-NLM and BM3D results to pickle
    save_pickle(array_gnlm_bm3d_moderate_filtereds, dir_out_results, name_pickle_results_gnlm_bm3d_output_moderate)

    # Export all results (NLM, GEO-NLM, BM3D) to an XLSX spreadsheet
    save_results_to_xlsx(
        array_gnlm_bm3d_moderate_filtereds,
        dir_out_results,
        name_results_xlsx_nlm_gnlm_bm3d_output_moderate
    )
