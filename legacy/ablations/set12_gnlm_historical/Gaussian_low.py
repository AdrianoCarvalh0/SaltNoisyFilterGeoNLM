import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from skimage.restoration import estimate_sigma
import numpy as np
import skimage
from Utils import (read_directories, save_pickle, save_results_to_xlsx, load_pickle, is_low_noise_or, get_multiplier)
from noisy_functions import (add_low_noise_gaussian, add_moderate_noise_gaussian, add_high_noise_gaussian)
from nlm_functions import (compute_adaptive_q, select_best_h_using_adaptive_q)
from geonlm_functions import run_geonlm_pipeline
import time
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def generate_gaussian_experiment_low(parameters):
    """
    Run the low-noise Gaussian denoising experiment using NLM, GEO-NLM, and BM3D.
    
    The function:
      1. Reads all images in dir_images_general.
      2. Adds low-level Gaussian noise.
      3. Estimates sigma and computes an adaptive NLM parameter h.
      4. Runs NLM and stores intermediate results in a pickle file.
      5. Reloads those results to run GEO-NLM and BM3D.
      6. Computes PSNR, SSIM, and a custom score for each method.
      7. Saves filtered images and metrics (pickle + XLSX).
    """

    # Unpack configuration parameters    
    dir_images_set12 = parameters['dir_images_set12']    
    dir_out_geonlm = parameters['dir_out_geonlm']    
    dir_out_results = parameters['dir_out_results']
    pickle_results_summary_low = parameters['pickle_results_summary_low']
    name_results_xlsx_gnlm_output_low = parameters['name_results_xlsx_gnlm_output_low']
    name_pickle_results_gnlm_output_low = parameters['name_pickle_results_gnlm_output_low']
  
    f = parameters['f']        # Patch radius (NLM / GEO-NLM)
    t = parameters['t']        # Search window radius (NLM / GEO-NLM)
    alpha = parameters['alpha']  # Weight parameter for adaptive score / geometry

    # List all input image filenames in the set12 image directory
    array_dir = read_directories(dir_images_set12)

 
    array_gnlm_low_filtereds = []

    # Reload the NLM results to use them as input for GEO-NLM and BM3D
    vector = load_pickle('array_pickle_nlm', pickle_results_summary_low)

    for array in vector:
        # Retrieve data saved during the NLM phase
        img_noisse_gaussian_np = array['img_noisse_gaussian_np']
        nlm_h = array['nlm_h']
        file_name = array['file_name']  
        estimated_sigma_gaussian = array['estimated_sigma_gaussian']

        # Reload the original (clean) image from disk
        img = skimage.io.imread(f'{dir_images_set12}/{file_name}')
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

        #import pdb; pdb.set_trace()  # Debug breakpoint to inspect variables before running GEO-NLM

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

        end = time.time()
        time_geonlm = end - ini  # GEO-NLM execution time (not stored, but kept here if needed)      
        
        print(f"PSNR = {psnr_gnlm:.2f} | SSIM = {ssim_gnlm:.4f} | Score = {score_gnlm:.4f} | time_geonlm = {time_geonlm:.2f}s | Image: {file_name}")
        # Save GEO-NLM-filtered image
        skimage.io.imsave(
            f'{dir_out_geonlm}/{file_name}',
            img_filtered_gnlm.astype(np.uint8)
        )         

        # Collect all metrics for this image
        dict = {
            'nlm_h': nlm_h,
            'h_gnlm': h_gnlm,
            'estimated_sigma_gaussian': estimated_sigma_gaussian,
            'ssim_gnlm': ssim_gnlm,
            'psnr_gnlm': psnr_gnlm,
            'score_gnlm': score_gnlm,
            'time_geonlm': time_geonlm,     
            'file_name': file_name  
        }

        array_gnlm_low_filtereds.append(dict)

    # Save the combined GEO-NLM and BM3D results to pickle
    save_pickle(array_gnlm_low_filtereds, dir_out_results, name_pickle_results_gnlm_output_low)

    # Export all results (NLM, GEO-NLM, BM3D) to an XLSX spreadsheet
    save_results_to_xlsx(
        array_gnlm_low_filtereds,
        dir_out_results,
        name_results_xlsx_gnlm_output_low
    )

