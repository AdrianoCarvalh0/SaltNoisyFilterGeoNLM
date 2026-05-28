import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import cupy as cp

from skimage.restoration import estimate_sigma
import numpy as np
import skimage
from functions.Utils import (read_directories, save_pickle, save_results_to_xlsx, load_pickle, is_low_noise_or, get_multiplier)
from functions.noisy_functions import (add_low_noise_gaussian, add_moderate_noise_gaussian, add_high_noise_gaussian)
from functions.nlm_functions import (compute_adaptive_q, select_best_h_using_adaptive_q)
from functions.geonlm_functions import run_geonlm_pipeline
import time
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from skimage.transform import resize

def resize_image(img, target_size):
    """
    Resize image to (target_size, target_size) using bilinear interpolation,
    preserving intensity range.

    Parameters
    ----------
    img : ndarray
        Input image read with skimage (H x W) or (H x W x C)
    target_size : int
        Target spatial resolution (e.g., 256, 384, 512, 768, 1024)

    Returns
    -------
    img_resized : ndarray
        Resized image with original data range preserved
    """
    img_resized = resize(
        img,
        (target_size, target_size),
        order=1,              # bilinear interpolation
        anti_aliasing=True,   # safe for both up/downscale
        preserve_range=True
    )
    return img_resized.astype(img.dtype)



def generate_gaussian_experiment(parameters):
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
    dir_out_nlm = parameters['dir_out_nlm']   
    dir_out_geonlm = parameters['dir_out_geonlm']    
    dir_out_results = parameters['dir_out_results']
   
    name_results_xlsx_gnlm_output = parameters['name_results_xlsx_gnlm_output']
    name_pickle_results_gnlm_output = parameters['name_pickle_results_gnlm_output']
    name_pickle_results_nlm_output = parameters['name_pickle_results_nlm_output']
    pickle_regional_experiment_summary = parameters['pickle_regional_experiment_summary']
  
    f = parameters['f']        # Patch radius (NLM / GEO-NLM)
    t = parameters['t']        # Search window radius (NLM / GEO-NLM)
    alpha = parameters['alpha']  # Weight parameter for adaptive score / geometry
    nn = parameters['nn']  # Number of nearest neighbors / similar patches in GEO-NLM

    # List all input image filenames in the set12 image directory
    array_dir = read_directories(dir_images_set12)

 
    array_gnlm_filtereds = []
    array_nlm_filtereds = []

    # Reload the NLM results to use them as input for GEO-NLM and BM3D
    vector = load_pickle('dataset_pickle', pickle_regional_experiment_summary) 

    vector = load_pickle(dir_out_results, pickle_regional_experiment_summary)

    array_nlm_filtereds = []

    for tipo in vector:
        for idx, sample in enumerate(vector[tipo]):

            img = sample['clean'].astype(np.float32)

            for sigma in sample['noisy']:

                noised = sample['noisy'][sigma].astype(np.float32)

                estimated_sigma = estimate_sigma(noised)

                img_gpu = cp.array(noised)

                h_nlm = compute_adaptive_q(estimated_sigma)

                if sigma == 5:
                    q_nlm_candidates = np.array([h_nlm + delta for delta in range(-100, 25)])

                elif sigma == 50:
                    q_nlm_candidates = np.array([h_nlm + delta for delta in range(200, 400)])
                    
                img_filtered_nlm, nlm_h, psnr_nlm, ssim_nlm, score_nlm = \
                    select_best_h_using_adaptive_q(
                        image=img,
                        image_gpu=img_gpu,
                        q_nlm_candidates=q_nlm_candidates,
                        f=f,
                        t=t,
                        alpha=alpha,
                    )

                dct = {
                    'tipo': tipo,
                    'index': idx,
                    'sigma': sigma,

                    'clean': img,
                    'noisy': noised,
                    'img_filtered_nlm': img_filtered_nlm,

                    'estimated_sigma': estimated_sigma,
                    'nlm_h': nlm_h,

                    'psnr_nlm': psnr_nlm,
                    'ssim_nlm': ssim_nlm,
                    'score_nlm': score_nlm,
                }

                array_nlm_filtereds.append(dct)

    save_pickle(array_nlm_filtereds, dir_out_results, 'regional_nlm_results.pkl')
    # ---------------------------------------
    # 2) BM3D AND GEO-NLM PHASE (LOW NOISE)
    # ---------------------------------------

    vector_nlm = load_pickle(dir_out_results, 'regional_nlm_results.pkl')

    array_gnlm_filtereds = []

    for sample in vector_nlm:

        img = sample['clean']
        noised = sample['noisy']
        nlm_h = sample['nlm_h']
        estimated_sigma = sample['estimated_sigma']

        tipo = sample['tipo']
        idx = sample['index']
        sigma = sample['sigma']

        mult = get_multiplier(nlm_h, estimated_sigma)

        img_filtered_gnlm, h_gnlm, psnr_gnlm, ssim_gnlm, score_gnlm = \
            run_geonlm_pipeline(
                img,
                nlm_h,
                noised,
                f,
                t,
                mult,
                nn
            )

        dict_out = {
            'tipo': tipo,
            'index': idx,
            'sigma': sigma,

            'psnr_nlm': sample['psnr_nlm'],
            'ssim_nlm': sample['ssim_nlm'],
            'score_nlm': sample['score_nlm'],

            'psnr_gnlm': psnr_gnlm,
            'ssim_gnlm': ssim_gnlm,
            'score_gnlm': score_gnlm,

            'h_nlm': nlm_h,
            'h_gnlm': h_gnlm
        }

        array_gnlm_filtereds.append(dict_out)

    save_pickle(array_gnlm_filtereds, dir_out_results, 'regional_final_results.pkl')

       

   
    