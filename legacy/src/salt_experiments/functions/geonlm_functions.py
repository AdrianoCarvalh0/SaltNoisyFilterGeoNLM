import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import sklearn.neighbors as sknn
import networkx as nx
from joblib import Parallel, delayed
from functions.nlm_functions import (mirror_cpu)   
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def process_pixel(i, j, img_n, f, t, h, nn, m, n):
    """
    Compute the GEONLM (graph/knn-based) filtered value for a single pixel (i, j).

    Parameters
    ----------
    i, j : int
        Pixel coordinates in the original (unpadded) image domain.
    img_n : np.ndarray
        Padded image (mirror padding) of shape (m + 2*f, n + 2*f).
    f : int
        Patch radius. Each patch has size (2*f + 1) x (2*f + 1).
    t : int
        Search window radius around (i, j) in the padded domain.
    h : float
        Filtering parameter (controls decay of similarity weights).
    nn : int
        Number of neighbors for the KNN graph.
    m, n : int
        Original image height and width (without padding).

    Returns
    -------
    float
        Filtered value for pixel (i, j).
    """
    # Map (i, j) from original image to padded coordinates
    im = i + f
    jn = j + f

    # Central patch (reference) around (im, jn)
    patch_central = img_n[im - f:im + f + 1, jn - f:jn + f + 1]
    central = patch_central.ravel()  # flatten to 1D

    # Search window bounds in padded coordinates (clamped to valid region)
    rmin = max(im - t, f)
    rmax = min(im + t, m + f)
    smin = max(jn - t, f)
    smax = min(jn + t, n + f)

    # Total number of candidate patches inside the search window
    n_patches = (rmax - rmin) * (smax - smin)
    patch_size = (2 * f + 1) ** 2

    # Allocate matrix of flattened patches (dataset) and corresponding center pixels
    dataset = np.empty((n_patches, patch_size), dtype=np.float32)
    pixels_search = np.empty(n_patches, dtype=np.float32)

    # 'source' will be the index of the central patch inside 'dataset' (for Dijkstra)
    source = -1
    k = 0

    # Iterate over the search window, collect neighbor patches and center pixels
    for r in range(rmin, rmax):
        for s in range(smin, smax):
            W = img_n[r - f:r + f + 1, s - f:s + f + 1]
            neighbor = W.ravel()
            dataset[k, :] = neighbor
            pixels_search[k] = img_n[r, s]
            # Detect which row corresponds to the central patch
            if np.array_equal(central, neighbor):
                source = k
            k += 1

    # Fallback: if central was not found (numerical quirks), anchor to first element
    if source == -1:
        source = 0

    # Build KNN graph (weighted by Euclidean distances between patches)
    knnGraph = sknn.kneighbors_graph(dataset, n_neighbors=nn, mode='distance')
    G = nx.from_scipy_sparse_array(knnGraph)

    # Shortest paths from 'source' to all nodes (Dijkstra on weighted graph)
    length, _ = nx.single_source_dijkstra(G, source)

    # Extract nodes and distances in aligned arrays
    points = list(length.keys())
    distancias = np.array(list(length.values()), dtype=np.float32)

    # Convert graph distances to similarities via Gaussian kernel with bandwidth h
    similarity_weights = np.exp(-distancias ** 2 / (h ** 2))

    # Gather corresponding pixel intensities for the candidate nodes
    pixels = pixels_search[points]

    # Non-local weighted average: NL / Z
    NL = np.sum(similarity_weights * pixels)
    Z = np.sum(similarity_weights)

    # If Z == 0 (degenerate case), return the original pixel value
    return NL / Z if Z > 0 else img_n[im, jn]


def Parallel_GEONLM(img_n, f, t, h, nn):
    """
    Apply the GEONLM filter in parallel over all pixels of the original image domain.

    Parameters
    ----------
    img_n : np.ndarray
        Padded image (mirror padding) of shape (m + 2*f, n + 2*f).
    f : int
        Patch radius. Each patch has size (2*f + 1) x (2*f + 1).
    t : int
        Search window radius.
    h : float
        Filtering parameter (similarity decay).
    nn : int
        Number of neighbors in the KNN graph.

    Returns
    -------
    np.ndarray
        Filtered image of shape (m, n).
    """
    # Padded image dimensions -> original domain dims
    print(f'img_n.shape: {img_n.shape}')
    m = img_n.shape[0] - 2 * f
    n = img_n.shape[1] - 2 * f
    print(f'M: {m}, N: {n}')

    # Parallel evaluation over all (i, j) in the original domain
    filtered = Parallel(n_jobs=-1)(
        delayed(process_pixel)(i, j, img_n, f, t, h, nn, m, n)
        for i in range(m)
        for j in range(n)
    )

    # Reshape flat list back to (m, n)
    filtered_geo = np.array(filtered).reshape((m, n))
    return filtered_geo


def run_geonlm_pipeline(img_original, h_base, img_noisy, f, t, mult, nn=10):

    img_noisy_mirror = mirror_cpu(img_noisy, f)    
   
    img_n_geo = np.pad(img_noisy_mirror, ((f, f), (f, f)), 'symmetric')  
   
   
    h_geo = (h_base) * mult
    print(f"\nExecutando GEONLM com h = {h_geo:.2f} (base {h_base} * {mult})")

    img_geo = Parallel_GEONLM(img_n_geo, f=f, t=t, h=h_geo, nn=nn)

    img_geo_no_pad = img_geo[f:-f, f:-f]  # Remove 'f' pixels de cada lado
    img_geo_no_pad = np.clip(img_geo_no_pad, 0, 255).astype(np.uint8)

    img_ref = np.clip(img_original, 0, 255).astype(np.uint8)   

    psnr = peak_signal_noise_ratio(img_ref, img_geo_no_pad, data_range=255)
    ssim = structural_similarity(img_ref, img_geo_no_pad, data_range=255)

    score = 0.5 * psnr + 0.5 * (ssim * 100)
    print(f"→ PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, Score: {score:.2f}")    

   
    return img_geo_no_pad, h_geo, psnr, ssim, score