import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import networkx as nx
import sklearn.neighbors as sknn
from joblib import Parallel, delayed
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.nlm_functions import mirror_cpu


def _score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


def _robust_center_value(patch, center_value, z_alpha=1.96, outlier_pixel_alpha=0.0):
    median = np.median(patch)
    std = patch.std()
    low = median - z_alpha * std
    high = median + z_alpha * std
    if center_value > low and center_value < high:
        return center_value
    return outlier_pixel_alpha * center_value + (1.0 - outlier_pixel_alpha) * median


def _aswmf_spatial_weight(di, dj, weight_diag_1=1.0, weight_diag_2=1.0, weight_other=10.0):
    if di == dj:
        return weight_diag_1
    if di + dj == 0:
        return weight_diag_2
    return weight_other


def _aswmf_local_fallback(
    img_n,
    im,
    jn,
    radius,
    weight_diag_1=1.0,
    weight_diag_2=1.0,
    weight_other=10.0,
):
    weighted_sum = 0.0
    weight_sum = 0.0
    fallback_sum = 0.0
    fallback_count = 0

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            value = img_n[im + di, jn + dj]
            fallback_sum += value
            fallback_count += 1

            if value == 0.0 or value == 255.0:
                continue

            weight = _aswmf_spatial_weight(
                di,
                dj,
                weight_diag_1=weight_diag_1,
                weight_diag_2=weight_diag_2,
                weight_other=weight_other,
            )
            weighted_sum += weight * value
            weight_sum += weight

    if weight_sum > 0.0:
        return weighted_sum / weight_sum
    return fallback_sum / fallback_count


def process_pixel_geonlm_medians(
    i,
    j,
    img_n,
    f,
    t,
    h,
    nn,
    m,
    n,
    z_alpha=1.96,
    outlier_pixel_alpha=0.0,
    switch_impulse_only=False,
    reject_impulse_candidates=False,
    use_aswmf_spatial_weights=False,
    aswmf_weight_diag_1=1.0,
    aswmf_weight_diag_2=1.0,
    aswmf_weight_other=10.0,
):
    im = i + f
    jn = j + f
    center_value = img_n[im, jn]

    if switch_impulse_only and center_value != 0.0 and center_value != 255.0:
        return center_value

    patch_central = img_n[im - f:im + f + 1, jn - f:jn + f + 1]
    central = patch_central.ravel()

    rmin = max(im - t, f)
    rmax = min(im + t, m + f)
    smin = max(jn - t, f)
    smax = min(jn + t, n + f)

    n_patches = (rmax - rmin) * (smax - smin)
    patch_size = (2 * f + 1) ** 2
    n_neighbors = min(nn, max(1, n_patches - 1))

    dataset = np.empty((n_patches, patch_size), dtype=np.float32)
    pixels_search = np.empty(n_patches, dtype=np.float32)
    valid_candidates = np.ones(n_patches, dtype=bool)
    spatial_weights = np.ones(n_patches, dtype=np.float32)

    source = -1
    k = 0
    for r in range(rmin, rmax):
        for s in range(smin, smax):
            patch = img_n[r - f:r + f + 1, s - f:s + f + 1]
            candidate_center = img_n[r, s]
            dataset[k, :] = patch.ravel()
            pixels_search[k] = _robust_center_value(
                patch,
                candidate_center,
                z_alpha=z_alpha,
                outlier_pixel_alpha=outlier_pixel_alpha,
            )
            if reject_impulse_candidates and (
                candidate_center == 0.0 or candidate_center == 255.0
            ):
                valid_candidates[k] = False
            if use_aswmf_spatial_weights:
                spatial_weights[k] = _aswmf_spatial_weight(
                    r - im,
                    s - jn,
                    weight_diag_1=aswmf_weight_diag_1,
                    weight_diag_2=aswmf_weight_diag_2,
                    weight_other=aswmf_weight_other,
                )
            if r == im and s == jn:
                source = k
            k += 1

    if source == -1:
        source = 0

    knn_graph = sknn.kneighbors_graph(
        dataset,
        n_neighbors=n_neighbors,
        mode="distance",
    )
    graph = nx.from_scipy_sparse_array(knn_graph)
    lengths, _ = nx.single_source_dijkstra(graph, source)

    points = list(lengths.keys())
    distances = np.array(list(lengths.values()), dtype=np.float32)
    similarity_weights = np.exp(-(distances ** 2) / (h ** 2))
    pixels = pixels_search[points]
    valid = valid_candidates[points]
    similarity_weights = similarity_weights * valid.astype(np.float32)
    if use_aswmf_spatial_weights:
        similarity_weights = similarity_weights * spatial_weights[points]

    z_value = np.sum(similarity_weights)
    if z_value <= 0:
        return _aswmf_local_fallback(
            img_n,
            im,
            jn,
            radius=f,
            weight_diag_1=aswmf_weight_diag_1,
            weight_diag_2=aswmf_weight_diag_2,
            weight_other=aswmf_weight_other,
        )
    return np.sum(similarity_weights * pixels) / z_value


def Parallel_GEONLMedians(
    img_n,
    f,
    t,
    h,
    nn=10,
    z_alpha=1.96,
    outlier_pixel_alpha=0.0,
    switch_impulse_only=False,
    reject_impulse_candidates=False,
    use_aswmf_spatial_weights=False,
    aswmf_weight_diag_1=1.0,
    aswmf_weight_diag_2=1.0,
    aswmf_weight_other=10.0,
    n_jobs=-1,
):
    m = img_n.shape[0] - 2 * f
    n = img_n.shape[1] - 2 * f
    filtered = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel_geonlm_medians)(
            i,
            j,
            img_n,
            f,
            t,
            h,
            nn,
            m,
            n,
            z_alpha,
            outlier_pixel_alpha,
            switch_impulse_only,
            reject_impulse_candidates,
            use_aswmf_spatial_weights,
            aswmf_weight_diag_1,
            aswmf_weight_diag_2,
            aswmf_weight_other,
        )
        for i in range(m)
        for j in range(n)
    )
    return np.array(filtered).reshape((m, n))


def Parallel_Switch_GEONLMedians(
    img_n,
    f,
    t,
    h,
    nn=10,
    z_alpha=1.96,
    outlier_pixel_alpha=0.0,
    reject_impulse_candidates=True,
    use_aswmf_spatial_weights=True,
    aswmf_weight_diag_1=1.0,
    aswmf_weight_diag_2=1.0,
    aswmf_weight_other=10.0,
    n_jobs=-1,
):
    m = img_n.shape[0] - 2 * f
    n = img_n.shape[1] - 2 * f
    filtered = img_n[f:f + m, f:f + n].copy()
    impulse_coords = [
        (i, j)
        for i in range(m)
        for j in range(n)
        if filtered[i, j] == 0.0 or filtered[i, j] == 255.0
    ]

    values = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel_geonlm_medians)(
            i,
            j,
            img_n,
            f,
            t,
            h,
            nn,
            m,
            n,
            z_alpha,
            outlier_pixel_alpha,
            False,
            reject_impulse_candidates,
            use_aswmf_spatial_weights,
            aswmf_weight_diag_1,
            aswmf_weight_diag_2,
            aswmf_weight_other,
        )
        for i, j in impulse_coords
    )

    for (i, j), value in zip(impulse_coords, values):
        filtered[i, j] = value

    return filtered


def run_geonlm_medians_pipeline(
    img_original,
    h_base,
    img_noisy,
    f,
    t,
    mult,
    nn=10,
    z_alpha=1.96,
    outlier_pixel_alpha=0.0,
    switch_impulse_only=False,
    reject_impulse_candidates=False,
    use_aswmf_spatial_weights=False,
    aswmf_weight_diag_1=1.0,
    aswmf_weight_diag_2=1.0,
    aswmf_weight_other=10.0,
    n_jobs=-1,
):
    img_noisy_mirror = mirror_cpu(img_noisy.astype(np.float32), f)
    h_geonlm_medians = float(h_base) * float(mult)

    if switch_impulse_only:
        img_filtered = Parallel_Switch_GEONLMedians(
            img_noisy_mirror,
            f=f,
            t=t,
            h=h_geonlm_medians,
            nn=nn,
            z_alpha=z_alpha,
            outlier_pixel_alpha=outlier_pixel_alpha,
            reject_impulse_candidates=reject_impulse_candidates,
            use_aswmf_spatial_weights=use_aswmf_spatial_weights,
            aswmf_weight_diag_1=aswmf_weight_diag_1,
            aswmf_weight_diag_2=aswmf_weight_diag_2,
            aswmf_weight_other=aswmf_weight_other,
            n_jobs=n_jobs,
        )
    else:
        img_filtered = Parallel_GEONLMedians(
            img_noisy_mirror,
            f=f,
            t=t,
            h=h_geonlm_medians,
            nn=nn,
            z_alpha=z_alpha,
            outlier_pixel_alpha=outlier_pixel_alpha,
            switch_impulse_only=switch_impulse_only,
            reject_impulse_candidates=reject_impulse_candidates,
            use_aswmf_spatial_weights=use_aswmf_spatial_weights,
            aswmf_weight_diag_1=aswmf_weight_diag_1,
            aswmf_weight_diag_2=aswmf_weight_diag_2,
            aswmf_weight_other=aswmf_weight_other,
            n_jobs=n_jobs,
        )
    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    img_ref = np.clip(img_original, 0, 255).astype(np.uint8)

    psnr = peak_signal_noise_ratio(img_ref, img_filtered, data_range=255)
    ssim = structural_similarity(img_ref, img_filtered, data_range=255)
    score = _score(psnr, ssim)

    return img_filtered, h_geonlm_medians, psnr, ssim, score
