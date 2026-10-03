import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from numba import njit
from scipy.ndimage import generic_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.cluster import MiniBatchKMeans

from functions.nlm_functions import mirror_cpu


def _score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


def extract_patch_features_3x3(image):
    padded = np.pad(image.astype(np.float32), 1, mode="reflect")
    rows = []
    h, w = image.shape
    for di in range(3):
        for dj in range(3):
            rows.append(padded[di:di + h, dj:dj + w].reshape(-1))
    return np.stack(rows, axis=1).astype(np.float32)


def classify_pixels_from_predenoised(predenoised, n_clusters=6, random_state=0):
    features = extract_patch_features_3x3(predenoised)
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=4096,
        n_init=5,
    )
    labels = model.fit_predict(features)
    return labels.reshape(predenoised.shape).astype(np.int32)


def edge_mask_from_labels(labels):
    labels = labels.astype(np.int32)
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edge


def local_variance_3x3(image):
    image = image.astype(np.float32)
    mean = generic_filter(image, np.mean, size=3, mode="reflect")
    mean_sq = generic_filter(image * image, np.mean, size=3, mode="reflect")
    return np.maximum(mean_sq - mean * mean, 0.0).astype(np.float32)


def compute_ifd_map(image, k=1.0, p1=0.5, p2=0.5, eps=1e-6):
    """
    Paper-inspired IFD map for AdaptiveRegionNLM.

    The paper defines IFD from membership and hesitation functions. The PDF
    formula rendering is partially broken, so this implementation preserves
    the described ingredients: local 3x3 block, smooth zero template, local
    variance, center-pixel contrast, and max IFD in the 3x3 neighborhood.
    The output is normalized to [0, 1] for stable h scaling.
    """
    image = image.astype(np.float32)
    var = local_variance_3x3(image)
    padded_image = np.pad(image, 1, mode="reflect")
    padded_var = np.pad(var, 1, mode="reflect")
    ifd = np.zeros_like(image, dtype=np.float32)

    for di in range(3):
        for dj in range(3):
            neighbor = padded_image[di:di + image.shape[0], dj:dj + image.shape[1]]
            neighbor_var = padded_var[di:di + image.shape[0], dj:dj + image.shape[1]]
            diff = np.abs(neighbor - image)
            mu_a = 1.0 / (1.0 + k * diff / (neighbor_var + eps))
            pi_a = p1 * (1.0 - mu_a)

            mu_o = 0.0
            pi_o = p2

            # Symmetric, bounded divergence surrogate from membership and hesitation.
            div = np.abs(mu_a - mu_o) + np.abs(pi_a - pi_o)
            ifd = np.maximum(ifd, div.astype(np.float32))

    lo = float(np.min(ifd))
    hi = float(np.max(ifd))
    if hi > lo:
        ifd = (ifd - lo) / (hi - lo)
    return ifd.astype(np.float32)


def build_adaptive_region_maps(
    noisy,
    predenoised,
    h_base,
    n_clusters=6,
    h_ifd_strength=1.0,
    random_state=0,
):
    labels = classify_pixels_from_predenoised(
        predenoised,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    edge_mask = edge_mask_from_labels(labels)
    ifd_map = compute_ifd_map(noisy)

    h_map = np.full(noisy.shape, float(h_base), dtype=np.float32)
    h_map[~edge_mask] = float(h_base) * np.exp(h_ifd_strength * ifd_map[~edge_mask])

    return labels, edge_mask, ifd_map, h_map


@njit
def adaptive_region_nlm_cpu(
    img,
    labels,
    edge_mask,
    h_map,
    f_small,
    t_small,
    f_large,
    t_large,
    require_same_label,
):
    m, n = img.shape
    max_f = f_large if f_large > f_small else f_small
    img_n = mirror_cpu(img, max_f)
    filtered = np.zeros((m, n), dtype=np.float32)

    for i in range(m):
        for j in range(n):
            if edge_mask[i, j]:
                f = f_small
                t = t_small
            else:
                f = f_large
                t = t_large

            im = i + max_f
            jn = j + max_f
            center_label = labels[i, j]
            h = h_map[i, j]

            rmin = max(i - t, 0)
            rmax = min(i + t, m - 1)
            smin = max(j - t, 0)
            smax = min(j + t, n - 1)

            nl = 0.0
            z = 0.0

            for rr in range(rmin, rmax + 1):
                for ss in range(smin, smax + 1):
                    if require_same_label and labels[rr, ss] != center_label:
                        continue

                    rp = rr + max_f
                    sp = ss + max_f
                    d2 = 0.0

                    for u in range(-f, f + 1):
                        for v in range(-f, f + 1):
                            diff = img_n[im + u, jn + v] - img_n[rp + u, sp + v]
                            d2 += diff * diff

                    weight = np.exp(-d2 / (h * h))
                    z += weight
                    nl += weight * img_n[rp, sp]

            if z > 0.0:
                filtered[i, j] = nl / z
            else:
                filtered[i, j] = img_n[im, jn]

    return filtered


def run_adaptive_region_nlm(
    reference,
    noisy,
    predenoised,
    h_base,
    f_small=2,
    t_small=7,
    f_large=4,
    t_large=14,
    n_clusters=6,
    h_ifd_strength=1.0,
    require_same_label=True,
    random_state=0,
):
    labels, edge_mask, ifd_map, h_map = build_adaptive_region_maps(
        noisy=noisy,
        predenoised=predenoised,
        h_base=h_base,
        n_clusters=n_clusters,
        h_ifd_strength=h_ifd_strength,
        random_state=random_state,
    )

    filtered = adaptive_region_nlm_cpu(
        noisy.astype(np.float32),
        labels,
        edge_mask,
        h_map,
        f_small,
        t_small,
        f_large,
        t_large,
        require_same_label,
    )
    filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
    reference_uint8 = np.clip(reference, 0, 255).astype(np.uint8)

    psnr = peak_signal_noise_ratio(reference_uint8, filtered_uint8, data_range=255)
    ssim = structural_similarity(reference_uint8, filtered_uint8, data_range=255)

    return {
        "filtered": filtered_uint8,
        "labels": labels,
        "edge_mask": edge_mask,
        "ifd_map": ifd_map,
        "h_map": h_map,
        "psnr": psnr,
        "ssim": ssim,
        "score": _score(psnr, ssim),
        "edge_ratio": float(np.mean(edge_mask)),
        "h_min": float(np.min(h_map)),
        "h_max": float(np.max(h_map)),
        "h_mean": float(np.mean(h_map)),
    }
