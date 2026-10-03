import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from numba import njit, prange
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from functions.nlm_functions import mirror_cpu


def _score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


@njit
def _weighted_median(values, weights, count):
    order = np.argsort(values[:count])
    total = 0.0
    for i in range(count):
        total += weights[i]

    threshold = 0.5 * total
    acc = 0.0
    for i in range(count):
        idx = order[i]
        acc += weights[idx]
        if acc >= threshold:
            return values[idx]

    return values[order[count - 1]]


@njit
def _plain_median(values, count):
    sorted_values = np.sort(values[:count])
    mid = count // 2
    if count % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


@njit(parallel=True)
def nonlocal_median_cpu(img, h, f, t, weighted=True, top_k=0):
    """
    Non-local medians filter.

    The patch similarity is the same NLM-style exponential kernel, but the
    final aggregation is a median of candidate center pixels rather than a
    mean. For scalar grayscale images, the Euclidean median reduces to a
    median; the weighted form uses NLM weights as robust sample weights.
    """
    m, n = img.shape
    filtered = np.zeros((m, n), dtype=np.float32)
    img_n = mirror_cpu(img.astype(np.float32), f)

    for i in prange(m):
        for j in range(n):
            max_candidates = (2 * t + 1) * (2 * t + 1)
            values = np.empty(max_candidates, dtype=np.float32)
            weights = np.empty(max_candidates, dtype=np.float32)
            distances = np.empty(max_candidates, dtype=np.float32)
            top_values = np.empty(max_candidates, dtype=np.float32)
            top_weights = np.empty(max_candidates, dtype=np.float32)
            im = i + f
            jn = j + f

            rmin = max(im - t, f)
            rmax = min(im + t, m + f - 1)
            smin = max(jn - t, f)
            smax = min(jn + t, n + f - 1)

            count = 0
            for r in range(rmin, rmax + 1):
                for s in range(smin, smax + 1):
                    d2 = 0.0
                    for u in range(-f, f + 1):
                        for v in range(-f, f + 1):
                            diff = img_n[im + u, jn + v] - img_n[r + u, s + v]
                            d2 += diff * diff

                    distances[count] = d2
                    values[count] = img_n[r, s]
                    weights[count] = np.exp(-d2 / (h * h))
                    count += 1

            if top_k > 0 and top_k < count:
                order = np.argsort(distances[:count])
                for k in range(top_k):
                    idx = order[k]
                    top_values[k] = values[idx]
                    top_weights[k] = weights[idx]
                for k in range(top_k):
                    values[k] = top_values[k]
                    weights[k] = top_weights[k]
                count = top_k

            if weighted:
                filtered[i, j] = _weighted_median(values, weights, count)
            else:
                filtered[i, j] = _plain_median(values, count)

    return filtered


def run_nonlocal_median(
    reference,
    noisy,
    h,
    f=4,
    t=7,
    weighted=True,
    top_k=0,
):
    filtered = nonlocal_median_cpu(
        noisy.astype(np.float32),
        float(h),
        int(f),
        int(t),
        bool(weighted),
        int(top_k),
    )
    filtered_uint8 = np.clip(filtered, 0, 255).astype(np.uint8)
    reference_uint8 = np.clip(reference, 0, 255).astype(np.uint8)

    psnr = peak_signal_noise_ratio(reference_uint8, filtered_uint8, data_range=255)
    ssim = structural_similarity(reference_uint8, filtered_uint8, data_range=255)

    return {
        "filtered": filtered_uint8,
        "psnr": psnr,
        "ssim": ssim,
        "score": _score(psnr, ssim),
    }
