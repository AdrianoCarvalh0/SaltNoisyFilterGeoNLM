"""
Non-local medians filter from Levada, SIBGRAPI 2021.

Parameters
----------
img:
    Noisy grayscale input image.
h:
    Smoothing parameter. Higher values produce stronger smoothing.
f:
    Patch radius. The patch size is (2f + 1) x (2f + 1).
t:
    Search-window radius. The search size is (2t + 1) x (2t + 1).
"""

import numpy as np
from numba import njit, prange
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _score(psnr, ssim):
    return 0.5 * psnr + 0.5 * (ssim * 100)


@njit
def _symmetric_index(index, size):
    if index < 0:
        return -index - 1
    if index >= size:
        return 2 * size - index - 1
    return index


@njit
def _symmetric_pad(img, f):
    m, n = img.shape
    padded = np.empty((m + 2 * f, n + 2 * f), dtype=np.float32)
    for i in range(m + 2 * f):
        source_i = _symmetric_index(i - f, m)
        for j in range(n + 2 * f):
            source_j = _symmetric_index(j - f, n)
            padded[i, j] = img[source_i, source_j]
    return padded


@njit
def _median_patch(img, center_i, center_j, f):
    size = (2 * f + 1) * (2 * f + 1)
    values = np.empty(size, dtype=np.float32)
    count = 0
    for u in range(-f, f + 1):
        for v in range(-f, f + 1):
            values[count] = img[center_i + u, center_j + v]
            count += 1
    sorted_values = np.sort(values)
    mid = size // 2
    if size % 2 == 1:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


@njit
def _std_patch(img, center_i, center_j, f, mean):
    size = (2 * f + 1) * (2 * f + 1)
    acc = 0.0
    for u in range(-f, f + 1):
        for v in range(-f, f + 1):
            diff = img[center_i + u, center_j + v] - mean
            acc += diff * diff
    return np.sqrt(acc / size)


@njit
def _matrix_l1_norm_patch_diff(img, i1, j1, i2, j2, f):
    max_col_sum = 0.0
    for v in range(-f, f + 1):
        col_sum = 0.0
        for u in range(-f, f + 1):
            diff = img[i1 + u, j1 + v] - img[i2 + u, j2 + v]
            col_sum += abs(diff)
        if col_sum > max_col_sum:
            max_col_sum = col_sum
    return max_col_sum


@njit(parallel=True)
def _nlmedians_cpu(img, h, f, t):
    m, n = img.shape
    filtered = np.zeros((m, n), dtype=np.float32)
    img_n = _symmetric_pad(img.astype(np.float32), f)
    h2 = h * h

    for i in prange(m):
        for j in range(n):
            im = i + f
            jn = j + f

            rmin = max(im - t, f)
            rmax = min(im + t, m + f)
            smin = max(jn - t, f)
            smax = min(jn + t, n + f)

            nl_value = 0.0
            z_value = 0.0

            for r in range(rmin, rmax):
                for s in range(smin, smax):
                    mj = _median_patch(img_n, r, s, f)
                    dpj = _std_patch(img_n, r, s, f, mj)

                    low = mj - 1.96 * dpj
                    high = mj + 1.96 * dpj

                    d = _matrix_l1_norm_patch_diff(img_n, im, jn, r, s, f)
                    sij = np.exp(-d / h2)

                    z_value += sij
                    if img_n[r, s] > low and img_n[r, s] < high:
                        value = img_n[r, s]
                    else:
                        value = mj
                    nl_value += sij * value

            filtered[i, j] = nl_value / z_value

    return filtered


def NLMedians(img, h, f, t):
    return _nlmedians_cpu(
        img.astype(np.float32),
        float(h),
        int(f),
        int(t),
    )


def run_nlmedians(reference, noisy, h, f=4, t=7):
    filtered = NLMedians(noisy, h, f, t)
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
