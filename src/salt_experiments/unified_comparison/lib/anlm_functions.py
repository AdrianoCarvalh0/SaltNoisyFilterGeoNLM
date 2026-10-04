import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
from joblib import Parallel, delayed
from numba import njit
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .geonlm_medians_functions import (
    _aswmf_local_fallback,
    _aswmf_spatial_weight,
    _robust_center_value,
    _score,
)
from .nlm_functions import mirror_cpu


@njit
def _aswmf_spatial_weight_numba(di, dj, weight_diag_1, weight_diag_2, weight_other):
    if di == dj:
        return weight_diag_1
    if di + dj == 0:
        return weight_diag_2
    return weight_other


@njit
def _median_1d(values, count):
    work = np.empty(count, dtype=np.float32)
    for idx in range(count):
        work[idx] = values[idx]
    work.sort()
    mid = count // 2
    if count % 2 == 1:
        return work[mid]
    return 0.5 * (work[mid - 1] + work[mid])


@njit
def _robust_center_value_numba(
    img_n,
    r,
    s,
    f,
    center_value,
    z_alpha,
    outlier_pixel_alpha,
):
    max_size = (2 * f + 1) * (2 * f + 1)
    values = np.empty(max_size, dtype=np.float32)
    count = 0
    total = 0.0
    for di in range(-f, f + 1):
        for dj in range(-f, f + 1):
            value = img_n[r + di, s + dj]
            values[count] = value
            total += value
            count += 1

    median = _median_1d(values, count)
    mean = total / count
    var = 0.0
    for idx in range(count):
        diff = values[idx] - mean
        var += diff * diff
    std = np.sqrt(var / count)
    low = median - z_alpha * std
    high = median + z_alpha * std
    if center_value > low and center_value < high:
        return center_value
    return outlier_pixel_alpha * center_value + (1.0 - outlier_pixel_alpha) * median


@njit
def _aswmf_local_fallback_numba(
    img_n,
    im,
    jn,
    radius,
    weight_diag_1,
    weight_diag_2,
    weight_other,
    impulse_tolerance,
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
            if value <= impulse_tolerance or value >= 255.0 - impulse_tolerance:
                continue

            weight = _aswmf_spatial_weight_numba(
                di,
                dj,
                weight_diag_1,
                weight_diag_2,
                weight_other,
            )
            weighted_sum += weight * value
            weight_sum += weight

    if weight_sum > 0.0:
        return weighted_sum / weight_sum
    return fallback_sum / fallback_count


@njit
def _switch_anlm_numba(
    img_n,
    f,
    t,
    padding,
    h,
    z_alpha,
    outlier_pixel_alpha,
    reject_impulse_candidates,
    use_aswmf_spatial_weights,
    aswmf_weight_diag_1,
    aswmf_weight_diag_2,
    aswmf_weight_other,
    impulse_tolerance,
):
    m = img_n.shape[0] - 2 * padding
    n = img_n.shape[1] - 2 * padding
    filtered = img_n[padding:padding + m, padding:padding + n].copy()
    processed_count = 0
    fallback_count = 0

    for i in range(m):
        for j in range(n):
            im = i + padding
            jn = j + padding
            center_value = img_n[im, jn]
            if center_value > impulse_tolerance and center_value < 255.0 - impulse_tolerance:
                continue
            processed_count += 1

            # ``img_n`` has f+t symmetric padding, so every candidate patch
            # is in bounds and every target sees the complete (2t+1)^2 grid.
            rmin = im - t
            rmax = im + t + 1
            smin = jn - t
            smax = jn + t + 1

            weighted_sum = 0.0
            weight_sum = 0.0

            for r in range(rmin, rmax):
                for s in range(smin, smax):
                    candidate_center = img_n[r, s]
                    if reject_impulse_candidates and (
                        candidate_center <= impulse_tolerance
                        or candidate_center >= 255.0 - impulse_tolerance
                    ):
                        continue

                    d2 = 0.0
                    for di in range(-f, f + 1):
                        for dj in range(-f, f + 1):
                            diff = img_n[im + di, jn + dj] - img_n[r + di, s + dj]
                            d2 += diff * diff
                    similarity_weight = np.exp(-d2 / (h * h))

                    if use_aswmf_spatial_weights:
                        similarity_weight *= _aswmf_spatial_weight_numba(
                            r - im,
                            s - jn,
                            aswmf_weight_diag_1,
                            aswmf_weight_diag_2,
                            aswmf_weight_other,
                        )

                    pixel_value = _robust_center_value_numba(
                        img_n,
                        r,
                        s,
                        f,
                        candidate_center,
                        z_alpha,
                        outlier_pixel_alpha,
                    )
                    weighted_sum += similarity_weight * pixel_value
                    weight_sum += similarity_weight

            if weight_sum <= 0.0:
                fallback_count += 1
                filtered[i, j] = _aswmf_local_fallback_numba(
                    img_n,
                    im,
                    jn,
                    f,
                    aswmf_weight_diag_1,
                    aswmf_weight_diag_2,
                    aswmf_weight_other,
                    impulse_tolerance,
                )
            else:
                filtered[i, j] = weighted_sum / weight_sum

    return filtered, processed_count, fallback_count


def process_pixel_anlm(
    i,
    j,
    img_n,
    f,
    t,
    h,
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
    impulse_tolerance=0,
):
    """Adaptive NLM counterpart to GHNLM, using direct Euclidean patch distance.

    This intentionally mirrors the salt-and-pepper decisions used by GHNLM:
    impulse-only switching, robust candidate center value, optional rejection of
    impulse candidates, ASWMF-inspired spatial weights, and ASWMF fallback when
    the normalizer vanishes. The only difference is that similarity is computed
    from direct Euclidean patch distance, without building a graph.
    """
    im = i + f
    jn = j + f
    center_value = img_n[im, jn]

    if switch_impulse_only and (
        center_value > impulse_tolerance
        and center_value < 255.0 - impulse_tolerance
    ):
        return center_value

    central_patch = img_n[im - f:im + f + 1, jn - f:jn + f + 1]

    rmin = max(im - t, f)
    rmax = min(im + t, m + f)
    smin = max(jn - t, f)
    smax = min(jn + t, n + f)

    weighted_sum = 0.0
    weight_sum = 0.0

    for r in range(rmin, rmax):
        for s in range(smin, smax):
            candidate_patch = img_n[r - f:r + f + 1, s - f:s + f + 1]
            candidate_center = img_n[r, s]

            if reject_impulse_candidates and (
                candidate_center <= impulse_tolerance
                or candidate_center >= 255.0 - impulse_tolerance
            ):
                continue

            distance = np.linalg.norm(central_patch - candidate_patch)
            similarity_weight = np.exp(-(distance ** 2) / (h ** 2))

            if use_aswmf_spatial_weights:
                similarity_weight *= _aswmf_spatial_weight(
                    r - im,
                    s - jn,
                    weight_diag_1=aswmf_weight_diag_1,
                    weight_diag_2=aswmf_weight_diag_2,
                    weight_other=aswmf_weight_other,
                )

            pixel_value = _robust_center_value(
                candidate_patch,
                candidate_center,
                z_alpha=z_alpha,
                outlier_pixel_alpha=outlier_pixel_alpha,
            )
            weighted_sum += similarity_weight * pixel_value
            weight_sum += similarity_weight

    if weight_sum <= 0.0:
        return _aswmf_local_fallback(
            img_n,
            im,
            jn,
            radius=f,
            weight_diag_1=aswmf_weight_diag_1,
            weight_diag_2=aswmf_weight_diag_2,
            weight_other=aswmf_weight_other,
            impulse_tolerance=impulse_tolerance,
        )
    return weighted_sum / weight_sum


def Parallel_ANLM(
    img_n,
    f,
    t,
    h,
    z_alpha=1.96,
    outlier_pixel_alpha=0.0,
    switch_impulse_only=False,
    reject_impulse_candidates=False,
    use_aswmf_spatial_weights=False,
    aswmf_weight_diag_1=1.0,
    aswmf_weight_diag_2=1.0,
    aswmf_weight_other=10.0,
    n_jobs=-1,
    impulse_tolerance=0,
):
    m = img_n.shape[0] - 2 * f
    n = img_n.shape[1] - 2 * f
    filtered = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel_anlm)(
            i,
            j,
            img_n,
            f,
            t,
            h,
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
            impulse_tolerance,
        )
        for i in range(m)
        for j in range(n)
    )
    return np.array(filtered).reshape((m, n))


def Parallel_Switch_ANLM(
    img_n,
    f,
    t,
    h,
    z_alpha=1.96,
    outlier_pixel_alpha=0.0,
    reject_impulse_candidates=True,
    use_aswmf_spatial_weights=True,
    aswmf_weight_diag_1=1.0,
    aswmf_weight_diag_2=1.0,
    aswmf_weight_other=10.0,
    n_jobs=-1,
    impulse_tolerance=0,
    return_stats=False,
):
    """Switching IANLM on an unpadded image.

    The public wrapper owns the f+t padding requirement.  Keeping padding
    here prevents callers from silently supplying only f pixels of context
    while requesting a larger search radius.
    """
    padding = f + t
    img_n = mirror_cpu(np.asarray(img_n, dtype=np.float32), padding)
    filtered, processed_count, fallback_count = _switch_anlm_numba(
        img_n,
        f,
        t,
        padding,
        h,
        z_alpha,
        outlier_pixel_alpha,
        reject_impulse_candidates,
        use_aswmf_spatial_weights,
        aswmf_weight_diag_1,
        aswmf_weight_diag_2,
        aswmf_weight_other,
        impulse_tolerance,
    )
    if return_stats:
        return filtered, {
            "processed_pixels": int(processed_count),
            "fallback_pixels": int(fallback_count),
            "nonlocal_pixels": int(processed_count - fallback_count),
            "fallback_fraction": (
                float(fallback_count) / processed_count if processed_count else 0.0
            ),
        }
    return filtered


def Parallel_Switch_ANLM_python(
    img_n,
    f,
    t,
    h,
    z_alpha=1.96,
    outlier_pixel_alpha=0.0,
    reject_impulse_candidates=True,
    use_aswmf_spatial_weights=True,
    aswmf_weight_diag_1=1.0,
    aswmf_weight_diag_2=1.0,
    aswmf_weight_other=10.0,
    n_jobs=-1,
    impulse_tolerance=0,
):
    m = img_n.shape[0] - 2 * f
    n = img_n.shape[1] - 2 * f
    filtered = img_n[f:f + m, f:f + n].copy()
    impulse_coords = [
        (i, j)
        for i in range(m)
        for j in range(n)
        if (
            filtered[i, j] <= impulse_tolerance
            or filtered[i, j] >= 255.0 - impulse_tolerance
        )
    ]

    values = Parallel(n_jobs=n_jobs)(
        delayed(process_pixel_anlm)(
            i,
            j,
            img_n,
            f,
            t,
            h,
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
            impulse_tolerance,
        )
        for i, j in impulse_coords
    )

    for (i, j), value in zip(impulse_coords, values):
        filtered[i, j] = value

    return filtered


def run_anlm_pipeline(
    img_original,
    h_base,
    img_noisy,
    f,
    t,
    mult,
    z_alpha=1.96,
    outlier_pixel_alpha=0.0,
    switch_impulse_only=False,
    reject_impulse_candidates=False,
    use_aswmf_spatial_weights=False,
    aswmf_weight_diag_1=1.0,
    aswmf_weight_diag_2=1.0,
    aswmf_weight_other=10.0,
    n_jobs=-1,
    impulse_tolerance=0,
):
    h_anlm = float(h_base) * float(mult)
    # The non-switching legacy branch retains its original execution model.
    # The switching branch above owns f+t padding through its public wrapper.
    img_noisy_mirror = mirror_cpu(img_noisy.astype(np.float32), f)

    if switch_impulse_only:
        img_filtered = Parallel_Switch_ANLM(
            img_noisy,
            f=f,
            t=t,
            h=h_anlm,
            z_alpha=z_alpha,
            outlier_pixel_alpha=outlier_pixel_alpha,
            reject_impulse_candidates=reject_impulse_candidates,
            use_aswmf_spatial_weights=use_aswmf_spatial_weights,
            aswmf_weight_diag_1=aswmf_weight_diag_1,
            aswmf_weight_diag_2=aswmf_weight_diag_2,
            aswmf_weight_other=aswmf_weight_other,
            n_jobs=n_jobs,
            impulse_tolerance=impulse_tolerance,
        )
    else:
        img_filtered = Parallel_ANLM(
            img_noisy_mirror,
            f=f,
            t=t,
            h=h_anlm,
            z_alpha=z_alpha,
            outlier_pixel_alpha=outlier_pixel_alpha,
            switch_impulse_only=switch_impulse_only,
            reject_impulse_candidates=reject_impulse_candidates,
            use_aswmf_spatial_weights=use_aswmf_spatial_weights,
            aswmf_weight_diag_1=aswmf_weight_diag_1,
            aswmf_weight_diag_2=aswmf_weight_diag_2,
            aswmf_weight_other=aswmf_weight_other,
            n_jobs=n_jobs,
            impulse_tolerance=impulse_tolerance,
        )

    img_filtered = np.clip(img_filtered, 0, 255).astype(np.uint8)
    img_ref = np.clip(img_original, 0, 255).astype(np.uint8)

    psnr = peak_signal_noise_ratio(img_ref, img_filtered, data_range=255)
    ssim = structural_similarity(img_ref, img_filtered, data_range=255)
    score = _score(psnr, ssim)

    return img_filtered, h_anlm, psnr, ssim, score
