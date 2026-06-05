import numpy as np
from numba import njit


@njit
def aswmf_filter(image, radius=3, weight_diag_1=1.0, weight_diag_2=1.0, weight_other=10.0):
    """
    Adaptive Switching Weight Mean Filter for salt-and-pepper noise.

    The default radius=3 follows the ASWMF paper's 7x7 window. Pixels with
    values 0 or 255 are treated as salt-and-pepper candidates.
    """
    height, width = image.shape
    output = image.copy()

    for i in range(height):
        for j in range(width):
            center = image[i, j]

            window_min = 255.0
            window_max = 0.0

            for di in range(-radius, radius + 1):
                ii = i + di
                if ii < 0:
                    ii = -ii
                elif ii >= height:
                    ii = 2 * height - ii - 2

                for dj in range(-radius, radius + 1):
                    jj = j + dj
                    if jj < 0:
                        jj = -jj
                    elif jj >= width:
                        jj = 2 * width - jj - 2

                    value = image[ii, jj]
                    if value < window_min:
                        window_min = value
                    if value > window_max:
                        window_max = value

            if window_min < center < window_max:
                output[i, j] = center
                continue

            weighted_sum = 0.0
            weight_sum = 0.0
            fallback_sum = 0.0
            fallback_count = 0

            for di in range(-radius, radius + 1):
                ii = i + di
                if ii < 0:
                    ii = -ii
                elif ii >= height:
                    ii = 2 * height - ii - 2

                for dj in range(-radius, radius + 1):
                    jj = j + dj
                    if jj < 0:
                        jj = -jj
                    elif jj >= width:
                        jj = 2 * width - jj - 2

                    value = image[ii, jj]
                    fallback_sum += value
                    fallback_count += 1

                    if value == 0.0 or value == 255.0:
                        continue

                    if di == dj:
                        weight = weight_diag_1
                    elif di + dj == 0:
                        weight = weight_diag_2
                    else:
                        weight = weight_other

                    weighted_sum += weight * value
                    weight_sum += weight

            if weight_sum > 0.0:
                output[i, j] = weighted_sum / weight_sum
            else:
                output[i, j] = fallback_sum / fallback_count

    return output
