import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import cupy as cp
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from numba import njit

@njit
def mirror_cpu(A, f):
    """
    Symmetric (mirror) padding on CPU.

    Parameters
    ----------
    A : np.ndarray
        2D grayscale image of shape (n, m).
    f : int
        Patch radius (padding width). The output will have 2*f extra pixels
        on each side (top, bottom, left, right).

    Returns
    -------
    np.ndarray
        Padded image B of shape (n + 2*f, m + 2*f) using mirror boundaries.
    """
    n, m = A.shape
    nlin = n + 2*f
    ncol = m + 2*f
    B = np.zeros((nlin, ncol), dtype=A.dtype)

    # Center region (original image)
    B[f:n+f, f:m+f] = A

    # Top and bottom stripes (mirror vertically)
    B[0:f, f:m+f] = A[0:f, :][::-1, :]
    B[n+f:, f:m+f] = A[n-f:, :][::-1, :]

    # Left and right stripes (mirror horizontally)
    B[f:n+f, 0:f] = A[:, 0:f][:, ::-1]
    B[f:n+f, m+f:] = A[:, m-f:][:, ::-1]

    # Four corners (mirror both axes)
    B[0:f, 0:f] = A[0:f, 0:f][::-1, ::-1]
    B[0:f, m+f:] = A[0:f, m-f:][::-1, ::-1]
    B[n+f:, 0:f] = A[n-f:, 0:f][::-1, ::-1]
    B[n+f:, m+f:] = A[n-f:, m-f:][::-1, ::-1]

    return B


@njit
def NLM_fast_cpu(img, h, f, t):
    """
    Fast Non-Local Means (CPU) with symmetric padding.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image (m, n). Expected range typically [0, 255].
    h : float
        Filtering parameter (decay). Larger h => smoother result.
    f : int
        Patch radius. Patch size is (2*f + 1) x (2*f + 1).
    t : int
        Search window radius around each pixel (in padded coordinates).

    Returns
    -------
    np.ndarray
        Filtered image (m, n) with the same dtype as input.
    """
    m, n = img.shape
    filtered = np.zeros((m, n), dtype=img.dtype)

    # Mirror padding to safely extract patches at borders
    img_n = mirror_cpu(img, f)

    for i in range(m):         # spatial loop over original image coordinates
        for j in range(n):
            im = i + f         # position in padded image
            jn = j + f

            # Reference patch W1 around (im, jn) in padded image
            W1 = img_n[im-f:im+f+1, jn-f:jn+f+1]

            # Define search window bounds in padded coordinates
            rmin = max(im-t, f)
            rmax = min(im+t, m+f-1)
            smin = max(jn-t, f)
            smax = min(jn+t, n+f-1)

            NL = 0.0          # weighted intensity sum
            Z = 0.0           # weight normalization (sum of weights)

            # Loop over candidate pixels in the search window
            for r in range(rmin, rmax+1):
                for s in range(smin, smax+1):
                    # Candidate patch W2 around (r, s)
                    W2 = img_n[r-f:r+f+1, s-f:s+f+1]

                    # Squared Euclidean distance between patches
                    d2 = np.sum((W1 - W2)**2)

                    # Weight based on patch distance and parameter h
                    sij = np.exp(-d2/(h**2))
                    Z += sij
                    NL += sij * img_n[r, s]

            # Normalize to get the filtered value for (i, j)
            filtered[i, j] = NL / Z
    return filtered


def mirror_gpu(A, f):
    """
    Symmetric (mirror) padding on GPU using CuPy arrays.

    Parameters
    ----------
    A : cp.ndarray
        2D grayscale image of shape (n, m) on GPU.
    f : int
        Patch radius (padding width).

    Returns
    -------
    cp.ndarray
        Padded image of shape (n + 2*f, m + 2*f) using mirror boundaries.
    """
    n, m = A.shape
    B = cp.zeros((n + 2*f, m + 2*f), dtype=A.dtype)

    # Center region (original image)
    B[f:n+f, f:m+f] = A

    # Top and bottom stripes (mirror vertically)
    B[0:f, f:m+f] = A[0:f, :][::-1, :]
    B[n+f:, f:m+f] = A[n-f:, :][::-1, :]

    # Left and right stripes (mirror horizontally)
    B[f:n+f, 0:f] = A[:, 0:f][:, ::-1]
    B[f:n+f, m+f:] = A[:, m-f:][:, ::-1]

    # Four corners (mirror both axes)
    B[0:f, 0:f] = A[0:f, 0:f][::-1, ::-1]
    B[0:f, m+f:] = A[0:f, m-f:][::-1, ::-1]
    B[n+f:, 0:f] = A[n-f:, 0:f][::-1, ::-1]
    B[n+f:, m+f:] = A[n-f:, m-f:][::-1, ::-1]

    return B


nlm_kernel_global_code = r'''
// Global-memory NLM kernel (single-channel, float32).
// - img_n: mirrored padded image (shape (m+2*f, n+2*f) flattened row-major)
// - output: filtered image (shape (m, n) flattened row-major)
// - m, n: original image height and width
// - f: patch radius
// - t: search window radius
// - h: filtering parameter
// - padded_width: width (columns) of the padded image
extern "C" __global__
void nlm_kernel_global(
    const float* img_n, float* output,
    int m, int n, int f, int t, float h, int padded_width
) {
    // Compute (i, j) for the original (unpadded) image
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= m || j >= n)
        return;

    // Coordinates in the padded image
    int im = i + f;
    int jm = j + f;

    float NL = 0.0f;  // weighted intensity sum
    float Z = 0.0f;   // normalization (sum of weights)

    // Search window around (im, jm)
    for (int r = im - t; r <= im + t; ++r) {
        for (int s = jm - t; s <= jm + t; ++s) {

            // Patch distance accumulator
            float d2 = 0.0f;

            // Compare patches W1 (centered at im,jm) and W2 (centered at r,s)
            for (int u = -f; u <= f; ++u) {
                for (int v = -f; v <= f; ++v) {
                    int x1 = im + u;
                    int y1 = jm + v;
                    int x2 = r + u;
                    int y2 = s + v;

                    float val1 = img_n[x1 * padded_width + y1];
                    float val2 = img_n[x2 * padded_width + y2];
                    float diff = val1 - val2;
                    d2 += diff * diff;
                }
            }

            // Weight from patch distance and parameter h
            float weight = __expf(-d2 / (h * h));
            Z += weight;
            NL += weight * img_n[r * padded_width + s];
        }
    }

    // Write filtered value
    output[i * n + j] = NL / Z;
}
'''


def NLM_fast_cuda_global(img, h, f, t):
    """
    Fast Non-Local Means on GPU using a raw CUDA kernel with global memory.

    Parameters
    ----------
    img : cp.ndarray
        2D grayscale image (m, n) on GPU (will be cast to float32).
    h : float
        Filtering parameter (decay).
    f : int
        Patch radius. Patch size is (2*f + 1) x (2*f + 1).
    t : int
        Search window radius around each pixel (in padded coordinates).

    Returns
    -------
    cp.ndarray
        Filtered image (m, n) in float32 on GPU.
    """
    img = img.astype(cp.float32)
    m, n = img.shape

    # Mirror padding on GPU for safe patch extraction
    padded = mirror_gpu(img, f)

    # Compile CUDA kernel
    module = cp.RawModule(code=nlm_kernel_global_code, options=('-std=c++11',))
    kernel = module.get_function("nlm_kernel_global")

    # Allocate output
    output = cp.zeros((m, n), dtype=cp.float32)

    # Block/grid configuration (16x16 threads per block)
    threads_per_block = (16, 16)
    grid = ((n + 15) // 16, (m + 15) // 16)

    # Launch kernel
    kernel(
        grid, threads_per_block,
        (
            padded.ravel(), output.ravel(),
            cp.int32(m), cp.int32(n), cp.int32(f), cp.int32(t),
            cp.float32(h), cp.int32(padded.shape[1])
        )
    )
    return output


def compute_adaptive_q(sigma_est):
    """
    Compute an adaptive q parameter from a noise estimate (sigma_est).
    This mapping uses a smooth squashing via tanh, then clips and scales.

    Parameters
    ----------
    sigma_est : float
        Noise standard deviation estimate (arbitrary units).

    Returns
    -------
    int
        q_nlm parameter as an integer after scaling/clipping.
    """
    q_nlm = 0.8 + 0.5 * np.tanh(0.3 * (sigma_est - 1))
    q_nlm = int(np.clip(q_nlm, 0.7, 2.2) * 100)
    return q_nlm


def select_best_h_using_adaptive_q(image, image_gpu, q_nlm_candidates, f, t, alpha=0.5):
    """
    Grid-search over q/h candidates on GPU NLM and pick the best by a mixed PSNR/SSIM score.

    Parameters
    ----------
    image : np.ndarray
        Reference image on CPU (used for PSNR/SSIM evaluation).
    image_gpu : cp.ndarray
        Noisy image on GPU to be denoised.
    q_nlm_candidates : iterable
        Candidate values for h (or q-like parameter) to test.
    f : int
        Patch radius for NLM.
    t : int
        Search window radius for NLM.
    alpha : float, default=0.5
        Score mixing factor: score = alpha * PSNR + (1 - alpha) * (SSIM * 100).

    Returns
    -------
    tuple
        (melhor_resultado, melhor_q_nlm, melhor_psnr, melhor_ssim, melhor_score)
        - melhor_resultado : np.ndarray
            Best denoised image (CPU array) for the chosen candidate.
        - melhor_q_nlm : value from q_nlm_candidates
            The candidate that maximized the score.
        - melhor_psnr : float
        - melhor_ssim : float
        - melhor_score : float
    """
    best_score = -float('inf')
    best_q_nlm = None
    best_result = None
    best_psnr = None
    best_ssim = None

    for h_nlm in q_nlm_candidates:
        # Run GPU NLM for this candidate and synchronize
        result_gpu = NLM_fast_cuda_global(image_gpu, h_nlm, f, t)
        cp.cuda.Stream.null.synchronize()

        # Reference: original image clipped to [0, 255] (uint8)
        img_ref = np.clip(image, 0, 255).astype(np.uint8)

        # Move result back to CPU and quantize to uint8 for metrics
        result_processed = cp.asnumpy(result_gpu)
        result_uint8 = np.clip(result_processed, 0, 255).astype(np.uint8)

        # Quality metrics
        psnr = peak_signal_noise_ratio(img_ref, result_uint8, data_range=255)
        ssim = structural_similarity(img_ref, result_uint8, data_range=255)

        # Mixed score (PSNR + scaled SSIM)
        score = alpha * psnr + (1 - alpha) * (ssim * 100)
        print(f"h = {h_nlm:.2f} | PSNR = {psnr:.2f} | SSIM = {ssim:.4f} | Score = {score:.2f}")

        # Keep best
        if score > best_score:
            best_score = score
            best_q_nlm = h_nlm
            best_result = result_processed
            best_psnr = psnr
            best_ssim = ssim
    print(f"\n[SELECTED] H = {best_q_nlm:.2f} | PSNR = {best_psnr:.2f} | SSIM = {best_ssim:.4f} | SCORE = {best_score:.2f}")
    return best_result, best_q_nlm, best_psnr, best_ssim, best_score
