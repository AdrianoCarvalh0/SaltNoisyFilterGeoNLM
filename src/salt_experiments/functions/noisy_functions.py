import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

# Returns float array (no dtype cast here)




def add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01, seed=None):
    """
    Adds salt-and-pepper noise to a 2D grayscale image.

    Parameters
    ----------
    img : ndarray
        Input grayscale image, assumed to be in the range [0, 255].
    salt_prob : float, optional
        Probability of adding salt noise, i.e., white pixels. Default is 0.01.
    pepper_prob : float, optional
        Probability of adding pepper noise, i.e., black pixels. Default is 0.01.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    noisy_img : ndarray
        Image corrupted with salt-and-pepper noise.
    """

    if img.ndim != 2:
        raise ValueError("The input image must be a 2D grayscale image.")

    if not (0 <= salt_prob <= 1):
        raise ValueError("salt_prob must be between 0 and 1.")

    if not (0 <= pepper_prob <= 1):
        raise ValueError("pepper_prob must be between 0 and 1.")

    if salt_prob + pepper_prob > 1:
        raise ValueError("The sum of salt_prob and pepper_prob must not exceed 1.")

    rng = np.random.default_rng(seed)

    m, n = img.shape
    noisy_img = img.copy()

    # Salt noise: white pixels
    num_salt = int(np.ceil(salt_prob * m * n))
    salt_rows = rng.integers(0, m, num_salt)
    salt_cols = rng.integers(0, n, num_salt)
    noisy_img[salt_rows, salt_cols] = 255

    # Pepper noise: black pixels
    num_pepper = int(np.ceil(pepper_prob * m * n))
    pepper_rows = rng.integers(0, m, num_pepper)
    pepper_cols = rng.integers(0, n, num_pepper)
    noisy_img[pepper_rows, pepper_cols] = 0

    return noisy_img
