import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

# Returns float array (no dtype cast here)




def add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    """
    Adds salt-and-pepper noise to a 2D grayscale image.

    Parameters
    ----------
    img : ndarray
        Input grayscale image (assumed range [0, 255]).
    salt_prob : float, optional
        Probability of adding salt noise (default = 0.01).
    pepper_prob : float, optional
        Probability of adding pepper noise (default = 0.01).
    """
    m, n = img.shape
    noisy_img = img.copy()
    
    # Salt noise (white pixels)
    num_salt = np.ceil(salt_prob * m * n).astype(int)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy_img[salt_coords[0], salt_coords[1]] = 255
    
    # Pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * m * n).astype(int)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy_img[pepper_coords[0], pepper_coords[1]] = 0
    
    return noisy_img