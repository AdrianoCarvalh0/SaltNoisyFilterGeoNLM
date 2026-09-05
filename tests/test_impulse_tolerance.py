import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "salt_experiments"))

from functions.noisy_functions import add_near_extreme_impulse_noise
from functions.salt_filters import aswmf_filter


class ImpulseToleranceTests(unittest.TestCase):
    def test_generator_uses_requested_bands_and_exact_density(self):
        image = np.full((20, 20), 128, dtype=np.float32)
        noisy, mask = add_near_extreme_impulse_noise(
            image,
            salt_prob=0.05,
            pepper_prob=0.05,
            impulse_tolerance=4,
            seed=42,
            return_mask=True,
        )
        self.assertEqual(int(mask.sum()), 40)
        self.assertTrue(np.all((noisy[mask] <= 4) | (noisy[mask] >= 251)))
        np.testing.assert_array_equal(noisy[~mask], image[~mask])

    def test_zero_tolerance_preserves_original_aswmf_behavior(self):
        image = np.arange(81, dtype=np.float32).reshape(9, 9) * 3
        old_style = aswmf_filter(image)
        explicit_zero = aswmf_filter(image, impulse_tolerance=0)
        np.testing.assert_array_equal(old_style, explicit_zero)

    def test_aswmf_accepts_near_extreme_tolerance(self):
        image = np.full((9, 9), 128, dtype=np.float32)
        image[4, 4] = 2
        filtered = aswmf_filter(image, radius=1, impulse_tolerance=4)
        self.assertEqual(float(filtered[4, 4]), 128.0)

    def test_invalid_generator_tolerance_is_rejected(self):
        image = np.zeros((4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            add_near_extreme_impulse_noise(image, impulse_tolerance=128)


if __name__ == "__main__":
    unittest.main()
