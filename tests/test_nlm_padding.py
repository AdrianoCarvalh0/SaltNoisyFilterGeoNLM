import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from salt_experiments.unified_comparison.lib.nlm_functions import (
    NLM_fast_cpu,
    NLM_fast_cuda_global,
)


def full_window_reference(image, h, f, t):
    """Direct NLM reference with symmetric padding of exactly f+t."""
    image = np.asarray(image, dtype=np.float32)
    padding = f + t
    padded = np.pad(image, padding, mode="symmetric")
    output = np.empty_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            im, jm = i + padding, j + padding
            target = padded[im - f:im + f + 1, jm - f:jm + f + 1]
            weighted_sum = 0.0
            weight_sum = 0.0
            for r in range(im - t, im + t + 1):
                for s in range(jm - t, jm + t + 1):
                    candidate = padded[r - f:r + f + 1, s - f:s + f + 1]
                    distance_squared = np.sum((target - candidate) ** 2)
                    weight = np.exp(-distance_squared / (h * h))
                    weighted_sum += weight * padded[r, s]
                    weight_sum += weight
            output[i, j] = weighted_sum / weight_sum
    return output


class NLMPaddingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.array(
            [[0, 10, 20, 30], [40, 50, 60, 70],
             [80, 90, 100, 110], [120, 130, 140, 150]],
            dtype=np.float32,
        )

    def test_cpu_uses_full_window_with_f_plus_t_padding(self):
        expected = full_window_reference(self.image, h=80.0, f=1, t=2)
        actual = NLM_fast_cpu(self.image, h=80.0, f=1, t=2)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)

    def test_cuda_matches_cpu_at_image_boundaries(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("CUDA device unavailable")
            actual = cp.asnumpy(
                NLM_fast_cuda_global(cp.asarray(self.image), h=80.0, f=1, t=2)
            )
            cp.cuda.Stream.null.synchronize()
        except Exception as exc:  # pragma: no cover - environment dependent
            self.skipTest(f"CUDA unavailable: {exc}")

        expected = NLM_fast_cpu(self.image, h=80.0, f=1, t=2)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
