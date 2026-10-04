import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from salt_experiments.unified_comparison.lib import geonlm_functions as gnlm
from salt_experiments.unified_comparison.lib import geonlm_medians_functions as ghnlm
from salt_experiments.unified_comparison.lib.anlm_functions import Parallel_Switch_ANLM
from salt_experiments.unified_comparison.lib.nlm_functions import mirror_cpu
from salt_experiments.unified_comparison.lib.nlmedians import NLMedians


class NonLocalProtocolTests(unittest.TestCase):
    """Regression tests for the declared f=1, t=3 graph/switching protocol."""

    def setUp(self):
        self.f = 1
        self.t = 3
        self.padding = self.f + self.t
        self.image = np.arange(81, dtype=np.float32).reshape(9, 9) * 3
        self.padded = mirror_cpu(self.image, self.padding)

    def test_gnlm_uses_a_full_7_by_7_grid_at_corner_and_interior(self):
        seen_shapes = []
        original = gnlm.sknn.kneighbors_graph

        def record(dataset, *args, **kwargs):
            seen_shapes.append(dataset.shape)
            return original(dataset, *args, **kwargs)

        with patch.object(gnlm.sknn, "kneighbors_graph", side_effect=record):
            for i, j in ((0, 0), (4, 4)):
                gnlm.process_pixel(
                    i, j, self.padded, self.f, self.t, 100.0, 7,
                    *self.image.shape, self.padding,
                )
        self.assertEqual(seen_shapes, [(49, 9), (49, 9)])

    def test_ghnlm_uses_a_full_7_by_7_grid_at_corner_and_interior(self):
        seen_shapes = []
        original = ghnlm.sknn.kneighbors_graph

        def record(dataset, *args, **kwargs):
            seen_shapes.append(dataset.shape)
            return original(dataset, *args, **kwargs)

        with patch.object(ghnlm.sknn, "kneighbors_graph", side_effect=record):
            for i, j in ((0, 0), (4, 4)):
                ghnlm.process_pixel_geonlm_medians(
                    i, j, self.padded, self.f, self.t, 100.0, 7,
                    *self.image.shape, self.padding,
                )
        self.assertEqual(seen_shapes, [(49, 9), (49, 9)])

    def test_gnlm_anchors_dijkstra_to_target_coordinate_with_repeated_patches(self):
        repeated = np.full((9, 9), 128, dtype=np.float32)
        padded = mirror_cpu(repeated, self.padding)
        source_seen = []

        def record_source(_graph, source):
            source_seen.append(source)
            return {source: 0.0}, {}

        with patch.object(gnlm.nx, "single_source_dijkstra", side_effect=record_source):
            gnlm.process_pixel(
                4, 4, padded, self.f, self.t, 100.0, 7,
                *repeated.shape, self.padding,
            )
        # Row-major target location in a 7x7 candidate grid.
        self.assertEqual(source_seen, [24])

    def test_switching_ianlm_preserves_clean_pixels_and_accounts_for_fallback(self):
        image = np.full((9, 9), 128, dtype=np.float32)
        image[0, 0] = 0.0
        restored, stats = Parallel_Switch_ANLM(
            image, f=self.f, t=self.t, h=1.0, impulse_tolerance=0,
            return_stats=True,
        )
        self.assertEqual(restored.shape, image.shape)
        np.testing.assert_array_equal(restored[1:, :], image[1:, :])
        self.assertTrue(np.isfinite(restored).all())
        self.assertEqual(stats["processed_pixels"], 1)
        self.assertEqual(
            stats["processed_pixels"],
            stats["fallback_pixels"] + stats["nonlocal_pixels"],
        )

    def test_graph_pipelines_return_original_shape_and_finite_values(self):
        image = self.image.copy()
        image[0, 0] = 0.0
        gnlm_out, *_ = gnlm.run_geonlm_pipeline(
            self.image, 100.0, image, f=self.f, t=self.t, mult=1.4,
            nn=7, n_jobs=1,
        )
        ghnlm_out, *_ = ghnlm.run_geonlm_medians_pipeline(
            self.image, 1.0, image, f=self.f, t=self.t, mult=1.0, nn=7,
            n_jobs=1, switch_impulse_only=True,
            reject_impulse_candidates=True, use_aswmf_spatial_weights=True,
            impulse_tolerance=0,
        )
        for output in (gnlm_out, ghnlm_out):
            self.assertEqual(output.shape, image.shape)
            self.assertTrue(np.isfinite(output).all())

    def test_nlmedians_uses_full_window_with_f_plus_t_padding(self):
        image = np.array(
            [[0, 10, 20, 30], [40, 50, 60, 70],
             [80, 90, 100, 110], [120, 130, 140, 150]],
            dtype=np.float32,
        )
        f, t, h = 1, 2, 80.0
        padding = f + t
        padded = np.pad(image, padding, mode='symmetric')
        expected = np.empty_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                im, jn = i + padding, j + padding
                target = padded[im - f:im + f + 1, jn - f:jn + f + 1]
                weighted_sum = 0.0
                weight_sum = 0.0
                for r in range(im - t, im + t + 1):
                    for s in range(jn - t, jn + t + 1):
                        patch = padded[r - f:r + f + 1, s - f:s + f + 1]
                        median = np.median(patch)
                        # Match the published implementation: its dispersion
                        # is computed around the patch median.
                        dispersion = np.sqrt(np.mean((patch - median) ** 2))
                        column_l1 = np.max(np.sum(np.abs(target - patch), axis=0))
                        weight = np.exp(-column_l1 / (h * h))
                        value = padded[r, s]
                        if not median - 1.96 * dispersion < value < median + 1.96 * dispersion:
                            value = median
                        weighted_sum += weight * value
                        weight_sum += weight
                expected[i, j] = weighted_sum / weight_sum
        actual = NLMedians(image, h=h, f=f, t=t)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
