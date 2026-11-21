"""Missing measurements must not invent edges or physical displacement."""
import unittest

import numpy as np

from train_physics_fall import features, augment


class DropoutFeatures(unittest.TestCase):
    def test_connected_patch_split_has_no_spurious_gradient_or_velocity(self):
        raw = np.full((6, 8, 8), -1, np.int16)
        raw[:, 1:7, 1:7] = 1200
        raw[1:3, 3, 1:7] = -1  # disconnect the top and bottom of the same surface
        raw[3:5, 1:3, 1:7] = -1  # a missing top shifts the naive center down
        x = features(raw.reshape(6, 64))
        self.assertGreater(abs(float(x[3, 321]-x[0, 321])), .1)
        np.testing.assert_allclose(x[:, 128:320], 0, atol=1e-7)
        np.testing.assert_allclose(x[:, 326:332], 0, atol=1e-6)

    def test_valid_distance_gradient_and_change_are_kept(self):
        raw = np.tile((np.arange(64).reshape(8, 8)*20+1000)[None], (2, 1, 1)).astype(np.int16)
        raw[1] += 100
        x = features(raw.reshape(2, 64))
        np.testing.assert_allclose(x[:, 128:192], .005, atol=1e-7)
        np.testing.assert_allclose(x[:, 192:256], .04, atol=1e-7)
        np.testing.assert_allclose(x[1, 256:320], .025, atol=1e-7)
        self.assertAlmostEqual(float(x[1, 328]), .025, places=6)
        self.assertAlmostEqual(float(x[1, 329]), .025, places=6)

    def test_little_overlap_does_not_create_motion(self):
        raw = np.full((2, 64), -1, np.int16)
        raw[0, :32] = 1000
        raw[1, 28:] = 2000
        x = features(raw)
        np.testing.assert_allclose(x[1, 326:332], 0)

    def test_uint16_sentinel_and_batch_agree(self):
        raw = np.full((4, 64), 1200, np.int32)
        raw[1, 24:32] = 65535
        normalized = np.where(raw==65535, -1, raw)
        np.testing.assert_allclose(features(raw), features(normalized))
        np.testing.assert_allclose(features(raw), features(raw[None])[0])

    def test_augmentation_never_produces_unusable_frames(self):
        rng = np.random.default_rng(9)
        raw = np.full((24, 64), -1, np.int16)
        raw[:, :4] = 1500
        for _ in range(30):
            result = augment(raw, rng)
            self.assertTrue(np.all((result>=0).sum(-1)>=4))


if __name__ == '__main__':
    unittest.main()
