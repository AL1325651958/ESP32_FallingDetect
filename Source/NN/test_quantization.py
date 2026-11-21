"""Exercise quantizer rounding, scale, accumulation and determinism."""
import unittest
import warnings
import numpy as np
from quantize_physics_fall import quantize_rows, quantize_activation, matvec, quantized_weights


class QuantizationTests(unittest.TestCase):
    def test_zero_rows_have_finite_scale(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            q, scale = quantize_rows(np.zeros((3, 192), np.float32))
        self.assertEqual(q.dtype, np.int8)
        self.assertFalse(q.any())
        self.assertTrue(np.isfinite(scale).all())
        x, s = quantize_activation(np.zeros(192, np.float32))
        self.assertFalse(x.any())
        self.assertEqual(s, 0)

    def test_signed_half_rounds_away_and_symmetric_endpoints(self):
        raw = np.array([-127, -1.5, -.5, 0, .5, 1.5, 127], np.float32)
        q, scale = quantize_activation(raw)
        np.testing.assert_array_equal(q, [-127, -2, -1, 0, 1, 2, 127])
        self.assertEqual(scale, 1)
        weights, scales = quantize_rows(raw[None])
        np.testing.assert_array_equal(weights[0], q)
        self.assertEqual(scales[0], 1)

    def test_output_rows_have_independent_scales(self):
        raw = np.array([[1, -1], [.01, -.01]], np.float32)
        q, scale = quantize_rows(raw)
        np.testing.assert_array_equal(q, [[127, -127], [127, -127]])
        np.testing.assert_allclose(q.astype(np.float32)*scale[:, None], raw, atol=1e-8)

    def test_int32_accumulation_does_not_overflow_int8(self):
        q = {'w': np.full((1, 192), 127, np.int8), 'w.scale': np.ones(1, np.float32)}
        np.testing.assert_allclose(matvec('w', np.full(192, 127, np.float32), q), [192*127*127])

    def test_export_excludes_training_head_without_mutating_source(self):
        source = {'spatial.weight': np.array([[.1, .2]], np.float32),
            'local.weight': np.ones((1, 1), np.float32),
            'lstm.weight_ih_l0': np.ones((1, 1), np.float32),
            'lstm.weight_hh_l0': np.ones((1, 1), np.float32),
            'head.weight': np.ones((1, 1), np.float32),
            'out.weight': np.ones((1, 1), np.float32),
            'out.bias': np.ones(1, np.float32), 'velocity.weight': np.ones((1, 1), np.float32)}
        before = {k: v.copy() for k, v in source.items()}
        first, second = quantized_weights(source), quantized_weights(source)
        self.assertNotIn('velocity.weight', first)
        self.assertEqual(first['out.bias'].dtype, np.float32)
        for name in source:
            np.testing.assert_array_equal(source[name], before[name])
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


if __name__ == '__main__':
    unittest.main()
