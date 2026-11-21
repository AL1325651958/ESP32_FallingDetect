"""An inference self-test alone must not hide failed sensor startup."""
import unittest
from capture_fall_device import summarize


class CaptureDiagnostics(unittest.TestCase):
    def test_sensor_startup_failure_is_visible(self):
        report = summarize('LSTM self-test PASS\nE (669) falling_detect: VL53L8CX not detected (status=8, alive=0)\n')
        self.assertTrue(report['self_test_pass'])
        self.assertFalse(report['ready_10hz'])
        self.assertEqual(report['activity_frames'], 0)
        self.assertIsNone(report['full_inference_us'])
        self.assertIn('VL53L8CX not detected', report['warning_lines'][0])

    def test_only_new_full_window_inferences_enter_timing(self):
        text = '\n'.join([
            'LSTM self-test PASS', 'ready: 8x8 at 10 Hz',
            'activity buffered=23 inference_us=5 capture_ms=100 queue_ms=0 queue_drops=0 updated=0',
            'activity buffered=24 inference_us=15000 capture_ms=200 queue_ms=1 queue_drops=0 updated=1',
            'activity buffered=24 inference_us=40 capture_ms=300 queue_ms=0 queue_drops=0 updated=0'])
        report = summarize(text)
        self.assertEqual(report['full_inference_us']['count'], 1)
        self.assertEqual(report['full_inference_us']['median'], 15000)
        self.assertEqual(report['capture_interval_ms']['median'], 100)
        self.assertEqual(report['queue_wait_ms']['max'], 1)


if __name__ == '__main__':
    unittest.main()
