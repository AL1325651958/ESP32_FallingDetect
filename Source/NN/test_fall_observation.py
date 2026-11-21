"""Regression scenarios for missing pixels and disappearance after fall evidence.

Inject high fall scores in a host-only build so tests verify that decisions remain
safe even if the unchanged network is fooled by an artificial hole. These tests
do not measure real-world fall recall or establish that a scene is unoccupied.
"""
import ctypes as ct
from pathlib import Path
import subprocess
import unittest

import numpy as np

from verify_physics_fall import Result, State

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'NN/artifacts/observation_guard_20261111'


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OUT.mkdir(parents=True, exist_ok=True)
        dll = OUT / 'observation_tests.dll'
        subprocess.run(['gcc', '-O3', '-Wall', '-Wextra', '-Werror', '-std=c11',
            '-shared', '-DFALL_TESTING', '-I', str(ROOT / 'main'),
            str(ROOT / 'main/fall_classifier.c'), str(ROOT / 'tests/fall_test_bridge.c'),
            '-o', str(dll), '-lm'], check=True)
        cls.lib = ct.CDLL(str(dll))
        cls.lib.fall_reset.argtypes = [ct.POINTER(State)]
        cls.lib.fall_push.argtypes = [ct.POINTER(State), ct.c_void_p, ct.c_uint32, ct.POINTER(Result)]
        cls.lib.fall_test_scores.argtypes = [ct.c_float, ct.c_float, ct.c_float]
        cls.lib.fall_encode_for_test.argtypes = [ct.c_void_p, ct.c_uint16, ct.POINTER(Result)]
        assert cls.lib.fall_state_size() == ct.sizeof(State)

    def setUp(self):
        self.state, self.result, self.now = State(), Result(), 1000
        self.lib.fall_reset(ct.byref(self.state))
        self.scores(.05, .90, .05)

    def scores(self, motion, fall, still):
        self.lib.fall_test_scores(motion, fall, still)

    def push(self, frame, count=1):
        raw = np.ascontiguousarray(frame, dtype=np.int16).reshape(64)
        for _ in range(count):
            self.lib.fall_push(ct.byref(self.state), raw.ctypes.data,
                self.now & 0xffffffff, ct.byref(self.result))
            self.now += 100
        return self.result

    @staticmethod
    def empty():
        return np.full(64, -1, np.int16)

    @staticmethod
    def patch():
        raw = np.full((8, 8), -1, np.int16)
        raw[1:7, 1:7] = 1200
        return raw

    def move(self, count=26):
        for t in range(count):
            self.push(np.full(64, 500 + t * 70, np.int16))

    def test_cold_missing_is_unknown_even_after_long_wait(self):
        self.push(self.empty(), 80)
        self.assertEqual((self.result.raw, self.result.stable, self.result.buffered), (3, 3, 0))
        self.assertFalse(any(self.result.p))

    def test_missing_then_reacquisition_has_clean_warmup(self):
        self.push(self.empty())
        for _ in range(23):
            self.push(self.patch())
            self.assertEqual((self.result.raw, self.result.stable), (-1, -1))
        self.push(self.patch())
        self.assertEqual(self.result.buffered, 24)
        self.assertNotIn(self.result.stable, (1, 3, 4))

    def test_split_component_and_top_dropout_do_not_invent_fall(self):
        for t in range(120):
            raw = self.patch()
            if t % 8 in (1, 2):
                raw[3, 1:7] = -1
            if t % 8 in (3, 4):
                raw[1:3, 1:7] = -1
            self.push(raw)
            self.assertNotIn(self.result.stable, (1, 4))
        self.assertEqual(self.result.raw, 1)  # network is deliberately fooled
        self.assertEqual(self.state.motion_history, 0)

    def test_gradient_surface_with_holes_does_not_invent_motion(self):
        surface = np.arange(64, dtype=np.int16).reshape(8, 8) * 20 + 700
        for t in range(70):
            raw = surface.copy()
            raw[t % 8, 1:7] = -1
            self.push(raw)
            self.assertNotIn(self.result.stable, (1, 4))

    def test_small_sensor_jitter_does_not_confirm_high_fall_score(self):
        for t in range(55):
            self.push(np.full(64, 1200 + (10 if t % 2 else -10), np.int16))
        self.assertNotIn(self.result.stable, (1, 4))

    def test_prior_ordinary_motion_cannot_license_later_hole_alarm(self):
        self.scores(.9, .05, .05)
        self.move(26)
        self.scores(.05, .9, .05)
        for t in range(45):
            raw = np.full((8, 8), 2250, np.int16)
            raw[3, 1:7] = -1
            if t % 3 == 0:
                raw[1, 2:6] = -1
            self.push(raw)
            self.assertNotIn(self.result.stable, (1, 4))

    def test_nonoverlapping_returns_do_not_invent_motion(self):
        for t in range(55):
            raw = self.empty()
            raw[:28] = 1000
            if t % 2:
                raw = self.empty()
                raw[36:] = 2000
            self.push(raw)
        self.assertEqual(self.state.motion_history, 0)
        self.assertNotEqual(self.result.stable, 1)

    def test_confirmed_fall_survives_complete_loss_and_time_gap(self):
        self.move()
        self.assertEqual(self.result.stable, 1)
        self.push(self.empty(), 35)
        self.assertEqual((self.result.raw, self.result.stable), (3, 1))
        self.now += 10000
        self.push(self.empty())
        self.assertEqual(self.result.stable, 1)
        self.push(self.patch(), 10)
        self.assertEqual((self.result.raw, self.result.stable), (-1, 1))

    def test_small_coherent_target_is_not_diluted_by_valid_background(self):
        for t in range(26):
            raw = np.full((8, 8), 3500, np.int16)
            raw[4:6, 3:6] = 800 + t * 65
            self.push(raw)
        self.assertEqual(self.result.stable, 1)

    def test_isolated_bad_points_do_not_supply_motion_evidence(self):
        for t in range(50):
            raw = np.full(64, 1200, np.int16)
            if t % 2:
                raw[[2, 26, 53]] += 150
            self.push(raw)
        self.assertNotIn(self.result.stable, (1, 4))

    def test_candidate_then_disappearance_is_suspected_not_vacant(self):
        self.move(24)
        self.assertNotEqual(self.result.stable, 1)
        self.push(self.empty())
        self.assertEqual((self.result.raw, self.result.stable), (3, 4))
        self.push(self.empty(), 50)
        self.assertEqual(self.result.stable, 4)

    def test_motion_then_disappearance_is_unknown_not_automatically_fall(self):
        self.scores(.9, .05, .05)
        self.move(30)
        self.assertEqual(self.result.stable, 0)
        self.push(self.empty())
        self.assertEqual((self.result.raw, self.result.stable), (3, 3))

    def test_descent_before_full_window_preserves_suspicion(self):
        yy = np.arange(8, dtype=np.float32)[:, None]
        for t in range(8):
            raw = np.broadcast_to(1600 + (3.5 - yy) * (50 + t * 35), (8, 8))
            self.push(raw)
        self.assertTrue(self.state.has_downward)
        self.push(self.empty())
        self.assertEqual((self.result.raw, self.result.stable), (3, 4))

    def test_few_points_cannot_clear_confirmed_fall(self):
        self.move()
        raw = self.empty()
        raw[:3] = 900
        self.push(raw, 20)
        self.assertEqual((self.result.raw, self.result.stable, self.result.valid), (3, 1, 3))

    def test_recovery_needs_full_window_then_stable_nonfall(self):
        self.move(24)
        self.push(self.empty())
        self.scores(.9, .05, .05)
        self.push(self.patch(), 23)
        self.assertEqual(self.result.stable, 4)
        self.push(self.patch())
        self.assertEqual(self.result.stable, 4)
        self.push(self.patch(), 4)
        self.assertEqual(self.result.stable, 0)

    def test_stale_movement_does_not_create_late_suspicion(self):
        self.move(24)
        self.scores(.9, .05, .05)
        self.push(np.full(64, 2110, np.int16), 35)
        self.push(self.empty())
        self.assertEqual(self.result.stable, 3)

    def test_timer_wrap_preserves_recent_fall_evidence(self):
        self.now = 0xffffff00
        self.move(24)
        self.push(self.empty())
        self.assertEqual(self.result.stable, 4)

    def test_short_gap_keeps_recent_candidate_but_breaks_window(self):
        self.move(24)
        self.now += 1000
        self.push(np.full(64, 2400, np.int16))
        self.assertEqual(self.result.buffered, 1)
        self.push(self.empty())
        self.assertEqual(self.result.stable, 4)

    def test_old_candidate_expires_across_long_gap(self):
        self.move(24)
        self.now += 5000
        self.push(self.empty())
        self.assertEqual(self.result.stable, 3)

    def test_reset_starts_a_new_episode(self):
        self.move()
        self.lib.fall_reset(ct.byref(self.state))
        self.push(self.empty())
        self.assertEqual(self.result.stable, 3)

    def test_wire_vectors_cover_alert_loss_and_recovery(self):
        vectors = []
        def record():
            packet = (ct.c_uint8 * 20)()
            self.lib.fall_encode_for_test(packet, 65535, ct.byref(self.result))
            vectors.append(bytes(packet).hex())
        self.push(self.empty()); record()
        self.move(24); record()
        self.push(self.empty()); record()
        self.push(self.patch()); record()
        self.scores(.9, .05, .05)
        self.push(self.patch(), 30); record()
        self.lib.fall_reset(ct.byref(self.state)); self.scores(.05, .9, .05)
        self.move(); record()
        self.push(self.empty()); record()
        self.push(self.patch()); record()
        (OUT / 'packet_vectors.txt').write_text('\n'.join(vectors) + '\n', encoding='ascii')


if __name__ == '__main__':
    unittest.main()
