"""Synthetic behavioral checks; these are not real-road accuracy measurements."""
import unittest
import cv2
import numpy as np
from lane_detection_fusion.core import LaneFusion, FusionConfig, draw_result


def road(left=True, right=True, shift=0):
    image = np.full((360, 640, 3), 30, np.uint8)
    for visible, x in ((left, 160 + shift), (right, 480 + shift)):
        if visible:
            cv2.line(image, (x, 165), (x, 345), (240, 240, 240), 6)
    return image


class CoreTests(unittest.TestCase):
    def test_pair_establishes_center_and_straight_command(self):
        result = LaneFusion().process(road(), 1.0)
        self.assertEqual(result.state, "both")
        self.assertGreater(result.speed, 0)
        self.assertLess(abs(result.angular), .02)
        self.assertLess(abs(float(result.center[:, 0].mean()) - 320), 2)

    def test_known_left_and_right_keep_same_center_and_slow_down(self):
        for left, right, state in ((True, False, "left_only"), (False, True, "right_only")):
            tracker = LaneFusion()
            initial = tracker.process(road(), 1.0)
            result = tracker.process(road(left, right), 1.08)
            self.assertEqual(result.state, state)
            self.assertLess(result.speed, initial.speed)
            np.testing.assert_allclose(result.center, initial.center, atol=2)

    def test_new_single_boundary_never_invents_role(self):
        for left, right in ((True, False), (False, True)):
            result = LaneFusion().process(road(left, right), 1.0)
            self.assertEqual(result.state, "ambiguous")
            self.assertEqual((result.speed, result.angular), (0, 0))

    def test_lost_does_not_reuse_old_command(self):
        tracker = LaneFusion()
        tracker.process(road(shift=35), 1.0)
        result = tracker.process(road(False, False), 1.08)
        self.assertEqual(result.state, "lost")
        self.assertEqual((result.speed, result.angular), (0, 0))

    def test_long_gap_drops_boundary_identity(self):
        tracker = LaneFusion()
        tracker.process(road(), 1.0)
        result = tracker.process(road(True, False), 2.0)
        self.assertEqual(result.state, "ambiguous")
        self.assertTrue(result.debug.get("gap_reset"))

    def test_steering_sign_and_limits(self):
        for shift, sign in ((45, -1), (-45, 1)):
            result = LaneFusion().process(road(shift=shift), 1.0)
            self.assertGreater(sign * result.angular, 0)
            self.assertLessEqual(abs(result.angular), FusionConfig().max_angular)
            self.assertLessEqual(result.speed, FusionConfig().max_speed)

    def test_bad_observations_are_zero(self):
        tracker = LaneFusion()
        for stamp in (float("nan"), "bad"):
            result = tracker.process(road(), stamp)
            self.assertEqual(result.state, "invalid")
            self.assertEqual((result.speed, result.angular), (0, 0))
        tracker.process(road(), 2.0)
        self.assertEqual(tracker.process(road(), 1.9).state, "invalid")
        self.assertEqual(tracker.process(np.zeros((5, 5), np.uint8), 3).state, "invalid")

    def test_debug_does_not_modify_source(self):
        source = road()
        original = source.copy()
        overlay = draw_result(source, LaneFusion().process(source, 1.0))
        np.testing.assert_array_equal(source, original)
        self.assertEqual(overlay.shape, source.shape)


if __name__ == "__main__":
    unittest.main()
