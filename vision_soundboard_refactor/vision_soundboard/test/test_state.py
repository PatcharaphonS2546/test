import unittest
import numpy as np
from vision_soundboard.ui.state import AppState

class TestAppState(unittest.TestCase):
    def setUp(self):
        self.state = AppState()

    def test_calibration_targets(self):
        self.assertIsInstance(self.state.targets, list)
        self.assertTrue(all(isinstance(t, tuple) and len(t) == 2 for t in self.state.targets))

    def test_calib_qualities(self):
        # calib_qualities may be empty initially, but should be a list
        self.assertIsInstance(self.state.calib_qualities, list)
        # If populated, should be floats between 0 and 1
        for q in self.state.calib_qualities:
            self.assertTrue(isinstance(q, float) and 0 <= q <= 1)

    def test_update_gaze_and_history(self):
        x, y, quality = 0.3, 0.7, 0.9
        self.state.update_gaze(x, y, quality)
        self.assertAlmostEqual(self.state.gx, x)
        self.assertAlmostEqual(self.state.gy, y)
        # History only stores (x, y), not quality
        self.state.add_gaze_history(x, y)
        self.assertGreaterEqual(len(self.state.gaze_history), 1)
        last = self.state.gaze_history[-1]
        self.assertEqual(last, (x, y))

    def test_set_param(self):
        self.state.set_param('gx', 0.1)
        self.state.set_param('gy', 0.2)
        self.assertEqual(self.state.gx, 0.1)
        self.assertEqual(self.state.gy, 0.2)
        self.state.set_param('unknown_param', 123)  # Should not raise

if __name__ == "__main__":
    unittest.main()
