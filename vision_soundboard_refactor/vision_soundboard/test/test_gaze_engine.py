import unittest
import numpy as np
from vision_soundboard.engine.gaze_engine import GazeEngine
from vision_soundboard.engine.feature_extractor import FeatureExtractor

class DummyExtractor:
    def extract(self, frame_bgr):
        # Always return fixed feature for testing
        return {
            "eye_cx_norm": 0.6,
            "eye_cy_norm": 0.4,
            "yaw": 10.0,
            "pitch": 5.0,
            "quality": 1.0
        }

class TestGazeEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GazeEngine(DummyExtractor(), screen_w=1920, screen_h=1080)

    def test_set_comp_params(self):
        self.engine.set_comp_params(0.5, 20, 0.4, 15)
        self.assertEqual(self.engine.k_yaw, 0.5)
        self.assertEqual(self.engine.c_yaw, 20)
        self.assertEqual(self.engine.k_pitch, 0.4)
        self.assertEqual(self.engine.c_pitch, 15)

    def test_set_bias_gain(self):
        self.engine.set_bias(0.1, -0.1)
        self.engine.set_gain(1.2)
        self.assertEqual(self.engine.bias_x, 0.1)
        self.assertEqual(self.engine.bias_y, -0.1)
        self.assertEqual(self.engine.gain, 1.2)

    def test_set_smoothing(self):
        self.engine.set_smoothing(mincutoff=0.7, beta=0.02, dcutoff=1.5)
        self.assertEqual(self.engine.sess.smooth_x.mincutoff, 0.7)
        self.assertEqual(self.engine.sess.smooth_y.beta, 0.02)
        self.assertEqual(self.engine.sess.smooth_x.dcutoff, 1.5)

    def test_set_deadzone(self):
        self.engine.set_deadzone(0.05)
        self.assertEqual(self.engine.deadzone, 0.05)

    def test_process_frame_outlier(self):
        # Simulate outlier jump
        self.engine._last_xy = (0.0, 0.0)
        x, y = self.engine.process_frame(None, self.engine.ext)
        self.assertTrue(0.0 <= x <= 1.0)
        self.assertTrue(0.0 <= y <= 1.0)

    def test_calibration_finish(self):
        features = [[0.5,0.5,0,0],[0.6,0.4,10,5]]
        targets = [[0.5,0.5],[0.6,0.4]]
        qualities = [1.0, 0.8]
        self.engine.calibration_finish(features, targets, qualities)
        self.assertTrue(self.engine.sess.model_ready)
        self.assertEqual(self.engine.sess.affine.shape[0], 4)

if __name__ == "__main__":
    unittest.main()
