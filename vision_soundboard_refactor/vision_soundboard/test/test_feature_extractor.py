import unittest
import numpy as np
from vision_soundboard.engine.feature_extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureExtractor(use_mediapipe=False)

    def test_median_xy_no_outlier(self):
        pts = [(1,2,0), (2,3,0), (3,4,0), (4,5,0), (5,6,0)]
        ids = [0,1,2,3,4]
        median = self.fe._median_xy(ids, pts)
        self.assertEqual(median, (3.0, 4.0))

    def test_median_xy_with_outlier(self):
        pts = [(1,2,0), (2,3,0), (100,200,0), (3,4,0), (4,5,0)]
        ids = [0,1,2,3,4]
        median = self.fe._median_xy(ids, pts)
        self.assertAlmostEqual(median[0], 2.5, delta=1.0)
        self.assertAlmostEqual(median[1], 3.5, delta=1.0)

    def test_eye_box_metrics(self):
        pts = [(1,2,0), (2,3,0), (3,4,0), (4,5,0)]
        ids = [0,1,2,3]
        box = self.fe._eye_box_metrics(ids, pts)
        self.assertIsNotNone(box)
        self.assertEqual(box[0], 1)
        self.assertEqual(box[1], 4)
        self.assertEqual(box[2], 2)
        self.assertEqual(box[3], 5)

    def test_norm_in_box(self):
        p = (3,4)
        box = (1,4,2,5,3,3,0.5)
        norm = self.fe._norm_in_box(p, box)
        self.assertTrue(0.0 <= norm[0] <= 1.0)
        self.assertTrue(0.0 <= norm[1] <= 1.0)

    def test_extract_fallback(self):
        result = self.fe.extract(None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['quality'], 0.2)

if __name__ == '__main__':
    unittest.main()
