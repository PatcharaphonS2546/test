import unittest
from vision_soundboard.engine.filters import OneEuroFilter

class TestOneEuroFilter(unittest.TestCase):
    def test_basic_filter(self):
        f = OneEuroFilter(freq=60.0, mincutoff=0.5, beta=0.01, dcutoff=1.0)
        vals = [0.0, 0.1, 0.2, 0.3, 0.4]
        filtered = [f.filter(v) for v in vals]
        self.assertTrue(all(0.0 <= v <= 0.4 for v in filtered))
        self.assertTrue(filtered[-1] > filtered[0])

    def test_dynamic_params(self):
        f = OneEuroFilter(freq=60.0, mincutoff=0.5, beta=0.01, dcutoff=1.0)
        f.set_dynamic_params(quality=0.9)
        self.assertTrue(0.3 <= f.mincutoff <= 1.6)
        self.assertTrue(0.01 <= f.beta <= 0.13)
        f.set_dynamic_params(quality=0.1)
        self.assertTrue(0.3 <= f.mincutoff <= 1.6)
        self.assertTrue(0.01 <= f.beta <= 0.13)

    def test_set_params(self):
        f = OneEuroFilter(freq=60.0, mincutoff=0.5, beta=0.01, dcutoff=1.0)
        f.set_params(mincutoff=0.8, beta=0.05, dcutoff=2.0)
        self.assertEqual(f.mincutoff, 0.8)
        self.assertEqual(f.beta, 0.05)
        self.assertEqual(f.dcutoff, 2.0)

    def test_outlier(self):
        f = OneEuroFilter(freq=60.0, mincutoff=0.5, beta=0.01, dcutoff=1.0)
        vals = [0.0, 0.1, 0.2, 5.0, 0.3]
        filtered = [f.filter(v) for v in vals]
        self.assertTrue(abs(filtered[3] - filtered[2]) < 5.0)  # outlier should be smoothed

if __name__ == "__main__":
    unittest.main()
