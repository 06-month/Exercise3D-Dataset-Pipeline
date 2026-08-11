import unittest

import numpy as np

from tools.assess_sam_mode_c_escalation import bounded_clips, robust_threshold


class SamModeCEscalationTest(unittest.TestCase):
    def test_robust_threshold_rejects_isolated_large_value(self) -> None:
        values = np.asarray([1.0] * 20 + [10.0])
        threshold, median, mad = robust_threshold(values)
        self.assertEqual(median, 1.0)
        self.assertEqual(mad, 0.0)
        self.assertLess(threshold, 10.0)

    def test_bounded_clips_respects_fraction(self) -> None:
        candidate = np.zeros(100, dtype=np.bool_)
        candidate[[10, 50, 90]] = True
        severity = np.zeros(100)
        severity[[10, 50, 90]] = [1.0, 3.0, 2.0]
        selected, clips = bounded_clips(candidate, severity, padding=4, maximum_fraction=0.1)
        self.assertLessEqual(int(selected.sum()), 10)
        self.assertTrue(selected[50])
        self.assertTrue(clips)


if __name__ == "__main__":
    unittest.main()
