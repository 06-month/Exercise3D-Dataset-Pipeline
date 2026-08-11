import unittest

import numpy as np

from tools.evaluate_fit3d_metrics import evaluate_metrics


class Fit3dMetricsTest(unittest.TestCase):
    def test_scale_only_alignment_removes_scale(self) -> None:
        rng = np.random.default_rng(11)
        target = rng.normal(size=(4, 17, 3))
        prediction = target * 2.0 + np.asarray([3.0, -2.0, 1.0])
        result = evaluate_metrics(
            prediction, target, np.ones((4, 17), dtype=np.bool_), 0, 1000.0
        )
        self.assertGreater(result["mpjpe_mm"], 1.0)
        self.assertLess(result["n_mpjpe_mm"], 1e-9)
        self.assertLess(result["pa_mpjpe_mm"], 1e-9)

    def test_procrustes_removes_rotation_not_scale_only(self) -> None:
        rng = np.random.default_rng(13)
        target = rng.normal(size=(2, 17, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        prediction = 1.7 * (target @ q) + np.asarray([1.0, 2.0, -1.0])
        result = evaluate_metrics(
            prediction, target, np.ones((2, 17), dtype=np.bool_), 0, 1000.0
        )
        self.assertGreater(result["n_mpjpe_mm"], 1.0)
        self.assertLess(result["pa_mpjpe_mm"], 1e-9)

    def test_nonfinite_root_excludes_the_frame(self) -> None:
        rng = np.random.default_rng(17)
        target = rng.normal(size=(2, 17, 3))
        prediction = target.copy()
        prediction[1, 0] = np.nan
        result = evaluate_metrics(
            prediction, target, np.ones((2, 17), dtype=np.bool_), 0, 1000.0
        )
        self.assertEqual(result["evaluated_frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
