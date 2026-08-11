import unittest

import numpy as np

from tools.fit_sequence_body import evaluate_fit_gate, smooth_track, weighted_similarity


class SequenceBodyFitTest(unittest.TestCase):
    def test_weighted_similarity_recovers_known_transform_with_outlier(self) -> None:
        rng = np.random.default_rng(7)
        source = rng.normal(size=(20, 3))
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        target = 2.5 * (source @ q.T) + np.asarray([4.0, -2.0, 1.0])
        target[-1] += 10.0
        weights = np.ones(20)
        weights[-1] = 0.01
        scale, rotation, translation, _ = weighted_similarity(source, target, weights)
        predicted = scale * (source[:-1] @ rotation.T) + translation
        self.assertLess(float(np.max(np.abs(predicted - target[:-1]))), 0.03)

    def test_temporal_fit_preserves_anchors_and_reduces_second_difference(self) -> None:
        frames = np.arange(50, dtype=np.float64)
        clean = np.stack([frames, frames * 0.5, -frames], axis=1)
        noisy = clean.copy()
        noisy[1:-1:2] += np.asarray([0.3, -0.2, 0.1])
        weights = np.full(50, 8.0)
        fitted = smooth_track(noisy, weights, temporal_weight=0.25)
        before = np.linalg.norm(np.diff(noisy, n=2, axis=0), axis=1).mean()
        after = np.linalg.norm(np.diff(fitted, n=2, axis=0), axis=1).mean()
        self.assertLess(after, before)
        self.assertLess(float(np.max(np.linalg.norm(fitted - noisy, axis=1))), 0.1)

    def test_quality_gate_separates_review_and_fail(self) -> None:
        config = {
            "review_if_any": {
                "final_valid_joint_fraction_below": 0.95,
                "alignment_success_fraction_below": 0.9,
                "observation_displacement_p95_normalized_above": 0.05,
                "prior_only_joint_fraction_above": 0.02,
                "median_bone_length_cv_above": 0.1,
                "camera_status_not_pass": True,
            },
            "fail_if_any": {
                "final_valid_joint_fraction_below": 0.8,
                "observation_displacement_p95_normalized_above": 0.2,
            },
        }
        qa = {
            "final_valid_joint_fraction": 0.99,
            "alignment_success_fraction": 0.95,
            "prior_only_joint_fraction": 0.0,
            "observation_displacement_p95_normalized": 0.01,
            "median_bone_length_cv": 0.02,
            "anthropometry": {"reference_length_sequence_gauge": 1.0},
            "finite_valid_points": True,
            "invalid_points_are_nan": True,
            "triangulation_camera_status": "REVIEW_POSE_CAMERA_CONSISTENCY",
        }
        status, review, fail = evaluate_fit_gate(qa, config)
        self.assertEqual(status, "REVIEW_BODY_FIT_QUALITY")
        self.assertIn("CAMERA_UNCERTAINTY", review)
        self.assertFalse(fail)
        qa["final_valid_joint_fraction"] = 0.5
        status, _, fail = evaluate_fit_gate(qa, config)
        self.assertEqual(status, "FAIL_BODY_FIT_QUALITY")
        self.assertIn("FINAL_VALID_JOINT_FRACTION", fail)


if __name__ == "__main__":
    unittest.main()
