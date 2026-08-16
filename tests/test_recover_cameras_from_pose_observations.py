import unittest

import numpy as np

from tools.recover_cameras_from_pose_observations import (
    camera_record,
    heldout_split,
    recovery_gate,
)


class RecoverCamerasFromPoseObservationsTest(unittest.TestCase):
    def test_heldout_split_is_disjoint_deterministic_and_distributed(self) -> None:
        fit_a, held_a = heldout_split(101, 0.2, 7)
        fit_b, held_b = heldout_split(101, 0.2, 7)
        np.testing.assert_array_equal(fit_a, fit_b)
        np.testing.assert_array_equal(held_a, held_b)
        self.assertFalse(np.any(fit_a & held_a))
        self.assertTrue(np.all(fit_a | held_a))
        self.assertGreaterEqual(int(held_a[:50].sum()), 9)
        self.assertGreaterEqual(int(held_a[50:].sum()), 9)

    def test_recovery_gate_requires_no_go_and_heldout_improvement(self) -> None:
        fit = {"median_px": 5.0, "p90_px": 20.0}
        current = {"median_px": 40.0, "p90_px": 200.0}
        recovered = {
            "median_px": 6.0,
            "p90_px": 24.0,
            "valid_joint_count": 500,
        }
        accepted, reasons = recovery_gate(
            "NO_GO_TRIANGULATION", fit, current, recovered, 500, 400, 10.0
        )
        self.assertTrue(accepted, reasons)
        accepted, reasons = recovery_gate(
            "REVIEW_POSE_CAMERA_CONSISTENCY",
            fit,
            current,
            recovered,
            500,
            400,
            10.0,
        )
        self.assertFalse(accepted)
        self.assertTrue(any("not NO_GO" in reason for reason in reasons))

    def test_camera_record_has_consistent_inverse_and_center(self) -> None:
        angle = np.deg2rad(20.0)
        rotation = np.asarray(
            [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
        )
        translation = np.asarray([0.2, -0.1, 1.0])
        record = camera_record({"intrinsic": np.eye(3).tolist()}, rotation, translation)
        extrinsic = np.eye(4)
        extrinsic[:3] = np.asarray(record["extrinsic_world_to_camera"])
        camera_to_world = np.asarray(record["camera_to_world"])
        np.testing.assert_allclose(extrinsic @ camera_to_world, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(record["camera_center_world"], camera_to_world[:3, 3])


if __name__ == "__main__":
    unittest.main()
