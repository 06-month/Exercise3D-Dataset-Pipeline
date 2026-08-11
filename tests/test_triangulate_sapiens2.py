import json
import unittest
from pathlib import Path

import numpy as np

from tools.triangulate_sapiens2 import (
    interpolate_observations,
    load_canonical,
    pose_camera_consistency_status,
    triangulate_joint,
)


class TriangulateSapiens2Test(unittest.TestCase):
    def test_linear_trajectory_interpolation_uses_conservative_confidence(self) -> None:
        timestamps = np.asarray([0.0, 1.0], dtype=np.float64)
        xy = np.asarray([[[0.0, 0.0]], [[10.0, 20.0]]], dtype=np.float32)
        confidence = np.asarray([[0.9], [0.7]], dtype=np.float32)
        result = interpolate_observations(
            timestamps,
            xy,
            confidence,
            np.asarray([True, True]),
            np.asarray([0.25]),
            max_gap=2.0,
        )

        np.testing.assert_allclose(result["xy"][0, 0], [2.5, 5.0])
        self.assertAlmostEqual(float(result["confidence"][0, 0]), 0.7)
        self.assertTrue(result["interpolated"][0])
        self.assertEqual(float(result["pairing_error_ms"][0]), 0.0)

    def test_weighted_triangulation_recovers_known_point(self) -> None:
        intrinsic = np.asarray(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]]
        )
        extrinsic_a = np.eye(4)
        extrinsic_b = np.eye(4)
        extrinsic_b[0, 3] = -1.0
        extrinsics = np.stack([extrinsic_a, extrinsic_b])
        projections = np.stack(
            [intrinsic @ extrinsic_a[:3], intrinsic @ extrinsic_b[:3]]
        )
        observations = np.asarray([[0.0, 0.0], [-200.0, 0.0]])

        result = triangulate_joint(
            observations,
            np.asarray([0.9, 0.8]),
            projections,
            np.stack([intrinsic, intrinsic]),
            np.stack([np.eye(3), np.eye(3)]),
            extrinsics,
            min_confidence=0.3,
            huber_scale_px=10.0,
        )

        self.assertTrue(result["valid"])
        self.assertEqual(result["support"], 2)
        np.testing.assert_allclose(result["point"], [0.0, 0.0, 5.0], atol=1e-5)
        np.testing.assert_allclose(result["reprojection"], [0.0, 0.0], atol=1e-5)

    def test_canonical_mapping_names_and_indices_are_explicit(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "sapiens2_canonical_joints.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        names = [f"unused_{index}" for index in range(308)]
        for row in config["direct"]:
            names[int(row["source_index"])] = row["source_name"]

        loaded = load_canonical(config_path, names)

        direct = {row["canonical"]: row["source_index"] for row in loaded["direct"]}
        self.assertEqual(direct["left_wrist"], 62)
        self.assertEqual(direct["right_wrist"], 41)
        self.assertIn("pelvis_center", [row["canonical"] for row in loaded["derived"]])

    def test_pose_camera_gate_is_tied_to_robust_scale(self) -> None:
        self.assertEqual(
            pose_camera_consistency_status(5.0, 20.0, "PASS", 10.0), "PASS"
        )
        self.assertEqual(
            pose_camera_consistency_status(8.0, 40.0, "PASS", 10.0),
            "REVIEW_POSE_CAMERA_CONSISTENCY",
        )
        self.assertEqual(
            pose_camera_consistency_status(21.0, 50.0, "PASS", 10.0),
            "NO_GO_TRIANGULATION",
        )


if __name__ == "__main__":
    unittest.main()
