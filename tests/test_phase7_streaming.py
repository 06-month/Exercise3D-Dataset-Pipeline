import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.run_phase7_streaming import (
    pose_camera_ready,
    read_triangulation,
    recovery_accepted,
)


class Phase7StreamingTest(unittest.TestCase):
    def test_pose_readiness_requires_complete_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "data" / "final_frame" / "lift" / "lift_0000" / "cam1"
            frames.mkdir(parents=True)
            for index in range(2):
                (frames / f"{index:06d}.jpg").touch()
            output = root / "pose" / "lift_0000" / "cam1"
            output.mkdir(parents=True)
            (output / "metadata.json").write_text(
                json.dumps({"qa": {"status": "PASS"}}), encoding="utf-8"
            )
            np.savez_compressed(
                output / "poses_2d.npz",
                frame_index=np.asarray([0, 1], dtype=np.int32),
                keypoints_xy=np.zeros((2, 308, 2), dtype=np.float32),
                confidence=np.ones((2, 308), dtype=np.float32),
            )
            self.assertTrue(
                pose_camera_ready(root / "data", root / "pose", "lift_0000", "cam1")
            )

    def test_recovery_requires_disjoint_heldout_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "lift_0000"
            sequence.mkdir()
            payload = {
                "eligible_for_triangulation": True,
                "fit_heldout_overlap_count": 0,
                "recovered_pose_camera_status_heldout": "REVIEW_POSE_CAMERA_CONSISTENCY",
            }
            (sequence / "validation.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            self.assertTrue(recovery_accepted(root, "lift_0000"))
            payload["fit_heldout_overlap_count"] = 1
            (sequence / "validation.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            self.assertFalse(recovery_accepted(root, "lift_0000"))

    def test_triangulation_read_rejects_nonfinite_valid_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "lift_0000"
            sequence.mkdir()
            (sequence / "metadata.json").write_text(
                json.dumps({"qa": {"schema_status": "PASS"}}), encoding="utf-8"
            )
            np.savez_compressed(
                sequence / "canonical_3d.npz",
                keypoints_3d=np.asarray([[[np.nan, 0.0, 0.0]]], dtype=np.float32),
                valid_mask=np.asarray([[True]]),
            )
            self.assertIsNone(read_triangulation(root, "lift_0000"))


if __name__ == "__main__":
    unittest.main()
