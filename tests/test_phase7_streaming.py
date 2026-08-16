import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.run_phase7_streaming import (
    mark_triangulation_in_progress,
    pose_camera_ready,
    read_triangulation,
    recovery_accepted,
    triangulation_source_identity,
    write_triangulation_source_identity,
)


class Phase7StreamingTest(unittest.TestCase):
    def make_source_tree(self, root: Path) -> argparse.Namespace:
        dataset = root / "data"
        pose = root / "pose"
        camera = root / "camera"
        sequence = "lift_0000"
        for camera_name in ("cam1", "cam2", "cam3"):
            pose_dir = pose / sequence / camera_name
            pose_dir.mkdir(parents=True)
            (pose_dir / "metadata.json").write_text("{}", encoding="utf-8")
            (pose_dir / "poses_2d.npz").write_bytes(b"pose")
            frame_dir = (
                dataset
                / "final_frame"
                / "lift"
                / sequence
                / camera_name
            )
            frame_dir.mkdir(parents=True)
            (frame_dir / "000000.jpg").write_bytes(b"frame")
        camera_dir = camera / sequence
        camera_dir.mkdir(parents=True)
        (camera_dir / "cameras_refined.json").write_text("{}", encoding="utf-8")
        (camera_dir / "validation.json").write_text("{}", encoding="utf-8")
        temporal = dataset / "reports" / "temporal_alignment"
        temporal.mkdir(parents=True)
        (temporal / "pair_summary.csv").write_text("set_id\n", encoding="utf-8")
        vggt = dataset / "outputs" / "vggt" / "a" / "b" / sequence
        vggt.mkdir(parents=True)
        (vggt / "metadata.json").write_text("{}", encoding="utf-8")
        return argparse.Namespace(
            dataset_root=dataset,
            pose_root=pose,
            camera_root=camera,
        )

    def make_valid_triangulation(self, root: Path, sequence: str) -> None:
        output = root / sequence
        output.mkdir(parents=True)
        (output / "metadata.json").write_text(
            json.dumps({"qa": {"schema_status": "PASS"}}), encoding="utf-8"
        )
        np.savez_compressed(
            output / "canonical_3d.npz",
            keypoints_3d=np.zeros((1, 1, 3), dtype=np.float32),
            valid_mask=np.asarray([[True]]),
        )

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

    def test_source_identity_changes_with_pose_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_source_tree(root)
            before = triangulation_source_identity(
                args, "lift_0000", args.camera_root
            )
            pose = args.pose_root / "lift_0000" / "cam2" / "poses_2d.npz"
            pose.write_bytes(b"changed pose")
            after = triangulation_source_identity(
                args, "lift_0000", args.camera_root
            )
            self.assertNotEqual(
                before["dependency_signature_sha256"],
                after["dependency_signature_sha256"],
            )

    def test_triangulation_reuse_requires_matching_source_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_source_tree(root)
            output = root / "triangulation"
            sequence = "lift_0000"
            self.make_valid_triangulation(output, sequence)
            identity = triangulation_source_identity(args, sequence, args.camera_root)
            self.assertIsNone(
                read_triangulation(
                    output, sequence, identity, "PHASE5_BACKGROUND_BA"
                )
            )
            write_triangulation_source_identity(
                output, sequence, identity, "PHASE5_BACKGROUND_BA"
            )
            self.assertIsNotNone(
                read_triangulation(
                    output, sequence, identity, "PHASE5_BACKGROUND_BA"
                )
            )
            self.assertIsNone(
                read_triangulation(
                    output,
                    sequence,
                    identity,
                    "REVIEW_OBSERVATION_CONDITIONED",
                )
            )
            mark_triangulation_in_progress(
                output, sequence, "PHASE5_BACKGROUND_BA"
            )
            self.assertIsNone(
                read_triangulation(
                    output, sequence, identity, "PHASE5_BACKGROUND_BA"
                )
            )

    def test_source_identity_rejects_symlink_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_source_tree(root)
            pose = args.pose_root / "lift_0000" / "cam1" / "poses_2d.npz"
            target = root / "outside.npz"
            target.write_bytes(b"pose")
            pose.unlink()
            os.symlink(target, pose)
            with self.assertRaisesRegex(RuntimeError, "unsafe Phase 7 dependency"):
                triangulation_source_identity(args, "lift_0000", args.camera_root)


if __name__ == "__main__":
    unittest.main()
