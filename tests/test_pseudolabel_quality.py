import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_pseudolabel_quality import (
    FRAME_STATUS_CODES,
    QUALITY_FLAG_BITS,
    compute_quality_vectors,
    finite_row_median,
    quality_dependency_paths,
    quality_dependency_signature,
    summarize_quality_outputs,
)


class PseudolabelQualityTest(unittest.TestCase):
    def make_dependency_args(self, root: Path) -> argparse.Namespace:
        args = argparse.Namespace(
            selection_root=root / "selection",
            pose_root=root / "pose",
            triangulation_root=root / "triangulation",
            sam_prior_root=root / "sam_prior",
            sam_mode_c_review_root=root / "mode_c",
            body_fit_root=root / "body",
        )
        for label, path in quality_dependency_paths(args, "sequence").items():
            if label.startswith("tool/"):
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(label.encode("utf-8"))
        return args

    def test_quality_vector_preserves_component_reasons(self) -> None:
        frame_index = np.asarray([0, 1], dtype=np.int32)
        timestamp = np.asarray([0.0, 1.0 / 30.0], dtype=np.float64)
        selections = {}
        poses = {}
        priors = {}
        for camera_index, camera in enumerate(("cam1", "cam2", "cam3")):
            ambiguous = np.asarray([False, camera_index == 0])
            status = np.asarray(
                ["TARGET", "TARGET_AMBIGUOUS" if camera_index == 0 else "TARGET"]
            )
            selections[camera] = {
                "frame_index": frame_index,
                "target_status": status,
                "target_ambiguous": ambiguous,
                "no_target": np.zeros(2, dtype=np.bool_),
                "target_selection_confidence": np.asarray([0.9, 0.8], dtype=np.float32),
                "identity_switch_risk": ambiguous.copy(),
                "global_track_ambiguity": np.zeros(2, dtype=np.bool_),
                "association_ambiguity": np.zeros(2, dtype=np.bool_),
                "target_fragmentation_risk": np.zeros(2, dtype=np.bool_),
                "detector_duplicate_count": np.zeros(2, dtype=np.int16),
                "possible_reflection_count": np.zeros(2, dtype=np.int16),
            }
            poses[camera] = {
                "frame_index": frame_index,
                "valid_mask": np.ones((2, 2), dtype=np.bool_),
            }
            accepted = np.asarray([True, camera_index != 2])
            priors[camera] = {
                "source_frame_index": frame_index,
                "accepted_prior": accepted,
                "output_valid": accepted,
                "occlusion_risk": np.asarray([False, camera_index == 1]),
                "failure_reason": np.asarray(["", "" if accepted[1] else "MODEL_FAILURE"]),
            }

        canonical_valid = np.asarray([[True, True], [True, False]])
        canonical_points = np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0], [np.nan, np.nan, np.nan]],
            ],
            dtype=np.float32,
        )
        canonical = {
            "frame_index": frame_index,
            "timestamp_pts_seconds": timestamp,
            "joint_names": np.asarray(["joint_a", "joint_b"]),
            "keypoints_3d": canonical_points.copy(),
            "valid_mask": canonical_valid,
            "quality_score": np.asarray([[0.8, 0.7], [0.6, np.nan]], dtype=np.float32),
        }
        triangulated = {
            "frame_index": frame_index,
            "timestamp_pts_seconds": timestamp,
            "per_view_reprojection_px": np.ones((2, 2, 3), dtype=np.float32),
            "min_ray_angle_deg": np.full((2, 2), 20.0, dtype=np.float32),
        }
        body_valid = canonical_valid.copy()
        evidence = np.asarray([[2, 2], [3, 0]], dtype=np.uint8)
        body = {
            "frame_index": frame_index,
            "timestamp_pts_seconds": timestamp,
            "joint_names": canonical["joint_names"].copy(),
            "keypoints_3d": canonical_points.copy(),
            "valid_mask": body_valid,
            "confidence": np.asarray([[0.8, 0.7], [0.5, 0.0]], dtype=np.float32),
            "evidence_type": evidence,
            "observation_residual_sequence_gauge": np.asarray(
                [[0.1, 0.2], [0.3, np.nan]], dtype=np.float32
            ),
            "alignment_residual_sequence_gauge": np.asarray(
                [[0.1, 0.1, 0.1], [0.2, np.nan, 0.3]], dtype=np.float32
            ),
            "sam_time_error_ms": np.zeros((2, 3), dtype=np.float32),
            "sam_source_frame_index": np.tile(frame_index[:, None], (1, 3)),
        }
        triangulation_metadata = {
            "qa": {"camera_acceptance": "PASS", "quality_status": "PASS"}
        }
        body_metadata = {
            "evidence_type_codes": {"3": "PRIOR_ONLY"},
            "qa": {
                "status": "PASS",
                "anthropometry": {"reference_length_sequence_gauge": 1.0},
            },
        }
        mode_c_metadata = {
            "status": "REVIEW_MODE_C_CANDIDATE",
            "cameras": [
                {
                    "camera": "cam1",
                    "clips_reference_timeline": [
                        {"start_frame_index": 1, "end_frame_index": 1}
                    ],
                }
            ],
        }
        arrays, metadata = compute_quality_vectors(
            selections,
            poses,
            priors,
            triangulated,
            canonical,
            body,
            triangulation_metadata,
            body_metadata,
            mode_c_metadata,
        )
        self.assertEqual(arrays["frame_status_code"].tolist(), [0, 1])
        self.assertEqual(metadata["qa"]["sequence_status"], "REVIEW")
        self.assertEqual(metadata["qa"]["pass_frame_count"], 1)
        self.assertEqual(metadata["qa"]["review_frame_count"], 1)
        second = int(arrays["quality_flag_bits"][1])
        for name in (
            "TARGET_VIEW_MISSING_OR_ABSTAINED",
            "IDENTITY_RISK",
            "OCCLUSION_RISK",
            "SAM_PRIOR_REJECTED_OR_INVALID",
            "TRIANGULATION_JOINT_MISSING",
            "PRIOR_ONLY_JOINT_USED",
            "BODY_JOINT_MISSING",
            "MODE_C_REVIEW_CANDIDATE",
        ):
            self.assertTrue(second & QUALITY_FLAG_BITS[name], name)
        self.assertEqual(arrays["frame_status_code"][0], FRAME_STATUS_CODES["PASS"])

    def test_finite_row_median_keeps_all_nan_row(self) -> None:
        result = finite_row_median(np.asarray([[np.nan, np.nan], [1.0, 3.0]]))
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 2.0)

    def test_incremental_summary_keeps_all_materialized_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sequence, status, frames in (
                ("one", "PASS", 2),
                ("two", "REVIEW", 3),
            ):
                output = root / sequence
                output.mkdir()
                (output / "metadata.json").write_text(
                    json.dumps(
                        {
                            "sequence": sequence,
                            "stage": "PHASE11_PSEUDOLABEL_QUALITY_CONTROL",
                            "qa": {"sequence_status": status, "frame_count": frames},
                        }
                    ),
                    encoding="utf-8",
                )
            summary = summarize_quality_outputs(root)
            self.assertEqual(summary["sequence_count"], 2)
            self.assertEqual(summary["frame_count"], 5)
            self.assertEqual(summary["pass_count"], 1)
            self.assertEqual(summary["review_count"], 1)

    def test_quality_source_signature_changes_with_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_dependency_args(root)
            before = quality_dependency_signature(args, "sequence")
            dependency = (
                args.selection_root
                / "sequence"
                / "cam2"
                / "target_selection.npz"
            )
            dependency.write_bytes(b"changed selection")
            after = quality_dependency_signature(args, "sequence")
            self.assertNotEqual(before, after)

    def test_quality_source_signature_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.make_dependency_args(root)
            dependency = args.pose_root / "sequence" / "cam1" / "poses_2d.npz"
            target = root / "outside.npz"
            target.write_bytes(b"pose")
            dependency.unlink()
            os.symlink(target, dependency)
            with self.assertRaisesRegex(RuntimeError, "unsafe quality dependency"):
                quality_dependency_signature(args, "sequence")


if __name__ == "__main__":
    unittest.main()
