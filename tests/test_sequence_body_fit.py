import csv
import json
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np

from tools.fit_sequence_body import (
    body_fit_dependency_signature,
    evaluate_fit_gate,
    finite_view_consensus,
    smooth_track,
    validate_existing_body_fit,
    weighted_similarity,
)


class SequenceBodyFitTest(unittest.TestCase):
    @staticmethod
    def gate_config() -> dict:
        return {
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

    def test_finite_view_consensus_preserves_unsupported_joints_without_warning(self) -> None:
        values = np.full((2, 3, 2, 3), np.nan, dtype=np.float64)
        values[1, 0, 0] = [1.0, 2.0, 3.0]
        values[1, 1, 0] = [3.0, 4.0, 5.0]
        values[1, 2, 1] = [8.0, 9.0, 10.0]
        valid = np.isfinite(values).all(axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = finite_view_consensus(values, valid)
        self.assertTrue(np.isnan(result[0]).all())
        np.testing.assert_allclose(result[1, 0], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(result[1, 1], [8.0, 9.0, 10.0])

    def test_finite_view_consensus_requires_matching_shapes(self) -> None:
        with self.assertRaises(ValueError):
            finite_view_consensus(
                np.zeros((1, 2, 3, 3)), np.zeros((1, 2, 3, 1), dtype=np.bool_)
            )

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
        config = self.gate_config()
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

    def test_existing_body_fit_requires_source_binding_and_finite_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            timestamps = np.asarray([0.0, 1.0 / 30.0], dtype=np.float64)
            names = ["joint_a", "joint_b"]
            triangulated_valid = np.ones((2, 2), dtype=np.bool_)
            triangulated_quality = np.ones((2, 2), dtype=np.float32)
            valid = np.ones((2, 2), dtype=np.bool_)
            evidence = np.ones((2, 2), dtype=np.uint8)
            np.savez_compressed(
                output / "body_fit.npz",
                frame_index=np.arange(2, dtype=np.int32),
                timestamp_pts_seconds=timestamps,
                joint_names=np.asarray(names),
                keypoints_3d=np.ones((2, 2, 3), dtype=np.float32),
                valid_mask=valid,
                confidence=np.ones((2, 2), dtype=np.float32),
                evidence_type=evidence,
                triangulated_valid=triangulated_valid,
                triangulated_quality=triangulated_quality,
                shape_params_consensus=np.ones(2, dtype=np.float32),
                scale_params_consensus=np.ones(2, dtype=np.float32),
                body_pose_params_consensus=np.ones((2, 2), dtype=np.float32),
                body_pose_prior_view_count=np.ones(2, dtype=np.uint8),
                s0_names=np.asarray(["bone"]),
                s0=np.ones(1, dtype=np.float32),
            )
            qa = {
                "frame_count": 2,
                "joint_count": 2,
                "final_valid_joint_fraction": 1.0,
                "alignment_success_fraction": 1.0,
                "prior_only_joint_fraction": 0.0,
                "observation_displacement_p95_normalized": 0.01,
                "median_bone_length_cv": 0.01,
                "anthropometry": {"reference_length_sequence_gauge": 1.0},
                "finite_valid_points": True,
                "invalid_points_are_nan": True,
                "triangulation_camera_status": "PASS",
                "geometry_plus_prior_count": 0,
                "geometry_only_count": 4,
                "prior_only_count": 0,
                "missing_count": 0,
                "status": "PASS",
                "review_reasons": [],
                "fail_reasons": [],
            }
            (output / "metadata.json").write_text(
                json.dumps(
                    {
                        "sequence": "sequence",
                        "stage": "SEQUENCE_LEVEL_CANONICAL_BODY_FIT",
                        "source_dependency_signature": "source-v1",
                        "qa": qa,
                    }
                ),
                encoding="utf-8",
            )
            with (output / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["frame_index"])
                writer.writeheader()
                writer.writerows([{"frame_index": 0}, {"frame_index": 1}])

            valid_existing, _ = validate_existing_body_fit(
                output,
                "sequence",
                timestamps,
                names,
                triangulated_valid,
                triangulated_quality,
                self.gate_config(),
                "source-v1",
            )
            self.assertTrue(valid_existing)
            drifted, _ = validate_existing_body_fit(
                output,
                "sequence",
                timestamps,
                names,
                triangulated_valid,
                triangulated_quality,
                self.gate_config(),
                "source-v2",
            )
            self.assertFalse(drifted)

            with np.load(output / "body_fit.npz", allow_pickle=False) as payload:
                corrupted = {key: payload[key].copy() for key in payload.files}
            corrupted["keypoints_3d"][0, 0] = np.nan
            np.savez_compressed(output / "body_fit.npz", **corrupted)
            corrupted_valid, _ = validate_existing_body_fit(
                output,
                "sequence",
                timestamps,
                names,
                triangulated_valid,
                triangulated_quality,
                self.gate_config(),
                "source-v1",
            )
            self.assertFalse(corrupted_valid)

    def test_body_fit_dependency_signature_changes_with_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            triangulation = root / "triangulation" / "sequence"
            triangulation.mkdir(parents=True)
            (triangulation / "canonical_3d.npz").write_bytes(b"canonical")
            (triangulation / "metadata.json").write_text("{}", encoding="utf-8")
            gate = root / "gate.json"
            gate.write_text("{}", encoding="utf-8")
            for camera in ("cam1", "cam2", "cam3"):
                prior = root / "sam" / "sequence" / camera
                prior.mkdir(parents=True)
                (prior / "sam_body_prior.npz").write_bytes(camera.encode("ascii"))
                (prior / "metadata.json").write_text("{}", encoding="utf-8")
            args = Namespace(
                triangulation_root=root / "triangulation",
                sam_prior_root=root / "sam",
                gate_config=gate,
                max_time_gap_seconds=0.05,
                minimum_alignment_joints=8,
                geometry_weight=8.0,
                sam_weight_per_view=0.25,
                temporal_weight=0.25,
            )

            first = body_fit_dependency_signature(args, "sequence")
            (root / "sam" / "sequence" / "cam2" / "metadata.json").write_text(
                '{"changed":true}', encoding="utf-8"
            )
            second = body_fit_dependency_signature(args, "sequence")
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
