import json
import tempfile
import unittest
import warnings
from argparse import Namespace
from pathlib import Path

import numpy as np

from tools.assess_sam_mode_c_escalation import (
    assessment_dependency_signature,
    bounded_clips,
    finite_row_median,
    robust_threshold,
    validate_existing_assessment,
)


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

    def test_finite_row_median_preserves_abstention_without_warning(self) -> None:
        values = np.asarray(
            [
                [np.nan, np.nan, np.nan],
                [1.0, np.nan, 3.0],
                [np.inf, -np.inf, np.nan],
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = finite_row_median(values)
        self.assertEqual(caught, [])
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1], 2.0)
        self.assertTrue(np.isnan(result[2]))

    def test_finite_row_median_requires_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, "2D array"):
            finite_row_median(np.asarray([1.0, 2.0]))

    def test_existing_assessment_requires_source_and_candidate_consistency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mode_c_escalation.json"
            policy = {"clip_policy": {"padding_frames_each_side": 2}}
            result = {
                "schema_version": 1,
                "created_at_utc": "2026-08-12T00:00:00+00:00",
                "sequence": "sequence",
                "default_mode": "B",
                "mode_c_executed": False,
                "source_dependency_signature": "source-v1",
                "selected_reference_frame_count": 3,
                "status": "REVIEW_MODE_C_CANDIDATE",
                "policy": policy,
                "cameras": [
                    {
                        "camera": camera,
                        "reference_frame_count": 10,
                        "occlusion_reference_count": 1,
                        "missing_signal_count": 0,
                        "temporal_outlier_signal_count": 1,
                        "alignment_outlier_signal_count": 0,
                        "base_candidate_count": 1,
                        "selected_reference_frame_count": 3 if camera == "cam1" else 0,
                        "selected_source_frame_count": 3 if camera == "cam1" else 0,
                        "selected_source_frame_indices": [2, 3, 4] if camera == "cam1" else [],
                        "clips_reference_timeline": (
                            [{"start_frame_index": 2, "end_frame_index": 4}]
                            if camera == "cam1"
                            else []
                        ),
                        "temporal_delta": {
                            "median": 0.1,
                            "mad": 0.01,
                            "threshold_median_plus_5_scaled_mad": 0.2,
                        },
                        "alignment_residual_normalized": {
                            "median": 0.1,
                            "mad": 0.01,
                            "threshold_median_plus_5_scaled_mad": 0.2,
                        },
                    }
                    for camera in ("cam1", "cam2", "cam3")
                ],
            }
            path.write_text(json.dumps(result), encoding="utf-8")

            valid, _ = validate_existing_assessment(
                path, "sequence", policy, "source-v1"
            )
            self.assertTrue(valid)
            drifted, _ = validate_existing_assessment(
                path, "sequence", policy, "source-v2"
            )
            self.assertFalse(drifted)
            result["status"] = "PASS_MODE_B_FROZEN"
            path.write_text(json.dumps(result), encoding="utf-8")
            inconsistent, _ = validate_existing_assessment(
                path, "sequence", policy, "source-v1"
            )
            self.assertFalse(inconsistent)

    def test_assessment_dependency_signature_changes_with_body_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            body = root / "body" / "sequence"
            geometry = root / "geometry" / "sequence"
            body.mkdir(parents=True)
            geometry.mkdir(parents=True)
            (body / "body_fit.npz").write_bytes(b"body")
            (body / "metadata.json").write_text("{}", encoding="utf-8")
            (geometry / "triangulated_3d.npz").write_bytes(b"geometry")
            policy = root / "policy.json"
            canonical = root / "canonical.json"
            policy.write_text("{}", encoding="utf-8")
            canonical.write_text("{}", encoding="utf-8")
            for camera in ("cam1", "cam2", "cam3"):
                prior = root / "sam" / "sequence" / camera
                prior.mkdir(parents=True)
                (prior / "sam_body_prior.npz").write_bytes(camera.encode("ascii"))
            args = Namespace(
                body_fit_root=root / "body",
                triangulation_root=root / "geometry",
                sam_prior_root=root / "sam",
                policy_config=policy,
                canonical_config=canonical,
            )

            first = assessment_dependency_signature(args, "sequence")
            (body / "metadata.json").write_text('{"changed":true}', encoding="utf-8")
            second = assessment_dependency_signature(args, "sequence")
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
