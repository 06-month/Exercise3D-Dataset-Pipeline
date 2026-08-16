import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.consolidate_sam_body_prior import REQUIRED_PRIOR_FIELDS

from tools.materialize_inference_provenance import (
    materialize_pose,
    materialize_sam,
    propagate_sam_provenance,
)


class InferenceProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        reports = self.dataset / "reports"
        reports.mkdir(parents=True)
        (reports / "dataset_inventory.json").write_text('{"frames":3}', encoding="utf-8")
        self.selection = self.root / "selection"
        target = self.selection / "sequence" / "cam1"
        target.mkdir(parents=True)
        (target / "target_selection.npz").write_bytes(b"selection")
        self.handoff = {
            "resume_commands": {
                "sapiens2_target_pipeline.py": "sapiens command",
                "run_sam_body4d_full.py": "sam command",
            }
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_pose_sidecar_is_atomic_and_idempotent(self) -> None:
        pose = self.root / "pose"
        output = pose / "sequence" / "cam1"
        output.mkdir(parents=True)
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "created_at_utc": "2026-01-01T00:00:00+00:00",
                    "model": "facebook/sapiens2-pose-5b",
                    "batch_size": 16,
                    "chunk_size": 256,
                    "precision": "float32",
                    "flip_test": True,
                    "qa": {"status": "PASS"},
                }
            ),
            encoding="utf-8",
        )
        for name in ("poses_2d.npz", "bboxes.npz", "frames.csv"):
            (output / name).touch()
        self.assertTrue(
            materialize_pose(
                self.dataset,
                self.selection,
                pose,
                "sequence",
                "cam1",
                self.handoff,
            )
        )
        payload = json.loads((output / "run_provenance.json").read_text())
        self.assertEqual(payload["status"], "PASS_PROVENANCE")
        self.assertEqual(payload["exact_resume_command"], "sapiens command")
        self.assertFalse(
            materialize_pose(
                self.dataset,
                self.selection,
                pose,
                "sequence",
                "cam1",
                self.handoff,
            )
        )

    def test_sam_sidecar_requires_accepted_target_coverage(self) -> None:
        sam = self.root / "sam"
        output = sam / "sequence" / "cam1"
        numeric = output / "mode_b_private_output" / "mhr_numeric" / "1"
        mesh = output / "mode_b_private_output" / "mesh_4d_individual" / "1"
        numeric.mkdir(parents=True)
        mesh.mkdir(parents=True)
        (output / "sam_body_benchmark.csv").write_text(
            "status,frames_processed,created_at_utc,repository_revision,elapsed_wall_seconds,peak_nvidia_vram_mib,gpu_utilization_mean_pct,power_mean_w\n"
            "PASS,2,2026-01-01T00:00:00+00:00,revision,1,1,1,1\n",
            encoding="utf-8",
        )
        (output / "mode_b_profile.json").write_text(
            '{"input_frames":2,"frames_processed":2,"target_seed_count":1,"persons_processed":1}',
            encoding="utf-8",
        )
        np.savez_compressed(
            output / "mode_b_private_output" / "target_provenance.npz",
            frame_names=np.asarray(["0.jpg", "1.jpg"]),
            source_frame_names=np.asarray(["0.jpg", "1.jpg"]),
            source_frame_indices=np.asarray([0, 1], dtype=np.int32),
            target_bboxes_xyxy=np.asarray(
                [[1, 2, 11, 22], [np.nan, np.nan, np.nan, np.nan]],
                dtype=np.float32,
            ),
            target_valid=np.asarray([True, False]),
            target_selection_confidence=np.asarray([1.0, 0.0]),
            target_ambiguous=np.asarray([False, True]),
            no_target=np.asarray([False, False]),
            occlusion_risk=np.asarray([False, True]),
            timestamp_pts_seconds=np.asarray([0.0, 0.033333]),
        )
        self.assertFalse(
            materialize_sam(
                self.dataset,
                self.selection,
                sam,
                "sequence",
                "cam1",
                self.handoff,
            )
        )
        np.savez_compressed(
            numeric / "0.npz",
            object_id=np.asarray(1, dtype=np.int32),
            **{
                key: np.asarray(0, dtype=np.float32)
                for key in REQUIRED_PRIOR_FIELDS
            },
        )
        (mesh / "0.ply").touch()
        self.assertTrue(
            materialize_sam(
                self.dataset,
                self.selection,
                sam,
                "sequence",
                "cam1",
                self.handoff,
            )
        )
        payload = json.loads((output / "run_provenance.json").read_text())
        self.assertEqual(payload["configuration"]["mode"], "B")
        self.assertEqual(payload["exact_resume_command"], "sam command")
        prior = self.root / "prior" / "sequence" / "cam1"
        prior.mkdir(parents=True)
        (prior / "metadata.json").write_text('{"qa":{"status":"PASS"}}', encoding="utf-8")
        self.assertTrue(
            propagate_sam_provenance(sam, self.root / "prior", "sequence", "cam1")
        )
        copied = json.loads((prior / "inference_run_provenance.json").read_text())
        self.assertEqual(copied["configuration_sha256"], payload["configuration_sha256"])


if __name__ == "__main__":
    unittest.main()
