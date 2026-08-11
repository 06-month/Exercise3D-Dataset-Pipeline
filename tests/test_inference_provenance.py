import json
import tempfile
import unittest
from pathlib import Path

from tools.materialize_inference_provenance import materialize_pose, materialize_sam


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

    def test_sam_sidecar_requires_complete_counts(self) -> None:
        sam = self.root / "sam"
        output = sam / "sequence" / "cam1"
        numeric = output / "mode_b_private_output" / "mhr_numeric" / "1"
        mesh = output / "mode_b_private_output" / "mesh_4d_individual" / "1"
        numeric.mkdir(parents=True)
        mesh.mkdir(parents=True)
        (output / "sam_body_benchmark.csv").write_text(
            "status,frames_processed,created_at_utc,repository_revision\n"
            "PASS,1,2026-01-01T00:00:00+00:00,revision\n",
            encoding="utf-8",
        )
        (output / "mode_b_profile.json").write_text(
            '{"input_frames":1,"frames_processed":1}', encoding="utf-8"
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
        (numeric / "0.npz").touch()
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


if __name__ == "__main__":
    unittest.main()
