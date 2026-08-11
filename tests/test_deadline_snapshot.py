import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.run_deadline_snapshot import atomic_json, export_command, read_manifest


class DeadlineSnapshotTest(unittest.TestCase):
    def test_command_uses_separate_versioned_build(self) -> None:
        root = Path("/private")
        args = Namespace(
            dataset_root=root,
            selection_root=Path("selection"),
            pose_root=Path("pose"),
            triangulation_root=Path("triangulation"),
            sam_prior_root=Path("sam_prior"),
            sam_mode_c_review_root=Path("mode_c"),
            body_fit_root=Path("body"),
            output_root=Path("freeze"),
            build_id="deadline-build",
            sequences=["sequence"],
        )
        command = export_command(args)
        self.assertIn("deadline-build", command)
        self.assertIn("sequence", command)
        self.assertNotIn("--overwrite", command)

    def test_atomic_state_and_manifest_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            atomic_json(path, {"status": "WAITING_DEADLINE"})
            self.assertEqual(read_manifest(path)["status"], "WAITING_DEADLINE")
            path.write_text("partial", encoding="utf-8")
            self.assertIsNone(read_manifest(path))


if __name__ == "__main__":
    unittest.main()
