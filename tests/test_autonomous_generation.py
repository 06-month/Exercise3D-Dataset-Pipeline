import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.consolidate_sam_body_prior import REQUIRED_PRIOR_FIELDS
from tools.run_autonomous_generation import free_gib, process_alive, sam_smoke_complete


class AutonomousGenerationTest(unittest.TestCase):
    def test_process_alive_for_current_and_finished_process(self) -> None:
        self.assertTrue(process_alive(os.getpid()))
        process = subprocess.Popen(["true"])
        process.wait()
        self.assertFalse(process_alive(process.pid))
        self.assertGreater(free_gib(Path(__file__)), 0)

    def test_sam_smoke_requires_full_numeric_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "mode_b_private_output"
            numeric = private / "mhr_numeric" / "1"
            mesh = private / "mesh_4d_individual" / "1"
            numeric.mkdir(parents=True)
            mesh.mkdir(parents=True)
            (root / "mode_b_profile.json").write_text(
                '{"frames_processed":1,"target_seed_count":1}', encoding="utf-8"
            )
            (root / "sam_body_benchmark.csv").write_text(
                "status\nPASS\n", encoding="utf-8"
            )
            np.savez_compressed(
                private / "target_provenance.npz",
                frame_names=np.asarray(["00000000.jpg"]),
                timestamp_pts_seconds=np.asarray([0.0]),
            )
            np.savez_compressed(
                numeric / "00000000.npz",
                **{key: np.asarray(0.0) for key in REQUIRED_PRIOR_FIELDS},
            )
            (mesh / "00000000.ply").touch()
            self.assertTrue(sam_smoke_complete(root, 1))
            np.savez_compressed(numeric / "00000000.npz", focal_length=np.asarray(1.0))
            self.assertFalse(sam_smoke_complete(root, 1))


if __name__ == "__main__":
    unittest.main()
