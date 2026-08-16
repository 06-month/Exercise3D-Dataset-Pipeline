import os
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from tools.consolidate_sam_body_prior import REQUIRED_PRIOR_FIELDS
from tools.run_autonomous_generation import (
    acquire_instance_lock,
    free_gib,
    load_successful_rows,
    next_stream_sequence,
    pending_stream_retries,
    process_alive,
    quality_command,
    sam_smoke_complete,
    sapiens_progress,
)


class AutonomousGenerationTest(unittest.TestCase):
    def test_supervisor_instance_lock_refuses_a_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "supervisor.lock"
            first = acquire_instance_lock(path)
            self.assertIsNotNone(first)
            second = acquire_instance_lock(path)
            self.assertIsNone(second)
            assert first is not None
            first.close()
            third = acquire_instance_lock(path)
            self.assertIsNotNone(third)
            assert third is not None
            third.close()

    def test_resume_loads_only_successful_sequence_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            path.write_text(
                "sequence,status,failed_stage\n"
                "ready,PASS,\n"
                "review,REVIEW,\n"
                "retry,INCOMPLETE,SAM_MODE_B\n",
                encoding="utf-8",
            )
            rows = load_successful_rows(path)
            self.assertEqual([row["sequence"] for row in rows], ["ready", "review"])

    def test_stream_retry_backoff_uses_later_ready_work_without_hot_loop(self) -> None:
        sequences = ["first", "second", "third"]
        attempts = {"first": 1}
        retry_not_before = {"first": 100.0}
        readiness = {"first": True, "second": True, "third": False}

        self.assertEqual(
            next_stream_sequence(
                sequences,
                set(),
                attempts,
                retry_not_before,
                max_attempts=2,
                monotonic_now=99.0,
                is_ready=readiness.__getitem__,
            ),
            "second",
        )
        self.assertEqual(
            next_stream_sequence(
                sequences,
                set(),
                attempts,
                retry_not_before,
                max_attempts=2,
                monotonic_now=100.0,
                is_ready=readiness.__getitem__,
            ),
            "first",
        )

    def test_stream_retry_is_bounded_and_checkpointed(self) -> None:
        sequences = ["exhausted", "pending", "complete"]
        attempts = {"exhausted": 2, "pending": 1}
        retry_not_before = {"exhausted": 50.0, "pending": 120.0}

        self.assertIsNone(
            next_stream_sequence(
                sequences,
                {"complete"},
                attempts,
                retry_not_before,
                max_attempts=2,
                monotonic_now=100.0,
                is_ready=lambda _sequence: True,
            )
        )
        self.assertEqual(
            pending_stream_retries(
                sequences,
                attempts,
                retry_not_before,
                max_attempts=2,
                monotonic_now=100.0,
            ),
            [
                {
                    "sequence": "pending",
                    "attempts_completed": 1,
                    "retry_in_seconds": 20.0,
                }
            ],
        )

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

    def test_sapiens_progress_uses_recent_completed_camera_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pose = root / "outputs" / "pose"
            runtime = root / "outputs" / "runtime" / "autonomous"
            monitor = root / "outputs" / "runtime" / "phase6_full_target_inference"
            monitor.mkdir(parents=True)
            started = datetime.now(timezone.utc) - timedelta(seconds=30)
            (monitor / "target_only_pilot_gpu_utilization.csv").write_text(
                "timestamp_utc,elapsed_seconds\n"
                f"{started.isoformat()},0\n",
                encoding="utf-8",
            )
            for index, camera in enumerate(("cam1", "cam2", "cam3")):
                output = pose / "sequence" / camera
                output.mkdir(parents=True)
                (output / "metadata.json").write_text(
                    json.dumps(
                        {
                            "created_at_utc": (started + timedelta(seconds=10 * index)).isoformat(),
                            "pose_inference_performed_in_this_stage": True,
                            "qa": {"target_pose_count": 10},
                        }
                    ),
                    encoding="utf-8",
                )
            args = Namespace(
                sequences=["sequence"],
                pose_root=pose,
                runtime_dir=runtime,
                reused_target_crops=0,
                expected_target_crops=40,
                expected_sam_hours=2.0,
            )
            progress = sapiens_progress(args)
            self.assertEqual(progress["completed_target_crops"], 30)
            self.assertEqual(progress["complete_camera_count"], 3)
            self.assertEqual(progress["complete_sequence_count"], 1)
            self.assertAlmostEqual(
                progress["recent_completed_camera_crops_per_second"], 1.0
            )

    def test_quality_command_is_sequence_scoped(self) -> None:
        root = Path("root")
        args = Namespace(
            selection_root=root / "selection",
            pose_root=root / "pose",
            triangulation_root=root / "triangulation",
            sam_prior_root=root / "sam",
            sam_mode_c_review_root=root / "mode_c",
            body_fit_root=root / "body",
            quality_root=root / "quality",
        )
        command = quality_command(args, "sequence")
        self.assertIn("build_pseudolabel_quality.py", command[1])
        self.assertEqual(command[command.index("--sequences") + 1], "sequence")
        self.assertEqual(Path(command[command.index("--output-root") + 1]).name, "quality")


if __name__ == "__main__":
    unittest.main()
