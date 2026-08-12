import json
import tempfile
import unittest
from pathlib import Path

from tools.checkpoint_handoff_state import (
    acquire_instance_lock,
    atomic_json,
    merge_resume_commands,
    pose_progress,
    sam_progress,
)


class HandoffCheckpointTest(unittest.TestCase):
    def test_resume_commands_include_monitoring_plane_and_preserve_absent(self) -> None:
        previous = {"resume_commands": {"sapiens2_target_pipeline.py": "old"}}
        processes = [
            {
                "argv": ["python", "tools/monitor_autonomous_generation.py", "--quiet"],
                "command": "python tools/monitor_autonomous_generation.py --quiet",
            },
            {
                "argv": ["python", "tools/checkpoint_handoff_state.py", "--output", "x"],
                "command": "python tools/checkpoint_handoff_state.py --output x",
            },
            {
                "argv": ["python", "tools/run_monitoring_watchdog.py"],
                "command": "python tools/run_monitoring_watchdog.py",
            },
        ]
        commands = merge_resume_commands(previous, processes)
        self.assertEqual(commands["sapiens2_target_pipeline.py"], "old")
        self.assertIn("monitor_autonomous_generation.py", commands)
        self.assertIn("checkpoint_handoff_state.py", commands)
        self.assertIn("run_monitoring_watchdog.py", commands)

    def test_handoff_monitor_lifetime_lock_rejects_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "handoff.lock"
            first = acquire_instance_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_instance_lock(path))
            assert first is not None
            first.close()
            recovered = acquire_instance_lock(path)
            self.assertIsNotNone(recovered)
            assert recovered is not None
            recovered.close()

    def test_atomic_json_and_pose_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pose = root / "pose"
            for camera in ("cam1", "cam2", "cam3"):
                output = pose / "sequence" / camera
                output.mkdir(parents=True)
                atomic_json(
                    output / "metadata.json",
                    {"qa": {"status": "PASS", "target_pose_count": 2}},
                )
            progress = pose_progress(pose, ["sequence", "pending"])
            self.assertEqual(progress["completed_sequences"], ["sequence"])
            self.assertEqual(progress["completed_camera_count"], 3)
            self.assertEqual(progress["processed_target_crops"], 6)
            payload = json.loads((pose / "sequence" / "cam1" / "metadata.json").read_text())
            self.assertEqual(payload["qa"]["status"], "PASS")

    def test_sam_progress_requires_three_pass_cameras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for camera in ("cam1", "cam2"):
                output = root / "sequence" / camera
                numeric = output / "mode_b_private_output" / "mhr_numeric" / "1"
                mesh = output / "mode_b_private_output" / "mesh_4d_individual" / "1"
                numeric.mkdir(parents=True)
                mesh.mkdir(parents=True)
                atomic_json(
                    output / "mode_b_profile.json",
                    {"input_frames": 1, "frames_processed": 1},
                )
                (output / "sam_body_benchmark.csv").write_text(
                    "status,frames_processed\nPASS,1\n", encoding="utf-8"
                )
                (numeric / "00000000.npz").touch()
                (mesh / "00000000.ply").touch()
            partial = sam_progress(root, ["sequence"])
            self.assertEqual(partial["completed_camera_count"], 2)
            self.assertEqual(partial["completed_sequence_count"], 0)
            output = root / "sequence" / "cam3"
            numeric = output / "mode_b_private_output" / "mhr_numeric" / "1"
            mesh = output / "mode_b_private_output" / "mesh_4d_individual" / "1"
            numeric.mkdir(parents=True)
            mesh.mkdir(parents=True)
            atomic_json(
                output / "mode_b_profile.json",
                {"input_frames": 1, "frames_processed": 1},
            )
            (output / "sam_body_benchmark.csv").write_text(
                "status,frames_processed\nPASS,1\n", encoding="utf-8"
            )
            (numeric / "00000000.npz").touch()
            (mesh / "00000000.ply").touch()
            complete = sam_progress(root, ["sequence"])
            self.assertEqual(complete["completed_sequences"], ["sequence"])
            self.assertEqual(complete["processed_frames"], 3)


if __name__ == "__main__":
    unittest.main()
