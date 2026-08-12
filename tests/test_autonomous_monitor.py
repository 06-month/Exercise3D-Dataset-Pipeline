import argparse
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.monitor_autonomous_generation import (
    atomic_json,
    build_dashboard,
    export_progress,
    first_incomplete_camera,
    metadata_statuses,
    quality_progress,
)


class AutonomousMonitorTest(unittest.TestCase):
    def test_first_incomplete_camera_respects_frozen_order(self) -> None:
        self.assertEqual(
            first_incomplete_camera(
                ["first", "second"],
                ["first/cam1", "first/cam2", "first/cam3", "second/cam1"],
            ),
            ("second", "cam2"),
        )

    def test_atomic_json_replaces_complete_document(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dashboard.json"
            atomic_json(path, {"value": 1})
            atomic_json(path, {"value": 2, "complete": True})
            self.assertEqual(json.loads(path.read_text()), {"value": 2, "complete": True})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_deadline_export_does_not_promote_an_old_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = root / "streaming-smoke"
            smoke.mkdir()
            atomic_json(smoke / "dataset_manifest.json", {"freeze_eligible": True})
            (smoke / "sequence_status.csv").write_text(
                "sequence,status\none,REVIEW\n", encoding="utf-8"
            )
            progress = export_progress(root, "deadline-build")
            self.assertEqual(progress["status"], "NOT_STARTED")
            self.assertEqual(progress["latest_build_id"], "deadline-build")
            self.assertEqual(progress["latest_materialized_build_id"], "streaming-smoke")
            self.assertEqual(progress["completed_sequences"], 0)
            self.assertFalse(progress["freeze_eligible"])

    def test_triangulation_quality_status_is_classified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "one"
            output.mkdir()
            atomic_json(
                output / "metadata.json",
                {"qa": {"schema_status": "PASS", "quality_status": "REVIEW_CAMERA"}},
            )
            self.assertEqual(
                metadata_statuses(root, ["one"], ("quality_status", "status")),
                {"one": "REVIEW_CAMERA"},
            )

    def test_quality_progress_preserves_review_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sequence, status, frames in (
                ("one", "REVIEW", 3),
                ("two", "FAIL", 2),
            ):
                output = root / sequence
                output.mkdir()
                (output / "quality_vector.npz").touch()
                atomic_json(
                    output / "metadata.json",
                    {"qa": {"sequence_status": status, "frame_count": frames}},
                )
            progress = quality_progress(root, ["one", "two", "pending"])
            self.assertEqual(progress["completed_sequences"], 2)
            self.assertEqual(progress["completed_frames"], 5)
            self.assertEqual(progress["status_counts"]["REVIEW"], 1)
            self.assertEqual(progress["status_counts"]["FAIL"], 1)

    def test_dashboard_detects_dead_jobs_disk_and_validation_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = datetime(2026, 8, 12, tzinfo=timezone.utc)
            handoff = {
                "updated_at_utc": now.isoformat(),
                "deadline_utc": (now + timedelta(days=1)).isoformat(),
                "completed": ["one"],
                "remaining": ["two"],
                "pose": {
                    "completed_cameras": ["one/cam1"],
                    "completed_camera_count": 1,
                    "processed_target_crops": 10,
                    "total_target_crops": 100,
                    "estimated_completion_utc": (now + timedelta(hours=2)).isoformat(),
                },
                "sam": {
                    "completed_cameras": [],
                    "completed_camera_count": 0,
                    "processed_frames": 0,
                    "total_frames": 200,
                },
                "triangulation": {"count": 1, "status": {"one": "FAIL_SCHEMA"}},
                "body_fit": {"count": 1, "status": {"one": "REVIEW_BODY_FIT_QUALITY"}},
            }
            atomic_json(root / "handoff.json", handoff)
            atomic_json(
                root / "deadline.json",
                {"deadline_utc": (now + timedelta(days=1)).isoformat(), "status": "WAITING_DEADLINE"},
            )
            (root / "sequences.csv").write_text(
                "sequence,status,failed_stage\none,INCOMPLETE,BODY_FIT\n", encoding="utf-8"
            )
            args = self._args(root)
            args.minimum_free_gib = 10**9
            state = build_dashboard(
                args,
                now=now,
                processes=[],
                gpu={"available": True, "utilization_pct": 0.0, "devices": []},
            )
            codes = {row["code"] for row in state["attention_reasons"]}
            self.assertTrue(state["attention_required"])
            self.assertIn("SAPIENS_PROCESS_DEAD", codes)
            self.assertIn("SUPERVISOR_DEAD", codes)
            self.assertIn("QUALITY_FOLLOWER_DEAD", codes)
            self.assertIn("DISK_RESERVE_LOW", codes)
            self.assertIn("VALIDATION_FAIL", codes)
            self.assertIn("SEQUENCE_PIPELINE_FAILED", codes)

    def test_dashboard_healthy_live_jobs_preserve_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = datetime(2026, 8, 12, tzinfo=timezone.utc)
            handoff = {
                "updated_at_utc": now.isoformat(),
                "deadline_utc": (now + timedelta(days=2)).isoformat(),
                "completed": ["one"],
                "remaining": ["two"],
                "pose": {
                    "completed_cameras": ["one/cam1", "one/cam2", "one/cam3"],
                    "completed_camera_count": 3,
                    "processed_target_crops": 50,
                    "total_target_crops": 100,
                    "estimated_completion_utc": (now + timedelta(days=1)).isoformat(),
                    "recent_chunk_crops_per_second": 0.2,
                    "effective_new_crops_per_second": 0.19,
                },
                "sam": {
                    "completed_cameras": [],
                    "completed_camera_count": 0,
                    "processed_frames": 0,
                    "total_frames": 200,
                },
                "triangulation": {"count": 0, "status": {}},
                "body_fit": {"count": 0, "status": {}},
            }
            atomic_json(root / "handoff.json", handoff)
            atomic_json(
                root / "supervisor.json",
                {"stage": "WAIT_RUNNING_SAPIENS2", "active_sequence": None},
            )
            atomic_json(
                root / "deadline.json",
                {"deadline_utc": (now + timedelta(days=2)).isoformat(), "status": "WAITING_DEADLINE"},
            )
            atomic_json(
                root / "quality_follower.json",
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "RUNNING",
                    "failures": [],
                    "freeze_readiness": {
                        "ready_sequence_count": 0,
                        "status_counts": {"PASS": 0, "REVIEW": 0},
                        "waiting": [],
                        "failures": [],
                    },
                },
            )
            processes = [
                self._process(1, "sapiens"),
                self._process(2, "supervisor"),
                self._process(3, "handoff_monitor"),
                self._process(4, "deadline_sentinel"),
                self._process(5, "quality_follower"),
                self._process(6, "supervisor_watchdog"),
                self._process(7, "deadline_sentinel_watchdog"),
            ]
            args = self._args(root)
            atomic_json(
                args.supervisor_watchdog_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "RUNNING",
                    "attention_required": False,
                    "attention_reasons": [],
                    "last_event": "SUPERVISOR_OBSERVED",
                },
            )
            atomic_json(
                args.deadline_watchdog_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "RUNNING",
                    "attention_required": False,
                    "attention_reasons": [],
                    "last_event": "DEADLINE_SENTINEL_OBSERVED",
                },
            )
            state = build_dashboard(
                args,
                now=now,
                processes=processes,
                gpu={"available": True, "utilization_pct": 95.0, "devices": []},
            )
            self.assertFalse(state["attention_required"])
            self.assertEqual(state["overall_status"], "RUNNING")
            self.assertEqual(state["sam"]["mode"], "B")
            self.assertEqual(state["sam"]["mode_c_policy"], "SELECTIVE_ESCALATION_ONLY")
            self.assertEqual(state["quality_control"]["freeze_ready_sequences"], 0)

            atomic_json(args.output, state)
            transient = build_dashboard(
                args,
                now=now + timedelta(seconds=30),
                processes=processes,
                gpu={"available": False, "devices": []},
            )
            transient_codes = {row["code"] for row in transient["attention_reasons"]}
            self.assertNotIn("GPU_STATUS_UNAVAILABLE", transient_codes)
            self.assertFalse(transient["gpu"]["telemetry_fresh"])
            self.assertEqual(transient["gpu"]["utilization_pct"], 95.0)

            atomic_json(
                args.deadline_state,
                {
                    "deadline_utc": (now + timedelta(days=2)).isoformat(),
                    "status": "EXPORT_INTEGRITY_FAILED",
                    "integrity_errors": ["sha256_mismatch:payload"],
                },
            )
            failed_snapshot = build_dashboard(
                args,
                now=now + timedelta(seconds=60),
                processes=processes,
                gpu={"available": True, "utilization_pct": 95.0, "devices": []},
            )
            failed_codes = {
                row["code"] for row in failed_snapshot["attention_reasons"]
            }
            self.assertIn("DEADLINE_SNAPSHOT_FAILED", failed_codes)

            atomic_json(
                args.deadline_state,
                {
                    "deadline_utc": (now + timedelta(days=2)).isoformat(),
                    "status": "WAITING_DEADLINE",
                },
            )
            atomic_json(
                args.quality_follower_state,
                {
                    "updated_at_utc": (now + timedelta(seconds=90)).isoformat(),
                    "status": "ATTENTION",
                    "failures": [],
                    "freeze_readiness": {
                        "ready_sequence_count": 0,
                        "status_counts": {"PASS": 0, "REVIEW": 0},
                        "waiting": [],
                        "failures": [
                            {
                                "sequence": "one",
                                "reasons": ["missing:pose_provenance"],
                            }
                        ],
                    },
                },
            )
            failed_readiness = build_dashboard(
                args,
                now=now + timedelta(seconds=90),
                processes=processes,
                gpu={"available": True, "utilization_pct": 95.0, "devices": []},
            )
            readiness_codes = {
                row["code"] for row in failed_readiness["attention_reasons"]
            }
            self.assertIn("FREEZE_READINESS_FAILED", readiness_codes)

    @staticmethod
    def _process(pid: int, group: str) -> dict[str, object]:
        return {
            "pid": pid,
            "ppid": 0,
            "state": "S",
            "argv": ["python", f"{group}.py"],
            "groups": [group],
            "command_name": f"{group}.py",
        }

    @staticmethod
    def _args(root: Path) -> argparse.Namespace:
        outputs = root / "outputs"
        outputs.mkdir(exist_ok=True)
        runtime = root / "runtime"
        runtime.mkdir(exist_ok=True)
        return argparse.Namespace(
            handoff_state=root / "handoff.json",
            supervisor_state=root / "supervisor.json",
            supervisor_watchdog_state=root / "supervisor_watchdog.json",
            deadline_watchdog_state=root / "deadline_watchdog.json",
            deadline_state=root / "deadline.json",
            quality_follower_state=root / "quality_follower.json",
            sequence_status=root / "sequences.csv",
            autonomous_runtime_dir=runtime,
            runtime_dir=runtime,
            pose_root=outputs / "pose",
            sam_output_root=outputs / "sam",
            triangulation_root=outputs / "triangulation",
            body_fit_root=outputs / "body_fit",
            quality_root=outputs / "quality",
            export_root=outputs / "export",
            disk_path=outputs,
            output=root / "dashboard.json",
            deadline_utc="2026-08-14T04:00:00+00:00",
            refresh_seconds=10.0,
            stall_minutes=60.0,
            gpu_idle_minutes=10.0,
            gpu_idle_threshold_pct=5.0,
            state_stale_seconds=180.0,
            eta_worsening_minutes=30.0,
            minimum_free_gib=0.001,
            once=True,
            plain=False,
            quiet=False,
            exit_nonzero_on_attention=False,
        )


if __name__ == "__main__":
    unittest.main()
