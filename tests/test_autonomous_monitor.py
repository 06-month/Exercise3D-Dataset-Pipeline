import argparse
import csv
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.monitor_autonomous_generation import (
    acquire_instance_lock as acquire_dashboard_lock,
    atomic_json,
    build_dashboard,
    deadline_freeze_upper_bound,
    export_progress,
    first_incomplete_camera,
    metadata_statuses,
    observed_post_sam_overhead,
    quality_progress,
    sam_output_storage_forecast,
    selection_workloads,
)


class AutonomousMonitorTest(unittest.TestCase):
    def test_dashboard_lifetime_lock_rejects_duplicate_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dashboard.lock"
            first = acquire_dashboard_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_dashboard_lock(path))
            assert first is not None
            first.close()
            recovered = acquire_dashboard_lock(path)
            self.assertIsNotNone(recovered)
            assert recovered is not None
            recovered.close()

    def test_sam_storage_forecast_uses_nearest_rank_p90(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for camera, output_bytes in (("cam1", 1_000), ("cam2", 2_000)):
                path = root / "one" / camera / "sam_body_benchmark.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=["frames_processed", "output_bytes", "status"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "frames_processed": 10,
                            "output_bytes": output_bytes,
                            "status": "PASS",
                        }
                    )
            forecast = sam_output_storage_forecast(
                root,
                ["one/cam1", "one/cam2"],
                total_frames=100,
                produced_frames=30,
                free_gib=1.0,
                minimum_free_gib=0.25,
            )
            self.assertTrue(forecast["available"])
            self.assertEqual(forecast["sample_count"], 2)
            self.assertEqual(forecast["median_bytes_per_frame"], 150)
            self.assertEqual(forecast["p90_bytes_per_frame"], 200)
            self.assertEqual(forecast["remaining_frames"], 70)
            self.assertEqual(forecast["projected_remaining_output_bytes"], 14_000)
            self.assertGreater(forecast["projected_reserve_margin_gib"], 0)

    def test_selection_workloads_validate_small_summary_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for sequence in ("one", "two"):
                for index, camera in enumerate(("cam1", "cam2", "cam3"), start=1):
                    directory = root / sequence / camera
                    directory.mkdir(parents=True)
                    atomic_json(
                        directory / "summary.json",
                        {
                            "sequence": sequence,
                            "camera": camera,
                            "status": "REVIEW" if camera == "cam1" else "PASS",
                            "frame_count": 10 + index,
                            "target_only_sapiens_crops": 9 + index,
                        },
                    )
            rows, errors = selection_workloads(root, ["one", "two"])
            self.assertEqual(errors, [])
            self.assertEqual([row["sequence"] for row in rows], ["one", "two"])
            self.assertEqual(rows[0]["frames"], 36)
            self.assertEqual(rows[0]["target_crops"], 33)

    def test_deadline_forecast_is_explicit_optimistic_sequence_bound(self) -> None:
        now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        workloads = [
            {
                "sequence": sequence,
                "target_crops": 10,
                "frames": 10,
                "cameras": {
                    "cam1": {"target_crops": 4, "frames": 4},
                    "cam2": {"target_crops": 3, "frames": 3},
                    "cam3": {"target_crops": 3, "frames": 3},
                },
            }
            for sequence in ("one", "two", "three")
        ]
        forecast = deadline_freeze_upper_bound(
            workloads,
            terminal_sequences={"one"},
            accepted_sequences={"one"},
            completed_sam_cameras={"two/cam1"},
            current_sam=("two", "cam2"),
            current_sam_frames=1,
            pose_completed_crops=10,
            pose_rate=1.0,
            sam_rate=1.0,
            now=now,
            deadline=now + timedelta(seconds=25),
        )
        self.assertTrue(forecast["available"])
        self.assertEqual(forecast["kind"], "OPTIMISTIC_UPPER_BOUND")
        self.assertEqual(forecast["estimated_completed_sequences_by_deadline"], 2)
        self.assertEqual(forecast["first_sequence_after_deadline"], "three")
        self.assertIn("overhead excluded", forecast["assumptions"][-1])
        adjusted = deadline_freeze_upper_bound(
            workloads,
            terminal_sequences={"one"},
            accepted_sequences={"one"},
            completed_sam_cameras={"two/cam1"},
            current_sam=("two", "cam2"),
            current_sam_frames=1,
            pose_completed_crops=10,
            pose_rate=1.0,
            sam_rate=1.0,
            now=now,
            deadline=now + timedelta(seconds=25),
            per_sequence_overhead_seconds=11.0,
            kind="EMPIRICAL_P90_POST_SAM_ADJUSTED",
        )
        self.assertEqual(adjusted["estimated_completed_sequences_by_deadline"], 1)
        self.assertEqual(adjusted["per_sequence_overhead_seconds"], 11.0)

    def test_observed_post_sam_overhead_uses_terminal_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sam_root = root / "sam"
            body_root = root / "body"
            mode_c_root = root / "mode_c"
            base = datetime(2026, 8, 12, tzinfo=timezone.utc)
            for sequence, delay in (("one", 10), ("two", 30)):
                for camera_index, camera in enumerate(("cam1", "cam2", "cam3")):
                    directory = sam_root / sequence / camera
                    directory.mkdir(parents=True)
                    with (directory / "sam_body_benchmark.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=["created_at_utc"])
                        writer.writeheader()
                        writer.writerow(
                            {
                                "created_at_utc": (
                                    base + timedelta(seconds=camera_index)
                                ).isoformat()
                            }
                        )
                terminal = base + timedelta(seconds=2 + delay)
                atomic_json(
                    body_root / sequence / "metadata.json",
                    {"created_at_utc": terminal.isoformat()},
                )
                atomic_json(
                    mode_c_root / sequence / "mode_c_escalation.json",
                    {"created_at_utc": terminal.isoformat()},
                )
            result = observed_post_sam_overhead(
                sam_root, body_root, mode_c_root, ["one", "two"]
            )
            self.assertTrue(result["available"])
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["median_seconds"], 20)
            self.assertEqual(result["p90_seconds"], 30)
            self.assertEqual(result["errors"], [])

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
            self.assertEqual(
                progress["durable_checkpoint"]["status"], "NOT_AVAILABLE"
            )

    def test_export_progress_selects_largest_contract_v2_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_build(name: str, sequences: list[str], modified: float) -> None:
                build = root / name
                build.mkdir()
                rows = [{"sequence": sequence, "status": "REVIEW"} for sequence in sequences]
                with (build / "sequence_status.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=["sequence", "status"])
                    writer.writeheader()
                    writer.writerows(rows)
                atomic_json(
                    build / "dataset_manifest.json",
                    {
                        "freeze_contract_version": 2,
                        "requested_sequences": sequences,
                        "sequence_count": len(sequences),
                        "pass_count": 0,
                        "review_count": len(sequences),
                        "fail_count": 0,
                        "incomplete_count": 0,
                        "private_dataset": True,
                        "source_rgb_included": False,
                        "source_payload_modified": False,
                        "freeze_eligible": True,
                        "file_count": 10 * len(sequences),
                        "total_payload_bytes": 100 * len(sequences),
                    },
                )
                os.utime(build / "dataset_manifest.json", (modified, modified))

            write_build("larger-older", ["one", "two", "three"], 1.0)
            write_build("smaller-newer", ["one"], 2.0)
            progress = export_progress(root, "deadline-build")
            self.assertEqual(progress["status"], "NOT_STARTED")
            self.assertEqual(progress["completed_sequences"], 0)
            checkpoint = progress["durable_checkpoint"]
            self.assertEqual(checkpoint["build_id"], "larger-older")
            self.assertEqual(checkpoint["completed_sequences"], 3)
            self.assertEqual(checkpoint["status_counts"]["REVIEW"], 3)
            self.assertTrue(checkpoint["manifest_contract_consistent"])
            self.assertEqual(progress["latest_materialized_build_id"], "smaller-newer")

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
            self.assertIn("QUALITY_FOLLOWER_WATCHDOG_DEAD", codes)
            self.assertIn("MONITORING_WATCHDOG_DEAD", codes)
            self.assertIn("PREDEADLINE_CHECKPOINT_FOLLOWER_DEAD", codes)
            self.assertIn("PREDEADLINE_CHECKPOINT_FOLLOWER_WATCHDOG_DEAD", codes)
            self.assertIn("DISK_RESERVE_LOW", codes)
            self.assertIn("VALIDATION_FAIL", codes)
            self.assertIn("SEQUENCE_PIPELINE_FAILED", codes)

    def test_dashboard_exposes_empirical_adjusted_deadline_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = datetime(2026, 8, 12, tzinfo=timezone.utc)
            args = self._args(root)
            sequences = ["one", "two"]
            for sequence in sequences:
                for camera in ("cam1", "cam2", "cam3"):
                    atomic_json(
                        args.selection_root / sequence / camera / "summary.json",
                        {
                            "sequence": sequence,
                            "camera": camera,
                            "status": "PASS",
                            "frame_count": 10,
                            "target_only_sapiens_crops": 10,
                        },
                    )
            for index, camera in enumerate(("cam1", "cam2", "cam3")):
                directory = args.sam_output_root / "one" / camera
                directory.mkdir(parents=True)
                with (directory / "sam_body_benchmark.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=["created_at_utc"])
                    writer.writeheader()
                    writer.writerow(
                        {
                            "created_at_utc": (
                                now + timedelta(seconds=index)
                            ).isoformat()
                        }
                    )
            terminal = now + timedelta(seconds=12)
            atomic_json(
                args.body_fit_root / "one" / "metadata.json",
                {"created_at_utc": terminal.isoformat(), "status": "REVIEW"},
            )
            atomic_json(
                args.sam_mode_c_review_root / "one" / "mode_c_escalation.json",
                {"created_at_utc": terminal.isoformat()},
            )
            atomic_json(
                args.handoff_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "deadline_utc": (now + timedelta(seconds=20)).isoformat(),
                    "completed": ["one"],
                    "remaining": ["two"],
                    "pose": {
                        "completed_cameras": [
                            "one/cam1",
                            "one/cam2",
                            "one/cam3",
                        ],
                        "processed_target_crops": 30,
                        "total_target_crops": 60,
                        "recent_chunk_crops_per_second": 1.0,
                    },
                    "sam": {
                        "completed_cameras": [
                            "one/cam1",
                            "one/cam2",
                            "one/cam3",
                        ],
                        "processed_frames": 30,
                        "total_frames": 60,
                        "measured_frames_per_second": 1.0,
                    },
                    "triangulation": {"count": 1, "status": {"one": "PASS"}},
                    "body_fit": {"count": 1, "status": {"one": "REVIEW"}},
                },
            )
            atomic_json(
                args.deadline_state,
                {
                    "deadline_utc": (now + timedelta(seconds=20)).isoformat(),
                    "status": "WAITING_DEADLINE",
                },
            )
            state = build_dashboard(
                args,
                now=now,
                processes=[],
                gpu={"available": True, "utilization_pct": 0.0, "devices": []},
            )
            forecast = state["deadline"]["freeze_forecast"]
            self.assertTrue(forecast["available"])
            self.assertEqual(
                forecast["observed_post_sam_overhead"]["p90_seconds"], 10
            )
            self.assertEqual(
                forecast["empirical_p90_adjusted"]["kind"],
                "EMPIRICAL_P90_POST_SAM_ADJUSTED",
            )
            reasons = {
                row["code"]: row["message"] for row in state["attention_reasons"]
            }
            self.assertIn("Empirical p90", reasons["DEADLINE_FREEZE_COVERAGE_AT_RISK"])

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
                self._process(8, "checkpoint_follower"),
                self._process(9, "checkpoint_follower_watchdog"),
                self._process(10, "quality_follower_watchdog"),
                self._process(11, "monitoring_watchdog"),
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
            atomic_json(
                args.checkpoint_follower_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "RUNNING",
                    "attention_required": False,
                    "attention_reasons": [],
                    "last_event": "WAITING_FOR_NEW_FREEZE_READY_SEQUENCE",
                    "ready_sequence_count": 0,
                },
            )
            atomic_json(
                args.checkpoint_follower_watchdog_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "RUNNING",
                    "attention_required": False,
                    "attention_reasons": [],
                    "last_event": "CHECKPOINT_FOLLOWER_OBSERVED",
                    "restart_count_in_window": 0,
                },
            )
            atomic_json(
                args.quality_follower_watchdog_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "RUNNING",
                    "attention_required": False,
                    "attention_reasons": [],
                    "last_event": "QUALITY_FOLLOWER_OBSERVED",
                    "restart_count_in_window": 0,
                },
            )
            atomic_json(
                args.monitoring_watchdog_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "RUNNING",
                    "attention_required": False,
                    "attention_reasons": [],
                    "last_event": "DASHBOARD_OBSERVED;HANDOFF_MONITOR_OBSERVED",
                    "targets": {},
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

            atomic_json(
                args.monitoring_watchdog_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "ATTENTION",
                    "attention_required": True,
                    "attention_reasons": [
                        {"code": "DASHBOARD_RESTART_EXHAUSTED", "message": "limit"}
                    ],
                },
            )
            watchdog_attention = build_dashboard(
                args,
                now=now,
                processes=processes,
                gpu={"available": True, "utilization_pct": 95.0, "devices": []},
            )
            self.assertIn(
                "DASHBOARD_RESTART_EXHAUSTED",
                {row["code"] for row in watchdog_attention["attention_reasons"]},
            )
            atomic_json(
                args.monitoring_watchdog_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "status": "RUNNING",
                    "attention_required": False,
                    "attention_reasons": [],
                    "last_event": "DASHBOARD_OBSERVED;HANDOFF_MONITOR_OBSERVED",
                    "targets": {},
                },
            )

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

    def test_checkpoint_follower_is_not_required_after_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = datetime(2026, 8, 14, 5, tzinfo=timezone.utc)
            handoff = {
                "updated_at_utc": now.isoformat(),
                "deadline_utc": (now - timedelta(hours=1)).isoformat(),
                "completed": [],
                "remaining": ["one"],
                "pose": {
                    "completed_cameras": [],
                    "processed_target_crops": 0,
                    "total_target_crops": 100,
                },
                "sam": {
                    "completed_cameras": [],
                    "processed_frames": 0,
                    "total_frames": 200,
                },
                "triangulation": {"count": 0, "status": {}},
                "body_fit": {"count": 0, "status": {}},
            }
            atomic_json(root / "handoff.json", handoff)
            atomic_json(
                root / "deadline.json",
                {
                    "deadline_utc": (now - timedelta(hours=1)).isoformat(),
                    "status": "COMPLETE",
                },
            )
            state = build_dashboard(
                self._args(root),
                now=now,
                processes=[],
                gpu={"available": True, "utilization_pct": 0.0, "devices": []},
            )
            codes = {row["code"] for row in state["attention_reasons"]}
            self.assertNotIn("PREDEADLINE_CHECKPOINT_FOLLOWER_DEAD", codes)
            self.assertNotIn("PREDEADLINE_CHECKPOINT_FOLLOWER_STATE_STALE", codes)
            self.assertNotIn(
                "PREDEADLINE_CHECKPOINT_FOLLOWER_WATCHDOG_DEAD", codes
            )

    def test_checkpoint_follower_is_required_for_unexported_ready_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            now = datetime(2026, 8, 13, tzinfo=timezone.utc)
            handoff = {
                "updated_at_utc": now.isoformat(),
                "deadline_utc": (now + timedelta(days=1)).isoformat(),
                "completed": ["one"],
                "remaining": [],
                "pose": {
                    "completed_cameras": ["one/cam1", "one/cam2", "one/cam3"],
                    "processed_target_crops": 100,
                    "total_target_crops": 100,
                },
                "sam": {
                    "completed_cameras": ["one/cam1", "one/cam2", "one/cam3"],
                    "processed_frames": 200,
                    "total_frames": 200,
                },
                "triangulation": {"count": 1, "status": {"one": "PASS"}},
                "body_fit": {"count": 1, "status": {"one": "REVIEW"}},
            }
            atomic_json(root / "handoff.json", handoff)
            atomic_json(
                root / "deadline.json",
                {
                    "deadline_utc": (now + timedelta(days=1)).isoformat(),
                    "status": "WAITING_DEADLINE",
                },
            )
            args = self._args(root)
            quality = args.quality_root / "one"
            quality.mkdir(parents=True)
            (quality / "quality_vector.npz").touch()
            atomic_json(
                quality / "metadata.json",
                {"qa": {"sequence_status": "REVIEW", "frame_count": 10}},
            )
            atomic_json(
                args.quality_follower_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "freeze_readiness": {
                        "ready_sequence_count": 1,
                        "status_counts": {"PASS": 0, "REVIEW": 1},
                        "ready": [{"sequence": "one", "status": "REVIEW"}],
                        "waiting": [],
                        "failures": [],
                    },
                },
            )
            atomic_json(
                args.checkpoint_follower_state,
                {
                    "updated_at_utc": now.isoformat(),
                    "best_checkpoint": {"completed_sequence_count": 0},
                },
            )
            state = build_dashboard(
                args,
                now=now,
                processes=[],
                gpu={"available": True, "utilization_pct": 0.0, "devices": []},
            )
            codes = {row["code"] for row in state["attention_reasons"]}
            self.assertIn("PREDEADLINE_CHECKPOINT_FOLLOWER_DEAD", codes)
            self.assertIn("PREDEADLINE_CHECKPOINT_FOLLOWER_WATCHDOG_DEAD", codes)
            self.assertNotIn("QUALITY_FOLLOWER_WATCHDOG_DEAD", codes)

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
            checkpoint_follower_state=root / "checkpoint_follower.json",
            checkpoint_follower_watchdog_state=root
            / "checkpoint_follower_watchdog.json",
            deadline_state=root / "deadline.json",
            quality_follower_state=root / "quality_follower.json",
            quality_follower_watchdog_state=root / "quality_follower_watchdog.json",
            monitoring_watchdog_state=root / "monitoring_watchdog.json",
            sequence_status=root / "sequences.csv",
            autonomous_runtime_dir=runtime,
            runtime_dir=runtime,
            selection_root=outputs / "selection",
            pose_root=outputs / "pose",
            sam_output_root=outputs / "sam",
            sam_mode_c_review_root=outputs / "mode_c",
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
