import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from tools.run_deadline_snapshot import (
    LOADED_IMPLEMENTATION,
    acquire_sentinel_lock,
    atomic_json,
    export_command,
    deadline_state_base,
    read_manifest,
    run_export_with_retries,
    verified_manifest,
)


class DeadlineSnapshotTest(unittest.TestCase):
    def test_runtime_state_records_loaded_sentinel_and_exporter_code(self) -> None:
        args = Namespace(build_id="deadline-build")
        now = datetime(2026, 8, 12, tzinfo=timezone.utc)
        deadline = datetime(2026, 8, 14, 4, tzinfo=timezone.utc)
        state = deadline_state_base(args, deadline, now)
        self.assertEqual(state["implementation"], LOADED_IMPLEMENTATION)
        self.assertRegex(
            state["implementation"]["sentinel_tool_sha256"], r"^[0-9a-f]{64}$"
        )
        self.assertRegex(
            state["implementation"]["exporter_tool_sha256"], r"^[0-9a-f]{64}$"
        )

    def test_sentinel_lifetime_lock_refuses_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sentinel.lock"
            first = acquire_sentinel_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_sentinel_lock(path))
            assert first is not None
            first.close()
            recovered = acquire_sentinel_lock(path)
            self.assertIsNotNone(recovered)
            assert recovered is not None
            recovered.close()

    def test_sentinel_lock_refuses_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "keep.txt"
            target.write_text("keep", encoding="utf-8")
            link = root / "sentinel.lock"
            link.symlink_to(target)
            with self.assertRaises(OSError):
                acquire_sentinel_lock(link)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

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
            quality_root=Path("quality"),
            output_root=Path("freeze"),
            build_id="deadline-build",
            sequences=["sequence"],
            deadline_utc="2026-08-14T04:00:00+00:00",
        )
        command = export_command(args)
        self.assertIn("deadline-build", command)
        self.assertIn("sequence", command)
        self.assertIn("--quality-root", command)
        self.assertIn("--deadline-cutoff-utc", command)
        self.assertNotIn("--overwrite", command)
        deferred = export_command(args, defer_eligible_incomplete=True)
        self.assertIn("--defer-eligible-incomplete", deferred)

    def test_export_retries_resume_until_manifest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                dataset_root=root / "dataset",
                selection_root=root / "selection",
                pose_root=root / "pose",
                triangulation_root=root / "triangulation",
                sam_prior_root=root / "sam_prior",
                sam_mode_c_review_root=root / "mode_c",
                body_fit_root=root / "body",
                quality_root=root / "quality",
                output_root=root / "freeze",
                runtime_state=root / "state.json",
                build_id="deadline-build",
                sequences=["sequence"],
                deadline_utc="2026-08-14T04:00:00+00:00",
                export_retries=2,
                retry_seconds=1.0,
            )
            completed = {"build_id": "deadline-build", "freeze_eligible": False}
            with (
                patch(
                    "tools.run_deadline_snapshot.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess([], 1),
                        subprocess.CompletedProcess([], 2),
                    ],
                ) as run,
                patch(
                    "tools.run_deadline_snapshot.verified_manifest",
                    side_effect=[(None, []), (completed, [])],
                ),
                patch("tools.run_deadline_snapshot.time.sleep") as sleep,
            ):
                manifest, errors, exit_code, attempt = run_export_with_retries(
                    args,
                    datetime(2026, 8, 14, 4, tzinfo=timezone.utc),
                    root / "freeze" / "deadline-build" / "dataset_manifest.json",
                )
            self.assertEqual(manifest, completed)
            self.assertEqual(errors, [])
            self.assertEqual(exit_code, 2)
            self.assertEqual(attempt, 2)
            self.assertEqual(run.call_count, 2)
            sleep.assert_called_once_with(1.0)

    def test_final_attempt_publishes_truthful_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                dataset_root=root / "dataset",
                selection_root=root / "selection",
                pose_root=root / "pose",
                triangulation_root=root / "triangulation",
                sam_prior_root=root / "sam_prior",
                sam_mode_c_review_root=root / "mode_c",
                body_fit_root=root / "body",
                quality_root=root / "quality",
                output_root=root / "freeze",
                runtime_state=root / "state.json",
                build_id="deadline-build",
                sequences=["sequence"],
                deadline_utc="2026-08-14T04:00:00+00:00",
                export_retries=1,
                retry_seconds=1.0,
            )
            with (
                patch(
                    "tools.run_deadline_snapshot.subprocess.run",
                    side_effect=[
                        subprocess.CompletedProcess([], 75),
                        subprocess.CompletedProcess([], 2),
                    ],
                ) as run,
                patch(
                    "tools.run_deadline_snapshot.verified_manifest",
                    side_effect=[
                        (None, []),
                        ({"build_id": "deadline-build", "freeze_eligible": False}, []),
                    ],
                ),
                patch("tools.run_deadline_snapshot.time.sleep"),
            ):
                manifest, _, _, attempt = run_export_with_retries(
                    args,
                    datetime(2026, 8, 14, 4, tzinfo=timezone.utc),
                    root / "freeze" / "deadline-build" / "dataset_manifest.json",
                )
            self.assertIsNotNone(manifest)
            self.assertEqual(attempt, 2)
            first_command = run.call_args_list[0].args[0]
            final_command = run.call_args_list[1].args[0]
            self.assertIn("--defer-eligible-incomplete", first_command)
            self.assertNotIn("--defer-eligible-incomplete", final_command)

    def test_atomic_state_and_manifest_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            atomic_json(path, {"status": "WAITING_DEADLINE"})
            self.assertEqual(read_manifest(path)["status"], "WAITING_DEADLINE")
            path.write_text("partial", encoding="utf-8")
            self.assertIsNone(read_manifest(path))

    def test_parseable_but_unverified_manifest_is_not_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "deadline-build"
            root.mkdir()
            path = root / "dataset_manifest.json"
            atomic_json(
                path,
                {
                    "build_id": "deadline-build",
                    "private_dataset": True,
                    "source_rgb_included": False,
                    "source_payload_modified": False,
                    "files": [],
                },
            )
            manifest, errors = verified_manifest(path, "deadline-build")
            self.assertIsNone(manifest)
            self.assertTrue(errors)

    def test_manifest_verification_binds_the_exact_deadline_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "deadline-build"
            root.mkdir()
            path = root / "dataset_manifest.json"
            atomic_json(path, {"build_id": "deadline-build"})
            deadline = datetime(2026, 8, 14, 4, tzinfo=timezone.utc)
            verified = {
                "valid": True,
                "errors": [],
                "manifest": {"build_id": "deadline-build"},
            }
            with patch(
                "tools.run_deadline_snapshot.verify_frozen_build",
                return_value=verified,
            ) as verifier:
                manifest, errors = verified_manifest(
                    path,
                    "deadline-build",
                    ["one"],
                    deadline,
                )
            self.assertEqual(manifest, verified["manifest"])
            self.assertEqual(errors, [])
            verifier.assert_called_once_with(
                root,
                "deadline-build",
                ["one"],
                deadline,
            )


if __name__ == "__main__":
    unittest.main()
