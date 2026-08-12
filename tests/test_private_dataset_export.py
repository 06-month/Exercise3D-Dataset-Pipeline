import csv
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.export_private_dataset import (
    acquire_build_lock,
    copy_exact,
    deadline_eligibility,
    finite_nan_contract,
    git_provenance,
    prune_staging_tree,
    publish_staged_build,
    required_global_manifest_paths,
    required_sequence_manifest_paths,
    remove_staging_symlinks,
    sequence_order_sha256,
    sequence_dependencies,
    sha256,
    validate_path_component,
    verify_frozen_build,
    verify_deadline_marker_mtimes,
)


class PrivateDatasetExportTest(unittest.TestCase):
    def test_build_lock_is_scoped_by_build_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = acquire_build_lock(root, "build-a")
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_build_lock(root, "build-a"))
            independent = acquire_build_lock(root, "build-b")
            self.assertIsNotNone(independent)
            assert first is not None and independent is not None
            first.close()
            independent.close()
            recovered = acquire_build_lock(root, "build-a")
            self.assertIsNotNone(recovered)
            assert recovered is not None
            recovered.close()

    def test_build_lock_refuses_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_root = root / ".locks"
            lock_root.mkdir()
            target = root / "keep.txt"
            target.write_text("keep", encoding="utf-8")
            (lock_root / "build-a.lock").symlink_to(target)
            with self.assertRaises(OSError):
                acquire_build_lock(root, "build-a")
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def build_contract_v2(
        self, root: Path, *, omit_sequence_path: str | None = None
    ) -> Path:
        complete = "complete"
        pending = "pending"
        records = []

        def add_record(path: str, sequence: str, content: bytes) -> dict:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            record = {
                "sequence": sequence,
                "path": path,
                "bytes": len(content),
                "sha256": sha256(target),
            }
            records.append(record)
            return record

        for path in sorted(required_global_manifest_paths()):
            add_record(path, "", b"{}\n")
        sequence_manifest_path = f"sequences/{complete}/sequence_manifest.json"
        sequence_records = []
        for path in sorted(required_sequence_manifest_paths(complete)):
            if path in {sequence_manifest_path, omit_sequence_path}:
                continue
            sequence_records.append(add_record(path, complete, b"payload"))
        sequence_manifest = {
            "schema_version": 1,
            "sequence": complete,
            "status": "REVIEW",
            "files": sequence_records,
        }
        add_record(
            sequence_manifest_path,
            complete,
            (json.dumps(sequence_manifest) + "\n").encode(),
        )
        with (root / "sequence_status.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["sequence", "status"])
            writer.writeheader()
            writer.writerows(
                [
                    {"sequence": complete, "status": "REVIEW"},
                    {"sequence": pending, "status": "INCOMPLETE"},
                ]
            )
        requested = [complete, pending]
        manifest = {
            "schema_version": 1,
            "freeze_contract_version": 2,
            "build_id": "contract-v2",
            "requested_sequences": requested,
            "sequence_order_sha256": sequence_order_sha256(requested),
            "private_dataset": True,
            "source_rgb_included": False,
            "source_payload_modified": False,
            "sequence_count": 2,
            "pass_count": 0,
            "review_count": 1,
            "fail_count": 0,
            "incomplete_count": 1,
            "freeze_eligible": False,
            "file_count": len(records),
            "total_payload_bytes": sum(record["bytes"] for record in records),
            "files": records,
        }
        (root / "dataset_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return root

    def test_contract_v2_binds_requested_incomplete_sequence_universe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.build_contract_v2(Path(temporary))
            result = verify_frozen_build(
                root, "contract-v2", ["complete", "pending"]
            )
            self.assertTrue(result["valid"], result["errors"])

            with (root / "sequence_status.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["sequence", "status"])
                writer.writeheader()
                writer.writerow({"sequence": "complete", "status": "REVIEW"})
            manifest_path = root / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(
                {
                    "sequence_count": 1,
                    "review_count": 1,
                    "incomplete_count": 0,
                    "freeze_eligible": True,
                }
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            tampered = verify_frozen_build(
                root, "contract-v2", ["complete", "pending"]
            )
            self.assertFalse(tampered["valid"])
            self.assertIn(
                "sequence_status_order_or_universe_mismatch", tampered["errors"]
            )

    def test_contract_v2_rejects_omitted_required_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            omitted = "sequences/complete/quality/metadata.json"
            root = self.build_contract_v2(
                Path(temporary), omit_sequence_path=omitted
            )
            result = verify_frozen_build(
                root, "contract-v2", ["complete", "pending"]
            )
            self.assertFalse(result["valid"])
            self.assertIn(
                "required_sequence_payload_set_mismatch:complete",
                result["errors"],
            )

    def test_deadline_cutoff_excludes_post_deadline_terminal_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                body_fit_root=root / "body",
                sam_mode_c_review_root=root / "mode_c",
            )
            sequence = "sequence"
            markers = [
                args.body_fit_root / sequence / "body_fit.npz",
                args.body_fit_root / sequence / "metadata.json",
                args.sam_mode_c_review_root / sequence / "mode_c_escalation.json",
            ]
            cutoff = datetime(2026, 8, 14, 4, tzinfo=timezone.utc)
            before_ns = int(cutoff.timestamp() * 1_000_000_000) - 1_000_000
            after_ns = int(cutoff.timestamp() * 1_000_000_000) + 1_000_000
            for path in markers:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                path.chmod(0o600)
                os.utime(path, ns=(before_ns, before_ns))
            eligible, reasons, mtimes = deadline_eligibility(
                args, sequence, cutoff
            )
            self.assertTrue(eligible)
            self.assertEqual(reasons, [])
            self.assertEqual(len(mtimes), 3)

            os.utime(markers[-1], ns=(after_ns, after_ns))
            eligible, reasons, _ = deadline_eligibility(args, sequence, cutoff)
            self.assertFalse(eligible)
            self.assertEqual(
                reasons,
                ["deadline_after_cutoff:body/mode_c_escalation.json"],
            )

    def test_deadline_marker_provenance_verifier_rejects_late_marker(self) -> None:
        cutoff = datetime(2026, 8, 14, 4, tzinfo=timezone.utc)
        metadata = {
            "validation": {
                "deadline_terminal_marker_mtimes": {
                    "body/body_fit.npz": "2026-08-14T03:59:58+00:00",
                    "body/metadata.json": "2026-08-14T03:59:59+00:00",
                    "body/mode_c_escalation.json": "2026-08-14T04:00:01+00:00",
                }
            }
        }
        self.assertEqual(
            verify_deadline_marker_mtimes(metadata, cutoff),
            ["deadline_terminal_marker_after_cutoff:body/mode_c_escalation.json"],
        )
    def test_git_provenance_records_dirty_state_without_exposing_diff(self) -> None:
        results = [
            subprocess.CompletedProcess([], 0, stdout="a" * 40 + "\n", stderr=""),
            subprocess.CompletedProcess([], 0, stdout=b" M safe.py\0", stderr=b""),
            subprocess.CompletedProcess([], 0, stdout=b"synthetic diff", stderr=b""),
        ]
        with patch("tools.export_private_dataset.subprocess.run", side_effect=results):
            provenance = git_provenance()
        self.assertEqual(provenance["git_commit"], "a" * 40)
        self.assertTrue(provenance["git_worktree_dirty"])
        self.assertEqual(
            provenance["git_diff_sha256"],
            hashlib.sha256(b"synthetic diff").hexdigest(),
        )
        self.assertNotIn("synthetic diff", json.dumps(provenance))

    def test_copy_is_byte_exact_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.bin"
            destination = root / "nested" / "destination.bin"
            source.write_bytes(bytes(range(255)))
            first = copy_exact(source, destination)
            second = copy_exact(source, destination)
            self.assertEqual(sha256(source), sha256(destination))
            self.assertFalse(first["resume_skipped"])
            self.assertTrue(second["resume_skipped"])

    def test_finite_nan_contract_separates_invalid_payload(self) -> None:
        points = np.asarray([[[1.0, 2.0, 3.0]], [[np.nan, np.nan, np.nan]]])
        valid = np.asarray([[True], [False]])
        self.assertEqual(finite_nan_contract(points, valid), (True, True))
        points[1, 0, 0] = 0.0
        self.assertEqual(finite_nan_contract(points, valid), (True, False))

    def test_verified_build_is_published_by_directory_rename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / ".deadline-build.inprogress"
            final = root / "deadline-build"
            payload = staging / "provenance" / "source.json"
            payload.parent.mkdir(parents=True)
            payload.write_text("{}\n", encoding="utf-8")
            with (staging / "sequence_status.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["sequence", "status"])
                writer.writeheader()
                writer.writerow({"sequence": "pending", "status": "INCOMPLETE"})
            manifest = {
                "schema_version": 1,
                "build_id": "deadline-build",
                "private_dataset": True,
                "source_rgb_included": False,
                "source_payload_modified": False,
                "sequence_count": 1,
                "pass_count": 0,
                "review_count": 0,
                "fail_count": 0,
                "incomplete_count": 1,
                "freeze_eligible": False,
                "file_count": 1,
                "total_payload_bytes": payload.stat().st_size,
                "files": [
                    {
                        "sequence": "",
                        "path": "provenance/source.json",
                        "bytes": payload.stat().st_size,
                        "sha256": sha256(payload),
                    }
                ],
            }
            (staging / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            before = verify_frozen_build(staging, "deadline-build")
            self.assertTrue(before["valid"], before["errors"])
            publish_staged_build(staging, final, "deadline-build")
            self.assertFalse(staging.exists())
            self.assertTrue(final.is_dir())
            self.assertTrue(verify_frozen_build(final, "deadline-build")["valid"])

            (final / "provenance" / "source.json").write_text(
                "corrupt", encoding="utf-8"
            )
            corrupted = verify_frozen_build(final, "deadline-build")
            self.assertFalse(corrupted["valid"])
            self.assertTrue(
                any("mismatch" in error for error in corrupted["errors"]),
                corrupted["errors"],
            )

    def test_verifier_rejects_and_pruner_removes_unlisted_staging_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            staging = parent / ".deadline-build.inprogress"
            payload = staging / "provenance" / "source.json"
            stale = staging / "sequences" / "old" / "stale.bin"
            payload.parent.mkdir(parents=True)
            stale.parent.mkdir(parents=True)
            payload.write_text("{}\n", encoding="utf-8")
            stale.write_bytes(b"stale-private-staging-payload")
            with (staging / "sequence_status.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["sequence", "status"])
                writer.writeheader()
                writer.writerow({"sequence": "pending", "status": "INCOMPLETE"})
            manifest = {
                "schema_version": 1,
                "build_id": "deadline-build",
                "private_dataset": True,
                "source_rgb_included": False,
                "source_payload_modified": False,
                "sequence_count": 1,
                "pass_count": 0,
                "review_count": 0,
                "fail_count": 0,
                "incomplete_count": 1,
                "freeze_eligible": False,
                "file_count": 1,
                "total_payload_bytes": payload.stat().st_size,
                "files": [
                    {
                        "sequence": "",
                        "path": "provenance/source.json",
                        "bytes": payload.stat().st_size,
                        "sha256": sha256(payload),
                    }
                ],
            }
            (staging / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            before = verify_frozen_build(staging, "deadline-build")
            self.assertFalse(before["valid"])
            self.assertIn(
                "unlisted_file:sequences/old/stale.bin", before["errors"]
            )

            removed = prune_staging_tree(
                staging,
                "deadline-build",
                {
                    "dataset_manifest.json",
                    "sequence_status.csv",
                    "provenance/source.json",
                },
            )
            self.assertEqual(removed, ["sequences/old/stale.bin"])
            self.assertFalse(stale.exists())
            self.assertTrue(verify_frozen_build(staging, "deadline-build")["valid"])

    def test_staging_symlink_cleanup_never_touches_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            staging = parent / ".deadline-build.inprogress"
            staging.mkdir()
            outside = parent / "outside"
            outside.mkdir()
            marker = outside / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            link = staging / "sequences"
            link.symlink_to(outside, target_is_directory=True)

            removed = remove_staging_symlinks(staging, "deadline-build")
            self.assertEqual(removed, ["sequences"])
            self.assertFalse(link.exists())
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_verifier_rejects_symlink_build_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            target = parent / "target"
            target.mkdir()
            link = parent / ".deadline-build.inprogress"
            link.symlink_to(target, target_is_directory=True)
            result = verify_frozen_build(link, "deadline-build")
            self.assertFalse(result["valid"])
            self.assertEqual(result["errors"], ["build_root_symlink"])

    def test_staging_prune_refuses_non_staging_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ordinary-directory"
            root.mkdir()
            marker = root / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                prune_staging_tree(root, "deadline-build", set())
            self.assertTrue(marker.is_file())

    def test_verifier_rejects_payload_for_incomplete_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / ".deadline-build.inprogress"
            payload = staging / "sequences" / "pending" / "data.bin"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"not-freeze-eligible")
            with (staging / "sequence_status.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["sequence", "status"])
                writer.writeheader()
                writer.writerow({"sequence": "pending", "status": "INCOMPLETE"})
            manifest = {
                "schema_version": 1,
                "build_id": "deadline-build",
                "private_dataset": True,
                "source_rgb_included": False,
                "source_payload_modified": False,
                "sequence_count": 1,
                "pass_count": 0,
                "review_count": 0,
                "fail_count": 0,
                "incomplete_count": 1,
                "freeze_eligible": False,
                "file_count": 1,
                "total_payload_bytes": payload.stat().st_size,
                "files": [
                    {
                        "sequence": "pending",
                        "path": "sequences/pending/data.bin",
                        "bytes": payload.stat().st_size,
                        "sha256": sha256(payload),
                    }
                ],
            }
            (staging / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            result = verify_frozen_build(staging, "deadline-build")
            self.assertFalse(result["valid"])
            self.assertIn(
                "file_for_noncomplete_sequence:sequences/pending/data.bin",
                result["errors"],
            )

    def test_build_and_sequence_ids_are_single_components(self) -> None:
        validate_path_component("deadline-build", "build id")
        for unsafe in ("", ".", "..", "../escape", "nested/path"):
            with self.assertRaises(RuntimeError):
                validate_path_component(unsafe, "build id")

    def test_sequence_dependencies_include_quality_payload(self) -> None:
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
        dependencies = sequence_dependencies(args, "sequence")
        self.assertEqual(
            dependencies["quality/quality_vector.npz"].name,
            "quality_vector.npz",
        )
        self.assertEqual(
            dependencies["quality/metadata.json"].name,
            "metadata.json",
        )


if __name__ == "__main__":
    unittest.main()
