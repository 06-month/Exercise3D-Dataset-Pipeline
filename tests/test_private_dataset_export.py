import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.export_private_dataset import (
    copy_exact,
    finite_nan_contract,
    git_provenance,
    prune_staging_tree,
    publish_staged_build,
    remove_staging_symlinks,
    sequence_dependencies,
    sha256,
    validate_path_component,
    verify_frozen_build,
)


class PrivateDatasetExportTest(unittest.TestCase):
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
