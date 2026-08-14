import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.prepare_transfer_snapshot import (
    aggregate_inventory,
    checkpoint_gate,
    is_transient_name,
    rsync_command,
    scan_finalized_path,
)


class PrepareTransferSnapshotTest(unittest.TestCase):
    def test_checkpoint_gate_reuses_exact_verified_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            build = output / "checkpoint-024"
            build.mkdir()
            sequences = [f"sequence_{index:02d}" for index in range(24)]
            manifest = {
                "build_id": build.name,
                "freeze_contract_version": 2,
                "requested_sequences": sequences,
                "sequence_count": 24,
                "fail_count": 0,
                "incomplete_count": 0,
                "file_count": 100,
                "total_payload_bytes": 12345,
                "freeze_eligible": True,
            }
            (build / "dataset_manifest.json").write_text(json.dumps(manifest))
            state = {
                "best_checkpoint": {
                    "build_id": build.name,
                    "sequences": sequences,
                    "completed_sequence_count": 24,
                    "file_count": 100,
                    "total_payload_bytes": 12345,
                    "verified_file_count": 100,
                    "verified_payload_bytes": 12345,
                    "freeze_eligible": True,
                    "integrity_verified": True,
                }
            }
            checkpoint, reasons = checkpoint_gate(state, output, 24)
            self.assertFalse(reasons)
            assert checkpoint is not None
            self.assertEqual(checkpoint["sequence_count"], 24)
            self.assertEqual(checkpoint["integrity"], "PASS_REUSED_FOLLOWER_VERIFICATION")

            state["best_checkpoint"]["verified_payload_bytes"] = 12344
            checkpoint, reasons = checkpoint_gate(state, output, 24)
            self.assertIsNone(checkpoint)
            self.assertIn("checkpoint verified byte count differs", reasons)

    def test_checkpoint_gate_does_not_accept_incomplete_or_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            build = output / "checkpoint-023"
            build.mkdir()
            sequences = [f"s{index}" for index in range(23)]
            (build / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "build_id": build.name,
                        "freeze_contract_version": 2,
                        "requested_sequences": sequences,
                        "sequence_count": 23,
                        "fail_count": 0,
                        "incomplete_count": 0,
                        "file_count": 1,
                        "total_payload_bytes": 1,
                        "freeze_eligible": True,
                    }
                )
            )
            state = {
                "best_checkpoint": {
                    "build_id": build.name,
                    "sequences": sequences,
                    "completed_sequence_count": 23,
                    "file_count": 1,
                    "total_payload_bytes": 1,
                    "verified_file_count": 1,
                    "verified_payload_bytes": 1,
                    "freeze_eligible": True,
                    "integrity_verified": True,
                }
            }
            checkpoint, reasons = checkpoint_gate(state, output, 24)
            self.assertIsNone(checkpoint)
            self.assertIn("checkpoint count 23 is below 24", reasons)

    def test_inventory_excludes_transient_locks_staging_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "stable.bin").write_bytes(b"stable")
            (root / "state.json.1.tmp").write_bytes(b"temp")
            (root / "run.lock").write_bytes(b"lock")
            staging = root / ".build.inprogress"
            staging.mkdir()
            (staging / "payload.bin").write_bytes(b"staging")
            (root / "nested").mkdir()
            (root / "nested" / "final.npz").write_bytes(b"payload")
            (root / "link").symlink_to(root / "stable.bin")
            row = scan_finalized_path(root, label="root")
            self.assertEqual(row["file_count"], 2)
            self.assertEqual(row["logical_bytes"], len(b"stable") + len(b"payload"))
            self.assertEqual(row["transient_files_excluded"], 3)
            self.assertEqual(row["symlinks_excluded"], 1)
            total = aggregate_inventory([row])
            self.assertEqual(total["file_count"], 2)
            self.assertEqual(total["scan_error_count"], 0)

    def test_transient_names_and_rsync_policy_are_conservative(self) -> None:
        for name in (
            "output.1.tmp",
            "chunk.tmp.npz",
            "file.partial",
            "file.part",
            "runner.lock",
            ".build.inprogress",
            ".rsync-partial",
            "transfer_manifest.json",
        ):
            self.assertTrue(is_transient_name(name), name)
        self.assertFalse(is_transient_name("quality_vector.npz"))
        command = rsync_command(["outputs/one", "HANDOFF.md"], bwlimit_kib=1234)
        self.assertIn("--bwlimit=1234", command)
        self.assertIn("--partial-dir=.rsync-partial", command)
        self.assertIn("--exclude='*.lock'", command)
        self.assertNotIn("--delete", command)
        self.assertIn("/./outputs/one", command)


if __name__ == "__main__":
    unittest.main()
