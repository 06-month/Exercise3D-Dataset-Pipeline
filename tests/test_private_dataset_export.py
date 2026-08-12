import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.export_private_dataset import (
    copy_exact,
    finite_nan_contract,
    publish_staged_build,
    sha256,
    validate_path_component,
    verify_frozen_build,
)


class PrivateDatasetExportTest(unittest.TestCase):
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

    def test_build_and_sequence_ids_are_single_components(self) -> None:
        validate_path_component("deadline-build", "build id")
        for unsafe in ("", ".", "..", "../escape", "nested/path"):
            with self.assertRaises(RuntimeError):
                validate_path_component(unsafe, "build id")


if __name__ == "__main__":
    unittest.main()
