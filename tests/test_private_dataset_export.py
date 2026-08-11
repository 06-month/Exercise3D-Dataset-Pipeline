import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.export_private_dataset import copy_exact, finite_nan_contract, sha256


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


if __name__ == "__main__":
    unittest.main()
