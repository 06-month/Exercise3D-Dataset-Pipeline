import unittest

import numpy as np

from tools.verify_mhr_parameter_replay import maximum_absolute


class MhrParameterReplayTest(unittest.TestCase):
    def test_maximum_absolute_checks_shape_and_value(self) -> None:
        reference = np.asarray([[1.0, 2.0]], dtype=np.float32)
        actual = np.asarray([[1.0, 2.000001]], dtype=np.float32)
        self.assertLess(maximum_absolute(reference, actual), 2e-6)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            maximum_absolute(reference, actual.reshape(2, 1))


if __name__ == "__main__":
    unittest.main()
