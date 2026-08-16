from pathlib import Path
import unittest

from tools.check_publication_safety import PROJECT_ROOT, PUBLIC_SHOWCASE_PATHS, inspect


class PublicationSafetyTest(unittest.TestCase):
    def test_exact_public_showcase_allowlist_passes(self) -> None:
        for relative in PUBLIC_SHOWCASE_PATHS:
            self.assertEqual(inspect(PROJECT_ROOT / relative), [], relative)

    def test_arbitrary_video_remains_forbidden(self) -> None:
        errors = inspect(PROJECT_ROOT / "docs/assets/showcase/not_reviewed.mp4")
        self.assertTrue(any("payload 확장자 금지" in error for error in errors))

    def test_allowlist_contains_only_requested_sequences(self) -> None:
        expected = {
            "benchpress_0004", "deadlift_0001", "barbellrow_0003",
            "latpulldown_0003", "squat_0002",
        }
        actual = {Path(path).stem.removesuffix("_mhr_mesh") for path in PUBLIC_SHOWCASE_PATHS}
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
