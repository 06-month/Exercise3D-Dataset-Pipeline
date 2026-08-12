import unittest
from unittest.mock import Mock, patch

from tools.run_predeadline_checkpoint_follower import (
    checkpoint_action,
    checkpoint_build_id,
    largest_verified_checkpoint,
    readiness_sequences,
)


class PredeadlineCheckpointFollowerTest(unittest.TestCase):
    def test_readiness_is_reordered_to_frozen_sequence_order(self) -> None:
        state = {
            "freeze_readiness": {
                "ready_sequence_count": 2,
                "ready": [
                    {"sequence": "second", "status": "REVIEW"},
                    {"sequence": "first", "status": "PASS"},
                ],
                "failures": [],
            }
        }
        ready, errors = readiness_sequences(state, ["first", "second", "third"])
        self.assertEqual(errors, [])
        self.assertEqual(ready, ["first", "second"])

    def test_readiness_rejects_count_duplicate_failure_and_bad_status(self) -> None:
        state = {
            "freeze_readiness": {
                "ready_sequence_count": 3,
                "ready": [
                    {"sequence": "one", "status": "REVIEW"},
                    {"sequence": "one", "status": "PASS"},
                    {"sequence": "two", "status": "FAIL"},
                ],
                "failures": [{"sequence": "two"}],
            }
        }
        ready, errors = readiness_sequences(state, ["one", "two"])
        self.assertIsNone(ready)
        text = ";".join(errors)
        self.assertIn("duplicate", text)
        self.assertIn("non-exportable", text)
        self.assertIn("count mismatch", text)
        self.assertIn("contains failures", text)

    def test_action_requires_strict_superset_and_increment(self) -> None:
        best = {"sequences": ["one", "two"]}
        self.assertEqual(checkpoint_action(["one", "two"], best, 1), ("WAIT", None))
        self.assertEqual(
            checkpoint_action(["one", "three"], best, 1)[0], "ATTENTION"
        )
        self.assertEqual(
            checkpoint_action(["one", "two", "three"], best, 2)[0], "WAIT"
        )
        self.assertEqual(
            checkpoint_action(["one", "two", "three"], best, 1)[0], "EXPORT"
        )

    def test_build_id_is_deterministic_and_count_bound(self) -> None:
        first = checkpoint_build_id("prefix", ["one", "two"])
        second = checkpoint_build_id("prefix", ["one", "two"])
        changed = checkpoint_build_id("prefix", ["two", "one"])
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("prefix-002-"))
        self.assertNotEqual(first, changed)

    @patch("tools.run_predeadline_checkpoint_follower.verify_frozen_build")
    @patch("tools.run_predeadline_checkpoint_follower.quick_checkpoint_candidates")
    def test_largest_checkpoint_skips_invalid_candidate(
        self, candidates_mock, verify_mock
    ) -> None:
        candidates_mock.return_value = [
            (3, 2.0, Mock(name="bad"), ["one", "two", "three"]),
            (2, 1.0, Mock(name="good"), ["one", "two"]),
        ]
        candidates_mock.return_value[0][2].name = "bad"
        candidates_mock.return_value[1][2].name = "good"
        verify_mock.side_effect = [
            {"valid": False, "errors": ["sha256_mismatch"]},
            {
                "valid": True,
                "manifest": {
                    "file_count": 20,
                    "total_payload_bytes": 200,
                    "freeze_eligible": True,
                },
                "verified_file_count": 20,
                "verified_payload_bytes": 200,
            },
        ]

        best, integrity_errors = largest_verified_checkpoint(
            Mock(), ["one", "two", "three"]
        )
        self.assertEqual(best["build_id"], "good")
        self.assertEqual(best["sequences"], ["one", "two"])
        self.assertTrue(best["integrity_verified"])
        self.assertEqual(integrity_errors, ["bad:sha256_mismatch"])


if __name__ == "__main__":
    unittest.main()
