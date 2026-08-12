import json
import tempfile
import unittest
from pathlib import Path

from tools.run_autonomous_supervisor_watchdog import command_sha256
from tools.run_deadline_sentinel_watchdog import (
    SENTINEL_SCRIPT,
    persisted_resume_argv,
    snapshot_complete,
)


class DeadlineSentinelWatchdogTest(unittest.TestCase):
    def test_resume_identity_is_exact_repository_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "handoff.json"
            command = "python tools/run_deadline_snapshot.py --poll-seconds 30"
            state.write_text(
                json.dumps(
                    {"resume_commands": {"run_deadline_snapshot.py": command}}
                ),
                encoding="utf-8",
            )
            argv, error = persisted_resume_argv(state, SENTINEL_SCRIPT)
            self.assertIsNone(error)
            self.assertEqual(argv, command.split())
            assert argv is not None
            self.assertEqual(command_sha256(argv), command_sha256(command.split()))

    def test_complete_requires_manifest_and_truthful_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "dataset_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            complete = {
                "status": "COMPLETE",
                "manifest": str(manifest),
                "pass_count": 0,
                "review_count": 11,
                "fail_count": 0,
                "incomplete_count": 15,
            }
            self.assertTrue(snapshot_complete(complete))
            self.assertFalse(snapshot_complete({**complete, "manifest": "missing"}))
            self.assertFalse(snapshot_complete({**complete, "status": "EXPORT_FAILED"}))
            self.assertFalse(
                snapshot_complete(
                    {
                        **complete,
                        "pass_count": 0,
                        "review_count": 0,
                        "fail_count": 0,
                        "incomplete_count": 0,
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()
