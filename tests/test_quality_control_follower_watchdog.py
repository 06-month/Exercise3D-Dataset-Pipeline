import json
import tempfile
import unittest
from pathlib import Path

from tools.run_autonomous_supervisor_watchdog import command_sha256
from tools.run_quality_control_follower_watchdog import (
    FOLLOWER_SCRIPT,
    persisted_resume_argv,
    quality_complete,
    validate_recovery_command,
)


class QualityControlFollowerWatchdogTest(unittest.TestCase):
    def test_resume_identity_is_exact_repository_follower(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "handoff.json"
            command = (
                "python tools/run_quality_control_follower.py "
                "--sequences one,two --poll-seconds 30"
            )
            state.write_text(
                json.dumps(
                    {
                        "resume_commands": {
                            "run_quality_control_follower.py": command
                        }
                    }
                ),
                encoding="utf-8",
            )
            argv, error = persisted_resume_argv(state, FOLLOWER_SCRIPT)
            self.assertIsNone(error)
            self.assertEqual(argv, command.split())
            assert argv is not None
            self.assertEqual(command_sha256(argv), command_sha256(command.split()))

    def test_complete_requires_all_quality_and_freeze_readiness(self) -> None:
        complete = {
            "status": "COMPLETE",
            "completed_sequence_count": 26,
            "total_sequence_count": 26,
            "failures": [],
            "freeze_readiness": {
                "ready_sequence_count": 26,
                "total_sequence_count": 26,
                "failures": [],
            },
        }
        self.assertTrue(quality_complete(complete))
        self.assertFalse(quality_complete({**complete, "status": "RUNNING"}))
        self.assertFalse(
            quality_complete({**complete, "completed_sequence_count": 25})
        )
        self.assertFalse(
            quality_complete(
                {
                    **complete,
                    "freeze_readiness": {
                        **complete["freeze_readiness"],
                        "ready_sequence_count": 25,
                    },
                }
            )
        )
        self.assertFalse(quality_complete({**complete, "failures": ["failure"]}))

    def test_recovery_command_requires_persistent_sequence_set(self) -> None:
        valid = [
            "python",
            "tools/run_quality_control_follower.py",
            "--sequences",
            "one,two",
        ]
        self.assertIsNone(validate_recovery_command(valid))
        self.assertIn("--once", validate_recovery_command([*valid, "--once"]) or "")
        self.assertIn(
            "no --sequences", validate_recovery_command(valid[:2]) or ""
        )


if __name__ == "__main__":
    unittest.main()
