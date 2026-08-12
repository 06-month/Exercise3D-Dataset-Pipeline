import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from tools.run_autonomous_supervisor_watchdog import command_sha256
from tools.run_predeadline_checkpoint_follower_watchdog import (
    FOLLOWER_SCRIPT,
    command_option,
    parse_utc,
    persisted_resume_argv,
    validate_recovery_command,
)


class PredeadlineCheckpointFollowerWatchdogTest(unittest.TestCase):
    def test_resume_identity_is_exact_repository_follower(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "handoff.json"
            command = (
                "python tools/run_predeadline_checkpoint_follower.py "
                "--deadline-utc 2026-08-14T04:00:00+00:00"
            )
            state.write_text(
                json.dumps(
                    {
                        "resume_commands": {
                            "run_predeadline_checkpoint_follower.py": command
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

    def test_recovery_command_requires_matching_deadline_and_persistent_mode(self) -> None:
        deadline = datetime(2026, 8, 14, 4, tzinfo=timezone.utc)
        valid = [
            "python",
            "tools/run_predeadline_checkpoint_follower.py",
            "--deadline-utc",
            "2026-08-14T04:00:00+00:00",
        ]
        self.assertIsNone(validate_recovery_command(valid, deadline))
        self.assertIn(
            "--once",
            validate_recovery_command([*valid, "--once"], deadline) or "",
        )
        mismatched = [*valid[:-1], "2026-08-14T05:00:00+00:00"]
        self.assertIn(
            "differs", validate_recovery_command(mismatched, deadline) or ""
        )
        self.assertIn(
            "no --deadline",
            validate_recovery_command(valid[:2], deadline) or "",
        )

    def test_deadline_parser_and_option_are_strict(self) -> None:
        parsed = parse_utc("2026-08-14T13:00:00+09:00")
        self.assertEqual(parsed, datetime(2026, 8, 14, 4, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            parse_utc("2026-08-14T04:00:00")
        self.assertEqual(command_option(["x", "--value", "one"], "--value"), "one")
        self.assertIsNone(command_option(["x", "--value"], "--value"))


if __name__ == "__main__":
    unittest.main()
