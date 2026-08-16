import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.run_autonomous_supervisor_watchdog import (
    PROJECT_ROOT,
    acquire_singleton_lock,
    adoptable_live_command_sha,
    command_sha256,
    recent_restart_history,
    recovery_decision,
    resume_argv,
    supervisor_complete,
    supervisor_processes,
)


class SupervisorWatchdogTest(unittest.TestCase):
    def test_explicit_identity_adoption_requires_one_exact_live_resume_match(self) -> None:
        live = [{"command_sha256": "new"}]
        self.assertEqual(
            adoptable_live_command_sha(live, "new", None), ("new", None)
        )
        adopted, error = adoptable_live_command_sha([], "new", None)
        self.assertIsNone(adopted)
        self.assertIn("exactly one", str(error))
        adopted, error = adoptable_live_command_sha(live * 2, "new", None)
        self.assertIsNone(adopted)
        self.assertIn("found 2", str(error))
        adopted, error = adoptable_live_command_sha(live, "other", None)
        self.assertIsNone(adopted)
        self.assertIn("differ", str(error))
        adopted, error = adoptable_live_command_sha(live, None, "bad resume")
        self.assertIsNone(adopted)
        self.assertEqual(error, "bad resume")

    def test_recovery_requires_confirmed_absence_and_valid_identity(self) -> None:
        base = dict(
            supervisor_count=0,
            complete=False,
            required_observations=3,
            command_valid=True,
            recent_restart_count=0,
            max_restarts=3,
        )
        self.assertEqual(
            recovery_decision(missing_observations=2, **base),
            "CONFIRM_MISSING",
        )
        self.assertEqual(
            recovery_decision(missing_observations=3, **base),
            "RESTART",
        )
        self.assertEqual(
            recovery_decision(
                missing_observations=3,
                **{**base, "command_valid": False},
            ),
            "ATTENTION_COMMAND",
        )
        self.assertEqual(
            recovery_decision(
                missing_observations=3,
                **{**base, "recent_restart_count": 3},
            ),
            "ATTENTION_RESTART_EXHAUSTED",
        )

    def test_live_duplicate_and_terminal_complete_never_restart(self) -> None:
        common = dict(
            missing_observations=10,
            required_observations=3,
            command_valid=True,
            recent_restart_count=0,
            max_restarts=3,
        )
        self.assertEqual(
            recovery_decision(supervisor_count=1, complete=False, **common),
            "OBSERVE",
        )
        self.assertEqual(
            recovery_decision(supervisor_count=2, complete=False, **common),
            "ATTENTION_DUPLICATE",
        )
        self.assertEqual(
            recovery_decision(supervisor_count=0, complete=True, **common),
            "COMPLETE",
        )

    def test_singleton_lock_prevents_two_watchdogs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "watchdog.lock"
            first = acquire_singleton_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_singleton_lock(path))
            assert first is not None
            first.close()

    def test_resume_command_must_resolve_to_repository_supervisor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "handoff.json"
            command = "python tools/run_autonomous_generation.py --poll-seconds 30"
            state.write_text(
                json.dumps(
                    {"resume_commands": {"run_autonomous_generation.py": command}}
                ),
                encoding="utf-8",
            )
            argv, error = resume_argv(state)
            self.assertIsNone(error)
            self.assertEqual(argv, command.split())
            assert argv is not None
            self.assertEqual(command_sha256(argv), command_sha256(command.split()))

            state.write_text(
                json.dumps(
                    {
                        "resume_commands": {
                            "run_autonomous_generation.py": "python /tmp/run_autonomous_generation.py"
                        }
                    }
                ),
                encoding="utf-8",
            )
            argv, error = resume_argv(state)
            self.assertIsNone(argv)
            self.assertIn("does not resolve", str(error))

    def test_proc_scan_requires_exact_script_and_repository_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            good = proc / "101"
            good.mkdir()
            (good / "cmdline").write_bytes(
                b"python\0tools/run_autonomous_generation.py\0--poll-seconds\0"
                b"30\0"
            )
            (good / "stat").write_text("101 (python) S 1\n", encoding="utf-8")
            os.symlink(PROJECT_ROOT, good / "cwd")

            unrelated = proc / "102"
            unrelated.mkdir()
            (unrelated / "cmdline").write_bytes(
                b"python\0tools/run_autonomous_supervisor_watchdog.py\0"
            )
            (unrelated / "stat").write_text("102 (python) S 1\n", encoding="utf-8")
            os.symlink(PROJECT_ROOT, unrelated / "cwd")

            rows = supervisor_processes(proc, PROJECT_ROOT)
            self.assertEqual([row["pid"] for row in rows], [101])

    def test_completion_and_restart_window_are_strict(self) -> None:
        self.assertTrue(
            supervisor_complete(
                {
                    "stage": "COMPLETE",
                    "final_status": "PASS_OR_REVIEW",
                    "completed_body_fit_count": 26,
                    "sequence_count": 26,
                    "failed_or_incomplete_count": 0,
                }
            )
        )
        self.assertFalse(
            supervisor_complete(
                {
                    "stage": "COMPLETE",
                    "final_status": "INCOMPLETE_OR_FAIL",
                    "completed_body_fit_count": 25,
                    "sequence_count": 26,
                    "failed_or_incomplete_count": 1,
                }
            )
        )
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(seconds=10)).isoformat()
        old = (now - timedelta(hours=2)).isoformat()
        self.assertEqual(recent_restart_history([recent, old, "bad"], now, 60), [recent])


if __name__ == "__main__":
    unittest.main()
