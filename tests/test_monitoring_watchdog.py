import json
import tempfile
import unittest
from pathlib import Path

from tools.run_autonomous_supervisor_watchdog import (
    command_sha256,
    persisted_resume_argv,
)
from tools.run_monitoring_watchdog import (
    DASHBOARD_SCRIPT,
    HANDOFF_SCRIPT,
    validate_target_command,
)


class MonitoringWatchdogTest(unittest.TestCase):
    def test_resume_identities_are_exact_repository_monitors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "handoff.json"
            commands = {
                "monitor_autonomous_generation.py": (
                    "python tools/monitor_autonomous_generation.py "
                    "--quiet --refresh-seconds 30"
                ),
                "checkpoint_handoff_state.py": (
                    "python tools/checkpoint_handoff_state.py --output state.json "
                    "--sequences one,two"
                ),
            }
            state.write_text(
                json.dumps({"resume_commands": commands}), encoding="utf-8"
            )
            for script in (DASHBOARD_SCRIPT, HANDOFF_SCRIPT):
                argv, error = persisted_resume_argv(state, script)
                self.assertIsNone(error)
                self.assertEqual(
                    command_sha256(argv or []),
                    command_sha256(commands[script.name].split()),
                )

    def test_recovery_commands_require_persistent_safe_modes(self) -> None:
        dashboard = [
            "python",
            "tools/monitor_autonomous_generation.py",
            "--quiet",
        ]
        handoff = [
            "python",
            "tools/checkpoint_handoff_state.py",
            "--output",
            "state.json",
            "--sequences",
            "one,two",
        ]
        self.assertIsNone(validate_target_command("dashboard", dashboard))
        self.assertIsNone(validate_target_command("handoff_monitor", handoff))
        self.assertIn(
            "--quiet",
            validate_target_command("dashboard", dashboard[:2]) or "",
        )
        self.assertIn(
            "--once",
            validate_target_command("dashboard", [*dashboard, "--once"]) or "",
        )
        self.assertIn(
            "has no",
            validate_target_command("handoff_monitor", handoff[:2]) or "",
        )


if __name__ == "__main__":
    unittest.main()
