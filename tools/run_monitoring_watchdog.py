#!/usr/bin/env python3
"""Recover the CPU-only dashboard and handoff monitor without duplication.

Both monitored processes publish operational metadata only.  The watchdog pins
each live argv digest to the exact command persisted in ``handoff_state.json``.
It never signals a live process and launches only after consecutive absence
observations plus a final scan.  Target lifetime locks close launch races.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from tools.run_autonomous_supervisor_watchdog import (
        PROJECT_ROOT,
        acquire_singleton_lock,
        atomic_json,
        command_sha256,
        persisted_resume_argv,
        read_json,
        recent_restart_history,
        recovery_decision,
        script_processes,
        utc_now,
    )
except ModuleNotFoundError:
    from run_autonomous_supervisor_watchdog import (
        PROJECT_ROOT,
        acquire_singleton_lock,
        atomic_json,
        command_sha256,
        persisted_resume_argv,
        read_json,
        recent_restart_history,
        recovery_decision,
        script_processes,
        utc_now,
    )


DASHBOARD_SCRIPT = (PROJECT_ROOT / "tools" / "monitor_autonomous_generation.py").resolve()
HANDOFF_SCRIPT = (PROJECT_ROOT / "tools" / "checkpoint_handoff_state.py").resolve()
TARGETS = {
    "dashboard": DASHBOARD_SCRIPT,
    "handoff_monitor": HANDOFF_SCRIPT,
}


def validate_target_command(target: str, argv: list[str] | None) -> str | None:
    if argv is None:
        return f"{target} resume command is missing"
    if "--once" in argv:
        return f"{target} resume command must not contain --once"
    if target == "dashboard" and "--quiet" not in argv:
        return "detached dashboard recovery requires --quiet"
    if target == "handoff_monitor":
        for required in ("--sequences", "--output"):
            if required not in argv:
                return f"handoff monitor resume command has no {required}"
    return None


def launch_target(argv: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    return process.pid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handoff-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "handoff_state.json",
    )
    parser.add_argument(
        "--runtime-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "monitoring_watchdog_state.json",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "monitoring_watchdog.lock",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "monitoring_recovery.log",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--missing-confirmations", type=int, default=3)
    parser.add_argument("--prelaunch-confirm-seconds", type=float, default=2.0)
    parser.add_argument("--restart-window-seconds", type=float, default=3600.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.poll_seconds <= 0
        or args.missing_confirmations < 2
        or args.prelaunch_confirm_seconds < 1
        or args.restart_window_seconds <= 0
        or args.max_restarts <= 0
    ):
        raise RuntimeError("watchdog intervals/restart limits are invalid")
    singleton = acquire_singleton_lock(args.lock_path.resolve())
    if singleton is None:
        print("monitoring watchdog is already running", flush=True)
        return 3

    previous = read_json(args.runtime_state.resolve())
    previous_targets = previous.get("targets", {})
    expected_sha: dict[str, str | None] = {
        target: (
            str(previous_targets.get(target, {}).get("expected_command_sha256"))
            if previous_targets.get(target, {}).get("expected_command_sha256")
            else None
        )
        for target in TARGETS
    }
    missing = {
        target: int(previous_targets.get(target, {}).get("missing_observations", 0) or 0)
        for target in TARGETS
    }
    restart_history: dict[str, list[str]] = {
        target: recent_restart_history(
            previous_targets.get(target, {}).get("restart_history_utc", []),
            utc_now(),
            args.restart_window_seconds,
        )
        for target in TARGETS
    }

    while True:
        now = utc_now()
        target_states: dict[str, dict[str, Any]] = {}
        all_attention: list[dict[str, str]] = []
        events: list[str] = []

        for target, script in TARGETS.items():
            processes = script_processes(script)
            argv, command_error = persisted_resume_argv(
                args.handoff_state.resolve(), script
            )
            validation_error = validate_target_command(target, argv)
            if command_error is None and validation_error is not None:
                command_error = validation_error
            resume_sha = command_sha256(argv) if argv is not None else None
            event = f"{target.upper()}_OBSERVED"
            launched_pid: int | None = None
            attention: list[dict[str, str]] = []

            if expected_sha[target] is None and len(processes) == 1 and resume_sha:
                if (
                    processes[0]["command_sha256"] == resume_sha
                    and command_error is None
                ):
                    expected_sha[target] = resume_sha
                    event = f"{target.upper()}_IDENTITY_PINNED"
                else:
                    command_error = (
                        command_error
                        or f"live {target} and persisted resume identities differ"
                    )
            live_identity_ok = bool(
                len(processes) == 1
                and expected_sha[target]
                and processes[0]["command_sha256"] == expected_sha[target]
            )
            command_valid = bool(
                argv is not None
                and expected_sha[target]
                and resume_sha == expected_sha[target]
                and command_error is None
            )
            if processes:
                missing[target] = 0
            else:
                missing[target] += 1
            restart_history[target] = recent_restart_history(
                restart_history[target], now, args.restart_window_seconds
            )
            decision = recovery_decision(
                supervisor_count=len(processes),
                complete=False,
                missing_observations=missing[target],
                required_observations=args.missing_confirmations,
                command_valid=command_valid,
                recent_restart_count=len(restart_history[target]),
                max_restarts=args.max_restarts,
            )
            if len(processes) == 1 and (not live_identity_ok or not command_valid):
                decision = "ATTENTION_COMMAND"
                command_error = command_error or (
                    f"live {target} does not match pinned identity"
                )

            if decision == "ATTENTION_DUPLICATE":
                attention.append(
                    {
                        "code": f"DUPLICATE_{target.upper()}",
                        "message": (
                            f"{len(processes)} {target} processes are alive; "
                            "watchdog will not signal any process"
                        ),
                    }
                )
                event = f"{target.upper()}_DUPLICATE_NO_ACTION"
            elif decision == "ATTENTION_COMMAND":
                attention.append(
                    {
                        "code": f"{target.upper()}_RECOVERY_IDENTITY_INVALID",
                        "message": command_error or "resume identity is not pinned",
                    }
                )
                event = f"{target.upper()}_RECOVERY_REFUSED_IDENTITY"
            elif decision == "ATTENTION_RESTART_EXHAUSTED":
                attention.append(
                    {
                        "code": f"{target.upper()}_RESTART_EXHAUSTED",
                        "message": (
                            f"{len(restart_history[target])} launches occurred within "
                            f"{args.restart_window_seconds:.0f} seconds"
                        ),
                    }
                )
                event = f"{target.upper()}_RECOVERY_REFUSED_RESTART_LIMIT"
            elif decision == "CONFIRM_MISSING":
                event = f"CONFIRMING_{target.upper()}_ABSENCE"
            elif decision == "RESTART":
                time.sleep(args.prelaunch_confirm_seconds)
                final_processes = script_processes(script)
                if final_processes:
                    processes = final_processes
                    missing[target] = 0
                    event = f"{target.upper()}_RECOVERY_CANCELLED_PROCESS_APPEARED"
                    if len(final_processes) > 1:
                        attention.append(
                            {
                                "code": f"DUPLICATE_{target.upper()}",
                                "message": (
                                    f"{len(final_processes)} {target} processes appeared "
                                    "during final confirmation; watchdog did not launch"
                                ),
                            }
                        )
                elif argv is not None:
                    launched_pid = launch_target(argv, args.log_path.resolve())
                    restart_history[target].append(now.isoformat())
                    missing[target] = 0
                    event = f"{target.upper()}_RECOVERY_LAUNCHED"

            target_state = {
                "script": str(script.relative_to(PROJECT_ROOT)),
                "process_count": len(processes),
                "observed_pids": [int(row["pid"]) for row in processes],
                "missing_observations": missing[target],
                "expected_command_sha256": expected_sha[target],
                "resume_command_sha256": resume_sha,
                "live_identity_exact": live_identity_ok,
                "restart_history_utc": restart_history[target],
                "restart_count_in_window": len(restart_history[target]),
                "launched_pid": launched_pid,
                "last_event": event,
                "attention_reasons": attention,
            }
            target_states[target] = target_state
            events.append(event)
            all_attention.extend(attention)

        payload = {
            "schema_version": 1,
            "updated_at_utc": now.isoformat(),
            "status": "ATTENTION" if all_attention else "RUNNING",
            "pid": os.getpid(),
            "attention_required": bool(all_attention),
            "attention_reasons": all_attention,
            "last_event": ";".join(events),
            "targets": target_states,
            "restart_window_seconds": args.restart_window_seconds,
            "max_restarts": args.max_restarts,
        }
        atomic_json(args.runtime_state.resolve(), payload)
        if args.once:
            return int(bool(all_attention))
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
