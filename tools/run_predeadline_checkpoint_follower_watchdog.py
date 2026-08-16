#!/usr/bin/env python3
"""Recover the CPU-only checkpoint follower without duplicating a live job."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

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


FOLLOWER_SCRIPT = (
    PROJECT_ROOT / "tools" / "run_predeadline_checkpoint_follower.py"
).resolve()


def follower_processes() -> list[dict[str, object]]:
    return script_processes(FOLLOWER_SCRIPT)


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must include a timezone")
    return parsed.astimezone(timezone.utc)


def command_option(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def validate_recovery_command(
    argv: list[str] | None, expected_deadline: datetime
) -> str | None:
    if argv is None:
        return "checkpoint follower resume command is missing"
    if "--once" in argv:
        return "checkpoint follower resume command must not contain --once"
    value = command_option(argv, "--deadline-utc")
    if value is None:
        return "checkpoint follower resume command has no --deadline-utc"
    try:
        command_deadline = parse_utc(value)
    except ValueError as error:
        return f"checkpoint follower deadline is invalid: {error}"
    if command_deadline != expected_deadline:
        return "checkpoint follower deadline differs from watchdog deadline"
    return None


def launch_follower(argv: list[str], log_path: Path) -> int:
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
        "--follower-state",
        type=Path,
        default=PROJECT_ROOT
        / ".runtime"
        / "predeadline_checkpoint_follower_state.json",
    )
    parser.add_argument(
        "--runtime-state",
        type=Path,
        default=PROJECT_ROOT
        / ".runtime"
        / "predeadline_checkpoint_follower_watchdog_state.json",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=PROJECT_ROOT
        / ".runtime"
        / "predeadline_checkpoint_follower_watchdog.lock",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PROJECT_ROOT
        / ".runtime"
        / "predeadline_checkpoint_follower_recovery.log",
    )
    parser.add_argument("--deadline-utc", default="2026-08-14T04:00:00+00:00")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--missing-confirmations", type=int, default=3)
    parser.add_argument("--prelaunch-confirm-seconds", type=float, default=2.0)
    parser.add_argument("--restart-window-seconds", type=float, default=3600.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    deadline = parse_utc(args.deadline_utc)
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
        print("predeadline checkpoint follower watchdog is already running", flush=True)
        return 3

    previous = read_json(args.runtime_state.resolve())
    expected_sha = previous.get("expected_command_sha256")
    expected_sha = str(expected_sha) if expected_sha else None
    missing_observations = 0
    restart_history = recent_restart_history(
        previous.get("restart_history_utc", []),
        utc_now(),
        args.restart_window_seconds,
    )

    while True:
        now = utc_now()
        processes = follower_processes()
        follower_state = read_json(args.follower_state.resolve())
        deadline_reached = now >= deadline
        argv, command_error = persisted_resume_argv(
            args.handoff_state.resolve(), FOLLOWER_SCRIPT
        )
        if argv is None and command_error is not None:
            command_error = (
                "persisted checkpoint follower resume command is invalid: "
                + command_error
            )
        validation_error = validate_recovery_command(argv, deadline)
        if command_error is None and validation_error is not None:
            command_error = validation_error
        resume_sha = command_sha256(argv) if argv is not None else None
        attention_reasons: list[dict[str, str]] = []
        launched_pid: int | None = None
        last_event = "CHECKPOINT_FOLLOWER_OBSERVED"

        if expected_sha is None and len(processes) == 1 and resume_sha:
            if processes[0]["command_sha256"] == resume_sha and command_error is None:
                expected_sha = resume_sha
                last_event = "COMMAND_IDENTITY_PINNED"
            else:
                command_error = (
                    command_error
                    or "live follower and persisted resume command identities differ"
                )
        live_identity_ok = bool(
            len(processes) == 1
            and expected_sha
            and processes[0]["command_sha256"] == expected_sha
        )
        command_valid = bool(
            argv is not None
            and expected_sha
            and resume_sha == expected_sha
            and command_error is None
        )
        if processes:
            missing_observations = 0
        elif not deadline_reached:
            missing_observations += 1
        restart_history = recent_restart_history(
            restart_history, now, args.restart_window_seconds
        )
        decision = recovery_decision(
            supervisor_count=len(processes),
            complete=deadline_reached,
            missing_observations=missing_observations,
            required_observations=args.missing_confirmations,
            command_valid=command_valid,
            recent_restart_count=len(restart_history),
            max_restarts=args.max_restarts,
        )
        if len(processes) == 1 and not live_identity_ok:
            decision = "ATTENTION_COMMAND"
            command_error = "live checkpoint follower does not match pinned identity"

        if decision == "ATTENTION_DUPLICATE":
            attention_reasons.append(
                {
                    "code": "DUPLICATE_PREDEADLINE_CHECKPOINT_FOLLOWER",
                    "message": (
                        f"{len(processes)} checkpoint followers are alive; "
                        "watchdog will not signal either"
                    ),
                }
            )
            last_event = "DUPLICATE_DETECTED_NO_ACTION"
        elif decision == "ATTENTION_COMMAND":
            attention_reasons.append(
                {
                    "code": "CHECKPOINT_FOLLOWER_RECOVERY_IDENTITY_INVALID",
                    "message": command_error or "resume command identity is not pinned",
                }
            )
            last_event = "RECOVERY_REFUSED_IDENTITY"
        elif decision == "ATTENTION_RESTART_EXHAUSTED":
            attention_reasons.append(
                {
                    "code": "CHECKPOINT_FOLLOWER_RESTART_EXHAUSTED",
                    "message": (
                        f"{len(restart_history)} launches occurred within "
                        f"{args.restart_window_seconds:.0f} seconds"
                    ),
                }
            )
            last_event = "RECOVERY_REFUSED_RESTART_LIMIT"
        elif decision == "CONFIRM_MISSING":
            last_event = "CONFIRMING_CHECKPOINT_FOLLOWER_ABSENCE"
        elif decision == "COMPLETE":
            missing_observations = 0
            last_event = "DEADLINE_REACHED_NO_RESTART"
        elif decision == "RESTART":
            time.sleep(args.prelaunch_confirm_seconds)
            final_processes = follower_processes()
            if final_processes:
                processes = final_processes
                missing_observations = 0
                last_event = "RECOVERY_CANCELLED_PROCESS_APPEARED"
                if len(final_processes) > 1:
                    attention_reasons.append(
                        {
                            "code": "DUPLICATE_PREDEADLINE_CHECKPOINT_FOLLOWER",
                            "message": (
                                f"{len(final_processes)} followers appeared during final "
                                "confirmation; watchdog did not launch"
                            ),
                        }
                    )
            else:
                try:
                    launched_pid = launch_follower(argv or [], args.log_path.resolve())
                    restart_history.append(now.isoformat())
                    missing_observations = 0
                    last_event = "CHECKPOINT_FOLLOWER_RESTART_LAUNCHED"
                except OSError as error:
                    attention_reasons.append(
                        {
                            "code": "CHECKPOINT_FOLLOWER_RESTART_FAILED",
                            "message": (
                                f"follower launch failed: {type(error).__name__}: {error}"
                            ),
                        }
                    )
                    last_event = "CHECKPOINT_FOLLOWER_RESTART_FAILED"

        status = (
            "ATTENTION"
            if attention_reasons
            else "COMPLETE"
            if decision == "COMPLETE"
            else "RECOVERED"
            if launched_pid is not None
            else "RUNNING"
        )
        state = {
            "schema_version": 1,
            "updated_at_utc": now.isoformat(),
            "status": status,
            "pid": os.getpid(),
            "follower_alive": bool(processes) or launched_pid is not None,
            "observed_follower_pids": [row["pid"] for row in processes],
            "launched_follower_pid": launched_pid,
            "missing_observations": missing_observations,
            "required_missing_observations": args.missing_confirmations,
            "expected_command_sha256": expected_sha,
            "resume_command_sha256": resume_sha,
            "restart_history_utc": restart_history,
            "restart_count_in_window": len(restart_history),
            "restart_window_seconds": args.restart_window_seconds,
            "max_restarts": args.max_restarts,
            "deadline_utc": deadline.isoformat(),
            "follower_status": follower_state.get("status", "UNKNOWN"),
            "last_event": last_event,
            "attention_required": bool(attention_reasons),
            "attention_reasons": attention_reasons,
            "policy": (
                "never signal live jobs; pin exact live/resume identity; require consecutive "
                "absence; final rescan; singleton watchdog; follower lifetime lock; capped "
                "detached recovery; no restart at or after deadline"
            ),
        }
        atomic_json(args.runtime_state.resolve(), state)
        if args.once or decision == "COMPLETE":
            return int(bool(attention_reasons))
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
