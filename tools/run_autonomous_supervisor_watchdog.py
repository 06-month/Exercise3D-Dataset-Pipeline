#!/usr/bin/env python3
"""Safely recover the Exercise3D supervisor without duplicating a live job.

The watchdog is deliberately CPU-only.  It observes the exact repository-local
``run_autonomous_generation.py`` process, pins its argv digest against the
persisted handoff resume command while that process is alive, and only relaunches
after multiple consecutive absence observations plus a final pre-launch scan.
The supervisor's own lifetime advisory lock closes the remaining launch race.
This tool never signals or replaces a live supervisor or inference process.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_SCRIPT = (PROJECT_ROOT / "tools" / "run_autonomous_generation.py").resolve()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def command_sha256(argv: list[str]) -> str:
    canonical = json.dumps(argv, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def script_from_argv(
    argv: list[str], cwd: Path, expected_script: Path
) -> Path | None:
    for token in argv[1:]:
        candidate = Path(token)
        if candidate.name != expected_script.name:
            continue
        try:
            return (candidate if candidate.is_absolute() else cwd / candidate).resolve()
        except OSError:
            return None
    return None


def supervisor_script_from_argv(argv: list[str], cwd: Path) -> Path | None:
    return script_from_argv(argv, cwd, SUPERVISOR_SCRIPT)


def valid_script_resume_argv(
    argv: list[str], expected_script: Path, cwd: Path = PROJECT_ROOT
) -> bool:
    if len(argv) < 2 or script_from_argv(argv, cwd, expected_script) != expected_script:
        return False
    executable = Path(argv[0])
    if executable.is_absolute():
        return executable.is_file() and os.access(executable, os.X_OK)
    return shutil.which(argv[0]) is not None


def valid_resume_argv(argv: list[str], cwd: Path = PROJECT_ROOT) -> bool:
    return valid_script_resume_argv(argv, SUPERVISOR_SCRIPT, cwd)


def persisted_resume_argv(
    handoff_state: Path, expected_script: Path
) -> tuple[list[str] | None, str | None]:
    command = (
        read_json(handoff_state)
        .get("resume_commands", {})
        .get(expected_script.name)
    )
    if not isinstance(command, str) or not command.strip():
        return None, "persisted supervisor resume command is missing"
    try:
        argv = shlex.split(command)
    except ValueError as error:
        return None, f"persisted supervisor resume command is not parseable: {error}"
    if not valid_script_resume_argv(argv, expected_script):
        return None, (
            "persisted resume command does not resolve to the expected repository script"
        )
    return argv, None


def resume_argv(handoff_state: Path) -> tuple[list[str] | None, str | None]:
    return persisted_resume_argv(handoff_state, SUPERVISOR_SCRIPT)


def script_processes(
    expected_script: Path,
    proc_root: Path = Path("/proc"),
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return rows
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            argv = [
                item.decode(errors="replace")
                for item in (entry / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            if not argv:
                continue
            cwd = (entry / "cwd").resolve()
            fields = (entry / "stat").read_text(encoding="utf-8").split()
            if fields[2] == "Z" or cwd != project_root.resolve():
                continue
            if script_from_argv(argv, cwd, expected_script) != expected_script:
                continue
            rows.append(
                {
                    "pid": int(entry.name),
                    "ppid": int(fields[3]),
                    "state": fields[2],
                    "argv": argv,
                    "command_sha256": command_sha256(argv),
                }
            )
        except (OSError, IndexError, ValueError):
            continue
    return sorted(rows, key=lambda row: row["pid"])


def supervisor_processes(
    proc_root: Path = Path("/proc"), project_root: Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    return script_processes(SUPERVISOR_SCRIPT, proc_root, project_root)


def supervisor_complete(state: dict[str, Any]) -> bool:
    try:
        return bool(
            state.get("stage") == "COMPLETE"
            and state.get("final_status") == "PASS_OR_REVIEW"
            and int(state.get("completed_body_fit_count", -1))
            == int(state.get("sequence_count", -2))
            and int(state.get("failed_or_incomplete_count", -1)) == 0
        )
    except (TypeError, ValueError):
        return False


def acquire_singleton_lock(path: Path) -> BinaryIO | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    handle = os.fdopen(descriptor, "r+b", buffering=0)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n".encode("ascii"))
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def recent_restart_history(
    values: Any, now: datetime, window_seconds: float
) -> list[str]:
    result: list[str] = []
    if not isinstance(values, list):
        return result
    for value in values:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            if (now - parsed.astimezone(timezone.utc)).total_seconds() <= window_seconds:
                result.append(parsed.astimezone(timezone.utc).isoformat())
        except ValueError:
            continue
    return result


def recovery_decision(
    *,
    supervisor_count: int,
    complete: bool,
    missing_observations: int,
    required_observations: int,
    command_valid: bool,
    recent_restart_count: int,
    max_restarts: int,
) -> str:
    if supervisor_count > 1:
        return "ATTENTION_DUPLICATE"
    if supervisor_count == 1:
        return "OBSERVE"
    if complete:
        return "COMPLETE"
    if missing_observations < required_observations:
        return "CONFIRM_MISSING"
    if not command_valid:
        return "ATTENTION_COMMAND"
    if recent_restart_count >= max_restarts:
        return "ATTENTION_RESTART_EXHAUSTED"
    return "RESTART"


def launch_supervisor(argv: list[str], log_path: Path) -> int:
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
        "--supervisor-state",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "runtime"
        / "autonomous_generation"
        / "autonomous_generation_state.json",
    )
    parser.add_argument(
        "--runtime-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "supervisor_watchdog_state.json",
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "supervisor_watchdog.lock",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "autonomous_supervisor_recovery.log",
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
        print("supervisor watchdog is already running", flush=True)
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
        processes = supervisor_processes()
        complete = supervisor_complete(read_json(args.supervisor_state.resolve()))
        argv, command_error = resume_argv(args.handoff_state.resolve())
        resume_sha = command_sha256(argv) if argv is not None else None
        attention_reasons: list[dict[str, str]] = []
        last_event = "SUPERVISOR_OBSERVED"
        launched_pid: int | None = None

        if expected_sha is None and len(processes) == 1 and resume_sha:
            if processes[0]["command_sha256"] == resume_sha:
                expected_sha = resume_sha
                last_event = "COMMAND_IDENTITY_PINNED"
            else:
                command_error = "live supervisor and persisted resume command identities differ"

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
        elif not complete:
            missing_observations += 1

        restart_history = recent_restart_history(
            restart_history, now, args.restart_window_seconds
        )
        decision = recovery_decision(
            supervisor_count=len(processes),
            complete=complete,
            missing_observations=missing_observations,
            required_observations=args.missing_confirmations,
            command_valid=command_valid,
            recent_restart_count=len(restart_history),
            max_restarts=args.max_restarts,
        )

        if len(processes) == 1 and not live_identity_ok:
            decision = "ATTENTION_COMMAND"
            command_error = "live supervisor command does not match the pinned identity"

        if decision == "ATTENTION_DUPLICATE":
            attention_reasons.append(
                {
                    "code": "DUPLICATE_SUPERVISOR",
                    "message": f"{len(processes)} supervisors are alive; watchdog will not signal either process",
                }
            )
            last_event = "DUPLICATE_DETECTED_NO_ACTION"
        elif decision == "ATTENTION_COMMAND":
            attention_reasons.append(
                {
                    "code": "SUPERVISOR_RECOVERY_IDENTITY_INVALID",
                    "message": command_error or "resume command identity is not pinned",
                }
            )
            last_event = "RECOVERY_REFUSED_IDENTITY"
        elif decision == "ATTENTION_RESTART_EXHAUSTED":
            attention_reasons.append(
                {
                    "code": "SUPERVISOR_RESTART_EXHAUSTED",
                    "message": (
                        f"{len(restart_history)} launches occurred within "
                        f"{args.restart_window_seconds:.0f} seconds"
                    ),
                }
            )
            last_event = "RECOVERY_REFUSED_RESTART_LIMIT"
        elif decision == "CONFIRM_MISSING":
            last_event = "CONFIRMING_SUPERVISOR_ABSENCE"
        elif decision == "COMPLETE":
            missing_observations = 0
            last_event = "PIPELINE_COMPLETE_NO_RESTART"
        elif decision == "RESTART":
            time.sleep(args.prelaunch_confirm_seconds)
            final_processes = supervisor_processes()
            if final_processes:
                processes = final_processes
                missing_observations = 0
                last_event = "RECOVERY_CANCELLED_PROCESS_APPEARED"
                if len(final_processes) > 1:
                    attention_reasons.append(
                        {
                            "code": "DUPLICATE_SUPERVISOR",
                            "message": (
                                f"{len(final_processes)} supervisors appeared during final confirmation; "
                                "watchdog did not launch"
                            ),
                        }
                    )
            else:
                try:
                    launched_pid = launch_supervisor(argv or [], args.log_path.resolve())
                    restart_history.append(now.isoformat())
                    missing_observations = 0
                    last_event = "SUPERVISOR_RESTART_LAUNCHED"
                except OSError as error:
                    attention_reasons.append(
                        {
                            "code": "SUPERVISOR_RESTART_FAILED",
                            "message": f"supervisor launch failed: {type(error).__name__}: {error}",
                        }
                    )
                    last_event = "SUPERVISOR_RESTART_FAILED"

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
            "supervisor_alive": bool(processes) or launched_pid is not None,
            "observed_supervisor_pids": [row["pid"] for row in processes],
            "launched_supervisor_pid": launched_pid,
            "missing_observations": missing_observations,
            "required_missing_observations": args.missing_confirmations,
            "expected_command_sha256": expected_sha,
            "resume_command_sha256": resume_sha,
            "restart_history_utc": restart_history,
            "restart_count_in_window": len(restart_history),
            "restart_window_seconds": args.restart_window_seconds,
            "max_restarts": args.max_restarts,
            "last_event": last_event,
            "attention_required": bool(attention_reasons),
            "attention_reasons": attention_reasons,
            "policy": (
                "never signal live jobs; pin exact live/resume identity; require consecutive absence; "
                "final rescan; singleton watchdog; supervisor lifetime lock; capped detached recovery"
            ),
        }
        atomic_json(args.runtime_state.resolve(), state)
        if args.once:
            return int(bool(attention_reasons))
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
