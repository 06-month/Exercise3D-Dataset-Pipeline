#!/usr/bin/env python3
"""Write a private, non-destructive dataset snapshot at the fixed deadline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.export_private_dataset import verify_frozen_build
except ModuleNotFoundError:
    from export_private_dataset import verify_frozen_build


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty sequence list")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("sequence list contains duplicates")
    return result


def export_command(
    args: argparse.Namespace, *, defer_eligible_incomplete: bool = False
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "export_private_dataset.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--selection-root", str(args.selection_root.resolve()),
        "--pose-root", str(args.pose_root.resolve()),
        "--triangulation-root", str(args.triangulation_root.resolve()),
        "--sam-prior-root", str(args.sam_prior_root.resolve()),
        "--sam-mode-c-review-root", str(args.sam_mode_c_review_root.resolve()),
        "--body-fit-root", str(args.body_fit_root.resolve()),
        "--quality-root", str(args.quality_root.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--build-id", args.build_id,
        "--sequences", ",".join(args.sequences),
    ]
    if getattr(args, "deadline_utc", None):
        command.extend(["--deadline-cutoff-utc", str(args.deadline_utc)])
    if defer_eligible_incomplete:
        command.append("--defer-eligible-incomplete")
    return command


def read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def verified_manifest(
    path: Path,
    expected_build_id: str,
    expected_sequences: list[str] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    manifest = read_manifest(path)
    if manifest is None:
        return None, []
    result = verify_frozen_build(
        path.parent, expected_build_id, expected_sequences
    )
    if not result["valid"]:
        return None, list(result["errors"])
    return result["manifest"], []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--sam-mode-c-review-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument(
        "--quality-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "quality_control_full",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-state", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--deadline-utc", default="2026-08-14T04:00:00+00:00")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--export-retries", type=int, default=3)
    parser.add_argument("--retry-seconds", type=float, default=30.0)
    return parser


def deadline_state_base(
    args: argparse.Namespace, deadline: datetime, now: datetime
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at_utc": now.isoformat(),
        "deadline_utc": deadline.isoformat(),
        "build_id": args.build_id,
        "point_in_time_policy": (
            "terminal body-fit and Mode-C marker mtimes must not exceed deadline; "
            "post-deadline sequences remain INCOMPLETE"
        ),
    }


def run_export_with_retries(
    args: argparse.Namespace,
    deadline: datetime,
    manifest_path: Path,
) -> tuple[dict[str, Any] | None, list[str], int, int]:
    last_exit_code = -1
    integrity_errors: list[str] = []
    for attempt in range(1, args.export_retries + 2):
        defer_eligible_incomplete = attempt <= args.export_retries
        command = export_command(
            args,
            defer_eligible_incomplete=defer_eligible_incomplete,
        )
        now = datetime.now(timezone.utc)
        atomic_json(
            args.runtime_state.resolve(),
            {
                **deadline_state_base(args, deadline, now),
                "status": "EXPORTING_DEADLINE_SNAPSHOT",
                "attempt": attempt,
                "maximum_attempts": args.export_retries + 1,
                "command": command,
                "defer_cutoff_eligible_incomplete": defer_eligible_incomplete,
            },
        )
        process = subprocess.run(command, cwd=PROJECT_ROOT)
        last_exit_code = process.returncode
        manifest, integrity_errors = verified_manifest(
            manifest_path, args.build_id, args.sequences
        )
        if manifest is not None or integrity_errors:
            return manifest, integrity_errors, last_exit_code, attempt
        if attempt <= args.export_retries:
            now = datetime.now(timezone.utc)
            atomic_json(
                args.runtime_state.resolve(),
                {
                    **deadline_state_base(args, deadline, now),
                    "status": "EXPORT_RETRY_WAIT",
                    "attempt": attempt,
                    "maximum_attempts": args.export_retries + 1,
                    "export_exit_code": last_exit_code,
                    "retry_in_seconds": args.retry_seconds,
                    "staging_resume": True,
                },
            )
            time.sleep(args.retry_seconds)
    return None, integrity_errors, last_exit_code, args.export_retries + 1


def main() -> int:
    args = build_parser().parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if (
        deadline.tzinfo is None
        or args.poll_seconds <= 0
        or args.export_retries < 0
        or args.retry_seconds <= 0
    ):
        raise RuntimeError(
            "deadline must be timezone-aware, intervals positive, and retries nonnegative"
        )
    deadline = deadline.astimezone(timezone.utc)
    manifest_path = args.output_root.resolve() / args.build_id / "dataset_manifest.json"
    while True:
        now = datetime.now(timezone.utc)
        manifest, integrity_errors = verified_manifest(
            manifest_path, args.build_id, args.sequences
        )
        if integrity_errors:
            atomic_json(
                args.runtime_state.resolve(),
                {
                    **deadline_state_base(args, deadline, now),
                    "status": "EXISTING_BUILD_INVALID",
                    "manifest": str(manifest_path),
                    "integrity_errors": integrity_errors,
                    "recovery": "preserve immutable build and choose a new build id",
                },
            )
            return 2
        if manifest is not None:
            atomic_json(
                args.runtime_state.resolve(),
                {
                    **deadline_state_base(args, deadline, now),
                    "status": "COMPLETE",
                    "manifest": str(manifest_path),
                    "freeze_eligible": manifest.get("freeze_eligible"),
                    "pass_count": manifest.get("pass_count"),
                    "review_count": manifest.get("review_count"),
                    "fail_count": manifest.get("fail_count"),
                    "incomplete_count": manifest.get("incomplete_count"),
                },
            )
            return 0
        if now < deadline:
            atomic_json(
                args.runtime_state.resolve(),
                {
                    **deadline_state_base(args, deadline, now),
                    "status": "WAITING_DEADLINE",
                    "remaining_wall_hours": (deadline - now).total_seconds() / 3600,
                    "command": " ".join(sys.argv),
                    "success_condition": "versioned manifest exists; PASS/REVIEW/FAIL/INCOMPLETE retained",
                    "next_stage": "continue autonomous generation after snapshot",
                },
            )
            time.sleep(args.poll_seconds)
            continue
        manifest, integrity_errors, export_exit_code, attempt = run_export_with_retries(
            args,
            deadline,
            manifest_path,
        )
        atomic_json(
            args.runtime_state.resolve(),
            {
                **deadline_state_base(args, deadline, datetime.now(timezone.utc)),
                "status": (
                    "COMPLETE"
                    if manifest is not None
                    else "EXPORT_INTEGRITY_FAILED"
                    if integrity_errors
                    else "EXPORT_FAILED"
                ),
                "export_exit_code": export_exit_code,
                "attempt": attempt,
                "maximum_attempts": args.export_retries + 1,
                "manifest": str(manifest_path),
                "freeze_eligible": manifest.get("freeze_eligible") if manifest else False,
                "incomplete_count": manifest.get("incomplete_count") if manifest else None,
                "integrity_errors": integrity_errors,
            },
        )
        return 0 if manifest is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
