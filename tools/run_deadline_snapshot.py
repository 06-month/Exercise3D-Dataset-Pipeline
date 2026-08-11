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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty sequence list")
    return result


def export_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "export_private_dataset.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--selection-root", str(args.selection_root.resolve()),
        "--pose-root", str(args.pose_root.resolve()),
        "--triangulation-root", str(args.triangulation_root.resolve()),
        "--sam-prior-root", str(args.sam_prior_root.resolve()),
        "--sam-mode-c-review-root", str(args.sam_mode_c_review_root.resolve()),
        "--body-fit-root", str(args.body_fit_root.resolve()),
        "--output-root", str(args.output_root.resolve()),
        "--build-id", args.build_id,
        "--sequences", ",".join(args.sequences),
    ]


def read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--sam-mode-c-review-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-state", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--deadline-utc", default="2026-08-14T04:00:00+00:00")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None or args.poll_seconds <= 0:
        raise RuntimeError("deadline must be timezone-aware and poll positive")
    deadline = deadline.astimezone(timezone.utc)
    manifest_path = args.output_root.resolve() / args.build_id / "dataset_manifest.json"
    while True:
        now = datetime.now(timezone.utc)
        manifest = read_manifest(manifest_path)
        if manifest is not None:
            atomic_json(
                args.runtime_state.resolve(),
                {
                    "schema_version": 1,
                    "status": "COMPLETE",
                    "updated_at_utc": now.isoformat(),
                    "deadline_utc": deadline.isoformat(),
                    "build_id": args.build_id,
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
                    "schema_version": 1,
                    "status": "WAITING_DEADLINE",
                    "updated_at_utc": now.isoformat(),
                    "deadline_utc": deadline.isoformat(),
                    "remaining_wall_hours": (deadline - now).total_seconds() / 3600,
                    "build_id": args.build_id,
                    "command": " ".join(sys.argv),
                    "success_condition": "versioned manifest exists; PASS/REVIEW/FAIL/INCOMPLETE retained",
                    "next_stage": "continue autonomous generation after snapshot",
                },
            )
            time.sleep(args.poll_seconds)
            continue
        command = export_command(args)
        atomic_json(
            args.runtime_state.resolve(),
            {
                "schema_version": 1,
                "status": "EXPORTING_DEADLINE_SNAPSHOT",
                "updated_at_utc": now.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "build_id": args.build_id,
                "command": command,
            },
        )
        process = subprocess.run(command, cwd=PROJECT_ROOT)
        manifest = read_manifest(manifest_path)
        atomic_json(
            args.runtime_state.resolve(),
            {
                "schema_version": 1,
                "status": "COMPLETE" if manifest is not None else "EXPORT_FAILED",
                "updated_at_utc": utc_now(),
                "deadline_utc": deadline.isoformat(),
                "build_id": args.build_id,
                "export_exit_code": process.returncode,
                "manifest": str(manifest_path),
                "freeze_eligible": manifest.get("freeze_eligible") if manifest else False,
                "incomplete_count": manifest.get("incomplete_count") if manifest else None,
            },
        )
        return 0 if manifest is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
