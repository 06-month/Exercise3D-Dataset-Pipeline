#!/usr/bin/env python3
"""Freeze every newly enlarged, export-ready Exercise3D sequence set.

This CPU-only follower consumes the existing quality follower readiness state.
It never launches inference and never rewrites an immutable checkpoint.  A new
deterministic build is exported only when the ready set is a strict superset of
the largest byte-verified checkpoint and the fixed deadline has not passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.export_private_dataset import fsync_directory, verify_frozen_build
    from tools.run_autonomous_supervisor_watchdog import acquire_singleton_lock
except ModuleNotFoundError:
    from export_private_dataset import fsync_directory, verify_frozen_build
    from run_autonomous_supervisor_watchdog import acquire_singleton_lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    fsync_directory(path.parent)


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("expected a non-empty unique sequence list")
    return result


def readiness_sequences(
    state: dict[str, Any], frozen_order: list[str]
) -> tuple[list[str] | None, list[str]]:
    readiness = state.get("freeze_readiness", {})
    ready = readiness.get("ready", []) if isinstance(readiness, dict) else []
    errors: list[str] = []
    if not isinstance(ready, list):
        return None, ["freeze_readiness.ready is not a list"]
    rows: dict[str, str] = {}
    for row in ready:
        if not isinstance(row, dict):
            errors.append("freeze readiness contains a non-object row")
            continue
        sequence = str(row.get("sequence", ""))
        status = str(row.get("status", ""))
        if not sequence or sequence not in frozen_order:
            errors.append(f"unknown freeze-ready sequence:{sequence}")
        elif sequence in rows:
            errors.append(f"duplicate freeze-ready sequence:{sequence}")
        elif status not in {"PASS", "REVIEW"}:
            errors.append(f"non-exportable freeze-ready status:{sequence}:{status}")
        else:
            rows[sequence] = status
    declared = readiness.get("ready_sequence_count") if isinstance(readiness, dict) else None
    try:
        if int(declared) != len(rows):
            errors.append(
                f"freeze-ready count mismatch:declared={declared}:actual={len(rows)}"
            )
    except (TypeError, ValueError):
        errors.append("freeze-ready count is invalid")
    failures = readiness.get("failures", []) if isinstance(readiness, dict) else []
    if failures:
        errors.append("freeze readiness contains failures")
    if errors:
        return None, errors
    return [sequence for sequence in frozen_order if sequence in rows], []


def checkpoint_build_id(prefix: str, sequences: list[str]) -> str:
    digest = hashlib.sha256(
        json.dumps(sequences, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    return f"{prefix}-{len(sequences):03d}-{digest}"


def quick_checkpoint_candidates(
    output_root: Path, frozen_order: list[str]
) -> list[tuple[int, float, Path, list[str]]]:
    candidates: list[tuple[int, float, Path, list[str]]] = []
    order_index = {sequence: index for index, sequence in enumerate(frozen_order)}
    for manifest_path in output_root.glob("*/dataset_manifest.json"):
        manifest = read_json(manifest_path)
        requested = manifest.get("requested_sequences")
        if not isinstance(requested, list) or not requested:
            continue
        sequences = [str(sequence) for sequence in requested]
        if len(set(sequences)) != len(sequences):
            continue
        if any(sequence not in order_index for sequence in sequences):
            continue
        if sequences != sorted(sequences, key=order_index.__getitem__):
            continue
        try:
            consistent = bool(
                manifest.get("freeze_contract_version") == 2
                and int(manifest.get("sequence_count", -1)) == len(sequences)
                and int(manifest.get("fail_count", -1)) == 0
                and int(manifest.get("incomplete_count", -1)) == 0
                and int(manifest.get("pass_count", -1))
                + int(manifest.get("review_count", -1))
                == len(sequences)
                and bool(manifest.get("freeze_eligible"))
            )
            modified = manifest_path.stat().st_mtime
        except (OSError, TypeError, ValueError):
            continue
        if consistent:
            candidates.append((len(sequences), modified, manifest_path.parent, sequences))
    return sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True)


def largest_verified_checkpoint(
    output_root: Path, frozen_order: list[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    invalid_larger_candidates: list[str] = []
    for count, _, directory, sequences in quick_checkpoint_candidates(
        output_root, frozen_order
    ):
        try:
            result = verify_frozen_build(directory, directory.name, sequences)
        except (OSError, RuntimeError, ValueError) as error:
            invalid_larger_candidates.append(
                f"{directory.name}:{type(error).__name__}:{error}"
            )
            continue
        if result["valid"]:
            manifest = result["manifest"]
            return (
                {
                    "build_id": directory.name,
                    "sequences": sequences,
                    "completed_sequence_count": count,
                    "file_count": int(manifest.get("file_count", 0) or 0),
                    "total_payload_bytes": int(
                        manifest.get("total_payload_bytes", 0) or 0
                    ),
                    "verified_file_count": int(result["verified_file_count"]),
                    "verified_payload_bytes": int(result["verified_payload_bytes"]),
                    "freeze_eligible": bool(manifest.get("freeze_eligible")),
                    "integrity_verified": True,
                },
                invalid_larger_candidates,
            )
        invalid_larger_candidates.append(
            f"{directory.name}:" + ";".join(result.get("errors", []))
        )
    return None, invalid_larger_candidates


def checkpoint_action(
    ready: list[str], best: dict[str, Any] | None, minimum_increment: int
) -> tuple[str, str | None]:
    prior = list(best.get("sequences", [])) if best else []
    if len(ready) < len(prior):
        return "WAIT", "readiness count is below the durable checkpoint"
    if len(ready) == len(prior):
        if ready != prior:
            return "ATTENTION", "readiness set changed without increasing count"
        return "WAIT", None
    if not set(prior).issubset(ready):
        return "ATTENTION", "new readiness set is not a superset of durable checkpoint"
    if len(ready) - len(prior) < minimum_increment:
        return "WAIT", None
    return "EXPORT", None


def export_command(args: argparse.Namespace, build_id: str, sequences: list[str]) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "export_private_dataset.py"),
        "--dataset-root",
        str(args.dataset_root.resolve()),
        "--selection-root",
        str(args.selection_root.resolve()),
        "--pose-root",
        str(args.pose_root.resolve()),
        "--triangulation-root",
        str(args.triangulation_root.resolve()),
        "--sam-prior-root",
        str(args.sam_prior_root.resolve()),
        "--sam-mode-c-review-root",
        str(args.sam_mode_c_review_root.resolve()),
        "--body-fit-root",
        str(args.body_fit_root.resolve()),
        "--quality-root",
        str(args.quality_root.resolve()),
        "--output-root",
        str(args.output_root.resolve()),
        "--build-id",
        build_id,
        "--sequences",
        ",".join(sequences),
    ]


def base_state(args: argparse.Namespace, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at_utc": utc_now().isoformat(),
        "pid": os.getpid(),
        "cwd": str(PROJECT_ROOT),
        "command": [sys.executable, *sys.argv],
        "gpu_work": False,
        "deadline_utc": args.deadline_utc,
        "frozen_sequence_count": len(args.sequences),
        "minimum_increment": args.minimum_increment,
        "build_prefix": args.build_prefix,
        "policy": (
            "export only a strict readiness superset before deadline; deterministic immutable "
            "build ID; exact verifier; never launch inference"
        ),
        **extra,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--sam-mode-c-review-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument("--quality-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--quality-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "quality_follower_state.json",
    )
    parser.add_argument(
        "--runtime-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "predeadline_checkpoint_follower_state.json",
    )
    parser.add_argument(
        "--instance-lock",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "predeadline_checkpoint_follower.lock",
    )
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--build-prefix", default="exercise3d-predeadline-auto")
    parser.add_argument("--minimum-increment", type=int, default=1)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retry-seconds", type=float, default=300.0)
    parser.add_argument("--deadline-utc", default="2026-08-14T04:00:00+00:00")
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    deadline = datetime.fromisoformat(args.deadline_utc)
    if (
        deadline.tzinfo is None
        or args.minimum_increment <= 0
        or args.minimum_free_gib <= 0
        or args.poll_seconds <= 0
        or args.retry_seconds <= 0
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.build_prefix) is None
    ):
        raise RuntimeError("invalid follower configuration")
    deadline = deadline.astimezone(timezone.utc)
    singleton = acquire_singleton_lock(args.instance_lock.resolve())
    if singleton is None:
        print("predeadline checkpoint follower is already running", flush=True)
        return 3

    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    best, checkpoint_integrity_errors = largest_verified_checkpoint(
        args.output_root.resolve(), args.sequences
    )
    retry_after = 0.0
    retry_attention: dict[str, str] | None = None
    integrity_retry_after = (
        time.monotonic() + args.retry_seconds if checkpoint_integrity_errors else 0.0
    )
    while True:
        now = utc_now()
        if checkpoint_integrity_errors and time.monotonic() >= integrity_retry_after:
            best, checkpoint_integrity_errors = largest_verified_checkpoint(
                args.output_root.resolve(), args.sequences
            )
            integrity_retry_after = (
                time.monotonic() + args.retry_seconds
                if checkpoint_integrity_errors
                else 0.0
            )
        ready, readiness_errors = readiness_sequences(
            read_json(args.quality_state.resolve()), args.sequences
        )
        attention: list[dict[str, str]] = []
        last_event = "WAITING_FOR_NEW_FREEZE_READY_SEQUENCE"
        status = "RUNNING"
        attempted_build_id: str | None = None
        export_exit_code: int | None = None
        export_output_tail: str | None = None
        if now >= deadline:
            status = "COMPLETE"
            last_event = "DEADLINE_REACHED_NO_MORE_CHECKPOINTS"
        elif checkpoint_integrity_errors:
            status = "ATTENTION"
            last_event = "EXISTING_CHECKPOINT_INTEGRITY_FAILED"
            attention = [
                {
                    "code": "EXISTING_CHECKPOINT_INTEGRITY_FAILED",
                    "message": error,
                }
                for error in checkpoint_integrity_errors
            ]
        elif readiness_errors:
            status = "ATTENTION"
            last_event = "READINESS_STATE_INVALID"
            attention = [
                {"code": "CHECKPOINT_READINESS_INVALID", "message": error}
                for error in readiness_errors
            ]
        elif ready is not None:
            action, reason = checkpoint_action(ready, best, args.minimum_increment)
            if action == "ATTENTION":
                status = "ATTENTION"
                last_event = "CHECKPOINT_SET_REGRESSION"
                attention = [
                    {"code": "CHECKPOINT_SET_REGRESSION", "message": str(reason)}
                ]
            elif action == "EXPORT" and time.monotonic() >= retry_after:
                free_gib = shutil.disk_usage(args.output_root.resolve()).free / (1024**3)
                if free_gib < args.minimum_free_gib:
                    status = "ATTENTION"
                    last_event = "CHECKPOINT_DISK_RESERVE_LOW"
                    attention = [
                        {
                            "code": "CHECKPOINT_DISK_RESERVE_LOW",
                            "message": (
                                f"free disk {free_gib:.2f} GiB is below "
                                f"{args.minimum_free_gib:.2f} GiB"
                            ),
                        }
                    ]
                else:
                    attempted_build_id = checkpoint_build_id(args.build_prefix, ready)
                    command = export_command(args, attempted_build_id, ready)
                    atomic_json(
                        args.runtime_state.resolve(),
                        base_state(
                            args,
                            status="EXPORTING",
                            last_event="EXPORTING_NEW_CHECKPOINT",
                            ready_sequences=ready,
                            best_checkpoint=best,
                            attempted_build_id=attempted_build_id,
                            export_command=command,
                            attention_required=False,
                            attention_reasons=[],
                        ),
                    )
                    try:
                        process = subprocess.run(
                            command,
                            cwd=PROJECT_ROOT,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                        )
                        export_exit_code = process.returncode
                        export_output_tail = process.stdout[-4000:]
                        candidate = args.output_root.resolve() / attempted_build_id
                        verification = verify_frozen_build(
                            candidate, attempted_build_id, ready
                        )
                    except (OSError, RuntimeError, ValueError) as error:
                        export_exit_code = 1
                        export_output_tail = f"{type(error).__name__}: {error}"
                        verification = {
                            "valid": False,
                            "manifest": None,
                            "errors": [export_output_tail],
                        }
                    if verification["valid"] and bool(
                        verification["manifest"].get("freeze_eligible")
                    ):
                        manifest = verification["manifest"]
                        best = {
                            "build_id": attempted_build_id,
                            "sequences": ready,
                            "completed_sequence_count": len(ready),
                            "file_count": int(manifest.get("file_count", 0) or 0),
                            "total_payload_bytes": int(
                                manifest.get("total_payload_bytes", 0) or 0
                            ),
                            "verified_file_count": int(
                                verification["verified_file_count"]
                            ),
                            "verified_payload_bytes": int(
                                verification["verified_payload_bytes"]
                            ),
                            "freeze_eligible": True,
                            "integrity_verified": True,
                        }
                        last_event = "CHECKPOINT_PUBLISHED_AND_VERIFIED"
                        retry_after = 0.0
                        retry_attention = None
                    elif export_exit_code == 75:
                        # The exporter uses 75 for a held build lock.  This is a
                        # recoverable coordination event, not a validation failure.
                        status = "RUNNING"
                        last_event = "CHECKPOINT_EXPORT_LOCK_RETRY_WAIT"
                        retry_after = time.monotonic() + args.retry_seconds
                        retry_attention = None
                    else:
                        status = "ATTENTION"
                        last_event = "CHECKPOINT_EXPORT_FAILED"
                        retry_after = time.monotonic() + args.retry_seconds
                        retry_attention = {
                            "code": "CHECKPOINT_EXPORT_FAILED",
                            "message": (
                                f"export exit={export_exit_code}; integrity="
                                + ";".join(verification.get("errors", []))
                            ),
                        }
                        attention = [retry_attention]
            elif action == "EXPORT":
                if retry_attention is not None:
                    status = "ATTENTION"
                    last_event = "CHECKPOINT_FAILURE_RETRY_WAIT"
                    attention = [retry_attention]
                else:
                    last_event = "CHECKPOINT_RETRY_WAIT"

        state = base_state(
            args,
            status=status,
            last_event=last_event,
            ready_sequence_count=len(ready or []),
            ready_sequences=ready or [],
            best_checkpoint=best,
            attempted_build_id=attempted_build_id,
            export_exit_code=export_exit_code,
            export_output_tail=export_output_tail,
            retry_in_seconds=max(0.0, retry_after - time.monotonic()),
            integrity_retry_in_seconds=max(
                0.0, integrity_retry_after - time.monotonic()
            ),
            attention_required=bool(attention),
            attention_reasons=attention,
        )
        atomic_json(args.runtime_state.resolve(), state)
        if args.once or status == "COMPLETE":
            return int(bool(attention))
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
