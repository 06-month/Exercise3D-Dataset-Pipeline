#!/usr/bin/env python3
"""Materialize Phase 11 quality for newly completed sequences without GPU work."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.build_pseudolabel_quality import (
        atomic_json,
        build_sequence_quality,
        quality_dependency_signature,
        summarize_quality_outputs,
        validate_quality_output,
    )
except ModuleNotFoundError:
    from build_pseudolabel_quality import (
        atomic_json,
        build_sequence_quality,
        quality_dependency_signature,
        summarize_quality_outputs,
        validate_quality_output,
    )

try:
    from tools.export_private_dataset import (
        sequence_dependencies as export_sequence_dependencies,
        validate_sequence as validate_export_sequence,
    )
except ModuleNotFoundError:
    from export_private_dataset import (
        sequence_dependencies as export_sequence_dependencies,
        validate_sequence as validate_export_sequence,
    )

try:
    from tools.run_autonomous_supervisor_watchdog import acquire_singleton_lock
except ModuleNotFoundError:
    from run_autonomous_supervisor_watchdog import acquire_singleton_lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("cam1", "cam2", "cam3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty sequence list")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("sequence list contains duplicates")
    return result


def dependency_paths(args: argparse.Namespace, sequence: str) -> dict[str, Path]:
    paths = {
        "triangulation/triangulated_3d.npz": (
            args.triangulation_root.resolve() / sequence / "triangulated_3d.npz"
        ),
        "triangulation/canonical_3d.npz": (
            args.triangulation_root.resolve() / sequence / "canonical_3d.npz"
        ),
        "triangulation/metadata.json": (
            args.triangulation_root.resolve() / sequence / "metadata.json"
        ),
        "body/body_fit.npz": args.body_fit_root.resolve() / sequence / "body_fit.npz",
        "body/metadata.json": args.body_fit_root.resolve() / sequence / "metadata.json",
        "body/mode_c_escalation.json": (
            args.sam_mode_c_review_root.resolve()
            / sequence
            / "mode_c_escalation.json"
        ),
    }
    for camera in CAMERAS:
        paths[f"selection/{camera}"] = (
            args.selection_root.resolve()
            / sequence
            / camera
            / "target_selection.npz"
        )
        paths[f"pose/{camera}"] = (
            args.pose_root.resolve() / sequence / camera / "poses_2d.npz"
        )
        paths[f"sam_prior/{camera}"] = (
            args.sam_prior_root.resolve()
            / sequence
            / camera
            / "sam_body_prior.npz"
        )
    return paths


def missing_dependencies(args: argparse.Namespace, sequence: str) -> list[str]:
    return [
        label
        for label, path in dependency_paths(args, sequence).items()
        if not path.is_file()
    ]


def quality_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        selection_root=args.selection_root,
        pose_root=args.pose_root,
        triangulation_root=args.triangulation_root,
        sam_prior_root=args.sam_prior_root,
        sam_mode_c_review_root=args.sam_mode_c_review_root,
        body_fit_root=args.body_fit_root,
        output_root=args.output_root,
    )


def export_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        selection_root=args.selection_root,
        pose_root=args.pose_root,
        triangulation_root=args.triangulation_root,
        sam_prior_root=args.sam_prior_root,
        sam_mode_c_review_root=args.sam_mode_c_review_root,
        body_fit_root=args.body_fit_root,
        quality_root=args.output_root,
    )


def assess_freeze_readiness(
    args: argparse.Namespace, sequence: str
) -> dict[str, Any]:
    result = validate_export_sequence(export_args(args), sequence)
    status = str(result.get("status", "UNKNOWN"))
    if status not in {"PASS", "REVIEW", "FAIL", "INCOMPLETE"}:
        raise RuntimeError(f"unexpected export-readiness status {status}")
    reasons = result.get("reasons", [])
    if not isinstance(reasons, list):
        raise RuntimeError("export-readiness reasons must be a list")
    return {
        "sequence": sequence,
        "status": status,
        "reasons": [str(reason) for reason in reasons],
        "reference_frame_count": int(result.get("reference_frame_count", 0) or 0),
        "body_fit_status": str(result.get("body_fit_status", "")),
        "camera_geometry_status": str(result.get("camera_geometry_status", "")),
        "sam_mode_c_review_status": str(
            result.get("sam_mode_c_review_status", "")
        ),
        "dependency_signature": export_dependency_signature(args, sequence),
    }


def export_dependency_signature(
    args: argparse.Namespace, sequence: str
) -> str | None:
    rows: list[tuple[str, int, int]] = []
    for label, path in sorted(
        export_sequence_dependencies(export_args(args), sequence).items()
    ):
        try:
            stat = path.stat()
        except OSError:
            return None
        if not path.is_file() or path.is_symlink():
            return None
        rows.append((label, stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_existing(args: argparse.Namespace, sequence: str) -> tuple[bool, str]:
    output = args.output_root.resolve() / sequence
    body_path = args.body_fit_root.resolve() / sequence / "body_fit.npz"
    signature = quality_dependency_signature(quality_args(args), sequence)
    valid, reasons, metadata = validate_quality_output(output, body_path, signature)
    if not valid or metadata is None:
        return False, ";".join(reasons) or "invalid quality output"
    status = str(metadata.get("qa", {}).get("sequence_status", "UNKNOWN"))
    if status not in {"PASS", "REVIEW"}:
        return False, f"quality sequence status is {status}"
    return True, status


def completed_quality_still_current(
    args: argparse.Namespace, sequence: str
) -> bool:
    """Fast source check for a persisted completion.

    Unsigned outputs already recorded complete before source-bound resume was
    introduced remain grandfathered.  Newly signed outputs must continue to
    match every input before the follower skips their full validation.
    """

    output = args.output_root.resolve() / sequence
    vector = output / "quality_vector.npz"
    metadata_path = output / "metadata.json"
    if not vector.is_file() or vector.is_symlink():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(metadata, dict):
        return False
    signature = metadata.get("source_dependency_signature")
    if signature is None:
        return True
    try:
        return signature == quality_dependency_signature(
            quality_args(args), sequence
        )
    except RuntimeError:
        return False


def run_cycle(
    args: argparse.Namespace,
    completed: dict[str, str],
    retry_state: dict[str, dict[str, Any]],
    freeze_ready: dict[str, dict[str, Any]] | None = None,
    readiness_state: dict[str, dict[str, Any]] | None = None,
    *,
    monotonic_now: float | None = None,
) -> dict[str, Any]:
    now = time.monotonic() if monotonic_now is None else monotonic_now
    freeze_ready = {} if freeze_ready is None else freeze_ready
    readiness_state = {} if readiness_state is None else readiness_state
    waiting: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    newly_validated: list[str] = []
    materialized: list[str] = []
    newly_freeze_ready: list[str] = []
    builder_args = quality_args(args)

    for sequence in args.sequences:
        if sequence in completed and completed_quality_still_current(args, sequence):
            continue
        completed.pop(sequence, None)

        missing = missing_dependencies(args, sequence)
        if missing:
            waiting.append({"sequence": sequence, "missing": missing})
            continue
        prior_failure = retry_state.get(sequence, {})
        retry_after = float(prior_failure.get("not_before_monotonic", 0.0))
        if retry_after > now:
            failures.append(
                {
                    "sequence": sequence,
                    "reason": str(prior_failure.get("reason", "prior failure")),
                    "retry_in_seconds": max(0.0, retry_after - now),
                }
            )
            continue

        try:
            built = False
            valid, status = validate_existing(args, sequence)
            if not valid:
                result = build_sequence_quality(builder_args, sequence)
                built = not bool(result.get("resume_skipped", False))
                valid, status = validate_existing(args, sequence)
                if not valid:
                    raise RuntimeError(status)
                result_status = str(result.get("qa", {}).get("sequence_status", status))
                if result_status not in {"PASS", "REVIEW"}:
                    raise RuntimeError(f"quality build returned status {result_status}")
            completed[sequence] = status
            retry_state.pop(sequence, None)
            newly_validated.append(sequence)
            if built:
                materialized.append(sequence)
        except (
            OSError,
            EOFError,
            ValueError,
            KeyError,
            RuntimeError,
            json.JSONDecodeError,
        ) as error:
            reason = f"{type(error).__name__}: {error}"
            retry_state[sequence] = {
                "not_before_monotonic": now + args.retry_seconds,
                "reason": reason,
            }
            failures.append(
                {
                    "sequence": sequence,
                    "reason": reason,
                    "retry_in_seconds": args.retry_seconds,
                }
            )

    if newly_validated:
        summary = summarize_quality_outputs(args.output_root.resolve())
        atomic_json(args.output_root.resolve() / "quality_summary.json", summary)

    readiness_waiting: list[dict[str, Any]] = []
    readiness_failures: list[dict[str, Any]] = []
    for sequence in args.sequences:
        if sequence not in completed:
            freeze_ready.pop(sequence, None)
            readiness_state.pop(sequence, None)
            continue
        if sequence in freeze_ready:
            current_signature = export_dependency_signature(args, sequence)
            unchanged = (
                current_signature is not None
                and current_signature
                == freeze_ready[sequence].get("dependency_signature")
            )
            if unchanged:
                continue
            freeze_ready.pop(sequence, None)
        prior = readiness_state.get(sequence, {})
        next_attempt = float(prior.get("next_attempt_monotonic", 0.0))
        if next_attempt > now:
            row = {
                "sequence": sequence,
                "status": str(prior.get("status", "UNKNOWN")),
                "reasons": list(prior.get("reasons", [])),
                "age_seconds": max(
                    0.0, now - float(prior.get("first_seen_monotonic", now))
                ),
                "retry_in_seconds": next_attempt - now,
            }
        else:
            try:
                row = assess_freeze_readiness(args, sequence)
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                json.JSONDecodeError,
            ) as error:
                row = {
                    "sequence": sequence,
                    "status": "FAIL",
                    "reasons": [f"{type(error).__name__}: {error}"],
                    "reference_frame_count": 0,
                }
            if row["status"] in {"PASS", "REVIEW"}:
                freeze_ready[sequence] = row
                readiness_state.pop(sequence, None)
                newly_freeze_ready.append(sequence)
                continue
            first_seen = float(prior.get("first_seen_monotonic", now))
            row["age_seconds"] = max(0.0, now - first_seen)
            row["retry_in_seconds"] = args.retry_seconds
            readiness_state[sequence] = {
                "status": row["status"],
                "reasons": row["reasons"],
                "first_seen_monotonic": first_seen,
                "next_attempt_monotonic": now + args.retry_seconds,
            }
        if row["age_seconds"] >= args.readiness_grace_seconds or row["status"] == "FAIL":
            readiness_failures.append(row)
        else:
            readiness_waiting.append(row)

    completed_rows = [
        {"sequence": sequence, "status": completed[sequence]}
        for sequence in args.sequences
        if sequence in completed
    ]
    freeze_ready_rows = [
        freeze_ready[sequence]
        for sequence in args.sequences
        if sequence in freeze_ready
    ]
    freeze_status_counts = {
        status_name: sum(row["status"] == status_name for row in freeze_ready_rows)
        for status_name in ("PASS", "REVIEW")
    }
    if failures or readiness_failures:
        status = "ATTENTION"
    elif len(freeze_ready_rows) == len(args.sequences):
        status = "COMPLETE"
    else:
        status = "RUNNING"
    return {
        "schema_version": 1,
        "stage": "PHASE11_QUALITY_FOLLOWER",
        "status": status,
        "updated_at_utc": utc_now(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "command": [sys.executable, *sys.argv],
        "gpu_work": False,
        "resume_policy": (
            "validate existing quality; build only complete dependencies; validate export "
            "readiness; retry failures without recomputing valid output"
        ),
        "completed_sequence_count": len(completed_rows),
        "total_sequence_count": len(args.sequences),
        "completed": completed_rows,
        "newly_validated": newly_validated,
        "materialized": materialized,
        "waiting": waiting,
        "failures": failures,
        "freeze_readiness": {
            "ready_sequence_count": len(freeze_ready_rows),
            "total_sequence_count": len(args.sequences),
            "status_counts": freeze_status_counts,
            "ready": freeze_ready_rows,
            "newly_ready": newly_freeze_ready,
            "waiting": readiness_waiting,
            "failures": readiness_failures,
            "grace_seconds": args.readiness_grace_seconds,
        },
        "last_event": (
            f"FREEZE_READY:{newly_freeze_ready[-1]}"
            if newly_freeze_ready
            else f"QUALITY_MATERIALIZED:{materialized[-1]}"
            if materialized
            else f"QUALITY_VALIDATED:{newly_validated[-1]}"
            if newly_validated
            else "WAITING_FOR_BODY_FIT"
            if waiting
            else "ALL_QUALITY_COMPLETE"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "target_selection_full",
    )
    parser.add_argument(
        "--pose-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sapiens2_target_only_full",
    )
    parser.add_argument(
        "--triangulation-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "triangulation_final",
    )
    parser.add_argument(
        "--sam-prior-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sam_body_prior_full",
    )
    parser.add_argument(
        "--sam-mode-c-review-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sam_mode_c_review_full",
    )
    parser.add_argument(
        "--body-fit-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "body_fit_full",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "quality_control_full",
    )
    parser.add_argument(
        "--runtime-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "quality_follower_state.json",
    )
    parser.add_argument(
        "--instance-lock",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "quality_follower.lock",
    )
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retry-seconds", type=float, default=300.0)
    parser.add_argument("--readiness-grace-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.poll_seconds <= 0
        or args.retry_seconds <= 0
        or args.readiness_grace_seconds < 0
    ):
        raise RuntimeError(
            "poll-seconds/retry-seconds must be positive and readiness grace nonnegative"
        )
    singleton = acquire_singleton_lock(args.instance_lock.resolve())
    if singleton is None:
        print("quality control follower is already running", flush=True)
        return 3
    completed: dict[str, str] = {}
    retry_state: dict[str, dict[str, Any]] = {}
    freeze_ready: dict[str, dict[str, Any]] = {}
    readiness_state: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {}
    try:
        while True:
            state = run_cycle(
                args,
                completed,
                retry_state,
                freeze_ready,
                readiness_state,
            )
            atomic_json(args.runtime_state.resolve(), state)
            if args.once or state["status"] == "COMPLETE":
                return 0 if state["status"] != "ATTENTION" else 2
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        stopped = {
            **state,
            "status": "STOPPED",
            "updated_at_utc": utc_now(),
            "last_event": "STOPPED_BY_OPERATOR",
        }
        atomic_json(args.runtime_state.resolve(), stopped)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
