#!/usr/bin/env python3
"""Materialize Phase 11 quality for newly completed sequences without GPU work."""

from __future__ import annotations

import argparse
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
        summarize_quality_outputs,
        validate_quality_output,
    )
except ModuleNotFoundError:
    from build_pseudolabel_quality import (
        atomic_json,
        build_sequence_quality,
        summarize_quality_outputs,
        validate_quality_output,
    )


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


def validate_existing(args: argparse.Namespace, sequence: str) -> tuple[bool, str]:
    output = args.output_root.resolve() / sequence
    body_path = args.body_fit_root.resolve() / sequence / "body_fit.npz"
    valid, reasons, metadata = validate_quality_output(output, body_path)
    if not valid or metadata is None:
        return False, ";".join(reasons) or "invalid quality output"
    status = str(metadata.get("qa", {}).get("sequence_status", "UNKNOWN"))
    if status not in {"PASS", "REVIEW"}:
        return False, f"quality sequence status is {status}"
    return True, status


def run_cycle(
    args: argparse.Namespace,
    completed: dict[str, str],
    retry_state: dict[str, dict[str, Any]],
    *,
    monotonic_now: float | None = None,
) -> dict[str, Any]:
    now = time.monotonic() if monotonic_now is None else monotonic_now
    waiting: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    newly_validated: list[str] = []
    materialized: list[str] = []
    builder_args = quality_args(args)

    for sequence in args.sequences:
        vector = args.output_root.resolve() / sequence / "quality_vector.npz"
        metadata = args.output_root.resolve() / sequence / "metadata.json"
        if sequence in completed and vector.is_file() and metadata.is_file():
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
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
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

    completed_rows = [
        {"sequence": sequence, "status": completed[sequence]}
        for sequence in args.sequences
        if sequence in completed
    ]
    if failures:
        status = "ATTENTION"
    elif len(completed_rows) == len(args.sequences):
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
        "resume_policy": "validate existing quality; build only complete dependencies; retry failures",
        "completed_sequence_count": len(completed_rows),
        "total_sequence_count": len(args.sequences),
        "completed": completed_rows,
        "newly_validated": newly_validated,
        "materialized": materialized,
        "waiting": waiting,
        "failures": failures,
        "last_event": (
            f"QUALITY_MATERIALIZED:{materialized[-1]}"
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
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--retry-seconds", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0 or args.retry_seconds <= 0:
        raise RuntimeError("poll-seconds and retry-seconds must be positive")
    completed: dict[str, str] = {}
    retry_state: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {}
    try:
        while True:
            state = run_cycle(args, completed, retry_state)
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
