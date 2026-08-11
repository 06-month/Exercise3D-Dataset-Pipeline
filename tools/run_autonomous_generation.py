#!/usr/bin/env python3
"""Supervise the remaining private Exercise3D generation critical path.

The supervisor watches an already-running Sapiens2 process and immediately
streams every pose-complete sequence through Phase 7, SAM Mode B, prior
consolidation and sequence fitting.  This overlaps the low-duty-cycle SAM
pipeline with Sapiens while retaining sequence-level resume.  If Sapiens exits
before all selection-bound camera outputs are complete, the same frozen
configuration is resumed.  A failure is recorded per sequence and does not
silently promote incomplete data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.consolidate_sam_body_prior import REQUIRED_PRIOR_FIELDS
    from tools.run_phase7_streaming import sequence_ready
except ModuleNotFoundError:
    from consolidate_sam_body_prior import REQUIRED_PRIOR_FIELDS
    from run_phase7_streaming import sequence_ready


CAMERAS = "cam1,cam2,cam3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--initial-triangulation-root", type=Path, required=True)
    parser.add_argument("--recovery-camera-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-output-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--sam-mode-c-review-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--wait-sapiens-pid", type=int)
    parser.add_argument("--sapiens-python", type=Path, required=True)
    parser.add_argument("--body4d-python", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--sam-body4d-root", type=Path, required=True)
    parser.add_argument("--sapiens-retries", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--deadline-utc", default="2026-08-14T04:00:00+00:00")
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--expected-target-crops", type=int, default=65430)
    parser.add_argument("--reused-target-crops", type=int, default=9725)
    parser.add_argument("--expected-sam-hours", type=float, default=20.8)
    return parser


def process_alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()[2]
        return state != "Z"
    except (OSError, IndexError):
        return False


def free_gib(path: Path) -> float:
    existing = path.resolve()
    while not existing.exists():
        existing = existing.parent
    return shutil.disk_usage(existing).free / (1024**3)


def sam_smoke_complete(output_dir: Path, expected_frames: int) -> bool:
    private = output_dir / "mode_b_private_output"
    profile_path = output_dir / "mode_b_profile.json"
    benchmark_path = output_dir / "sam_body_benchmark.csv"
    provenance_path = private / "target_provenance.npz"
    numeric = sorted((private / "mhr_numeric" / "1").glob("*.npz"))
    meshes = sorted((private / "mesh_4d_individual" / "1").glob("*.ply"))
    if (
        not profile_path.is_file()
        or not benchmark_path.is_file()
        or not provenance_path.is_file()
        or len(numeric) != expected_frames
        or len(meshes) != expected_frames
    ):
        return False
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        with benchmark_path.open(newline="", encoding="utf-8") as handle:
            benchmark = list(csv.DictReader(handle))
        with np.load(provenance_path, allow_pickle=False) as provenance:
            provenance_ok = (
                len(provenance["frame_names"]) == expected_frames
                and len(provenance["timestamp_pts_seconds"]) == expected_frames
            )
        with np.load(numeric[0], allow_pickle=False) as payload:
            numeric_ok = set(REQUIRED_PRIOR_FIELDS) <= set(payload.files)
        return bool(
            provenance_ok
            and numeric_ok
            and len(benchmark) == 1
            and benchmark[0]["status"] == "PASS"
            and int(profile["frames_processed"]) == expected_frames
            and int(profile["target_seed_count"]) == 1
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def sapiens_progress(args: argparse.Namespace) -> dict[str, Any]:
    completed_crops = 0
    complete_cameras = 0
    complete_sequences = 0
    new_camera_events: list[tuple[datetime, int]] = []
    for sequence in args.sequences:
        sequence_complete = True
        for camera in CAMERAS.split(","):
            camera_dir = args.pose_root.resolve() / sequence / camera
            metadata_path = camera_dir / "metadata.json"
            if metadata_path.is_file():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    camera_crops = int(metadata["qa"]["target_pose_count"])
                    completed_crops += camera_crops
                    complete_cameras += 1
                    if metadata.get("pose_inference_performed_in_this_stage"):
                        new_camera_events.append(
                            (datetime.fromisoformat(metadata["created_at_utc"]), camera_crops)
                        )
                    continue
                except (OSError, KeyError, ValueError, json.JSONDecodeError):
                    pass
            sequence_complete = False
            for chunk in sorted((camera_dir / "chunks").glob("*.npz")):
                try:
                    with np.load(chunk, allow_pickle=False) as payload:
                        completed_crops += int(payload["target_present"].sum())
                except (OSError, KeyError, ValueError):
                    continue
        complete_sequences += int(sequence_complete)
    inferred = max(0, completed_crops - args.reused_target_crops)
    monitor_path = (
        args.pose_root.resolve().parent
        / "runtime"
        / "phase6_full_target_inference"
        / "target_only_pilot_gpu_utilization.csv"
    )
    # The standard layout above may not match an injected output root.
    injected_monitor = args.runtime_dir.resolve().parent / "phase6_full_target_inference" / (
        "target_only_pilot_gpu_utilization.csv"
    )
    if injected_monitor.is_file():
        monitor_path = injected_monitor
    started = None
    if monitor_path.is_file():
        try:
            with monitor_path.open(newline="", encoding="utf-8") as handle:
                first = next(csv.DictReader(handle))
            started = datetime.fromisoformat(first["timestamp_utc"])
        except (OSError, StopIteration, KeyError, ValueError):
            started = None
    now = datetime.now(timezone.utc)
    elapsed = (now - started.astimezone(timezone.utc)).total_seconds() if started else 0.0
    rate = inferred / elapsed if inferred > 0 and elapsed > 0 else 0.0
    new_camera_events.sort()
    recent_rate = 0.0
    if len(new_camera_events) >= 2:
        recent_elapsed = (
            new_camera_events[-1][0] - new_camera_events[0][0]
        ).total_seconds()
        recent_crops = sum(event[1] for event in new_camera_events[1:])
        recent_rate = recent_crops / recent_elapsed if recent_elapsed > 0 else 0.0
    remaining = max(0, args.expected_target_crops - completed_crops)
    projection_rate = recent_rate or rate
    if recent_rate and new_camera_events:
        remaining_after_first_new_camera = max(
            0,
            args.expected_target_crops
            - args.reused_target_crops
            - new_camera_events[0][1],
        )
        projected_completion = datetime.fromtimestamp(
            new_camera_events[0][0].timestamp()
            + remaining_after_first_new_camera / recent_rate,
            tz=timezone.utc,
        )
        eta_hours = max(0.0, (projected_completion - now).total_seconds() / 3600.0)
    elif projection_rate > 0:
        eta_hours = remaining / projection_rate / 3600.0
        projected_completion = datetime.fromtimestamp(
            now.timestamp() + eta_hours * 3600, tz=timezone.utc
        )
    else:
        eta_hours = None
        projected_completion = None
    return {
        "expected_target_crops": args.expected_target_crops,
        "completed_target_crops": completed_crops,
        "remaining_target_crops": remaining,
        "reused_target_crops": args.reused_target_crops,
        "new_inferred_crops": inferred,
        "effective_new_crops_per_second": rate,
        "recent_completed_camera_crops_per_second": recent_rate or None,
        "eta_projection_crops_per_second": projection_rate or None,
        "complete_camera_count": complete_cameras,
        "complete_sequence_count": complete_sequences,
        "elapsed_hours": elapsed / 3600.0,
        "estimated_remaining_sapiens_hours": eta_hours,
        "estimated_sapiens_completion_utc": (
            projected_completion.isoformat()
            if projected_completion is not None
            else None
        ),
        "downstream_expected_sam_hours": args.expected_sam_hours,
    }
def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    fields = list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_successful_rows(path: Path) -> list[dict[str, Any]]:
    """Load only durable completed rows; incomplete work is safe to retry."""
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [row for row in rows if row.get("status") in {"PASS", "REVIEW"}]
    except (OSError, csv.Error):
        return []


def missing_pose_sequences(args: argparse.Namespace) -> list[str]:
    return [
        sequence
        for sequence in args.sequences
        if not sequence_ready(
            args.dataset_root.resolve(), args.pose_root.resolve(), sequence
        )
    ]


def update_state(
    args: argparse.Namespace,
    stage: str,
    rows: list[dict[str, Any]],
    **extra: Any,
) -> None:
    deadline = datetime.fromisoformat(args.deadline_utc)
    if deadline.tzinfo is None:
        raise RuntimeError("deadline must include a timezone")
    now = datetime.now(timezone.utc)
    state = {
        "schema_version": 1,
        "updated_at_utc": now.isoformat(),
        "deadline_utc": deadline.astimezone(timezone.utc).isoformat(),
        "remaining_wall_hours": (deadline.astimezone(timezone.utc) - now).total_seconds()
        / 3600.0,
        "stage": stage,
        "sequence_count": len(args.sequences),
        "completed_body_fit_count": sum(row["status"] in {"PASS", "REVIEW"} for row in rows),
        "failed_or_incomplete_count": sum(
            row["status"] not in {"PASS", "REVIEW"} for row in rows
        ),
        **extra,
    }
    atomic_text(
        args.runtime_dir.resolve() / "autonomous_generation_state.json",
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
    )


def run(command: list[str]) -> int:
    print(json.dumps({"command": command, "started_at_utc": utc_now()}), flush=True)
    process = subprocess.run(command, cwd=PROJECT_ROOT)
    print(
        json.dumps(
            {
                "exit_code": process.returncode,
                "finished_at_utc": utc_now(),
                "executable": command[0],
                "tool": command[1] if len(command) > 1 else "",
            }
        ),
        flush=True,
    )
    return process.returncode


def sapiens_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.sapiens_python.resolve()),
        str(PROJECT_ROOT / "tools" / "sapiens2_target_pipeline.py"),
        "infer",
        "--dataset-root", str(args.dataset_root.resolve()),
        "--selection-root", str(args.selection_root.resolve()),
        "--output-root", str(args.pose_root.resolve()),
        "--runtime-dir", str((args.runtime_dir.resolve() / "phase6_resume")),
        "--sequences", ",".join(args.sequences),
        "--cameras", CAMERAS,
        "--batch-size", "16",
        "--chunk-size", "256",
        "--loader-workers", "8",
        "--prefetch-batches", "4",
        "--retry-failures", "1",
        "--save-overlays", "0",
    ]


def phase7_command(args: argparse.Namespace, sequence: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_phase7_streaming.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--pose-root", str(args.pose_root.resolve()),
        "--camera-root", str(args.camera_root.resolve()),
        "--initial-output-root", str(args.initial_triangulation_root.resolve()),
        "--recovery-camera-root", str(args.recovery_camera_root.resolve()),
        "--final-output-root", str(args.triangulation_root.resolve()),
        "--runtime-dir", str(args.runtime_dir.resolve() / "phase7" / sequence),
        "--sequences", sequence,
        "--poll-seconds", str(args.poll_seconds),
    ]


def sam_command(args: argparse.Namespace, sequence: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_sam_body4d_full.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--selection-root", str(args.selection_root.resolve()),
        "--output-root", str(args.sam_output_root.resolve()),
        "--runtime-dir", str(args.runtime_dir.resolve() / "phase8" / sequence),
        "--checkpoint-root", str(args.checkpoint_root.resolve()),
        "--sam-body4d-root", str(args.sam_body4d_root.resolve()),
        "--body4d-python", str(args.body4d_python.resolve()),
        "--sequences", sequence,
        "--cameras", CAMERAS,
        "--retry-failures", "1",
    ]


def sam_smoke_command(
    args: argparse.Namespace, sequence: str, camera: str, frame_count: int
) -> list[str]:
    exercise = sequence.rsplit("_", 1)[0]
    frames = (
        args.dataset_root.resolve() / "final_frame" / exercise / sequence / camera
    )
    selection = args.selection_root.resolve() / sequence / camera / "target_selection.npz"
    return [
        str(args.body4d_python.resolve()),
        str(PROJECT_ROOT / "tools" / "benchmark_sam_body4d.py"),
        "--mode", "B",
        "--sam-body4d-root", str(args.sam_body4d_root.resolve()),
        "--checkpoint-root", str(args.checkpoint_root.resolve()),
        "--input-frames", str(frames),
        "--target-selection", str(selection),
        "--source-start-index", "0",
        "--frame-count", str(frame_count),
        "--output-dir", str(args.runtime_dir.resolve() / "sam_numeric_smoke"),
        "--python", str(args.body4d_python.resolve()),
        "--run",
    ]


def consolidate_command(args: argparse.Namespace, sequence: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "consolidate_sam_body_prior.py"),
        "--sam-root", str(args.sam_output_root.resolve()),
        "--output-root", str(args.sam_prior_root.resolve()),
        "--runtime-dir", str(args.runtime_dir.resolve() / "phase8_prior" / sequence),
        "--sequences", sequence,
        "--cameras", CAMERAS,
    ]


def fit_command(args: argparse.Namespace, sequence: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "fit_sequence_body.py"),
        "--triangulation-root", str(args.triangulation_root.resolve()),
        "--sam-prior-root", str(args.sam_prior_root.resolve()),
        "--output-root", str(args.body_fit_root.resolve()),
        "--runtime-dir", str(args.runtime_dir.resolve() / "phase9" / sequence),
        "--sequences", sequence,
    ]


def mode_c_assessment_command(args: argparse.Namespace, sequence: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "assess_sam_mode_c_escalation.py"),
        "--sam-prior-root", str(args.sam_prior_root.resolve()),
        "--body-fit-root", str(args.body_fit_root.resolve()),
        "--triangulation-root", str(args.triangulation_root.resolve()),
        "--output-root", str(args.sam_mode_c_review_root.resolve()),
        "--sequences", sequence,
    ]


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
        "--output-root", str(args.export_root.resolve()),
        "--build-id", args.build_id,
        "--sequences", ",".join(args.sequences),
    ]


def ensure_sam_smoke(args: argparse.Namespace, sequence: str) -> bool:
    smoke_frames = 8
    smoke_dir = args.runtime_dir.resolve() / "sam_numeric_smoke"
    if sam_smoke_complete(smoke_dir, smoke_frames):
        return True
    update_state(
        args,
        "SAM_MODE_B_NUMERIC_SMOKE",
        [],
        smoke_sequence=sequence,
        smoke_camera="cam1",
        smoke_frames=smoke_frames,
        concurrent_with_sapiens=bool(
            args.wait_sapiens_pid is not None
            and process_alive(args.wait_sapiens_pid)
        ),
    )
    run(sam_smoke_command(args, sequence, "cam1", smoke_frames))
    return sam_smoke_complete(smoke_dir, smoke_frames)


def run_sequence_pipeline(
    args: argparse.Namespace, sequence: str, sam_smoke_ok: bool
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "sequence": sequence,
        "status": "INCOMPLETE",
        "failed_stage": "",
        "started_at_utc": utc_now(),
        "finished_at_utc": "",
        "elapsed_seconds": 0.0,
    }
    if not sam_smoke_ok:
        row["failed_stage"] = "SAM_MODE_B_NUMERIC_SMOKE"
    elif run(phase7_command(args, sequence)) != 0:
        row["failed_stage"] = "PHASE7"
    elif free_gib(args.sam_output_root) < args.minimum_free_gib:
        row["failed_stage"] = "DISK_RESERVE"
    elif run(sam_command(args, sequence)) != 0:
        row["failed_stage"] = "SAM_MODE_B"
    elif run(consolidate_command(args, sequence)) != 0:
        row["failed_stage"] = "SAM_PRIOR_CONSOLIDATION"
    elif run(fit_command(args, sequence)) != 0:
        row["failed_stage"] = "BODY_FIT"
    elif run(mode_c_assessment_command(args, sequence)) != 0:
        row["failed_stage"] = "MODE_C_ASSESSMENT"
    else:
        metadata_path = args.body_fit_root.resolve() / sequence / "metadata.json"
        try:
            body_status = json.loads(metadata_path.read_text(encoding="utf-8"))["qa"][
                "status"
            ]
            mode_c_status = json.loads(
                (
                    args.sam_mode_c_review_root.resolve()
                    / sequence
                    / "mode_c_escalation.json"
                ).read_text(encoding="utf-8")
            )["status"]
            row["status"] = (
                "REVIEW"
                if str(body_status).startswith("REVIEW")
                or mode_c_status == "REVIEW_MODE_C_CANDIDATE"
                else "PASS"
            )
        except (OSError, KeyError, json.JSONDecodeError):
            row["failed_stage"] = "BODY_FIT_VALIDATION"
    row["finished_at_utc"] = utc_now()
    row["elapsed_seconds"] = time.perf_counter() - started
    return row


def upsert_row(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    rows[:] = [existing for existing in rows if existing.get("sequence") != row["sequence"]]
    rows.append(row)


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.poll_seconds <= 0
        or args.sapiens_retries < 0
        or args.minimum_free_gib <= 0
        or args.expected_target_crops <= 0
        or args.reused_target_crops < 0
        or args.expected_sam_hours <= 0
    ):
        raise RuntimeError("invalid poll/retry configuration")
    sequence_csv = args.runtime_dir.resolve() / "autonomous_sequences.csv"
    rows = load_successful_rows(sequence_csv)
    successful = {row["sequence"] for row in rows}
    stream_attempted: set[str] = set()
    sam_smoke_ok = False
    if args.wait_sapiens_pid is not None:
        while process_alive(args.wait_sapiens_pid):
            ready = next(
                (
                    sequence
                    for sequence in args.sequences
                    if sequence not in successful
                    and sequence not in stream_attempted
                    and sequence_ready(
                        args.dataset_root.resolve(), args.pose_root.resolve(), sequence
                    )
                ),
                None,
            )
            if ready is not None:
                sam_smoke_ok = sam_smoke_ok or ensure_sam_smoke(args, ready)
                update_state(
                    args,
                    "STREAM_SEQUENCE_PIPELINE",
                    rows,
                    active_sequence=ready,
                    monitored_pid=args.wait_sapiens_pid,
                    concurrent_with_sapiens=True,
                    sapiens_progress=sapiens_progress(args),
                    free_storage_gib=free_gib(args.sam_output_root),
                )
                row = run_sequence_pipeline(args, ready, sam_smoke_ok)
                upsert_row(rows, row)
                atomic_csv(sequence_csv, rows)
                stream_attempted.add(ready)
                if row["status"] in {"PASS", "REVIEW"}:
                    successful.add(ready)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                # A failed sequence has already consumed its configured retry;
                # do not hot-loop it while the long teacher process is alive.
                continue
            update_state(
                args,
                "WAIT_RUNNING_SAPIENS2",
                rows,
                monitored_pid=args.wait_sapiens_pid,
                sapiens_progress=sapiens_progress(args),
                free_storage_gib=free_gib(args.sam_output_root),
            )
            time.sleep(args.poll_seconds)

    for attempt in range(args.sapiens_retries + 1):
        missing = missing_pose_sequences(args)
        if not missing:
            break
        update_state(
            args,
            "RESUME_SAPIENS2",
            rows,
            sapiens_attempt=attempt + 1,
            missing_pose_sequences=missing,
        )
        run(sapiens_command(args))
    pose_missing = missing_pose_sequences(args)

    if not sam_smoke_ok:
        smoke_sequence = next(
            (
                sequence
                for sequence in args.sequences
                if sequence_ready(
                    args.dataset_root.resolve(), args.pose_root.resolve(), sequence
                )
            ),
            None,
        )
        sam_smoke_ok = bool(
            smoke_sequence is not None and ensure_sam_smoke(args, smoke_sequence)
        )

    for sequence in args.sequences:
        if sequence in successful:
            continue
        if sequence in pose_missing:
            now = utc_now()
            row = {
                "sequence": sequence,
                "status": "INCOMPLETE",
                "failed_stage": "SAPIENS2",
                "started_at_utc": now,
                "finished_at_utc": now,
                "elapsed_seconds": 0.0,
            }
        else:
            row = run_sequence_pipeline(args, sequence, sam_smoke_ok)
        upsert_row(rows, row)
        atomic_csv(sequence_csv, rows)
        update_state(
            args,
            "SEQUENCE_PIPELINE",
            rows,
            active_sequence=sequence,
            remaining_sequences=[
                item
                for item in args.sequences
                if item not in {existing["sequence"] for existing in rows}
            ],
            mode_b_default=True,
            mode_c_automatic=False,
            free_storage_gib=free_gib(args.sam_output_root),
            minimum_free_gib=args.minimum_free_gib,
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    export_exit = run(export_command(args))
    final_status = (
        "PASS_OR_REVIEW"
        if export_exit == 0 and all(row["status"] in {"PASS", "REVIEW"} for row in rows)
        else "INCOMPLETE_OR_FAIL"
    )
    update_state(
        args,
        "COMPLETE",
        rows,
        final_status=final_status,
        export_exit_code=export_exit,
        mode_c_automatic=False,
    )
    return 0 if final_status == "PASS_OR_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
