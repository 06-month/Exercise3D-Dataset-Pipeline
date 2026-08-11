#!/usr/bin/env python3
"""Atomically checkpoint live private operational state for agent handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.materialize_inference_provenance import materialize_all
except ModuleNotFoundError:
    from materialize_inference_provenance import materialize_all


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_MARKERS = (
    "sapiens2_target_pipeline.py",
    "run_autonomous_generation.py",
    "run_sam_body4d_full.py",
    "benchmark_sam_body4d.py",
    "checkpoint_handoff_state.py",
    "run_deadline_snapshot.py",
)
RESUMABLE_MARKERS = tuple(
    marker for marker in PROCESS_MARKERS if marker != "checkpoint_handoff_state.py"
)
CAMERAS = ("cam1", "cam2", "cam3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def active_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().split(b"\0")
            command = " ".join(part.decode(errors="replace") for part in parts if part)
            if not command or not any(marker in command for marker in PROCESS_MARKERS):
                continue
            rows.append(
                {
                    "pid": int(entry.name),
                    "command": command,
                    "cwd": str((entry / "cwd").resolve()),
                    "process_dir_ctime_utc": datetime.fromtimestamp(
                        entry.stat().st_ctime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        except (OSError, ValueError):
            continue
    return sorted(rows, key=lambda row: row["pid"])


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def pose_progress(pose_root: Path, sequences: list[str]) -> dict[str, Any]:
    completed_cameras: list[str] = []
    completed_sequences: list[str] = []
    crops = 0
    chunk_events: list[tuple[datetime, int]] = []
    for sequence in sequences:
        sequence_ok = True
        for camera in CAMERAS:
            metadata = read_json(pose_root / sequence / camera / "metadata.json")
            ok = False
            try:
                ok = metadata is not None and metadata["qa"]["status"] == "PASS"
                if ok:
                    crops += int(metadata["qa"]["target_pose_count"])
                    completed_cameras.append(f"{sequence}/{camera}")
                else:
                    sequence_ok = False
            except (KeyError, TypeError, ValueError):
                sequence_ok = False
            camera_dir = pose_root / sequence / camera
            chunks = sorted((camera_dir / "chunks").glob("*.npz"))
            for chunk in chunks:
                try:
                    with np.load(chunk, allow_pickle=False) as payload:
                        chunk_crops = int(payload["target_present"].sum())
                    chunk_events.append(
                        (
                            datetime.fromtimestamp(chunk.stat().st_mtime, tz=timezone.utc),
                            chunk_crops,
                        )
                    )
                    if not ok:
                        crops += chunk_crops
                except (OSError, KeyError, ValueError):
                    continue
        if sequence_ok:
            completed_sequences.append(sequence)
    monitor = (
        pose_root.parent
        / "runtime"
        / "phase6_full_target_inference"
        / "target_only_pilot_gpu_utilization.csv"
    )
    started: datetime | None = None
    try:
        with monitor.open(newline="", encoding="utf-8") as handle:
            first = next(csv.DictReader(handle))
        started = datetime.fromisoformat(first["timestamp_utc"])
    except (OSError, StopIteration, KeyError, ValueError, csv.Error):
        pass
    if started is not None:
        chunk_events = [event for event in chunk_events if event[0] >= started]
    chunk_events.sort()
    recent = chunk_events[-6:]
    recent_rate = 0.0
    if len(recent) >= 2:
        seconds = (recent[-1][0] - recent[0][0]).total_seconds()
        recent_rate = sum(event[1] for event in recent[1:]) / seconds if seconds > 0 else 0.0
    effective_rate = 0.0
    if started is not None:
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        effective_rate = max(0, crops - 9725) / elapsed if elapsed > 0 else 0.0
    remaining = max(0, 65430 - crops)
    projection_rate = recent_rate or effective_rate
    eta_hours = remaining / projection_rate / 3600 if projection_rate > 0 else None
    return {
        "completed_cameras": completed_cameras,
        "completed_camera_count": len(completed_cameras),
        "completed_sequences": completed_sequences,
        "completed_sequence_count": len(completed_sequences),
        "processed_target_crops": crops,
        "total_target_crops": 65430,
        "remaining_target_crops": remaining,
        "effective_new_crops_per_second": effective_rate or None,
        "recent_chunk_crops_per_second": recent_rate or None,
        "eta_projection_crops_per_second": projection_rate or None,
        "estimated_remaining_hours": eta_hours,
        "estimated_completion_utc": (
            datetime.fromtimestamp(
                datetime.now(timezone.utc).timestamp() + eta_hours * 3600,
                tz=timezone.utc,
            ).isoformat()
            if eta_hours is not None
            else None
        ),
    }


def downstream_progress(root: Path, sequences: list[str]) -> dict[str, Any]:
    completed: list[str] = []
    status: dict[str, str] = {}
    for sequence in sequences:
        metadata = read_json(root / sequence / "metadata.json")
        if metadata is None:
            continue
        value = str(metadata.get("qa", {}).get("status", metadata.get("status", "UNKNOWN")))
        status[sequence] = value
        completed.append(sequence)
    return {"completed": completed, "count": len(completed), "status": status}


def sam_progress(root: Path, sequences: list[str]) -> dict[str, Any]:
    completed_cameras: list[str] = []
    completed_sequences: list[str] = []
    processed_frames = 0
    elapsed_wall_seconds = 0.0
    for sequence in sequences:
        sequence_ok = True
        for camera in CAMERAS:
            output = root / sequence / camera
            benchmark_path = output / "sam_body_benchmark.csv"
            profile = read_json(output / "mode_b_profile.json")
            try:
                with benchmark_path.open(newline="", encoding="utf-8") as handle:
                    benchmark_rows = list(csv.DictReader(handle))
                benchmark = benchmark_rows[0] if len(benchmark_rows) == 1 else {}
                expected = int(profile["input_frames"]) if profile is not None else 0
                mesh_count = len(
                    list((output / "mode_b_private_output" / "mesh_4d_individual" / "1").glob("*.ply"))
                )
                numeric_count = len(
                    list((output / "mode_b_private_output" / "mhr_numeric" / "1").glob("*.npz"))
                )
                passed = bool(
                    expected > 0
                    and benchmark.get("status") == "PASS"
                    and int(float(benchmark.get("frames_processed") or 0)) == expected
                    and int(profile["frames_processed"]) == expected
                    and mesh_count == expected
                    and numeric_count == expected
                )
            except (OSError, KeyError, ValueError, csv.Error):
                passed = False
                expected = 0
            if not passed:
                sequence_ok = False
                continue
            completed_cameras.append(f"{sequence}/{camera}")
            processed_frames += expected
            elapsed_wall_seconds += float(benchmark.get("elapsed_wall_seconds") or 0)
        if sequence_ok:
            completed_sequences.append(sequence)
    return {
        "completed_cameras": completed_cameras,
        "completed_camera_count": len(completed_cameras),
        "completed_sequences": completed_sequences,
        "completed_sequence_count": len(completed_sequences),
        "processed_frames": processed_frames,
        "total_frames": 65595,
        "measured_frames_per_second": (
            processed_frames / elapsed_wall_seconds if elapsed_wall_seconds > 0 else None
        ),
        "summed_camera_wall_seconds": elapsed_wall_seconds,
    }


def gpu_state() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return [
            {
                "uuid": fields[0],
                "utilization_gpu_pct": fields[1],
                "memory_used_mib": fields[2],
                "power_draw_w": fields[3],
            }
            for line in output.splitlines()
            if line.strip()
            for fields in [[item.strip() for item in line.split(",")]]
        ]
    except (OSError, subprocess.SubprocessError):
        return []


def build_state(args: argparse.Namespace) -> dict[str, Any]:
    sequences = [item for item in args.sequences.split(",") if item]
    configs = [
        PROJECT_ROOT / "configs" / "sapiens2_pose_5b_environment.json",
        PROJECT_ROOT / "configs" / "phase9_body_fit.json",
        PROJECT_ROOT / "configs" / "sam_mode_c_escalation.json",
    ]
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_head = "UNKNOWN"
    try:
        git_diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"],
            cwd=PROJECT_ROOT,
            timeout=10,
        )
        git_diff_sha256 = hashlib.sha256(git_diff).hexdigest()
    except (OSError, subprocess.SubprocessError):
        git_diff_sha256 = "UNKNOWN"
    deadline = datetime.fromisoformat(args.deadline_utc).astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    processes = active_processes()
    pose = pose_progress(args.pose_root.resolve(), sequences)
    sam = sam_progress(args.sam_output_root.resolve(), sequences)
    in_progress = [
        process["command"]
        for process in processes
        if any(marker in process["command"] for marker in RESUMABLE_MARKERS)
    ]
    remaining = [
        sequence for sequence in sequences if sequence not in pose["completed_sequences"]
    ]
    return {
        "schema_version": 1,
        "phase": "PHASE6_7_8_STREAMING",
        "status": "RUNNING" if in_progress else "STOPPED",
        "completed": pose["completed_sequences"],
        "in_progress": in_progress,
        "remaining": remaining,
        "last_completed_item": (
            pose["completed_cameras"][-1] if pose["completed_cameras"] else None
        ),
        "processed_frames": pose["processed_target_crops"],
        "total_frames": pose["total_target_crops"],
        "updated_at_utc": now.isoformat(),
        "deadline_utc": deadline.isoformat(),
        "remaining_wall_hours": (deadline - now).total_seconds() / 3600.0,
        "git_commit": git_head,
        "git_diff_sha256": git_diff_sha256,
        "config_hash": sha256_files(configs),
        "model_checkpoint": {
            "sapiens2": "facebook/sapiens2-pose-5b; configs/sapiens2_pose_5b_environment.json",
            "sam": "SAM-Body4D Mode B; metadata/results/sam_body4d_checkpoint_integrity.csv",
        },
        "source_manifest": "private inventory: 26 sequences / 78 cameras / 65,595 frames",
        "camera_geometry_version": "PHASE5_BACKGROUND_BA_RECOVERED",
        "temporal_metadata_version": "PHASE2_PTS_AUDIT",
        "active_processes": processes,
        "gpu": gpu_state(),
        "pose": pose,
        "triangulation": downstream_progress(args.triangulation_root.resolve(), sequences),
        "sam": sam,
        "body_fit": downstream_progress(args.body_fit_root.resolve(), sequences),
        "autonomous_supervisor": read_json(args.supervisor_state.resolve()),
        "resume_policy": "validate PASS metadata/schema/checksum; skip complete; rerun only incomplete/corrupt item",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-output-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument("--supervisor-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / ".runtime" / "handoff_state.json")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--deadline-utc", default="2026-08-14T04:00:00+00:00")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise RuntimeError("poll-seconds must be positive")
    while True:
        output = args.output.resolve()
        previous = read_json(output) or {}
        state = build_state(args)
        resume_commands = dict(previous.get("resume_commands", {}))
        for process in state["active_processes"]:
            for marker in RESUMABLE_MARKERS:
                if marker in process["command"]:
                    resume_commands[marker] = process["command"]
        state["resume_commands"] = resume_commands
        atomic_json(output, state)
        provenance_args = argparse.Namespace(
            dataset_root=args.dataset_root,
            selection_root=args.selection_root,
            pose_root=args.pose_root,
            sam_output_root=args.sam_output_root,
            sam_prior_root=args.sam_prior_root,
            handoff_state=output,
            sequences=[item for item in args.sequences.split(",") if item],
        )
        state["provenance_materialized"] = materialize_all(provenance_args)
        atomic_json(output, state)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
