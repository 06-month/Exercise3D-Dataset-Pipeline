#!/usr/bin/env python3
"""Render and atomically persist the Exercise3D autonomous-run dashboard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9), name="KST")
CAMERAS = ("cam1", "cam2", "cam3")
PROCESS_MARKERS = {
    "sapiens": {"sapiens2_target_pipeline.py", "sapiens2_pose_pipeline.py"},
    "sam": {
        "run_sam_body4d_full.py",
        "benchmark_sam_body4d.py",
        "sam_body_primary_target_runner.py",
    },
    "supervisor": {"run_autonomous_generation.py"},
    "quality_follower": {"run_quality_control_follower.py"},
    "handoff_monitor": {"checkpoint_handoff_state.py"},
    "deadline_sentinel": {"run_deadline_snapshot.py"},
}


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


def parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def human_duration(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    seconds = abs(int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    prefix = f"{days}d " if days else ""
    return f"{sign}{prefix}{hours:02d}h {minutes:02d}m"


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if result == result else None
    except (TypeError, ValueError):
        return None


def process_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        entries = list(Path("/proc").iterdir())
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
            names = {Path(item).name for item in argv if "\n" not in item}
            groups = [
                group
                for group, markers in PROCESS_MARKERS.items()
                if names & markers
            ]
            if not groups:
                continue
            stat_fields = (entry / "stat").read_text(encoding="utf-8").split()
            rows.append(
                {
                    "pid": int(entry.name),
                    "ppid": int(stat_fields[3]),
                    "state": stat_fields[2],
                    "argv": argv,
                    "groups": groups,
                    "command_name": next(
                        (
                            name
                            for name in names
                            if any(name in markers for markers in PROCESS_MARKERS.values())
                        ),
                        Path(argv[0]).name,
                    ),
                }
            )
        except (OSError, IndexError, ValueError):
            continue
    return sorted(rows, key=lambda row: row["pid"])


def root_processes(processes: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    members = [row for row in processes if group in row["groups"] and row["state"] != "Z"]
    member_pids = {row["pid"] for row in members}
    return [row for row in members if row["ppid"] not in member_pids]


def command_flag(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def sequence_order(handoff: dict[str, Any]) -> list[str]:
    commands = handoff.get("resume_commands", {})
    for key in ("sapiens2_target_pipeline.py", "run_autonomous_generation.py"):
        command = commands.get(key)
        if not command:
            continue
        try:
            value = command_flag(shlex.split(str(command)), "--sequences")
        except ValueError:
            value = None
        if value:
            return [item for item in value.split(",") if item]
    ordered = list(handoff.get("completed", [])) + list(handoff.get("remaining", []))
    return list(dict.fromkeys(str(item) for item in ordered))


def first_incomplete_camera(
    sequences: list[str], completed: Iterable[str]
) -> tuple[str | None, str | None]:
    done = set(completed)
    for sequence in sequences:
        for camera in CAMERAS:
            if f"{sequence}/{camera}" not in done:
                return sequence, camera
    return None, None


def current_sam_camera(
    processes: list[dict[str, Any]],
    supervisor: dict[str, Any],
    sequences: list[str],
    completed: Iterable[str],
) -> tuple[str | None, str | None]:
    for process in processes:
        if "sam" not in process["groups"]:
            continue
        frames = command_flag(process["argv"], "--input-frames")
        if frames:
            path = Path(frames)
            return path.parent.name, path.name
        sequence = command_flag(process["argv"], "--sequences")
        if sequence:
            selected = sequence.split(",", 1)[0]
            camera = first_incomplete_camera([selected], completed)[1]
            return selected, camera
    active = supervisor.get("active_sequence")
    if active:
        return str(active), first_incomplete_camera([str(active)], completed)[1]
    return first_incomplete_camera(sequences, completed)


def query_gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10)
        devices = []
        for line in output.splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) != 7:
                continue
            devices.append(
                {
                    "index": int(fields[0]),
                    "uuid": fields[1],
                    "utilization_pct": float(fields[2]),
                    "memory_used_mib": float(fields[3]),
                    "memory_total_mib": float(fields[4]),
                    "power_draw_w": float(fields[5]),
                    "temperature_c": float(fields[6]),
                }
            )
        return {
            "available": bool(devices),
            "devices": devices,
            "utilization_pct": max((row["utilization_pct"] for row in devices), default=None),
            "memory_used_mib": sum(row["memory_used_mib"] for row in devices),
            "memory_total_mib": sum(row["memory_total_mib"] for row in devices),
            "power_draw_w": sum(row["power_draw_w"] for row in devices),
            "temperature_c": max((row["temperature_c"] for row in devices), default=None),
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"available": False, "devices": []}


def load_sequence_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def status_counts(values: Iterable[Any]) -> dict[str, int]:
    result = {"PASS": 0, "REVIEW": 0, "FAIL": 0, "INCOMPLETE": 0, "UNKNOWN": 0}
    for raw in values:
        value = str(raw or "UNKNOWN").upper()
        if value.startswith("PASS"):
            result["PASS"] += 1
        elif value.startswith("REVIEW"):
            result["REVIEW"] += 1
        elif value.startswith("FAIL") or value.startswith("NO_GO"):
            result["FAIL"] += 1
        elif value.startswith("INCOMPLETE"):
            result["INCOMPLETE"] += 1
        else:
            result["UNKNOWN"] += 1
    return result


def export_progress(root: Path, target_build_id: str | None = None) -> dict[str, Any]:
    builds: list[tuple[float, Path, dict[str, Any], list[dict[str, str]]]] = []
    try:
        manifests = list(root.glob("*/dataset_manifest.json"))
    except OSError:
        manifests = []
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        rows = load_sequence_rows(manifest_path.parent / "sequence_status.csv")
        try:
            modified = manifest_path.stat().st_mtime
        except OSError:
            continue
        builds.append((modified, manifest_path.parent, manifest, rows))
    if not builds or (
        target_build_id
        and not (root / target_build_id / "dataset_manifest.json").is_file()
    ):
        latest_materialized = max(builds, key=lambda row: row[0])[1].name if builds else None
        return {
            "status": "NOT_STARTED",
            "build_count": len(builds),
            "latest_build_id": target_build_id,
            "latest_materialized_build_id": latest_materialized,
            "completed_sequences": 0,
            "status_counts": status_counts([]),
            "freeze_eligible": False,
        }
    selected = (
        next(row for row in builds if row[1].name == target_build_id)
        if target_build_id
        else max(builds, key=lambda row: row[0])
    )
    _, directory, manifest, rows = selected
    counts = status_counts(row.get("status") for row in rows)
    return {
        "status": str(manifest.get("status", "MATERIALIZED")),
        "build_count": len(builds),
        "latest_build_id": directory.name,
        "latest_materialized_build_id": directory.name,
        "completed_sequences": counts["PASS"] + counts["REVIEW"],
        "status_counts": counts,
        "freeze_eligible": bool(manifest.get("freeze_eligible", False)),
    }


def metadata_statuses(
    root: Path, sequences: list[str], keys: tuple[str, ...]
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for sequence in sequences:
        metadata = read_json(root / sequence / "metadata.json")
        qa = metadata.get("qa", {}) if isinstance(metadata.get("qa", {}), dict) else {}
        for key in keys:
            value = qa.get(key) if key in qa else metadata.get(key)
            if value:
                statuses[sequence] = str(value)
                break
    return statuses


def quality_progress(root: Path, sequences: list[str]) -> dict[str, Any]:
    statuses: dict[str, str] = {}
    frame_count = 0
    for sequence in sequences:
        output = root / sequence
        metadata = read_json(output / "metadata.json")
        if not (output / "quality_vector.npz").is_file() or not metadata:
            continue
        qa = metadata.get("qa", {})
        status = str(qa.get("sequence_status", "UNKNOWN"))
        statuses[sequence] = status
        try:
            frame_count += int(qa.get("frame_count", 0))
        except (TypeError, ValueError):
            pass
    return {
        "completed_sequences": len(statuses),
        "completed_frames": frame_count,
        "status_counts": status_counts(statuses.values()),
        "statuses": statuses,
    }


def sam_retry_summary(runtime_dir: Path) -> tuple[int, list[dict[str, Any]]]:
    retry_count = 0
    failures: list[dict[str, Any]] = []
    for path in runtime_dir.glob("phase8/*/sam_body4d_full.csv"):
        for row in load_sequence_rows(path):
            try:
                retry_count += int(row.get("attempt", "1")) > 1
            except ValueError:
                pass
            if row.get("status") not in {"PASS", "SKIPPED"}:
                failures.append(
                    {
                        "source": str(path.relative_to(PROJECT_ROOT)),
                        "sequence": row.get("sequence"),
                        "camera": row.get("camera"),
                        "status": row.get("status"),
                        "reason": row.get("reason"),
                    }
                )
    return retry_count, failures


def scan_runtime_errors(runtime_dir: Path) -> list[dict[str, Any]]:
    patterns = (
        ("TRACEBACK", re.compile(r"Traceback \(most recent call last\):", re.I)),
        ("CUDA_OOM", re.compile(r"CUDA out of memory|CUDNN_STATUS_ALLOC_FAILED", re.I)),
        ("RETRY_EXHAUSTED", re.compile(r"retr(?:y|ies).*(?:exhaust|failed)", re.I)),
        (
            "NAN_INF_GATE_FAIL",
            re.compile(r"(?:nan|inf).*(?:gate|validation).*fail|(?:gate|validation).*fail.*(?:nan|inf)", re.I),
        ),
    )
    errors: list[dict[str, Any]] = []
    for path in runtime_dir.glob("*.log"):
        try:
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 262144))
                text = handle.read().decode(errors="replace")
        except OSError:
            continue
        for code, pattern in patterns:
            matches = list(pattern.finditer(text))
            if matches:
                line = text[max(0, text.rfind("\n", 0, matches[-1].start()) + 1) :]
                line = line.splitlines()[0][:500]
                errors.append({"code": code, "source": path.name, "message": line})
    return errors


def latest_mtime(paths: Iterable[Path]) -> datetime | None:
    latest: float | None = None
    for path in paths:
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        latest = modified if latest is None else max(latest, modified)
    return datetime.fromtimestamp(latest, tz=timezone.utc) if latest is not None else None


def progress_artifacts(
    args: argparse.Namespace,
    sapiens_current: tuple[str | None, str | None],
    sam_current: tuple[str | None, str | None],
    handoff: dict[str, Any],
) -> tuple[list[Path], int]:
    paths: list[Path] = [args.sequence_status, args.supervisor_state]
    pose_sequence, pose_camera = sapiens_current
    if pose_sequence and pose_camera:
        camera_dir = args.pose_root / pose_sequence / pose_camera
        paths.extend((camera_dir / "chunks").glob("*.npz"))
        paths.append(camera_dir / "metadata.json")
    last_pose = handoff.get("last_completed_item")
    if last_pose and "/" in str(last_pose):
        sequence, camera = str(last_pose).split("/", 1)
        paths.append(args.pose_root / sequence / camera / "metadata.json")
    current_sam_frames = 0
    sam_sequence, sam_camera = sam_current
    if sam_sequence and sam_camera:
        numeric_dir = (
            args.sam_output_root
            / sam_sequence
            / sam_camera
            / "mode_b_private_output"
            / "mhr_numeric"
            / "1"
        )
        numeric = list(numeric_dir.glob("*.npz"))
        current_sam_frames = len(numeric)
        paths.extend(numeric)
        paths.append(args.sam_output_root / sam_sequence / sam_camera / "mode_b_profile.json")
    return paths, current_sam_frames


def build_dashboard(
    args: argparse.Namespace,
    *,
    now: datetime | None = None,
    processes: list[dict[str, Any]] | None = None,
    gpu: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = (now or utc_now()).astimezone(timezone.utc)
    handoff = read_json(args.handoff_state)
    supervisor = read_json(args.supervisor_state)
    deadline_state = read_json(args.deadline_state)
    quality_follower_state = read_json(args.quality_follower_state)
    freeze_readiness = dict(quality_follower_state.get("freeze_readiness", {}))
    previous = read_json(args.output)
    sequences = sequence_order(handoff)
    total_sequences = len(sequences) or 26
    processes = process_table() if processes is None else processes
    gpu = query_gpu() if gpu is None else gpu
    roots = {group: root_processes(processes, group) for group in PROCESS_MARKERS}

    previous_gpu = previous.get("gpu", {})
    if gpu.get("available"):
        gpu["telemetry_fresh"] = True
        gpu["last_success_at"] = now.isoformat()
    else:
        last_success = parse_datetime(previous_gpu.get("last_success_at"))
        if previous_gpu.get("available"):
            cached = dict(previous_gpu)
            cached["telemetry_fresh"] = False
            cached["query_failed_at"] = now.isoformat()
            gpu = cached
        else:
            gpu["telemetry_fresh"] = False
            gpu["last_success_at"] = last_success.isoformat() if last_success else None

    pose = dict(handoff.get("pose", {}))
    sam_source = dict(handoff.get("sam", {}))
    pose_completed = list(pose.get("completed_cameras", []))
    sam_completed = list(sam_source.get("completed_cameras", []))
    sapiens_current = first_incomplete_camera(sequences, pose_completed)
    sam_current = current_sam_camera(processes, supervisor, sequences, sam_completed)
    artifact_paths, current_sam_frames = progress_artifacts(
        args, sapiens_current, sam_current, handoff
    )

    deadline = parse_datetime(
        deadline_state.get("deadline_utc")
        or handoff.get("deadline_utc")
        or args.deadline_utc
    )
    if deadline is None:
        deadline = datetime.fromisoformat(args.deadline_utc).astimezone(timezone.utc)
    remaining_seconds = (deadline - now).total_seconds()
    eta = parse_datetime(pose.get("estimated_completion_utc"))

    sapiens_total = int(pose.get("total_target_crops", 65430))
    sapiens_done = int(pose.get("processed_target_crops", 0))
    sam_total = int(sam_source.get("total_frames", 65595))
    sam_done = int(sam_source.get("processed_frames", 0))
    sam_rate = safe_float(sam_source.get("measured_frames_per_second"))
    sam_remaining_hours = (
        max(0, sam_total - sam_done) / sam_rate / 3600
        if sam_rate and sam_rate > 0
        else None
    )
    sam_eta = now + timedelta(hours=sam_remaining_hours) if sam_remaining_hours is not None else None

    sequence_rows = load_sequence_rows(args.sequence_status)
    retry_count, sam_failures = sam_retry_summary(args.autonomous_runtime_dir)
    runtime_errors = scan_runtime_errors(args.runtime_dir)
    row_failures = [
        {
            "code": "SEQUENCE_PIPELINE_FAILED",
            "source": "autonomous_sequences.csv",
            "sequence": row.get("sequence"),
            "message": row.get("failed_stage") or row.get("status"),
        }
        for row in sequence_rows
        if row.get("status") not in {"PASS", "REVIEW"}
    ]
    errors = runtime_errors + sam_failures + row_failures

    triangulation_source = dict(handoff.get("triangulation", {}))
    body_source = dict(handoff.get("body_fit", {}))
    triangulation_status = metadata_statuses(
        args.triangulation_root,
        sequences,
        ("quality_status", "status", "schema_status"),
    ) or dict(triangulation_source.get("status", {}))
    body_status = metadata_statuses(
        args.body_fit_root, sequences, ("status",)
    ) or dict(body_source.get("status", {}))
    triangulation_counts = status_counts(triangulation_status.values())
    body_counts = status_counts(body_status.values())
    quality = quality_progress(args.quality_root, sequences)
    export = export_progress(args.export_root, deadline_state.get("build_id"))
    deadline_snapshot_status = str(deadline_state.get("status", "UNKNOWN"))
    export["deadline_snapshot_status"] = deadline_snapshot_status

    try:
        disk_usage = shutil.disk_usage(args.disk_path)
        disk = {
            "path": str(args.disk_path),
            "free_gib": disk_usage.free / (1024**3),
            "total_gib": disk_usage.total / (1024**3),
            "minimum_free_gib": args.minimum_free_gib,
        }
    except OSError:
        disk = {
            "path": str(args.disk_path),
            "free_gib": None,
            "total_gib": None,
            "minimum_free_gib": args.minimum_free_gib,
        }

    counters = {
        "sapiens_crops": sapiens_done,
        "sapiens_cameras": int(pose.get("completed_camera_count", 0)),
        "sam_frames": sam_done,
        "sam_current_camera_frames": current_sam_frames,
        "sam_cameras": int(sam_source.get("completed_camera_count", 0)),
        "triangulation_sequences": int(triangulation_source.get("count", 0)),
        "body_fit_sequences": int(body_source.get("count", 0)),
        "quality_sequences": int(quality["completed_sequences"]),
        "export_sequences": int(export["completed_sequences"]),
    }
    signature = hashlib.sha256(
        json.dumps(counters, sort_keys=True).encode("utf-8")
    ).hexdigest()
    previous_signature = previous.get("monitoring", {}).get("progress_signature")
    if previous_signature == signature:
        last_progress = parse_datetime(previous.get("last_progress_timestamp"))
    else:
        last_progress = now
    if last_progress is None:
        last_progress = latest_mtime(artifact_paths) or now

    gpu_work_expected = bool(roots["sapiens"] or roots["sam"])
    utilization = safe_float(gpu.get("utilization_pct"))
    previous_idle = parse_datetime(previous.get("gpu", {}).get("idle_since"))
    if gpu_work_expected and utilization is not None and utilization < args.gpu_idle_threshold_pct:
        idle_since = previous_idle or now
    else:
        idle_since = None
    gpu["workload_expected"] = gpu_work_expected
    gpu["idle_since"] = idle_since.isoformat() if idle_since else None

    reasons: list[dict[str, Any]] = []

    def attention(code: str, message: str, severity: str = "ERROR") -> None:
        reasons.append({"code": code, "severity": severity, "message": message})

    incomplete_pose = sapiens_done < sapiens_total
    incomplete_body = int(body_source.get("count", 0)) < total_sequences
    incomplete_quality = quality["completed_sequences"] < total_sequences
    if incomplete_pose and not roots["sapiens"]:
        attention("SAPIENS_PROCESS_DEAD", "Sapiens output is incomplete but no matching live process exists.")
    if (incomplete_pose or incomplete_body) and not roots["supervisor"]:
        attention("SUPERVISOR_DEAD", "Autonomous generation is incomplete but the supervisor is not alive.")
    if (incomplete_pose or incomplete_body) and not roots["handoff_monitor"]:
        attention("HANDOFF_MONITOR_DEAD", "Persistent handoff checkpoint monitor is not alive.")
    if remaining_seconds > 0 and not roots["deadline_sentinel"]:
        attention("DEADLINE_SENTINEL_DEAD", "Deadline snapshot sentinel is not alive.")
    if incomplete_quality and not roots["quality_follower"]:
        attention(
            "QUALITY_FOLLOWER_DEAD",
            "Phase 11 quality output is incomplete but the CPU follower is not alive.",
        )
    quality_follower_updated = parse_datetime(quality_follower_state.get("updated_at_utc"))
    if (
        incomplete_quality
        and roots["quality_follower"]
        and (
            quality_follower_updated is None
            or (now - quality_follower_updated).total_seconds() > args.state_stale_seconds
        )
    ):
        age = (
            "unknown"
            if quality_follower_updated is None
            else human_duration((now - quality_follower_updated).total_seconds())
        )
        attention(
            "QUALITY_FOLLOWER_STATE_STALE",
            f"quality_follower_state.json age is {age}.",
        )
    quality_failures = quality_follower_state.get("failures", [])
    readiness_failures = freeze_readiness.get("failures", [])
    if quality_failures:
        failure_text = "; ".join(
            f"{row.get('sequence')}: {row.get('reason')}"
            for row in quality_failures
        )
        attention(
            "QUALITY_FOLLOWER_FAILURE",
            failure_text or "Phase 11 quality follower reported ATTENTION.",
        )
    if readiness_failures:
        failure_text = "; ".join(
            f"{row.get('sequence')}: {','.join(map(str, row.get('reasons', [])))}"
            for row in readiness_failures
        )
        attention(
            "FREEZE_READINESS_FAILED",
            failure_text or "One or more completed sequences are not export-ready.",
        )
    if (
        quality_follower_state.get("status") == "ATTENTION"
        and not quality_failures
        and not readiness_failures
    ):
        attention(
            "QUALITY_FOLLOWER_FAILURE",
            "Phase 11 quality follower reported ATTENTION without a structured reason.",
        )
    for group in (
        "sapiens",
        "supervisor",
        "quality_follower",
        "handoff_monitor",
        "deadline_sentinel",
    ):
        if len(roots[group]) > 1:
            attention(
                "DUPLICATE_PROCESS",
                f"{group} has {len(roots[group])} independent root processes.",
            )

    handoff_updated = parse_datetime(handoff.get("updated_at_utc"))
    if handoff_updated is None or (now - handoff_updated).total_seconds() > args.state_stale_seconds:
        age = "unknown" if handoff_updated is None else human_duration((now - handoff_updated).total_seconds())
        attention("HANDOFF_STATE_STALE", f"handoff_state.json age is {age}.")
    if (
        (incomplete_pose or incomplete_body)
        and (now - last_progress).total_seconds() > args.stall_minutes * 60
    ):
        attention(
            "PROGRESS_STALLED",
            f"No monitored progress counter changed for {human_duration((now - last_progress).total_seconds())}.",
        )
    free_gib = safe_float(disk.get("free_gib"))
    if free_gib is None:
        attention("DISK_STATUS_UNAVAILABLE", "Disk free-space query failed.")
    elif free_gib < args.minimum_free_gib:
        attention(
            "DISK_RESERVE_LOW",
            f"Disk free space {free_gib:.2f} GiB is below {args.minimum_free_gib:.2f} GiB.",
        )
    gpu_last_success = parse_datetime(gpu.get("last_success_at"))
    if (
        gpu_work_expected
        and not gpu.get("telemetry_fresh")
        and (
            gpu_last_success is None
            or (now - gpu_last_success).total_seconds() > args.state_stale_seconds
        )
    ):
        attention(
            "GPU_STATUS_UNAVAILABLE",
            "A GPU workload is alive but nvidia-smi telemetry remained unavailable beyond the stale window.",
        )
    if idle_since and (now - idle_since).total_seconds() > args.gpu_idle_minutes * 60:
        attention(
            "GPU_UNEXPECTEDLY_IDLE",
            f"GPU utilization stayed below {args.gpu_idle_threshold_pct:.1f}% for {human_duration((now - idle_since).total_seconds())}.",
        )
    if eta and eta > deadline:
        attention(
            "DEADLINE_ETA_AT_RISK",
            f"Sapiens ETA is {human_duration((eta - deadline).total_seconds())} after the deadline.",
            severity="WARNING",
        )
    if deadline_snapshot_status in {
        "EXPORT_FAILED",
        "EXPORT_INTEGRITY_FAILED",
        "EXISTING_BUILD_INVALID",
    }:
        integrity_errors = deadline_state.get("integrity_errors", [])
        detail = f" Integrity errors: {';'.join(map(str, integrity_errors))}." if integrity_errors else ""
        attention(
            "DEADLINE_SNAPSHOT_FAILED",
            f"Deadline snapshot state is {deadline_snapshot_status}.{detail}",
        )
    previous_eta = parse_datetime(previous.get("sapiens", {}).get("eta_utc"))
    if eta and previous_eta and (eta - previous_eta).total_seconds() > args.eta_worsening_minutes * 60:
        attention(
            "DEADLINE_ETA_WORSENED",
            f"Sapiens ETA worsened by {human_duration((eta - previous_eta).total_seconds())} since the previous snapshot.",
            severity="WARNING",
        )
    for error in errors:
        code = str(error.get("code", "RUNTIME_ERROR"))
        attention(code, str(error.get("message") or error.get("reason") or error), severity="ERROR")
    if body_counts["FAIL"] or triangulation_counts["FAIL"] or quality["status_counts"]["FAIL"]:
        attention(
            "VALIDATION_FAIL",
            "Validation FAIL counts: "
            f"triangulation={triangulation_counts['FAIL']}, "
            f"body_fit={body_counts['FAIL']}, quality={quality['status_counts']['FAIL']}.",
        )

    if int(body_source.get("count", 0)) >= total_sequences and export.get("freeze_eligible"):
        overall_status = "COMPLETE"
    elif any(
        roots[group] for group in ("sapiens", "sam", "supervisor", "quality_follower")
    ):
        overall_status = "RUNNING"
    else:
        overall_status = "STOPPED"

    stage = str(supervisor.get("stage", "UNKNOWN"))
    active_sequence = supervisor.get("active_sequence")
    last_event = f"{stage}: {active_sequence}" if active_sequence else stage
    stalled_jobs = [
        reason["code"]
        for reason in reasons
        if reason["code"] in {"PROGRESS_STALLED", "GPU_UNEXPECTEDLY_IDLE"}
    ]
    return {
        "schema_version": 1,
        "updated_at": now.isoformat(),
        "current_time_kst": now.astimezone(KST).isoformat(),
        "overall_status": overall_status,
        "attention_required": bool(reasons),
        "attention_reasons": reasons,
        "deadline": {
            "utc": deadline.isoformat(),
            "kst": deadline.astimezone(KST).isoformat(),
            "remaining_seconds": remaining_seconds,
            "remaining_human": human_duration(remaining_seconds),
            "passed": remaining_seconds < 0,
        },
        "sapiens": {
            "model": "Sapiens2-5B target-only",
            "alive": bool(roots["sapiens"]),
            "pid": roots["sapiens"][0]["pid"] if roots["sapiens"] else None,
            "root_process_count": len(roots["sapiens"]),
            "completed_cameras": int(pose.get("completed_camera_count", 0)),
            "total_cameras": 78,
            "completed_crops": sapiens_done,
            "total_crops": sapiens_total,
            "current_sequence": sapiens_current[0],
            "current_camera": sapiens_current[1],
            "recent_throughput_crops_per_second": safe_float(
                pose.get("recent_chunk_crops_per_second")
            ),
            "average_throughput_crops_per_second": safe_float(
                pose.get("effective_new_crops_per_second")
            ),
            "eta_utc": eta.isoformat() if eta else None,
            "eta_kst": eta.astimezone(KST).isoformat() if eta else None,
            "retry_count": int(supervisor.get("sapiens_attempt", 0) or 0),
            "error_count": sum(error.get("code") in {"CUDA_OOM", "RETRY_EXHAUSTED"} for error in errors),
        },
        "sam": {
            "mode": "B",
            "mode_c_policy": "SELECTIVE_ESCALATION_ONLY",
            "alive": bool(roots["sam"]),
            "pid": roots["sam"][0]["pid"] if roots["sam"] else None,
            "completed_cameras": int(sam_source.get("completed_camera_count", 0)),
            "total_cameras": 78,
            "completed_frames": sam_done,
            "total_frames": sam_total,
            "current_sequence": sam_current[0],
            "current_camera": sam_current[1],
            "current_camera_output_frames": current_sam_frames,
            "throughput_frames_per_second": sam_rate,
            "eta_utc": sam_eta.isoformat() if sam_eta else None,
            "eta_kst": sam_eta.astimezone(KST).isoformat() if sam_eta else None,
            "retry_count": retry_count,
            "error_count": len(sam_failures),
        },
        "triangulation": {
            "completed_sequences": int(triangulation_source.get("count", 0)),
            "total_sequences": total_sequences,
            "status_counts": triangulation_counts,
        },
        "body_fit": {
            "completed_sequences": int(body_source.get("count", 0)),
            "total_sequences": total_sequences,
            "status_counts": body_counts,
        },
        "quality_control": {
            **quality,
            "total_sequences": total_sequences,
            "freeze_ready_sequences": int(
                freeze_readiness.get("ready_sequence_count", 0) or 0
            ),
            "freeze_readiness_status_counts": dict(
                freeze_readiness.get("status_counts", {})
            ),
            "freeze_readiness_waiting_count": len(
                freeze_readiness.get("waiting", [])
            ),
            "freeze_readiness_failure_count": len(readiness_failures),
        },
        "quality_follower": {
            "alive": bool(roots["quality_follower"]),
            "pid": roots["quality_follower"][0]["pid"] if roots["quality_follower"] else None,
            "status": quality_follower_state.get("status", "UNKNOWN"),
            "updated_at": quality_follower_state.get("updated_at_utc"),
            "last_event": quality_follower_state.get("last_event"),
            "failures": quality_follower_state.get("failures", []),
        },
        "export": export,
        "supervisor": {
            "alive": bool(roots["supervisor"]),
            "pid": roots["supervisor"][0]["pid"] if roots["supervisor"] else None,
            "stage": stage,
            "active_sequence": active_sequence,
            "completed_sequence_rows": len(sequence_rows),
        },
        "handoff_monitor": {
            "alive": bool(roots["handoff_monitor"]),
            "pid": roots["handoff_monitor"][0]["pid"] if roots["handoff_monitor"] else None,
            "state_updated_at": handoff.get("updated_at_utc"),
        },
        "deadline_sentinel": {
            "alive": bool(roots["deadline_sentinel"]),
            "pid": roots["deadline_sentinel"][0]["pid"] if roots["deadline_sentinel"] else None,
            "status": deadline_state.get("status", "UNKNOWN"),
        },
        "gpu": gpu,
        "disk": disk,
        "last_progress_timestamp": last_progress.isoformat(),
        "last_event": last_event,
        "errors": errors,
        "stalled_jobs": stalled_jobs,
        "monitoring": {
            "progress_signature": signature,
            "progress_counters": counters,
            "stall_minutes": args.stall_minutes,
            "gpu_idle_minutes": args.gpu_idle_minutes,
            "state_stale_seconds": args.state_stale_seconds,
            "source_of_truth": str(args.handoff_state),
        },
    }


def cell(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def render_rich(state: dict[str, Any], console: Any | None = None) -> Any:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = console or Console()
    deadline = state["deadline"]
    attention = state["attention_required"]
    title = Text(
        f"Exercise3D  {state['overall_status']}  |  {state['current_time_kst']}  |  "
        f"deadline {deadline['remaining_human']}",
        style="bold red" if attention else "bold green",
    )
    table = Table(expand=True)
    table.add_column("Stage")
    table.add_column("Progress")
    table.add_column("Current")
    table.add_column("Rate")
    table.add_column("ETA / State")
    sapiens = state["sapiens"]
    table.add_row(
        f"Sapiens PID {sapiens['pid'] or '-'}",
        f"{sapiens['completed_cameras']}/{sapiens['total_cameras']} cams | "
        f"{sapiens['completed_crops']:,}/{sapiens['total_crops']:,} crops",
        f"{sapiens['current_sequence'] or '-'} / {sapiens['current_camera'] or '-'}",
        f"recent {cell(sapiens['recent_throughput_crops_per_second'])} | "
        f"avg {cell(sapiens['average_throughput_crops_per_second'])} crop/s",
        sapiens["eta_kst"] or "-",
    )
    sam = state["sam"]
    table.add_row(
        f"SAM Mode {sam['mode']} PID {sam['pid'] or '-'}",
        f"{sam['completed_cameras']}/{sam['total_cameras']} cams | "
        f"{sam['completed_frames']:,}/{sam['total_frames']:,} frames",
        f"{sam['current_sequence'] or '-'} / {sam['current_camera'] or '-'} "
        f"({sam['current_camera_output_frames']:,} durable)",
        f"{cell(sam['throughput_frames_per_second'])} frame/s",
        sam["eta_kst"] or "waiting",
    )
    for key, label in (
        ("triangulation", "Triangulation"),
        ("body_fit", "Body fit"),
        ("quality_control", "Quality control"),
    ):
        row = state[key]
        counts = row["status_counts"]
        current = f"PASS {counts['PASS']} REVIEW {counts['REVIEW']} FAIL {counts['FAIL']}"
        if key == "quality_control":
            follower = state.get("quality_follower", {})
            current += (
                f" | follower {follower.get('status', 'UNKNOWN')}"
                f" PID {follower.get('pid') or '-'}"
            )
        table.add_row(
            label,
            f"{row['completed_sequences']}/{row['total_sequences']} sequences",
            current,
            "-",
            (
                f"freeze-ready {row.get('freeze_ready_sequences', 0)}/"
                f"{row['total_sequences']}"
                if key == "quality_control"
                else "-"
            ),
        )
    export = state["export"]
    table.add_row(
        "Export / freeze",
        f"{export['completed_sequences']} sequences",
        export["latest_build_id"] or "-",
        "-",
        f"{export['deadline_snapshot_status']} | freeze={export['freeze_eligible']}",
    )
    gpu = state["gpu"]
    disk = state["disk"]
    telemetry_suffix = " (cached)" if not gpu.get("telemetry_fresh", True) else ""
    system = (
        f"GPU {cell(gpu.get('utilization_pct'), '%')}{telemetry_suffix} | "
        f"VRAM {cell(gpu.get('memory_used_mib'))}/{cell(gpu.get('memory_total_mib'))} MiB | "
        f"Power {cell(gpu.get('power_draw_w'))} W | Temp {cell(gpu.get('temperature_c'))} C\n"
        f"Disk free {cell(disk.get('free_gib'))} GiB | Last progress {state['last_progress_timestamp']} | "
        f"Last event {state['last_event']}"
    )
    parts: list[Any] = [Panel(title), table, Panel(system, title="System")]
    if attention:
        reasons = "\n".join(
            f"[{row['severity']}] {row['code']}: {row['message']}"
            for row in state["attention_reasons"]
        )
        parts.append(Panel(reasons, title="Attention required", border_style="red"))
    return Group(*parts)


def print_plain(state: dict[str, Any]) -> None:
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-state", type=Path, default=PROJECT_ROOT / ".runtime/handoff_state.json")
    parser.add_argument(
        "--supervisor-state",
        type=Path,
        default=PROJECT_ROOT / "outputs/runtime/autonomous_generation/autonomous_generation_state.json",
    )
    parser.add_argument(
        "--deadline-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime/deadline_snapshot_state.json",
    )
    parser.add_argument(
        "--quality-follower-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "quality_follower_state.json",
    )
    parser.add_argument(
        "--sequence-status",
        type=Path,
        default=PROJECT_ROOT / "outputs/runtime/autonomous_generation/autonomous_sequences.csv",
    )
    parser.add_argument(
        "--autonomous-runtime-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/runtime/autonomous_generation",
    )
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / ".runtime")
    parser.add_argument("--pose-root", type=Path, default=PROJECT_ROOT / "outputs/sapiens2_target_only_full")
    parser.add_argument("--sam-output-root", type=Path, default=PROJECT_ROOT / "outputs/sam_body4d_full")
    parser.add_argument("--triangulation-root", type=Path, default=PROJECT_ROOT / "outputs/triangulation_final")
    parser.add_argument("--body-fit-root", type=Path, default=PROJECT_ROOT / "outputs/body_fit_full")
    parser.add_argument("--quality-root", type=Path, default=PROJECT_ROOT / "outputs/quality_control_full")
    parser.add_argument("--export-root", type=Path, default=PROJECT_ROOT / "outputs/private_dataset_freeze")
    parser.add_argument("--disk-path", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / ".runtime/dashboard_state.json")
    parser.add_argument("--deadline-utc", default="2026-08-14T04:00:00+00:00")
    parser.add_argument("--refresh-seconds", type=float, default=10.0)
    parser.add_argument("--stall-minutes", type=float, default=60.0)
    parser.add_argument("--gpu-idle-minutes", type=float, default=10.0)
    parser.add_argument("--gpu-idle-threshold-pct", type=float, default=5.0)
    parser.add_argument("--state-stale-seconds", type=float, default=180.0)
    parser.add_argument("--eta-worsening-minutes", type=float, default=30.0)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--plain", action="store_true", help="print JSON instead of the Rich dashboard")
    parser.add_argument("--quiet", action="store_true", help="refresh state without terminal output")
    parser.add_argument("--exit-nonzero-on-attention", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.refresh_seconds <= 0
        or args.stall_minutes <= 0
        or args.gpu_idle_minutes <= 0
        or args.state_stale_seconds <= 0
        or args.minimum_free_gib <= 0
    ):
        raise RuntimeError("monitor thresholds and refresh interval must be positive")

    def snapshot() -> dict[str, Any]:
        state = build_dashboard(args)
        atomic_json(args.output, state)
        return state

    if args.once:
        state = snapshot()
        if args.quiet:
            pass
        elif args.plain:
            print_plain(state)
        else:
            try:
                from rich.console import Console

                Console().print(render_rich(state))
            except ImportError:
                print_plain(state)
        return int(args.exit_nonzero_on_attention and state["attention_required"])

    try:
        if args.quiet:
            while True:
                snapshot()
                time.sleep(args.refresh_seconds)
        elif args.plain:
            while True:
                print_plain(snapshot())
                time.sleep(args.refresh_seconds)
        else:
            try:
                from rich.live import Live

                with Live(render_rich(snapshot()), refresh_per_second=4) as live:
                    while True:
                        time.sleep(args.refresh_seconds)
                        live.update(render_rich(snapshot()), refresh=True)
            except ImportError:
                while True:
                    print_plain(snapshot())
                    time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
