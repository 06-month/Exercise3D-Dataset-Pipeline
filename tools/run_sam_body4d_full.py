#!/usr/bin/env python3
"""Run resumable primary-target SAM-Body4D Mode B one camera at a time."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import numpy as np

try:
    from tools.consolidate_sam_body_prior import REQUIRED_PRIOR_FIELDS
except ModuleNotFoundError:
    from consolidate_sam_body_prior import REQUIRED_PRIOR_FIELDS


CAMERAS = ("cam1", "cam2", "cam3")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
COORDINATOR_SCRIPT = Path(__file__).resolve()
BENCHMARK_SCRIPT = COORDINATOR_SCRIPT.with_name("benchmark_sam_body4d.py")
PRIMARY_RUNNER_SCRIPT = COORDINATOR_SCRIPT.with_name(
    "sam_body_primary_target_runner.py"
)


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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--sam-body4d-root", type=Path, required=True)
    parser.add_argument("--body4d-python", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--cameras", type=parse_list, default=list(CAMERAS))
    parser.add_argument("--retry-failures", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--instance-lock",
        type=Path,
        default=PROJECT_ROOT / ".runtime" / "sam_body4d_full.lock",
        help="Lifetime advisory lock preventing duplicate SAM Mode B jobs.",
    )
    return parser


def acquire_instance_lock(path: Path) -> BinaryIO | None:
    """Hold a singleton lock for one full SAM Mode B coordinator lifetime."""
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


def cli_option(argv: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return str(argv[index + 1])
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def resolved_cli_path(argv: Sequence[str], name: str, cwd: Path) -> Path | None:
    raw = cli_option(argv, name)
    if raw is None:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        return candidate.resolve()
    except OSError:
        return None


def command_matches_sam_job(
    argv: Sequence[str], cwd: Path, output_root: Path
) -> bool:
    """Match a coordinator or orphan GPU child bound below one output root."""
    script_values = [
        value
        for value in argv
        if value.endswith(
            (
                "run_sam_body4d_full.py",
                "benchmark_sam_body4d.py",
                "sam_body_primary_target_runner.py",
            )
        )
    ]
    if len(script_values) != 1:
        return False
    script = Path(script_values[0])
    if not script.is_absolute():
        script = cwd / script
    try:
        script = script.resolve()
        root = output_root.resolve()
    except OSError:
        return False

    if script == COORDINATOR_SCRIPT:
        candidate = resolved_cli_path(argv, "--output-root", cwd)
        return candidate == root

    if cli_option(argv, "--mode") != "B":
        return False
    candidate = resolved_cli_path(argv, "--output-dir", cwd)
    if candidate is None:
        return False
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    if not relative.parts:
        return False
    if script == BENCHMARK_SCRIPT:
        return "--run" in argv
    if script == PRIMARY_RUNNER_SCRIPT:
        return candidate.name == "mode_b_private_output"
    return False


def matching_sam_processes(
    output_root: Path,
    *,
    proc_root: Path = Path("/proc"),
    exclude_pid: int | None = None,
) -> list[int]:
    """Find live legacy coordinators and orphan children for this output root."""
    if not proc_root.is_dir():
        raise RuntimeError("process table is unavailable; refusing SAM launch")
    try:
        directories = list(proc_root.iterdir())
    except OSError as error:
        raise RuntimeError(
            "process table cannot be enumerated; refusing SAM launch"
        ) from error
    matches: list[int] = []
    for directory in directories:
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        if pid == exclude_pid:
            continue
        try:
            status = (directory / "stat").read_text(encoding="utf-8")
            state = status.rsplit(")", 1)[1].strip().split()[0]
            if state == "Z":
                continue
            raw = (directory / "cmdline").read_bytes()
            argv = [
                value.decode("utf-8", errors="surrogateescape")
                for value in raw.split(b"\0")
                if value
            ]
            cwd = (directory / "cwd").resolve(strict=True)
        except (OSError, IndexError):
            continue
        if command_matches_sam_job(argv, cwd, output_root):
            matches.append(pid)
    return sorted(matches)


def frame_directory(dataset_root: Path, sequence: str, camera: str) -> Path:
    exercise = sequence.rsplit("_", 1)[0]
    return dataset_root / "final_frame" / exercise / sequence / camera


def read_single_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one row: {path}")
    return rows[0]


def completion_status(output_dir: Path, expected_frames: int) -> dict[str, Any]:
    benchmark_path = output_dir / "sam_body_benchmark.csv"
    profile_path = output_dir / "mode_b_profile.json"
    private = output_dir / "mode_b_private_output"
    provenance_path = private / "target_provenance.npz"
    required = [benchmark_path, profile_path, provenance_path]
    missing = [str(path.name) for path in required if not path.is_file()]
    if missing:
        return {"status": "INCOMPLETE", "reason": f"missing {','.join(missing)}"}
    try:
        benchmark = read_single_csv(benchmark_path)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        with np.load(provenance_path, allow_pickle=False) as payload:
            required_provenance = {
                "frame_names",
                "source_frame_names",
                "source_frame_indices",
                "target_bboxes_xyxy",
                "target_valid",
                "target_selection_confidence",
                "target_ambiguous",
                "no_target",
                "occlusion_risk",
                "timestamp_pts_seconds",
            }
            provenance_schema = required_provenance <= set(payload.files)
            provenance_frames = len(payload["frame_names"])
            provenance_lengths = all(
                len(payload[key]) == expected_frames
                for key in required_provenance - {"target_bboxes_xyxy"}
            ) and payload["target_bboxes_xyxy"].shape == (expected_frames, 4)
            source_indices = payload["source_frame_indices"].astype(np.int64)
            target_valid = payload["target_valid"].astype(np.bool_)
            target_ambiguous = payload["target_ambiguous"].astype(np.bool_)
            no_target = payload["no_target"].astype(np.bool_)
            target_bboxes = payload["target_bboxes_xyxy"].astype(np.float64)
            timestamps = payload["timestamp_pts_seconds"].astype(np.float64)
            confidence = payload["target_selection_confidence"].astype(np.float64)
            target_valid_frames = int(target_valid.sum())
            provenance_source_indices = np.array_equal(
                source_indices, np.arange(expected_frames, dtype=np.int64)
            )
            provenance_abstention = bool(
                not np.any(target_valid & (target_ambiguous | no_target))
            )
            valid_boxes = target_bboxes[target_valid]
            invalid_boxes = target_bboxes[~target_valid]
            provenance_bboxes = bool(
                len(valid_boxes) > 0
                and np.isfinite(valid_boxes).all()
                and np.all(valid_boxes[:, 2:] > valid_boxes[:, :2])
                and np.isnan(invalid_boxes).all()
            )
            provenance_timestamps = bool(
                np.isfinite(timestamps).all()
                and (len(timestamps) < 2 or np.all(np.diff(timestamps) > 0))
            )
            provenance_confidence = bool(
                np.isfinite(confidence).all()
                and np.all((confidence >= 0) & (confidence <= 1))
            )
            provenance_seed = bool(len(target_valid) and target_valid[0])
    except (OSError, TypeError, ValueError, KeyError, RuntimeError) as error:
        return {"status": "INCOMPLETE", "reason": str(error)}
    mesh_count = len(list((private / "mesh_4d_individual" / "1").glob("*.ply")))
    numeric_count = len(list((private / "mhr_numeric" / "1").glob("*.npz")))
    numeric_paths = sorted((private / "mhr_numeric" / "1").glob("*.npz"))
    numeric_schema_complete = len(numeric_paths) == expected_frames
    if numeric_schema_complete:
        for path in numeric_paths:
            try:
                with np.load(path, allow_pickle=False) as payload:
                    if not set(REQUIRED_PRIOR_FIELDS) <= set(payload.files):
                        numeric_schema_complete = False
                        break
            except (OSError, ValueError):
                numeric_schema_complete = False
                break
    temporary_count = len(list(private.rglob("*.tmp*")))
    checks = {
        "benchmark_pass": benchmark.get("status") == "PASS",
        "benchmark_frames": int(float(benchmark.get("frames_processed") or 0))
        == expected_frames,
        "profile_frames": int(profile.get("frames_processed", 0)) == expected_frames,
        "input_frames": int(profile.get("input_frames", 0)) == expected_frames,
        "target_seed_one": int(profile.get("target_seed_count", 0)) == 1,
        "persons_one": int(profile.get("persons_processed", 0)) == 1,
        "provenance_frames": provenance_frames == expected_frames,
        "provenance_schema": provenance_schema,
        "provenance_lengths": provenance_lengths,
        "provenance_source_indices": provenance_source_indices,
        "provenance_abstention": provenance_abstention,
        "provenance_bboxes": provenance_bboxes,
        "provenance_timestamps": provenance_timestamps,
        "provenance_confidence": provenance_confidence,
        "provenance_seed": provenance_seed,
        "mesh_complete": mesh_count == expected_frames,
        "numeric_prior_complete": numeric_count == expected_frames,
        "numeric_prior_schema_complete": numeric_schema_complete,
        "no_temporary_files": temporary_count == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "INCOMPLETE",
        "reason": "" if all(checks.values()) else ";".join(
            key for key, value in checks.items() if not value
        ),
        "expected_frames": expected_frames,
        "target_valid_frames": target_valid_frames,
        "mesh_count": mesh_count,
        "numeric_prior_count": numeric_count,
        "checks": checks,
        "elapsed_wall_seconds": float(benchmark["elapsed_wall_seconds"]),
        "peak_nvidia_vram_mib": float(benchmark["peak_nvidia_vram_mib"]),
        "gpu_utilization_mean_pct": float(benchmark["gpu_utilization_mean_pct"]),
        "power_mean_w": float(benchmark["power_mean_w"]),
    }


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key != "checks"})
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key in fields}
            for row in rows
        )
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_full(args: argparse.Namespace) -> int:
    benchmark_tool = PROJECT_ROOT / "tools" / "benchmark_sam_body4d.py"
    rows = []
    for sequence in args.sequences:
        for camera in args.cameras:
            frames = frame_directory(args.dataset_root.resolve(), sequence, camera)
            frame_count = len(list(frames.glob("*.jpg")))
            output_dir = args.output_root.resolve() / sequence / camera
            selection = (
                args.selection_root.resolve()
                / sequence
                / camera
                / "target_selection.npz"
            )
            before = completion_status(output_dir, frame_count)
            if before["status"] == "PASS" and not args.overwrite:
                result = before
                result["resume_skipped"] = True
            else:
                result = before
                for attempt in range(args.retry_failures + 1):
                    command = [
                        str(args.body4d_python.resolve()),
                        str(benchmark_tool),
                        "--mode", "B",
                        "--sam-body4d-root", str(args.sam_body4d_root.resolve()),
                        "--checkpoint-root", str(args.checkpoint_root.resolve()),
                        "--input-frames", str(frames),
                        "--target-selection", str(selection),
                        "--source-start-index", "0",
                        "--frame-count", str(frame_count),
                        "--output-dir", str(output_dir),
                        "--python", str(args.body4d_python.resolve()),
                        "--run",
                    ]
                    process = subprocess.run(command, cwd=PROJECT_ROOT)
                    result = completion_status(output_dir, frame_count)
                    result["runner_exit_code"] = process.returncode
                    result["attempt"] = attempt + 1
                    if result["status"] == "PASS":
                        break
                result["resume_skipped"] = False
            result.update({"sequence": sequence, "camera": camera, "mode": "B"})
            rows.append(result)
            atomic_csv(args.runtime_dir.resolve() / "sam_body4d_full.csv", rows)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "mode": "B",
        "camera_count": len(rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "incomplete_count": sum(row["status"] != "PASS" for row in rows),
        "frame_count": sum(int(row.get("expected_frames", 0)) for row in rows),
        "target_valid_frame_count": sum(
            int(row.get("target_valid_frames", 0)) for row in rows
        ),
        "mesh_count": sum(int(row.get("mesh_count", 0)) for row in rows),
        "numeric_prior_count": sum(
            int(row.get("numeric_prior_count", 0)) for row in rows
        ),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "REVIEW",
    }
    atomic_json(args.runtime_dir.resolve() / "sam_body4d_full_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 2


def guarded_run(args: argparse.Namespace) -> int:
    """Refuse duplicate current or legacy jobs before spawning a GPU child."""
    instance_lock = acquire_instance_lock(args.instance_lock.resolve())
    if instance_lock is None:
        print(
            json.dumps(
                {
                    "status": "DUPLICATE_SAM_MODE_B_REFUSED",
                    "reason": "instance_lock_held",
                    "pid": os.getpid(),
                }
            ),
            flush=True,
        )
        return 3
    try:
        try:
            existing = matching_sam_processes(
                args.output_root.expanduser().resolve(), exclude_pid=os.getpid()
            )
        except RuntimeError as error:
            print(
                json.dumps(
                    {
                        "status": "SAM_MODE_B_PROCESS_DISCOVERY_FAILED",
                        "reason": str(error),
                        "pid": os.getpid(),
                    }
                ),
                flush=True,
            )
            return 4
        if existing:
            print(
                json.dumps(
                    {
                        "status": "EXISTING_SAM_MODE_B_REFUSED",
                        "matching_pids": existing,
                        "pid": os.getpid(),
                    }
                ),
                flush=True,
            )
            return 3
        return run_full(args)
    finally:
        instance_lock.close()


def main() -> int:
    return guarded_run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
