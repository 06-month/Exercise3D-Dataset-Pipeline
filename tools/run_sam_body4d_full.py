#!/usr/bin/env python3
"""Run resumable primary-target SAM-Body4D Mode B one camera at a time."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("cam1", "cam2", "cam3")


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
    return parser


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
            provenance_frames = len(payload["frame_names"])
            target_valid_frames = int(payload["target_valid"].sum())
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return {"status": "INCOMPLETE", "reason": str(error)}
    mesh_count = len(list((private / "mesh_4d_individual" / "1").glob("*.ply")))
    numeric_count = len(list((private / "mhr_numeric" / "1").glob("*.npz")))
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
        "mesh_complete": mesh_count == expected_frames,
        "numeric_prior_complete": numeric_count == expected_frames,
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


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    benchmark_tool = project_root / "tools" / "benchmark_sam_body4d.py"
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
                    process = subprocess.run(command, cwd=project_root)
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


if __name__ == "__main__":
    raise SystemExit(main())
