#!/usr/bin/env python3
"""Run Phase 7 as soon as each three-camera Sapiens2 sequence is complete.

The immutable Phase 5 camera result is always evaluated first.  A camera
candidate conditioned on pose observations is generated only for an explicit
NO_GO result, validated on the tool's deterministic held-out split, and kept
separate from Phase 5.  Final output records which camera source was used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("cam1", "cam2", "cam3")
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
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--initial-output-root", type=Path, required=True)
    parser.add_argument("--recovery-camera-root", type=Path, required=True)
    parser.add_argument("--final-output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def frame_directory(dataset_root: Path, sequence: str, camera: str) -> Path:
    exercise = sequence.rsplit("_", 1)[0]
    return dataset_root / "final_frame" / exercise / sequence / camera


def pose_camera_ready(
    dataset_root: Path, pose_root: Path, sequence: str, camera: str
) -> bool:
    camera_dir = pose_root / sequence / camera
    metadata_path = camera_dir / "metadata.json"
    pose_path = camera_dir / "poses_2d.npz"
    if not metadata_path.is_file() or not pose_path.is_file():
        return False
    expected = len(list(frame_directory(dataset_root, sequence, camera).glob("*.jpg")))
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(pose_path, allow_pickle=False) as payload:
            frame_count = len(payload["frame_index"])
            coordinates = payload["keypoints_xy"]
            confidence = payload["confidence"]
        return bool(
            expected > 0
            and frame_count == expected
            and coordinates.shape == (expected, 308, 2)
            and confidence.shape == (expected, 308)
            and metadata["qa"]["status"] == "PASS"
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def sequence_ready(dataset_root: Path, pose_root: Path, sequence: str) -> bool:
    return all(
        pose_camera_ready(dataset_root, pose_root, sequence, camera)
        for camera in CAMERAS
    )


def read_triangulation(output_root: Path, sequence: str) -> dict[str, Any] | None:
    metadata_path = output_root / sequence / "metadata.json"
    canonical_path = output_root / sequence / "canonical_3d.npz"
    if not metadata_path.is_file() or not canonical_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(canonical_path, allow_pickle=False) as payload:
            finite = np.isfinite(payload["keypoints_3d"][payload["valid_mask"]]).all()
        qa = metadata["qa"]
        if qa["schema_status"] != "PASS" or not finite:
            return None
        return metadata
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def recovery_accepted(root: Path, sequence: str) -> bool:
    path = root / sequence / "validation.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            payload["eligible_for_triangulation"]
            and payload["fit_heldout_overlap_count"] == 0
            and payload["recovered_pose_camera_status_heldout"]
            != "NO_GO_TRIANGULATION"
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def run_checked(command: list[str]) -> None:
    process = subprocess.run(command, cwd=PROJECT_ROOT)
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed with {process.returncode}: {' '.join(command)}"
        )


def triangulate_command(
    args: argparse.Namespace, sequence: str, camera_root: Path, output_root: Path, label: str
) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "triangulate_sapiens2.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--pose-root", str(args.pose_root.resolve()),
        "--camera-root", str(camera_root.resolve()),
        "--output-root", str(output_root.resolve()),
        "--runtime-dir", str((args.runtime_dir.resolve() / label / sequence)),
        "--sequences", sequence,
    ]


def recover_command(args: argparse.Namespace, sequence: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "recover_cameras_from_pose_observations.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--pose-root", str(args.pose_root.resolve()),
        "--camera-root", str(args.camera_root.resolve()),
        "--output-root", str(args.recovery_camera_root.resolve()),
        "--sequences", sequence,
    ]


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


def process_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    initial = read_triangulation(args.initial_output_root.resolve(), sequence)
    if initial is None:
        run_checked(
            triangulate_command(
                args,
                sequence,
                args.camera_root,
                args.initial_output_root,
                "initial",
            )
        )
        initial = read_triangulation(args.initial_output_root.resolve(), sequence)
    if initial is None:
        raise RuntimeError(f"invalid initial triangulation output: {sequence}")
    initial_status = initial["qa"]["pose_camera_consistency_status"]
    use_recovery = initial_status == "NO_GO_TRIANGULATION"
    if use_recovery and not recovery_accepted(args.recovery_camera_root.resolve(), sequence):
        run_checked(recover_command(args, sequence))
    recovery_ok = recovery_accepted(args.recovery_camera_root.resolve(), sequence)
    selected_camera_root = (
        args.recovery_camera_root if use_recovery and recovery_ok else args.camera_root
    )
    camera_source = (
        "REVIEW_OBSERVATION_CONDITIONED" if use_recovery and recovery_ok else "PHASE5_BACKGROUND_BA"
    )
    final = read_triangulation(args.final_output_root.resolve(), sequence)
    final_source = None if final is None else final.get("camera_uncertainty_provenance", {}).get(
        "observation_conditioned_acceptance_status"
    )
    must_rerun = final is None or (
        camera_source == "REVIEW_OBSERVATION_CONDITIONED"
        and final_source != "REVIEW_OBSERVATION_CONDITIONED"
    )
    if must_rerun:
        run_checked(
            triangulate_command(
                args,
                sequence,
                selected_camera_root,
                args.final_output_root,
                "final",
            )
        )
        final = read_triangulation(args.final_output_root.resolve(), sequence)
    if final is None:
        raise RuntimeError(f"invalid final triangulation output: {sequence}")
    return {
        "sequence": sequence,
        "completed_at_utc": utc_now(),
        "initial_pose_camera_status": initial_status,
        "camera_source": camera_source,
        "recovery_accepted": recovery_ok if use_recovery else False,
        "final_pose_camera_status": final["qa"]["pose_camera_consistency_status"],
        "final_schema_status": final["qa"]["schema_status"],
        "eligible_for_body_fitting": final["qa"]["eligible_for_body_fitting"],
        "frame_count": final["qa"]["frame_count"],
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_seconds <= 0:
        raise RuntimeError("poll interval must be positive")
    rows: list[dict[str, Any]] = []
    pending = list(args.sequences)
    while pending:
        progressed = False
        for sequence in list(pending):
            if not sequence_ready(
                args.dataset_root.resolve(), args.pose_root.resolve(), sequence
            ):
                continue
            row = process_sequence(args, sequence)
            rows.append(row)
            pending.remove(sequence)
            progressed = True
            atomic_csv(args.runtime_dir.resolve() / "phase7_streaming.csv", rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        state = {
            "schema_version": 1,
            "updated_at_utc": utc_now(),
            "sequence_count": len(args.sequences),
            "completed_count": len(rows),
            "pending_sequences": pending,
            "schema_pass_count": sum(row["final_schema_status"] == "PASS" for row in rows),
            "body_fitting_eligible_count": sum(row["eligible_for_body_fitting"] for row in rows),
            "status": "PASS" if not pending else "IN_PROGRESS",
        }
        atomic_text(
            args.runtime_dir.resolve() / "phase7_streaming_summary.json",
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        )
        if pending and not progressed:
            time.sleep(args.poll_seconds)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
