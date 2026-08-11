#!/usr/bin/env python3
"""Materialize atomic, camera-level config provenance for completed inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = ("cam1", "cam2", "cam3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "status": "MISSING"}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_single_csv(path: Path) -> dict[str, str] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows[0] if len(rows) == 1 else None
    except (OSError, csv.Error):
        return None


def git_tool_identity(path: Path) -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(PROJECT_ROOT))],
            cwd=PROJECT_ROOT,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        commit = "UNKNOWN"
    return {"last_commit": commit, "sha256": file_identity(path).get("sha256", "UNKNOWN")}


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_manifest_identity(dataset_root: Path) -> dict[str, Any]:
    candidates = (
        dataset_root / "reports" / "dataset_inventory.json",
        dataset_root / "manifest.json",
    )
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    return file_identity(path)


def command_for(marker: str, handoff_state: dict[str, Any] | None) -> str:
    if handoff_state is None:
        return "UNKNOWN"
    return str(handoff_state.get("resume_commands", {}).get(marker, "UNKNOWN"))


def pose_camera_complete(output: Path) -> bool:
    metadata = read_json(output / "metadata.json")
    return bool(
        metadata is not None
        and metadata.get("qa", {}).get("status") == "PASS"
        and (output / "poses_2d.npz").is_file()
        and (output / "bboxes.npz").is_file()
        and (output / "frames.csv").is_file()
    )


def sam_camera_complete(output: Path) -> bool:
    benchmark = read_single_csv(output / "sam_body_benchmark.csv")
    profile = read_json(output / "mode_b_profile.json")
    if benchmark is None or profile is None or benchmark.get("status") != "PASS":
        return False
    try:
        expected = int(profile["input_frames"])
        return bool(
            expected > 0
            and int(profile["frames_processed"]) == expected
            and int(float(benchmark["frames_processed"])) == expected
            and len(list((output / "mode_b_private_output" / "mesh_4d_individual" / "1").glob("*.ply"))) == expected
            and len(list((output / "mode_b_private_output" / "mhr_numeric" / "1").glob("*.npz"))) == expected
        )
    except (KeyError, ValueError):
        return False


def materialize_pose(
    dataset_root: Path,
    selection_root: Path,
    pose_root: Path,
    sequence: str,
    camera: str,
    handoff_state: dict[str, Any] | None,
) -> bool:
    output = pose_root / sequence / camera
    provenance_path = output / "run_provenance.json"
    if provenance_path.is_file() or not pose_camera_complete(output):
        return False
    metadata = read_json(output / "metadata.json") or {}
    config = {
        "model": metadata.get("model", "facebook/sapiens2-pose-5b"),
        "checkpoint": json.loads(
            (PROJECT_ROOT / "configs" / "sapiens2_pose_5b_environment.json").read_text(encoding="utf-8")
        )["pose_model"],
        "batch_size": metadata.get("batch_size"),
        "chunk_size": metadata.get("chunk_size"),
        "precision": metadata.get("precision"),
        "flip_test": metadata.get("flip_test"),
        "thresholds": "frozen upstream target selector metadata",
        "source_manifest": source_manifest_identity(dataset_root),
        "target_selection": file_identity(selection_root / sequence / camera / "target_selection.npz"),
        "camera_geometry_version": "PHASE5_BACKGROUND_BA_RECOVERED; not directly consumed by 2D teacher",
        "temporal_metadata_version": "PHASE2_PTS_AUDIT via target selection PTS",
    }
    payload = {
        "schema_version": 1,
        "status": "PASS_PROVENANCE",
        "materialized_at_utc": utc_now(),
        "output_created_at_utc": metadata.get("created_at_utc"),
        "phase": "PHASE6_TARGET_ONLY_SAPIENS2_5B",
        "sequence": sequence,
        "camera": camera,
        "configuration": config,
        "configuration_sha256": canonical_hash(config),
        "tool": git_tool_identity(PROJECT_ROOT / "tools" / "sapiens2_target_pipeline.py"),
        "exact_resume_command": command_for("sapiens2_target_pipeline.py", handoff_state),
        "completion_gate": "metadata QA PASS + poses/bboxes/frames present; full schema validator remains authoritative",
    }
    atomic_json(provenance_path, payload)
    return True


def materialize_sam(
    dataset_root: Path,
    selection_root: Path,
    sam_root: Path,
    sequence: str,
    camera: str,
    handoff_state: dict[str, Any] | None,
) -> bool:
    output = sam_root / sequence / camera
    provenance_path = output / "run_provenance.json"
    if provenance_path.is_file() or not sam_camera_complete(output):
        return False
    benchmark = read_single_csv(output / "sam_body_benchmark.csv") or {}
    profile = read_json(output / "mode_b_profile.json") or {}
    checkpoint_manifest = PROJECT_ROOT / "metadata" / "results" / "sam_body4d_checkpoint_integrity.csv"
    config = {
        "model": "SAM-Body4D",
        "implementation_revision": benchmark.get("repository_revision"),
        "mode": "B",
        "completion_enabled": False,
        "batch_size": "official upstream internal batching",
        "thresholds": "primary target bbox from frozen selector; no all-person initialization",
        "checkpoint_manifest": file_identity(checkpoint_manifest),
        "source_manifest": source_manifest_identity(dataset_root),
        "target_selection": file_identity(selection_root / sequence / camera / "target_selection.npz"),
        "camera_geometry_version": "PHASE5_BACKGROUND_BA_RECOVERED; consumed downstream, not by Mode B",
        "temporal_metadata_version": "PHASE2_PTS_AUDIT via target provenance",
        "input_frames": profile.get("input_frames"),
    }
    payload = {
        "schema_version": 1,
        "status": "PASS_PROVENANCE",
        "materialized_at_utc": utc_now(),
        "output_created_at_utc": benchmark.get("created_at_utc"),
        "phase": "PHASE8_SAM_BODY4D_MODE_B",
        "sequence": sequence,
        "camera": camera,
        "configuration": config,
        "configuration_sha256": canonical_hash(config),
        "tool": git_tool_identity(PROJECT_ROOT / "tools" / "benchmark_sam_body4d.py"),
        "exact_resume_command": command_for("run_sam_body4d_full.py", handoff_state),
        "completion_gate": "benchmark/profile PASS + exact mesh/numeric counts; full runner required-field validator authoritative",
    }
    atomic_json(provenance_path, payload)
    return True


def materialize_all(args: argparse.Namespace) -> dict[str, int]:
    handoff_state = read_json(args.handoff_state.resolve())
    counts = {"pose_created": 0, "sam_created": 0}
    for sequence in args.sequences:
        for camera in CAMERAS:
            counts["pose_created"] += int(
                materialize_pose(
                    args.dataset_root.resolve(),
                    args.selection_root.resolve(),
                    args.pose_root.resolve(),
                    sequence,
                    camera,
                    handoff_state,
                )
            )
            counts["sam_created"] += int(
                materialize_sam(
                    args.dataset_root.resolve(),
                    args.selection_root.resolve(),
                    args.sam_output_root.resolve(),
                    sequence,
                    camera,
                    handoff_state,
                )
            )
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--sam-output-root", type=Path, required=True)
    parser.add_argument("--handoff-state", type=Path, required=True)
    parser.add_argument("--sequences", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.sequences = [item for item in args.sequences.split(",") if item]
    print(json.dumps(materialize_all(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
