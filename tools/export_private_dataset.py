#!/usr/bin/env python3
"""Create a versioned, self-contained private Exercise3D dataset freeze candidate.

Source RGB/video is never copied or modified.  Frame names, indices, PTS and
the immutable source inventory are included as provenance.  Every copied stage
payload is byte-identical and SHA-256 checked.  Missing dependencies remain
explicit INCOMPLETE records rather than being promoted to PASS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
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
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--sam-mode-c-review-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    return parser


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def copy_exact(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_digest = sha256(source)
    if destination.is_file() and sha256(destination) == source_digest:
        return {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "sha256": source_digest,
            "resume_skipped": True,
        }
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    copied_digest = sha256(temporary)
    if copied_digest != source_digest or temporary.stat().st_size != source.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"byte-integrity mismatch while copying {source}")
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": copied_digest,
        "resume_skipped": False,
    }


def npz_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def finite_nan_contract(
    points: np.ndarray, valid: np.ndarray
) -> tuple[bool, bool]:
    return bool(np.isfinite(points[valid]).all()), bool(np.isnan(points[~valid]).all())


def sequence_dependencies(args: argparse.Namespace, sequence: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for camera in CAMERAS:
        selection = args.selection_root.resolve() / sequence / camera
        pose = args.pose_root.resolve() / sequence / camera
        sam = args.sam_prior_root.resolve() / sequence / camera
        files[f"view/{camera}_target_selection.npz"] = selection / "target_selection.npz"
        files[f"view/{camera}_target_metadata.json"] = selection / "metadata.json"
        files[f"view/{camera}_pose_2d.npz"] = pose / "poses_2d.npz"
        files[f"view/{camera}_pose_metadata.json"] = pose / "metadata.json"
        files[f"body/{camera}_sam_body_prior.npz"] = sam / "sam_body_prior.npz"
        files[f"body/{camera}_sam_body_metadata.json"] = sam / "metadata.json"
    triangulation = args.triangulation_root.resolve() / sequence
    body = args.body_fit_root.resolve() / sequence
    files["geometry/triangulated_3d.npz"] = triangulation / "triangulated_3d.npz"
    files["geometry/canonical_3d.npz"] = triangulation / "canonical_3d.npz"
    files["geometry/metadata.json"] = triangulation / "metadata.json"
    files["body/body_fit.npz"] = body / "body_fit.npz"
    files["body/metadata.json"] = body / "metadata.json"
    files["body/mode_c_escalation.json"] = (
        args.sam_mode_c_review_root.resolve() / sequence / "mode_c_escalation.json"
    )
    return files


def validate_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    missing = [
        label
        for label, path in sequence_dependencies(args, sequence).items()
        if not path.is_file()
    ]
    if missing:
        return {"status": "INCOMPLETE", "reasons": [f"missing:{item}" for item in missing]}
    reasons = []
    source_counts = {}
    for camera in CAMERAS:
        selection = npz_arrays(
            args.selection_root.resolve() / sequence / camera / "target_selection.npz"
        )
        pose = npz_arrays(args.pose_root.resolve() / sequence / camera / "poses_2d.npz")
        sam = npz_arrays(
            args.sam_prior_root.resolve() / sequence / camera / "sam_body_prior.npz"
        )
        frame_count = len(selection["frame_index"])
        source_counts[camera] = frame_count
        if len(pose["frame_index"]) != frame_count or len(sam["source_frame_index"]) != frame_count:
            reasons.append(f"{camera}:frame_count_mismatch")
        if not np.array_equal(selection["frame_index"], pose["frame_index"]):
            reasons.append(f"{camera}:frame_index_mismatch")
        if not np.array_equal(
            selection["timestamp_pts_seconds"], pose["timestamp_pts_seconds"]
        ):
            reasons.append(f"{camera}:pose_pts_mismatch")
        if not np.array_equal(
            selection["timestamp_pts_seconds"], sam["timestamp_pts_seconds"]
        ):
            reasons.append(f"{camera}:sam_pts_mismatch")
        finite, invalid_nan = finite_nan_contract(
            pose["keypoints_xy"], pose["valid_mask"]
        )
        if not finite or not invalid_nan:
            reasons.append(f"{camera}:pose_finite_nan_contract")
        if not np.isfinite(
            sam["mhr_keypoints_local_3d"][sam["output_valid"]]
        ).all():
            reasons.append(f"{camera}:sam_nonfinite_valid")
        if np.any(sam["accepted_prior"] & ~sam["target_valid"]):
            reasons.append(f"{camera}:sam_forced_invalid_target")

    triangulated = npz_arrays(
        args.triangulation_root.resolve() / sequence / "triangulated_3d.npz"
    )
    canonical = npz_arrays(
        args.triangulation_root.resolve() / sequence / "canonical_3d.npz"
    )
    body = npz_arrays(args.body_fit_root.resolve() / sequence / "body_fit.npz")
    tri_finite, tri_nan = finite_nan_contract(
        triangulated["keypoints_3d"], triangulated["valid_mask"]
    )
    body_finite, body_nan = finite_nan_contract(
        body["keypoints_3d"], body["valid_mask"]
    )
    if not tri_finite or not tri_nan:
        reasons.append("triangulation_finite_nan_contract")
    if not body_finite or not body_nan:
        reasons.append("body_finite_nan_contract")
    if not np.array_equal(
        canonical["timestamp_pts_seconds"], body["timestamp_pts_seconds"]
    ):
        reasons.append("body_pts_mismatch")
    if not np.array_equal(canonical["joint_names"], body["joint_names"]):
        reasons.append("body_joint_convention_mismatch")
    tri_metadata = json.loads(
        (args.triangulation_root.resolve() / sequence / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    body_metadata = json.loads(
        (args.body_fit_root.resolve() / sequence / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    mode_c_metadata = json.loads(
        (
            args.sam_mode_c_review_root.resolve()
            / sequence
            / "mode_c_escalation.json"
        ).read_text(encoding="utf-8")
    )
    if not tri_metadata["qa"]["eligible_for_body_fitting"]:
        reasons.append("triangulation_not_eligible")
    if str(body_metadata["qa"]["status"]).startswith("FAIL"):
        reasons.append("body_fit_fail")
    status = "FAIL" if reasons else (
        "REVIEW"
        if body_metadata["qa"]["status"].startswith("REVIEW")
        or mode_c_metadata["status"] == "REVIEW_MODE_C_CANDIDATE"
        else "PASS"
    )
    return {
        "status": status,
        "reasons": reasons,
        "source_frame_counts": source_counts,
        "reference_frame_count": len(body["frame_index"]),
        "body_fit_status": body_metadata["qa"]["status"],
        "sam_mode_c_review_status": mode_c_metadata["status"],
        "camera_geometry_status": tri_metadata["qa"]["pose_camera_consistency_status"],
        "valid_body_joint_fraction": float(body["valid_mask"].mean()),
        "prior_only_joint_count": int((body["evidence_type"] == 3).sum()),
    }


def git_commit() -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def main() -> int:
    args = build_parser().parse_args()
    if "/" in args.build_id or args.build_id in {".", ".."}:
        raise RuntimeError("build id must be one path component")
    build_root = args.output_root.resolve() / args.build_id
    if build_root.exists() and not build_root.is_dir():
        raise RuntimeError("build output exists and is not a directory")
    provenance_sources = {
        "provenance/source_inventory.json": args.dataset_root.resolve()
        / "reports"
        / "dataset_inventory.json",
        "provenance/temporal_audit.json": args.dataset_root.resolve()
        / "reports"
        / "temporal_alignment"
        / "audit.json",
        "provenance/temporal_camera_frame_mapping.csv": args.dataset_root.resolve()
        / "reports"
        / "temporal_alignment"
        / "camera_frame_mapping.csv",
    }
    for label, source in provenance_sources.items():
        if not source.is_file():
            raise RuntimeError(f"missing global source provenance: {source}")
        copy_exact(source, build_root / label)

    sequence_rows = []
    file_manifest = []
    for sequence in args.sequences:
        validation = validate_sequence(args, sequence)
        row = {
            "sequence": sequence,
            "status": validation["status"],
            "reasons": ";".join(validation["reasons"]),
            "reference_frame_count": validation.get("reference_frame_count", 0),
            "valid_body_joint_fraction": validation.get("valid_body_joint_fraction", ""),
            "body_fit_status": validation.get("body_fit_status", ""),
            "camera_geometry_status": validation.get("camera_geometry_status", ""),
            "sam_mode_c_review_status": validation.get("sam_mode_c_review_status", ""),
        }
        sequence_rows.append(row)
        if validation["status"] in {"INCOMPLETE", "FAIL"}:
            print(json.dumps({"sequence": sequence, **validation}, ensure_ascii=False), flush=True)
            continue
        sequence_root = build_root / "sequences" / sequence
        copied = []
        for label, source in sequence_dependencies(args, sequence).items():
            result = copy_exact(source, sequence_root / label)
            record = {
                "sequence": sequence,
                "path": str((Path("sequences") / sequence / label).as_posix()),
                "bytes": result["bytes"],
                "sha256": result["sha256"],
            }
            copied.append(record)
            file_manifest.append(record)
        sequence_metadata = {
            "schema_version": 1,
            "sequence": sequence,
            "status": validation["status"],
            "source_rgb_included": False,
            "source_rgb_reference": "frame name/index/PTS only; immutable private source remains external",
            "validation": validation,
            "files": copied,
        }
        atomic_text(
            sequence_root / "sequence_manifest.json",
            json.dumps(sequence_metadata, ensure_ascii=False, indent=2) + "\n",
        )
        print(json.dumps({"sequence": sequence, **validation}, ensure_ascii=False), flush=True)

    atomic_csv(build_root / "sequence_status.csv", sequence_rows)
    manifest = {
        "schema_version": 1,
        "build_id": args.build_id,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(),
        "private_dataset": True,
        "not_ground_truth": True,
        "source_payload_modified": False,
        "source_rgb_included": False,
        "sequence_count": len(sequence_rows),
        "pass_count": sum(row["status"] == "PASS" for row in sequence_rows),
        "review_count": sum(row["status"] == "REVIEW" for row in sequence_rows),
        "fail_count": sum(row["status"] == "FAIL" for row in sequence_rows),
        "incomplete_count": sum(row["status"] == "INCOMPLETE" for row in sequence_rows),
        "freeze_eligible": all(row["status"] in {"PASS", "REVIEW"} for row in sequence_rows),
        "file_count": len(file_manifest),
        "total_payload_bytes": sum(row["bytes"] for row in file_manifest),
        "files": file_manifest,
        "status_policy": "REVIEW is retained; FAIL/INCOMPLETE is never promoted",
    }
    atomic_text(
        build_root / "dataset_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["freeze_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
