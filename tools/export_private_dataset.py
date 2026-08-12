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
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

try:
    from tools.build_pseudolabel_quality import (
        build_sequence_quality,
        validate_quality_output,
    )
except ModuleNotFoundError:
    from build_pseudolabel_quality import build_sequence_quality, validate_quality_output


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
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--sam-mode-c-review-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument(
        "--quality-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "quality_control_full",
    )
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
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
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
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def validate_path_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise RuntimeError(f"{label} must be one non-empty path component")


def validate_staging_root(staging_root: Path, expected_build_id: str) -> Path:
    """Resolve only the dedicated hidden staging directory for one build id."""
    validate_path_component(expected_build_id, "build id")
    if staging_root.name != f".{expected_build_id}.inprogress":
        raise RuntimeError("refusing to prune a path that is not the expected staging root")
    if staging_root.is_symlink():
        raise RuntimeError("staging root must not be a symlink")
    root = staging_root.resolve()
    if root == root.parent or root.parent == Path(root.anchor):
        raise RuntimeError("refusing broad staging root")
    if root.exists() and root.is_mount():
        raise RuntimeError("staging root must not be a mount point")
    return root


def remove_staging_symlinks(
    staging_root: Path, expected_build_id: str
) -> list[str]:
    """Remove nested symlinks before resumable copies can traverse stale paths."""
    root = validate_staging_root(staging_root, expected_build_id)
    if not root.exists():
        return []
    if not root.is_dir():
        raise RuntimeError("staging root exists and is not a directory")
    removed: list[str] = []
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in list(directories):
            path = base / name
            if path.is_mount():
                raise RuntimeError(f"nested mount point in staging root: {path}")
            if path.is_symlink():
                removed.append(path.relative_to(root).as_posix())
                path.unlink()
                directories.remove(name)
        for name in files:
            path = base / name
            if path.is_symlink():
                removed.append(path.relative_to(root).as_posix())
                path.unlink()
    return sorted(removed)


def prune_staging_tree(
    staging_root: Path,
    expected_build_id: str,
    expected_files: set[str],
) -> list[str]:
    """Delete only unlisted artifacts inside a validated resumable staging root."""
    root = validate_staging_root(staging_root, expected_build_id)
    if not root.is_dir():
        raise RuntimeError("staging root does not exist or is not a directory")
    normalized: set[str] = set()
    for value in expected_files:
        relative = PurePosixPath(value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise RuntimeError(f"unsafe expected staging path: {value}")
        normalized.add(relative.as_posix())

    for directory, directories, _ in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in list(directories):
            path = base / name
            if path.is_mount():
                raise RuntimeError(f"nested mount point in staging root: {path}")
            if path.is_symlink():
                directories.remove(name)

    removed: list[str] = []
    for directory, directories, files in os.walk(root, topdown=False, followlinks=False):
        base = Path(directory)
        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or relative not in normalized:
                path.unlink()
                removed.append(relative)
        for name in directories:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_mount():
                raise RuntimeError(f"nested mount point in staging root: {path}")
            if path.is_symlink():
                path.unlink()
                removed.append(relative)
                continue
            try:
                path.rmdir()
            except OSError:
                pass
    return sorted(removed)


def verify_frozen_build(
    build_root: Path, expected_build_id: str | None = None
) -> dict[str, Any]:
    """Verify a published/staged build without mutating it."""
    if build_root.is_symlink():
        return {
            "valid": False,
            "errors": ["build_root_symlink"],
            "manifest": None,
        }
    root = build_root.resolve()
    if root.exists() and root.is_mount():
        return {
            "valid": False,
            "errors": ["build_root_mount_point"],
            "manifest": None,
        }
    errors: list[str] = []
    manifest = read_json(root / "dataset_manifest.json")
    if manifest is None:
        return {"valid": False, "errors": ["missing_or_invalid:dataset_manifest.json"], "manifest": None}
    if expected_build_id is not None and manifest.get("build_id") != expected_build_id:
        errors.append("build_id_mismatch")
    if manifest.get("private_dataset") is not True:
        errors.append("private_dataset_flag_invalid")
    if manifest.get("source_rgb_included") is not False:
        errors.append("source_rgb_policy_invalid")
    if manifest.get("source_payload_modified") is not False:
        errors.append("source_mutation_policy_invalid")
    commit = manifest.get("git_commit")
    if commit is not None and not re.fullmatch(r"[0-9a-f]{40,64}", str(commit)):
        errors.append("git_commit_invalid")
    if (
        "git_worktree_dirty" in manifest
        and not isinstance(manifest.get("git_worktree_dirty"), bool)
    ):
        errors.append("git_worktree_dirty_invalid")
    for key in ("git_status_sha256", "git_diff_sha256"):
        if key in manifest and not re.fullmatch(
            r"[0-9a-f]{64}", str(manifest.get(key))
        ):
            errors.append(f"{key}_invalid")

    records = manifest.get("files")
    if not isinstance(records, list):
        records = []
        errors.append("files_manifest_invalid")
    listed_paths: set[str] = set()
    manifest_records: dict[str, dict[str, Any]] = {}
    verified_bytes = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"file_record_invalid:{index}")
            continue
        value = record.get("path")
        if not isinstance(value, str):
            errors.append(f"file_path_invalid:{index}")
            continue
        relative = PurePosixPath(value)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            errors.append(f"unsafe_file_path:{value}")
            continue
        normalized = relative.as_posix()
        if normalized in listed_paths:
            errors.append(f"duplicate_file_path:{normalized}")
            continue
        listed_paths.add(normalized)
        manifest_records[normalized] = record
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            errors.append(f"missing_or_symlink:{normalized}")
            continue
        try:
            expected_bytes = int(record["bytes"])
            expected_digest = str(record["sha256"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"file_identity_invalid:{normalized}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            errors.append(f"byte_mismatch:{normalized}")
            continue
        if sha256(path) != expected_digest:
            errors.append(f"sha256_mismatch:{normalized}")
            continue
        verified_bytes += actual_bytes

    allowed_tree_files = listed_paths | {"dataset_manifest.json", "sequence_status.csv"}
    actual_tree_files: set[str] = set()
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        for name in list(directories):
            path = base / name
            if path.is_mount():
                relative = path.relative_to(root).as_posix()
                errors.append(f"unexpected_mount_point:{relative}")
                directories.remove(name)
                continue
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                errors.append(f"unexpected_symlink:{relative}")
                directories.remove(name)
        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                errors.append(f"unexpected_symlink:{relative}")
            else:
                actual_tree_files.add(relative)
    for relative in sorted(actual_tree_files - allowed_tree_files):
        errors.append(f"unlisted_file:{relative}")

    try:
        expected_file_count = int(manifest.get("file_count"))
        expected_total_bytes = int(manifest.get("total_payload_bytes"))
    except (TypeError, ValueError):
        expected_file_count = -1
        expected_total_bytes = -1
        errors.append("file_totals_invalid")
    if expected_file_count != len(records):
        errors.append("file_count_mismatch")
    if expected_total_bytes != verified_bytes:
        errors.append("total_payload_bytes_mismatch")

    status_path = root / "sequence_status.csv"
    try:
        with status_path.open(newline="", encoding="utf-8") as handle:
            status_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        status_rows = []
        errors.append("missing_or_invalid:sequence_status.csv")
    sequence_ids = [str(row.get("sequence", "")) for row in status_rows]
    if any(not sequence for sequence in sequence_ids) or len(set(sequence_ids)) != len(sequence_ids):
        errors.append("sequence_status_identity_invalid")
    allowed = {"PASS", "REVIEW", "FAIL", "INCOMPLETE"}
    statuses = [str(row.get("status", "")) for row in status_rows]
    if any(status not in allowed for status in statuses):
        errors.append("sequence_status_value_invalid")
    calculated = {
        "sequence_count": len(status_rows),
        "pass_count": statuses.count("PASS"),
        "review_count": statuses.count("REVIEW"),
        "fail_count": statuses.count("FAIL"),
        "incomplete_count": statuses.count("INCOMPLETE"),
    }
    for key, value in calculated.items():
        try:
            if int(manifest.get(key)) != value:
                errors.append(f"{key}_mismatch")
        except (TypeError, ValueError):
            errors.append(f"{key}_invalid")
    freeze_eligible = bool(status_rows) and all(
        status in {"PASS", "REVIEW"} for status in statuses
    )
    if manifest.get("freeze_eligible") is not freeze_eligible:
        errors.append("freeze_eligible_mismatch")
    status_by_sequence = {
        sequence: status for sequence, status in zip(sequence_ids, statuses)
    }
    for path, record in manifest_records.items():
        sequence = str(record.get("sequence", ""))
        path_sequence = (
            path.split("/", 2)[1]
            if path.startswith("sequences/") and len(path.split("/", 2)) >= 3
            else ""
        )
        if sequence:
            if sequence not in status_by_sequence:
                errors.append(f"file_sequence_unknown:{path}")
            elif status_by_sequence[sequence] not in {"PASS", "REVIEW"}:
                errors.append(f"file_for_noncomplete_sequence:{path}")
            if path_sequence != sequence:
                errors.append(f"file_sequence_path_mismatch:{path}")
        elif path_sequence:
            errors.append(f"file_sequence_owner_missing:{path}")
    for row in status_rows:
        if row.get("status") not in {"PASS", "REVIEW"}:
            continue
        sequence = str(row["sequence"])
        sequence_manifest = f"sequences/{sequence}/sequence_manifest.json"
        if sequence_manifest not in listed_paths:
            errors.append(f"sequence_manifest_unlisted:{sequence}")
            continue
        metadata = read_json(root / sequence_manifest)
        if (
            metadata is None
            or metadata.get("sequence") != sequence
            or metadata.get("status") != row.get("status")
        ):
            errors.append(f"sequence_manifest_status_mismatch:{sequence}")
            continue
        declared_files = metadata.get("files")
        if not isinstance(declared_files, list):
            errors.append(f"sequence_manifest_files_invalid:{sequence}")
            continue
        declared_paths: set[str] = set()
        for index, declared in enumerate(declared_files):
            if not isinstance(declared, dict) or not isinstance(declared.get("path"), str):
                errors.append(f"sequence_file_record_invalid:{sequence}:{index}")
                continue
            path = str(declared["path"])
            if path in declared_paths:
                errors.append(f"sequence_file_duplicate:{sequence}:{path}")
                continue
            declared_paths.add(path)
            global_record = manifest_records.get(path)
            if global_record is None:
                errors.append(f"sequence_file_unlisted:{sequence}:{path}")
                continue
            for key in ("sequence", "bytes", "sha256"):
                if declared.get(key) != global_record.get(key):
                    errors.append(f"sequence_file_identity_mismatch:{sequence}:{path}:{key}")
        expected_sequence_paths = {
            path
            for path, record in manifest_records.items()
            if str(record.get("sequence", "")) == sequence
            and path != sequence_manifest
        }
        if declared_paths != expected_sequence_paths:
            errors.append(f"sequence_file_set_mismatch:{sequence}")
    return {
        "valid": not errors,
        "errors": errors,
        "manifest": manifest,
        "verified_file_count": len(records),
        "verified_payload_bytes": verified_bytes,
        "sequence_count": len(status_rows),
    }


def publish_staged_build(
    staging_root: Path, final_root: Path, expected_build_id: str
) -> dict[str, Any]:
    integrity = verify_frozen_build(staging_root, expected_build_id)
    if not integrity["valid"]:
        raise RuntimeError(
            "staged build failed integrity verification: "
            + ";".join(integrity["errors"])
        )
    if final_root.exists():
        raise RuntimeError("final build appeared during staging; refusing to overwrite it")
    os.replace(staging_root, final_root)
    return integrity


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
        files[f"view/{camera}_pose_run_provenance.json"] = pose / "run_provenance.json"
        files[f"body/{camera}_sam_body_prior.npz"] = sam / "sam_body_prior.npz"
        files[f"body/{camera}_sam_body_metadata.json"] = sam / "metadata.json"
        files[f"body/{camera}_sam_inference_run_provenance.json"] = (
            sam / "inference_run_provenance.json"
        )
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
    quality = args.quality_root.resolve() / sequence
    files["quality/quality_vector.npz"] = quality / "quality_vector.npz"
    files["quality/metadata.json"] = quality / "metadata.json"
    return files


def ensure_quality_output(args: argparse.Namespace, sequence: str) -> dict[str, Any] | None:
    body_path = args.body_fit_root.resolve() / sequence / "body_fit.npz"
    mode_c_path = (
        args.sam_mode_c_review_root.resolve()
        / sequence
        / "mode_c_escalation.json"
    )
    if not body_path.is_file() or not mode_c_path.is_file():
        return None
    quality_args = argparse.Namespace(
        selection_root=args.selection_root,
        pose_root=args.pose_root,
        triangulation_root=args.triangulation_root,
        sam_prior_root=args.sam_prior_root,
        sam_mode_c_review_root=args.sam_mode_c_review_root,
        body_fit_root=args.body_fit_root,
        output_root=args.quality_root,
    )
    return build_sequence_quality(quality_args, sequence)


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
    quality_ok, quality_reasons, quality_metadata = validate_quality_output(
        args.quality_root.resolve() / sequence,
        args.body_fit_root.resolve() / sequence / "body_fit.npz",
    )
    if not quality_ok:
        reasons.extend(f"quality:{reason}" for reason in quality_reasons)
    elif str(quality_metadata["qa"]["sequence_status"]).startswith("FAIL"):
        reasons.append("quality_control_fail")
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


def git_provenance() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal", "-z"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    status_bytes = status.stdout if status.returncode == 0 else b""
    diff_bytes = diff.stdout if diff.returncode == 0 else b""
    return {
        "git_commit": head.stdout.strip() if head.returncode == 0 else None,
        "git_worktree_dirty": bool(status_bytes) if status.returncode == 0 else None,
        "git_status_sha256": (
            hashlib.sha256(status_bytes).hexdigest() if status.returncode == 0 else None
        ),
        "git_diff_sha256": (
            hashlib.sha256(diff_bytes).hexdigest() if diff.returncode == 0 else None
        ),
    }


def main() -> int:
    args = build_parser().parse_args()
    validate_path_component(args.build_id, "build id")
    for sequence in args.sequences:
        validate_path_component(sequence, "sequence id")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    final_root = output_root / args.build_id
    final_manifest = final_root / "dataset_manifest.json"
    if final_root.exists() and not final_root.is_dir():
        raise RuntimeError("build output exists and is not a directory")
    if final_manifest.is_file():
        existing = verify_frozen_build(final_root, args.build_id)
        if not existing["valid"]:
            raise RuntimeError(
                "immutable build exists but failed integrity verification: "
                + ";".join(existing["errors"])
            )
        manifest = existing["manifest"]
        print(
            json.dumps(
                {
                    "build_id": args.build_id,
                    "status": "IMMUTABLE_BUILD_REUSED",
                    "freeze_eligible": manifest["freeze_eligible"],
                    "verified_file_count": existing["verified_file_count"],
                    "verified_payload_bytes": existing["verified_payload_bytes"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0 if manifest["freeze_eligible"] else 2
    if final_root.exists():
        raise RuntimeError(
            "final build directory exists without a manifest; preserve it and use a new build id"
        )
    build_root = output_root / f".{args.build_id}.inprogress"
    if build_root.exists() and not build_root.is_dir():
        raise RuntimeError("staging build output exists and is not a directory")
    removed_symlinks = remove_staging_symlinks(build_root, args.build_id)
    if removed_symlinks:
        print(
            json.dumps(
                {
                    "build_id": args.build_id,
                    "status": "STAGING_SYMLINKS_REMOVED",
                    "removed_count": len(removed_symlinks),
                }
            ),
            flush=True,
        )
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
    file_manifest = []
    for label, source in provenance_sources.items():
        if not source.is_file():
            raise RuntimeError(f"missing global source provenance: {source}")
        result = copy_exact(source, build_root / label)
        file_manifest.append(
            {
                "sequence": "",
                "path": label,
                "bytes": result["bytes"],
                "sha256": result["sha256"],
            }
        )

    sequence_rows = []
    for sequence in args.sequences:
        try:
            ensure_quality_output(args, sequence)
        except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as error:
            print(
                json.dumps(
                    {
                        "sequence": sequence,
                        "stage": "PHASE11_QUALITY_CONTROL",
                        "status": "INCOMPLETE",
                        "reason": str(error),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
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
            "subject_id": None,
            "subject_mapping_status": "SUBJECT_MAPPING_UNAVAILABLE",
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
        sequence_manifest_path = sequence_root / "sequence_manifest.json"
        file_manifest.append(
            {
                "sequence": sequence,
                "path": str(
                    (Path("sequences") / sequence / "sequence_manifest.json").as_posix()
                ),
                "bytes": sequence_manifest_path.stat().st_size,
                "sha256": sha256(sequence_manifest_path),
            }
        )
        print(json.dumps({"sequence": sequence, **validation}, ensure_ascii=False), flush=True)

    atomic_csv(build_root / "sequence_status.csv", sequence_rows)
    manifest = {
        "schema_version": 1,
        "build_id": args.build_id,
        "created_at_utc": utc_now(),
        **git_provenance(),
        "private_dataset": True,
        "not_ground_truth": True,
        "declared_subject_count": 3,
        "subject_mapping_status": "SUBJECT_MAPPING_UNAVAILABLE",
        "subject_id_policy": "null; no appearance/shape-based cross-sequence inference",
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
    expected_tree_files = {
        "dataset_manifest.json",
        "sequence_status.csv",
        *(str(row["path"]) for row in file_manifest),
    }
    removed_stale = prune_staging_tree(
        build_root,
        args.build_id,
        expected_tree_files,
    )
    if removed_stale:
        print(
            json.dumps(
                {
                    "build_id": args.build_id,
                    "status": "STALE_STAGING_ARTIFACTS_REMOVED",
                    "removed_count": len(removed_stale),
                }
            ),
            flush=True,
        )
    publish_staged_build(build_root, final_root, args.build_id)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["freeze_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
