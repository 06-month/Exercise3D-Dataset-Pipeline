#!/usr/bin/env python3
"""Consolidate per-frame Mode B MHR outputs into a provenance-safe camera prior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("cam1", "cam2", "cam3")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PRIOR_FIELDS = (
    "bbox",
    "focal_length",
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_cam_t",
    "pred_pose_raw",
    "global_rot",
    "body_pose_params",
    "hand_pose_params",
    "scale_params",
    "shape_params",
    "expr_params",
    "pred_joint_coords",
    "pred_global_rots",
    "mhr_model_params",
)
REQUIRED_CONSOLIDATED_FIELDS = {
    "frame_name",
    "source_frame_name",
    "source_frame_index",
    "timestamp_pts_seconds",
    "output_valid",
    "accepted_prior",
    "target_valid",
    "target_selection_confidence",
    "target_ambiguous",
    "no_target",
    "occlusion_risk",
    "canonical_joint_names",
    "canonical_local_3d",
    "target_bbox_xyxy",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--cameras", type=parse_list, default=list(CAMERAS))
    parser.add_argument(
        "--canonical-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "mhr70_canonical_joints.json",
    )
    return parser


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_mapping(path: Path) -> dict[str, Any]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    indices = [int(row["source_index"]) for row in mapping["direct"]]
    names = [str(row["canonical"]) for row in mapping["direct"]]
    if len(indices) != len(set(indices)) or len(names) != len(set(names)):
        raise RuntimeError("MHR canonical mapping has duplicate index or name")
    if min(indices) < 0 or max(indices) >= 70:
        raise RuntimeError("MHR canonical mapping index is outside MHR70")
    return mapping


def canonical_from_mhr(
    keypoints: np.ndarray, mapping: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    direct_indices = np.asarray(
        [int(row["source_index"]) for row in mapping["direct"]], dtype=np.int32
    )
    names = [str(row["canonical"]) for row in mapping["direct"]]
    canonical = keypoints[:, direct_indices].copy()
    for row in mapping["derived"]:
        left = names.index(row["inputs"][0])
        right = names.index(row["inputs"][1])
        derived = (canonical[:, left] + canonical[:, right]) * 0.5
        canonical = np.concatenate([canonical, derived[:, None]], axis=1)
        names.append(str(row["canonical"]))
    return canonical, names


def robust_parameter_summary(array: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    selected = array[valid]
    if not len(selected):
        return {"valid_frames": 0, "median_l2": None, "mad_l2": None}
    median = np.median(selected, axis=0)
    distance = np.linalg.norm(selected - median, axis=-1)
    return {
        "valid_frames": len(selected),
        "median_l2": float(np.linalg.norm(median)),
        "mad_l2": float(np.median(np.abs(distance - np.median(distance)))),
    }


def consecutive_delta(array: np.ndarray, valid: np.ndarray) -> dict[str, float | None]:
    usable = valid[1:] & valid[:-1]
    if not usable.any():
        return {"median_l2": None, "p95_l2": None}
    delta = np.linalg.norm(array[1:][usable] - array[:-1][usable], axis=-1)
    return {
        "median_l2": float(np.median(delta)),
        "p95_l2": float(np.percentile(delta, 95)),
    }


def source_dependency_signature(sam_camera_dir: Path) -> str:
    """Bind a consolidated prior to the current provenance/numeric inventory."""
    private = sam_camera_dir / "mode_b_private_output"
    paths = [private / "target_provenance.npz"]
    paths.extend(sorted((private / "mhr_numeric" / "1").glob("*.npz")))
    rows: list[tuple[str, int, int, int]] = []
    for index, path in enumerate(paths):
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("SAM prior dependency is missing or symlinked")
        stat = path.stat()
        label = "target_provenance.npz" if index == 0 else f"numeric/{path.name}"
        rows.append((label, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_existing_prior(
    output_dir: Path,
    provenance: dict[str, np.ndarray],
    sequence: str,
    camera: str,
    mapping: dict[str, Any],
    dependency_signature: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Validate a source-bound PASS prior before allowing resume skip."""
    archive_path = output_dir / "sam_body_prior.npz"
    metadata_path = output_dir / "metadata.json"
    frames_path = output_dir / "frames.csv"
    if not archive_path.is_file() or not metadata_path.is_file() or not frames_path.is_file():
        return False, None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        qa = metadata["qa"]
        if (
            metadata.get("stage") != "SAM_BODY4D_MODE_B_PRIOR_CONSOLIDATION"
            or metadata.get("sequence") != sequence
            or metadata.get("camera") != camera
            or metadata.get("source_dependency_signature") != dependency_signature
            or metadata.get("canonical_mapping") != mapping
            or qa.get("status") != "PASS"
        ):
            return False, None
        with np.load(archive_path, allow_pickle=False) as payload:
            if not REQUIRED_CONSOLIDATED_FIELDS <= set(payload.files):
                return False, None
            frame_count = len(provenance["frame_names"])
            output_valid = payload["output_valid"].astype(np.bool_)
            target_valid = provenance["target_valid"].astype(np.bool_)
            accepted = payload["accepted_prior"].astype(np.bool_)
            comparisons = (
                np.array_equal(payload["frame_name"].astype(str), provenance["frame_names"].astype(str)),
                np.array_equal(
                    payload["source_frame_name"].astype(str),
                    provenance["source_frame_names"].astype(str),
                ),
                np.array_equal(
                    payload["source_frame_index"].astype(np.int32),
                    provenance["source_frame_indices"].astype(np.int32),
                ),
                np.array_equal(
                    payload["timestamp_pts_seconds"].astype(np.float64),
                    provenance["timestamp_pts_seconds"].astype(np.float64),
                ),
                np.array_equal(payload["target_valid"].astype(np.bool_), target_valid),
                np.array_equal(
                    payload["target_ambiguous"].astype(np.bool_),
                    provenance["target_ambiguous"].astype(np.bool_),
                ),
                np.array_equal(
                    payload["no_target"].astype(np.bool_),
                    provenance["no_target"].astype(np.bool_),
                ),
                np.array_equal(
                    payload["occlusion_risk"].astype(np.bool_),
                    provenance["occlusion_risk"].astype(np.bool_),
                ),
                np.array_equal(
                    payload["target_selection_confidence"].astype(np.float32),
                    provenance["target_selection_confidence"].astype(np.float32),
                ),
                np.array_equal(
                    payload["target_bbox_xyxy"].astype(np.float32),
                    provenance["target_bboxes_xyxy"].astype(np.float32),
                    equal_nan=True,
                ),
            )
            if (
                len(output_valid) != frame_count
                or not output_valid.all()
                or not np.array_equal(accepted, target_valid)
                or not all(comparisons)
                or not np.isfinite(payload["canonical_local_3d"][output_valid]).all()
            ):
                return False, None
        with frames_path.open(newline="", encoding="utf-8") as handle:
            frame_rows = list(csv.DictReader(handle))
        if (
            len(frame_rows) != frame_count
            or int(qa.get("frame_count", -1)) != frame_count
            or int(qa.get("output_valid_count", -1)) != frame_count
            or int(qa.get("accepted_prior_count", -1)) != int(target_valid.sum())
            or not bool(qa.get("finite_valid_payload"))
        ):
            return False, None
        return True, qa
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False, None


def consolidate_camera(
    sam_camera_dir: Path,
    output_dir: Path,
    sequence: str,
    camera: str,
    mapping: dict[str, Any],
) -> dict[str, Any]:
    private = sam_camera_dir / "mode_b_private_output"
    provenance_path = private / "target_provenance.npz"
    numeric_dir = private / "mhr_numeric" / "1"
    with np.load(provenance_path, allow_pickle=False) as payload:
        provenance = {key: payload[key].copy() for key in payload.files}
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
    missing_provenance = required_provenance - set(provenance)
    if missing_provenance:
        raise RuntimeError(f"missing SAM target provenance: {sorted(missing_provenance)}")
    dependency_signature = source_dependency_signature(sam_camera_dir)
    existing_valid, existing_qa = validate_existing_prior(
        output_dir,
        provenance,
        sequence,
        camera,
        mapping,
        dependency_signature,
    )
    if existing_valid and existing_qa is not None:
        return {**existing_qa, "resume_skipped": True}
    frame_names = provenance["frame_names"].astype(str)
    frame_count = len(frame_names)
    first_path = numeric_dir / f"{Path(frame_names[0]).stem}.npz"
    with np.load(first_path, allow_pickle=False) as first:
        missing = set(REQUIRED_PRIOR_FIELDS) - set(first.files)
        if missing:
            raise RuntimeError(f"missing compact MHR fields: {sorted(missing)}")
        shapes = {key: first[key].shape for key in REQUIRED_PRIOR_FIELDS}
    arrays = {
        key: np.full((frame_count, *shape), np.nan, dtype=np.float32)
        for key, shape in shapes.items()
    }
    output_valid = np.zeros(frame_count, dtype=np.bool_)
    failure_reason = np.full(frame_count, "MISSING_MHR_PRIOR", dtype="<U48")
    for frame, name in enumerate(frame_names):
        path = numeric_dir / f"{Path(name).stem}.npz"
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                for key in REQUIRED_PRIOR_FIELDS:
                    value = payload[key]
                    if value.shape != shapes[key] or not np.isfinite(value).all():
                        raise ValueError(f"invalid {key}")
                    arrays[key][frame] = value
            output_valid[frame] = True
            failure_reason[frame] = ""
        except (OSError, ValueError, KeyError):
            failure_reason[frame] = "INVALID_MHR_PRIOR"
    target_valid = provenance["target_valid"].astype(np.bool_)
    accepted = output_valid & target_valid
    camera_keypoints = arrays["pred_keypoints_3d"] + arrays["pred_cam_t"][:, None]
    canonical_local, canonical_names = canonical_from_mhr(
        arrays["pred_keypoints_3d"], mapping
    )
    canonical_camera, _ = canonical_from_mhr(camera_keypoints, mapping)
    atomic_npz(
        output_dir / "sam_body_prior.npz",
        frame_name=frame_names,
        source_frame_name=provenance["source_frame_names"].astype(str),
        source_frame_index=provenance["source_frame_indices"].astype(np.int32),
        timestamp_pts_seconds=provenance["timestamp_pts_seconds"].astype(np.float64),
        output_valid=output_valid,
        accepted_prior=accepted,
        target_valid=target_valid,
        target_selection_confidence=provenance["target_selection_confidence"].astype(np.float32),
        target_ambiguous=provenance["target_ambiguous"].astype(np.bool_),
        no_target=provenance["no_target"].astype(np.bool_),
        occlusion_risk=provenance["occlusion_risk"].astype(np.bool_),
        failure_reason=failure_reason,
        mhr_keypoints_local_3d=arrays["pred_keypoints_3d"],
        mhr_keypoints_camera_3d=camera_keypoints,
        mhr_keypoints_2d=arrays["pred_keypoints_2d"],
        canonical_joint_names=np.asarray(canonical_names),
        canonical_local_3d=canonical_local,
        canonical_camera_3d=canonical_camera,
        target_bbox_xyxy=provenance["target_bboxes_xyxy"].astype(np.float32),
        predicted_bbox_xyxy=arrays["bbox"],
        pred_cam_t=arrays["pred_cam_t"],
        focal_length=arrays["focal_length"],
        pred_pose_raw=arrays["pred_pose_raw"],
        global_rot=arrays["global_rot"],
        body_pose_params=arrays["body_pose_params"],
        hand_pose_params=arrays["hand_pose_params"],
        scale_params=arrays["scale_params"],
        shape_params=arrays["shape_params"],
        expression_params=arrays["expr_params"],
        mhr_joint_coords=arrays["pred_joint_coords"],
        mhr_joint_global_rotations=arrays["pred_global_rots"],
        mhr_model_params=arrays["mhr_model_params"],
    )
    rows = [
        {
            "frame_index": frame,
            "source_frame_index": int(provenance["source_frame_indices"][frame]),
            "source_frame_name": str(provenance["source_frame_names"][frame]),
            "output_valid": bool(output_valid[frame]),
            "accepted_prior": bool(accepted[frame]),
            "target_ambiguous": bool(provenance["target_ambiguous"][frame]),
            "no_target": bool(provenance["no_target"][frame]),
            "occlusion_risk": bool(provenance["occlusion_risk"][frame]),
            "failure_reason": str(failure_reason[frame]),
        }
        for frame in range(frame_count)
    ]
    atomic_csv(output_dir / "frames.csv", rows)
    status = "PASS" if output_valid.all() else "REVIEW_INCOMPLETE"
    qa = {
        "sequence": sequence,
        "camera": camera,
        "frame_count": frame_count,
        "output_valid_count": int(output_valid.sum()),
        "accepted_prior_count": int(accepted.sum()),
        "ambiguous_or_no_target_count": int((~target_valid).sum()),
        "occlusion_risk_count": int(provenance["occlusion_risk"].sum()),
        "finite_valid_payload": bool(
            all(np.isfinite(array[output_valid]).all() for array in arrays.values())
        ),
        "shape_consistency": robust_parameter_summary(arrays["shape_params"], accepted),
        "scale_consistency": robust_parameter_summary(arrays["scale_params"], accepted),
        "body_pose_temporal_delta": consecutive_delta(arrays["body_pose_params"], accepted),
        "status": status,
    }
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": sequence,
        "camera": camera,
        "stage": "SAM_BODY4D_MODE_B_PRIOR_CONSOLIDATION",
        "not_ground_truth": True,
        "source_mode": "B (completion disabled)",
        "source_dependency_signature": dependency_signature,
        "source_numeric_dir": str(numeric_dir),
        "coordinate_semantics": {
            "mhr_keypoints_local_3d": "MHR body prior before pred_cam_t",
            "mhr_keypoints_camera_3d": "local prior plus monocular pred_cam_t",
            "canonical_camera_3d": "MHR70 mapping; still monocular learned prior",
        },
        "canonical_mapping": mapping,
        "qa": qa,
    }
    atomic_text(
        output_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    return {**qa, "resume_skipped": False}


def main() -> int:
    args = build_parser().parse_args()
    mapping = load_mapping(args.canonical_config.resolve())
    rows = []
    for sequence in args.sequences:
        for camera in args.cameras:
            row = consolidate_camera(
                args.sam_root.resolve() / sequence / camera,
                args.output_root.resolve() / sequence / camera,
                sequence,
                camera,
                mapping,
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "camera_count": len(rows),
        "frame_count": sum(row["frame_count"] for row in rows),
        "output_valid_count": sum(row["output_valid_count"] for row in rows),
        "accepted_prior_count": sum(row["accepted_prior_count"] for row in rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "review_count": sum(row["status"] != "PASS" for row in rows),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "REVIEW",
    }
    atomic_csv(args.runtime_dir.resolve() / "sam_body_prior_qa.csv", rows)
    atomic_text(
        args.runtime_dir.resolve() / "sam_body_prior_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
