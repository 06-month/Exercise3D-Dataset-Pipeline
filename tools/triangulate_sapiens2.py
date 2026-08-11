#!/usr/bin/env python3
"""Timestamp-aware weighted multi-view triangulation for target-only Sapiens2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument(
        "--canonical-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sapiens2_canonical_joints.json",
    )
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--huber-scale-px", type=float, default=10.0)
    parser.add_argument("--max-bracket-gap-seconds", type=float, default=0.050)
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
    fields = list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def frame_dir(dataset_root: Path, sequence: str, camera: str) -> Path:
    exercise = sequence.rsplit("_", 1)[0]
    return dataset_root / "final_frame" / exercise / sequence / camera


def first_image_shape(path: Path) -> tuple[int, int]:
    images = sorted(path.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"no source frames: {path}")
    image = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode source frame: {images[0]}")
    return int(image.shape[0]), int(image.shape[1])


def temporal_models(report_root: Path, sequence: str) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in read_csv(report_root / "pair_summary.csv")
        if row["set_id"] == sequence
    ]
    if len(rows) != 3:
        raise RuntimeError(f"missing Phase 2 temporal models: {sequence}")
    return {
        row["pair_id"]: {
            "classification": row["classification"],
            "representative_offset_ms": float(
                row["representative_frame_pts_offset_ms"]
            ),
            "drift_ms_per_sec": float(row["fused_drift_ms_per_sec"])
            if row.get("fused_drift_ms_per_sec")
            else 0.0,
            "duration_sec": float(row["duration_sec"]),
            "timing_uncertainty_ms": float(row["frame_pts_offset_p95_abs_ms"]),
        }
        for row in rows
    }


def corrected_target_time(timestamp: float, model: dict[str, Any]) -> float:
    offset = model["representative_offset_ms"]
    if model["classification"] == "CLOCK_DRIFT_DETECTED":
        offset += model["drift_ms_per_sec"] * (
            timestamp - model["duration_sec"] * 0.5
        )
    return timestamp + offset / 1000.0


def interpolate_observations(
    timestamps: np.ndarray,
    xy: np.ndarray,
    confidence: np.ndarray,
    present: np.ndarray,
    targets: np.ndarray,
    max_gap: float,
) -> dict[str, np.ndarray]:
    frame_count, joint_count = len(targets), xy.shape[1]
    output_xy = np.full((frame_count, joint_count, 2), np.nan, dtype=np.float32)
    output_confidence = np.full((frame_count, joint_count), np.nan, dtype=np.float32)
    lower_out = np.full(frame_count, -1, dtype=np.int32)
    upper_out = np.full(frame_count, -1, dtype=np.int32)
    alpha_out = np.full(frame_count, np.nan, dtype=np.float32)
    pairing_error_ms = np.full(frame_count, np.nan, dtype=np.float32)
    interpolated = np.zeros(frame_count, dtype=np.bool_)
    for output_index, target in enumerate(targets):
        upper = int(np.searchsorted(timestamps, target, side="left"))
        lower = max(0, upper - 1)
        upper = min(upper, len(timestamps) - 1)
        nearest = min((lower, upper), key=lambda index: abs(timestamps[index] - target))
        lower_out[output_index] = lower
        upper_out[output_index] = upper
        span = float(timestamps[upper] - timestamps[lower])
        can_interpolate = (
            lower != upper
            and span > 0
            and span <= max_gap
            and bool(present[lower])
            and bool(present[upper])
            and timestamps[lower] <= target <= timestamps[upper]
        )
        if can_interpolate:
            alpha = float((target - timestamps[lower]) / span)
            valid = (
                np.isfinite(xy[lower]).all(axis=-1)
                & np.isfinite(xy[upper]).all(axis=-1)
                & np.isfinite(confidence[lower])
                & np.isfinite(confidence[upper])
            )
            output_xy[output_index, valid] = (
                (1.0 - alpha) * xy[lower, valid] + alpha * xy[upper, valid]
            )
            output_confidence[output_index, valid] = np.minimum(
                confidence[lower, valid], confidence[upper, valid]
            )
            alpha_out[output_index] = alpha
            pairing_error_ms[output_index] = 0.0
            interpolated[output_index] = True
        elif bool(present[nearest]):
            output_xy[output_index] = xy[nearest]
            output_confidence[output_index] = confidence[nearest]
            lower_out[output_index] = nearest
            upper_out[output_index] = nearest
            alpha_out[output_index] = 0.0
            pairing_error_ms[output_index] = abs(
                float(timestamps[nearest] - target) * 1000.0
            )
    return {
        "xy": output_xy,
        "confidence": output_confidence,
        "lower_index": lower_out,
        "upper_index": upper_out,
        "alpha": alpha_out,
        "pairing_error_ms": pairing_error_ms,
        "interpolated": interpolated,
    }


def weighted_dlt(
    observations: np.ndarray,
    projections: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    rows = []
    for point, projection, weight in zip(observations, projections, weights):
        scale = math.sqrt(max(float(weight), 1e-8))
        rows.append((point[0] * projection[2] - projection[0]) * scale)
        rows.append((point[1] * projection[2] - projection[1]) * scale)
    matrix = np.asarray(rows, dtype=np.float64)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    homogeneous = vt[-1]
    if abs(float(homogeneous[3])) < 1e-12:
        return np.full(3, np.nan), 0.0
    conditioning = float(singular[-2] / singular[0]) if len(singular) >= 2 else 0.0
    return homogeneous[:3] / homogeneous[3], conditioning


def reproject(point: np.ndarray, projections: np.ndarray) -> np.ndarray:
    homogeneous = np.append(point, 1.0)
    projected = projections @ homogeneous
    return projected[:, :2] / projected[:, 2:3]


def ray_angles(
    observations: np.ndarray,
    intrinsics: np.ndarray,
    rotations: np.ndarray,
) -> np.ndarray:
    rays = []
    for point, intrinsic, rotation in zip(observations, intrinsics, rotations):
        ray_camera = np.linalg.inv(intrinsic) @ np.asarray(
            [point[0], point[1], 1.0]
        )
        ray_world = rotation.T @ ray_camera
        rays.append(ray_world / np.linalg.norm(ray_world))
    angles = []
    for left in range(len(rays)):
        for right in range(left + 1, len(rays)):
            cosine = float(np.clip(np.dot(rays[left], rays[right]), -1.0, 1.0))
            angle = math.degrees(math.acos(abs(cosine)))
            angles.append(min(angle, 180.0 - angle))
    return np.asarray(angles, dtype=np.float32)


def triangulate_joint(
    observations: np.ndarray,
    confidence: np.ndarray,
    projections: np.ndarray,
    intrinsics: np.ndarray,
    rotations: np.ndarray,
    extrinsics: np.ndarray,
    min_confidence: float,
    huber_scale_px: float,
) -> dict[str, Any]:
    usable = np.isfinite(observations).all(axis=-1) & (confidence >= min_confidence)
    indices = np.flatnonzero(usable)
    result = {
        "point": np.full(3, np.nan, dtype=np.float32),
        "valid": False,
        "support": len(indices),
        "reprojection": np.full(len(observations), np.nan, dtype=np.float32),
        "min_ray_angle": np.nan,
        "conditioning": np.nan,
        "cheirality": False,
    }
    if len(indices) < 2:
        return result
    selected_xy = observations[indices]
    selected_projection = projections[indices]
    weights = confidence[indices].astype(np.float64)
    point, conditioning = weighted_dlt(selected_xy, selected_projection, weights)
    if not np.isfinite(point).all():
        return result
    for _ in range(2):
        errors = np.linalg.norm(
            reproject(point, selected_projection) - selected_xy, axis=-1
        )
        robust = np.minimum(1.0, huber_scale_px / np.maximum(errors, 1e-6))
        point, conditioning = weighted_dlt(
            selected_xy, selected_projection, weights * robust
        )
    projected = reproject(point, selected_projection)
    errors = np.linalg.norm(projected - selected_xy, axis=-1)
    depths = np.asarray(
        [
            (extrinsics[index, :3, :3] @ point + extrinsics[index, :3, 3])[2]
            for index in indices
        ]
    )
    angles = ray_angles(
        selected_xy, intrinsics[indices], rotations[indices]
    )
    result["point"] = point.astype(np.float32)
    result["reprojection"][indices] = errors.astype(np.float32)
    result["min_ray_angle"] = float(np.min(angles)) if len(angles) else np.nan
    result["conditioning"] = conditioning
    result["cheirality"] = bool(np.all(depths > 0))
    result["valid"] = bool(result["cheirality"] and np.isfinite(errors).all())
    return result


def percentile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, q)) if len(finite) else None


def pose_camera_consistency_status(
    median_px: float | None,
    p90_px: float | None,
    camera_acceptance: str,
    huber_scale_px: float,
) -> str:
    """Conservative gate tied to the declared robust-loss pixel scale.

    This is an observation-vs-camera consistency gate, not an independent
    camera accuracy estimate.  A NO_GO result keeps the raw proposal for
    diagnosis but excludes it from body fitting and pseudo-label export.
    """
    if median_px is None or p90_px is None:
        return "NO_GO_TRIANGULATION"
    if median_px > 2.0 * huber_scale_px or p90_px > 10.0 * huber_scale_px:
        return "NO_GO_TRIANGULATION"
    if (
        camera_acceptance != "PASS"
        or median_px > huber_scale_px
        or p90_px > 3.0 * huber_scale_px
    ):
        return "REVIEW_POSE_CAMERA_CONSISTENCY"
    return "PASS"


def load_canonical(path: Path, source_names: Sequence[str]) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for row in config["direct"]:
        index = int(row["source_index"])
        if index >= len(source_names) or source_names[index] != row["source_name"]:
            raise RuntimeError(f"canonical mapping mismatch: {row}")
    return config


def sequence_vggt_metadata(dataset_root: Path, sequence: str) -> dict[str, Any]:
    candidates = list(
        (dataset_root / "outputs" / "vggt").glob(f"*/*/{sequence}/metadata.json")
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected one VGGT metadata file for {sequence}")
    return json.loads(candidates[0].read_text(encoding="utf-8"))


def run_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    started = time.perf_counter()
    dataset_root = args.dataset_root.expanduser().resolve()
    pose_root = args.pose_root.expanduser().resolve()
    camera_root = args.camera_root.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / sequence
    pose_data: dict[str, dict[str, Any]] = {}
    source_names: list[str] | None = None
    image_shapes = {}
    for camera in CAMERAS:
        camera_dir = pose_root / sequence / camera
        metadata = json.loads((camera_dir / "metadata.json").read_text(encoding="utf-8"))
        names = list(metadata["keypoint_names"])
        if source_names is None:
            source_names = names
        elif names != source_names:
            raise RuntimeError(f"keypoint convention differs: {sequence}/{camera}")
        with np.load(camera_dir / "poses_2d.npz", allow_pickle=False) as payload:
            pose_data[camera] = {key: payload[key].copy() for key in payload.files}
        image_shapes[camera] = first_image_shape(frame_dir(dataset_root, sequence, camera))
    assert source_names is not None
    canonical = load_canonical(args.canonical_config.expanduser().resolve(), source_names)
    camera_payload = json.loads(
        (camera_root / sequence / "cameras_refined.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (camera_root / sequence / "validation.json").read_text(encoding="utf-8")
    )
    vggt = sequence_vggt_metadata(dataset_root, sequence)
    model_height = int(vggt["sequence_status"]["model_height"])
    model_width = int(vggt["sequence_status"]["model_width"])
    intrinsics = []
    rotations = []
    extrinsics = []
    projections = []
    scaled_camera_payload = {}
    for camera in CAMERAS:
        camera_item = camera_payload["cameras"][camera]
        intrinsic = np.asarray(camera_item["intrinsic"], dtype=np.float64)
        height, width = image_shapes[camera]
        intrinsic[0] *= width / model_width
        intrinsic[1] *= height / model_height
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :4] = np.asarray(
            camera_item["extrinsic_world_to_camera"], dtype=np.float64
        )
        intrinsics.append(intrinsic)
        rotations.append(extrinsic[:3, :3])
        extrinsics.append(extrinsic)
        projections.append(intrinsic @ extrinsic[:3])
        scaled_camera_payload[camera] = {
            "intrinsic_working_frame": intrinsic.tolist(),
            "extrinsic_world_to_camera": extrinsic[:3].tolist(),
            "working_frame_hw": [height, width],
            "source_intrinsic_canvas_hw": [model_height, model_width],
        }
    intrinsics_array = np.stack(intrinsics)
    rotations_array = np.stack(rotations)
    extrinsics_array = np.stack(extrinsics)
    projections_array = np.stack(projections)
    models = temporal_models(
        dataset_root / "reports" / "temporal_alignment", sequence
    )
    reference_timestamps = pose_data["cam1"]["timestamp_pts_seconds"].astype(
        np.float64
    )
    frame_count = len(reference_timestamps)
    joint_count = len(source_names)
    aligned_xy = np.full((frame_count, 3, joint_count, 2), np.nan, dtype=np.float32)
    aligned_confidence = np.full((frame_count, 3, joint_count), np.nan, dtype=np.float32)
    source_lower = np.full((frame_count, 3), -1, dtype=np.int32)
    source_upper = np.full((frame_count, 3), -1, dtype=np.int32)
    interpolation_alpha = np.full((frame_count, 3), np.nan, dtype=np.float32)
    pairing_error_ms = np.full((frame_count, 3), np.nan, dtype=np.float32)
    timing_uncertainty = np.zeros((frame_count, 3), dtype=np.float32)
    interpolation_used = np.zeros((frame_count, 3), dtype=np.bool_)
    for camera_index, camera in enumerate(CAMERAS):
        payload = pose_data[camera]
        if camera == "cam1":
            targets = reference_timestamps
            timing_uncertainty[:, camera_index] = 0.0
        else:
            model = models[f"cam1-{camera}"]
            targets = np.asarray(
                [corrected_target_time(value, model) for value in reference_timestamps]
            )
            timing_uncertainty[:, camera_index] = model["timing_uncertainty_ms"]
        aligned = interpolate_observations(
            payload["timestamp_pts_seconds"].astype(np.float64),
            payload["keypoints_xy"].astype(np.float32),
            payload["confidence"].astype(np.float32),
            payload["target_present"].astype(np.bool_),
            targets,
            args.max_bracket_gap_seconds,
        )
        aligned_xy[:, camera_index] = aligned["xy"]
        aligned_confidence[:, camera_index] = aligned["confidence"]
        source_lower[:, camera_index] = aligned["lower_index"]
        source_upper[:, camera_index] = aligned["upper_index"]
        interpolation_alpha[:, camera_index] = aligned["alpha"]
        pairing_error_ms[:, camera_index] = aligned["pairing_error_ms"]
        interpolation_used[:, camera_index] = aligned["interpolated"]

    points = np.full((frame_count, joint_count, 3), np.nan, dtype=np.float32)
    valid = np.zeros((frame_count, joint_count), dtype=np.bool_)
    support = np.zeros((frame_count, joint_count), dtype=np.uint8)
    reprojection = np.full((frame_count, joint_count, 3), np.nan, dtype=np.float32)
    ray_angle = np.full((frame_count, joint_count), np.nan, dtype=np.float32)
    conditioning = np.full((frame_count, joint_count), np.nan, dtype=np.float32)
    cheirality_failure = np.zeros((frame_count, joint_count), dtype=np.bool_)
    quality_score = np.full((frame_count, joint_count), np.nan, dtype=np.float32)
    for frame in range(frame_count):
        for joint in range(joint_count):
            result = triangulate_joint(
                aligned_xy[frame, :, joint],
                aligned_confidence[frame, :, joint],
                projections_array,
                intrinsics_array,
                rotations_array,
                extrinsics_array,
                args.min_confidence,
                args.huber_scale_px,
            )
            support[frame, joint] = result["support"]
            reprojection[frame, joint] = result["reprojection"]
            ray_angle[frame, joint] = result["min_ray_angle"]
            conditioning[frame, joint] = result["conditioning"]
            cheirality_failure[frame, joint] = (
                result["support"] >= 2 and not result["cheirality"]
            )
            if not result["valid"]:
                continue
            points[frame, joint] = result["point"]
            valid[frame, joint] = True
            used = np.isfinite(result["reprojection"])
            mean_confidence = float(np.mean(aligned_confidence[frame, used, joint]))
            median_error = float(np.median(result["reprojection"][used]))
            timing_ms = float(np.max(timing_uncertainty[frame, used]))
            quality_score[frame, joint] = (
                mean_confidence
                * (result["support"] / 3.0)
                * math.exp(-median_error / args.huber_scale_px)
                * min(1.0, float(result["min_ray_angle"]) / 5.0)
                * math.exp(-timing_ms / (1000.0 / 30.0))
            )

    direct_indices = np.asarray(
        [int(row["source_index"]) for row in canonical["direct"]], dtype=np.int32
    )
    canonical_names = [row["canonical"] for row in canonical["direct"]]
    canonical_points = points[:, direct_indices].copy()
    canonical_valid = valid[:, direct_indices].copy()
    canonical_quality = quality_score[:, direct_indices].copy()
    for row in canonical["derived"]:
        left = canonical_names.index(row["inputs"][0])
        right = canonical_names.index(row["inputs"][1])
        derived_valid = canonical_valid[:, left] & canonical_valid[:, right]
        derived = np.full((frame_count, 3), np.nan, dtype=np.float32)
        derived[derived_valid] = (
            canonical_points[derived_valid, left]
            + canonical_points[derived_valid, right]
        ) * 0.5
        derived_quality = np.full(frame_count, np.nan, dtype=np.float32)
        derived_quality[derived_valid] = np.minimum(
            canonical_quality[derived_valid, left],
            canonical_quality[derived_valid, right],
        )
        canonical_names.append(row["canonical"])
        canonical_points = np.concatenate(
            [canonical_points, derived[:, None]], axis=1
        )
        canonical_valid = np.concatenate(
            [canonical_valid, derived_valid[:, None]], axis=1
        )
        canonical_quality = np.concatenate(
            [canonical_quality, derived_quality[:, None]], axis=1
        )

    reprojection_count = np.isfinite(reprojection).sum(axis=-1)
    mean_reprojection = np.divide(
        np.nansum(reprojection, axis=-1),
        reprojection_count,
        out=np.full(reprojection_count.shape, np.nan, dtype=np.float32),
        where=reprojection_count > 0,
    )
    canonical_source_reprojection = reprojection[:, direct_indices]
    canonical_median = percentile(canonical_source_reprojection, 50)
    canonical_p90 = percentile(canonical_source_reprojection, 90)
    camera_acceptance = validation["ba_acceptance_status"]
    consistency_status = pose_camera_consistency_status(
        canonical_median,
        canonical_p90,
        camera_acceptance,
        args.huber_scale_px,
    )
    qa = {
        "sequence": sequence,
        "frame_count": frame_count,
        "teacher_joint_count": joint_count,
        "canonical_joint_count": len(canonical_names),
        "valid_teacher_joint_fraction": float(valid.mean()),
        "valid_canonical_joint_fraction": float(canonical_valid.mean()),
        "three_view_joint_fraction": float((support == 3).mean()),
        "two_view_joint_fraction": float((support == 2).mean()),
        "insufficient_view_joint_fraction": float((support < 2).mean()),
        "reprojection_median_px": percentile(mean_reprojection, 50),
        "reprojection_p90_px": percentile(mean_reprojection, 90),
        "reprojection_p95_px": percentile(mean_reprojection, 95),
        "canonical_source_reprojection_median_px": canonical_median,
        "canonical_source_reprojection_p90_px": canonical_p90,
        "canonical_source_reprojection_p95_px": percentile(
            canonical_source_reprojection, 95
        ),
        "canonical_source_per_camera_reprojection_median_px": {
            camera: percentile(canonical_source_reprojection[:, :, index], 50)
            for index, camera in enumerate(CAMERAS)
        },
        "canonical_source_per_camera_reprojection_p90_px": {
            camera: percentile(canonical_source_reprojection[:, :, index], 90)
            for index, camera in enumerate(CAMERAS)
        },
        "min_ray_angle_p05_deg": percentile(ray_angle[valid], 5),
        "pairing_error_p95_ms": percentile(pairing_error_ms, 95),
        "timing_uncertainty_max_ms": float(np.max(timing_uncertainty)),
        "cheirality_failure_count": int(cheirality_failure.sum()),
        "finite_valid_points": bool(np.isfinite(points[valid]).all()),
        "invalid_points_are_nan": bool(np.isnan(points[~valid]).all()),
        "camera_acceptance": camera_acceptance,
        "pose_camera_consistency_status": consistency_status,
        "eligible_for_body_fitting": consistency_status != "NO_GO_TRIANGULATION",
    }
    qa["schema_status"] = "PASS" if (
        qa["finite_valid_points"] and qa["invalid_points_are_nan"]
    ) else "FAIL"
    qa["quality_status"] = consistency_status
    atomic_npz(
        output_dir / "triangulated_3d.npz",
        frame_index=np.arange(frame_count, dtype=np.int32),
        timestamp_pts_seconds=reference_timestamps,
        keypoints_3d=points,
        valid_mask=valid,
        supporting_views=support,
        per_view_reprojection_px=reprojection,
        min_ray_angle_deg=ray_angle,
        dlt_conditioning=conditioning,
        quality_score=quality_score,
        source_confidence=aligned_confidence,
        source_lower_frame_index=source_lower,
        source_upper_frame_index=source_upper,
        interpolation_alpha=interpolation_alpha,
        pairing_error_ms=pairing_error_ms,
        timing_uncertainty_ms=timing_uncertainty,
        interpolation_used=interpolation_used,
    )
    atomic_npz(
        output_dir / "canonical_3d.npz",
        frame_index=np.arange(frame_count, dtype=np.int32),
        timestamp_pts_seconds=reference_timestamps,
        joint_names=np.asarray(canonical_names),
        keypoints_3d=canonical_points,
        valid_mask=canonical_valid,
        quality_score=canonical_quality,
    )
    frame_rows = []
    for frame in range(frame_count):
        frame_rows.append(
            {
                "frame_index": frame,
                "timestamp_pts_seconds": f"{reference_timestamps[frame]:.9f}",
                "valid_canonical_joints": int(canonical_valid[frame].sum()),
                "valid_teacher_joints": int(valid[frame].sum()),
                "mean_reprojection_px": f"{np.nanmean(mean_reprojection[frame]):.6f}"
                if np.isfinite(mean_reprojection[frame]).any()
                else "",
                "canonical_source_mean_reprojection_px": f"{np.nanmean(mean_reprojection[frame, direct_indices]):.6f}"
                if np.isfinite(mean_reprojection[frame, direct_indices]).any()
                else "",
                "minimum_ray_angle_deg": f"{np.nanmin(ray_angle[frame]):.6f}"
                if np.isfinite(ray_angle[frame]).any()
                else "",
                "camera_acceptance": validation["ba_acceptance_status"],
            }
        )
    atomic_csv(output_dir / "frames.csv", frame_rows)
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": sequence,
        "stage": "TIMESTAMP_AWARE_WEIGHTED_TRIANGULATION",
        "not_ground_truth": True,
        "world_gauge": camera_payload.get(
            "not_metric", True
        ) and "sequence-local arbitrary scale; cam1 identity gauge",
        "source_pose": "Sapiens2-5B target-only, official flip-test",
        "source_joint_names": source_names,
        "canonical_joint_names": canonical_names,
        "camera_geometry": scaled_camera_payload,
        "camera_uncertainty_provenance": validation,
        "temporal_models": models,
        "parameters": {
            "min_confidence": args.min_confidence,
            "huber_scale_px": args.huber_scale_px,
            "max_bracket_gap_seconds": args.max_bracket_gap_seconds,
            "reference_camera": "cam1",
            "rgb_interpolation_performed": False,
            "trajectory_interpolation": "linear 2D only when both brackets are valid",
            "quality_score_semantics": "heuristic QA score, not calibrated probability",
            "pose_camera_gate": {
                "basis": "canonical source-joint reprojection; multiples of declared Huber scale",
                "pass": "median <= 1x and p90 <= 3x Huber scale, plus camera PASS",
                "review": "within NO_GO bounds but outside PASS or camera REVIEW",
                "no_go": "median > 2x or p90 > 10x Huber scale",
                "semantics": "consistency check only; not independent camera accuracy",
            },
        },
        "qa": qa,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_text(
        output_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    return {**qa, "elapsed_seconds": metadata["elapsed_seconds"]}


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        raise RuntimeError("minimum confidence must be in [0,1]")
    if args.huber_scale_px <= 0 or args.max_bracket_gap_seconds <= 0:
        raise RuntimeError("robust scale and bracket gap must be positive")
    runtime = args.runtime_dir.expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence in args.sequences:
        row = run_sequence(args, sequence)
        rows.append(row)
        print(json.dumps(row), flush=True)
    atomic_csv(runtime / "triangulation_qa.csv", rows)
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence_count": len(rows),
        "frame_count": int(sum(row["frame_count"] for row in rows)),
        "schema_pass_count": int(sum(row["schema_status"] == "PASS" for row in rows)),
        "schema_fail_count": int(sum(row["schema_status"] == "FAIL" for row in rows)),
        "pose_camera_pass_count": int(
            sum(row["pose_camera_consistency_status"] == "PASS" for row in rows)
        ),
        "pose_camera_review_count": int(
            sum(
                row["pose_camera_consistency_status"]
                == "REVIEW_POSE_CAMERA_CONSISTENCY"
                for row in rows
            )
        ),
        "pose_camera_no_go_count": int(
            sum(
                row["pose_camera_consistency_status"] == "NO_GO_TRIANGULATION"
                for row in rows
            )
        ),
        "quality_gate": (
            "CAMERA_RECOVERY_REQUIRED"
            if any(
                row["pose_camera_consistency_status"] == "NO_GO_TRIANGULATION"
                for row in rows
            )
            else "PASS_OR_REVIEW"
        ),
        "elapsed_seconds": float(sum(row["elapsed_seconds"] for row in rows)),
    }
    atomic_text(
        runtime / "triangulation_summary.json", json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["schema_fail_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
