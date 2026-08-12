#!/usr/bin/env python3
"""Fit a sequence-level canonical body representation from geometry and MHR priors.

This is a staged evidence fusion, not a claim that either teacher is ground
truth.  Timestamp-aware triangulated joints remain the strongest observation.
Each accepted monocular MHR prior is robustly similarity-aligned to the
sequence-local geometry gauge.  A weak, explicitly correlated prior term and a
second-difference temporal term then produce the fitted canonical trajectory.
"""

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
CORE_ALIGNMENT_JOINTS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
    "pelvis_center",
    "shoulder_center",
)
BONES = (
    ("left_shoulder", "left_elbow", "left_upper_arm"),
    ("left_elbow", "left_wrist", "left_forearm"),
    ("right_shoulder", "right_elbow", "right_upper_arm"),
    ("right_elbow", "right_wrist", "right_forearm"),
    ("left_hip", "left_knee", "left_femur"),
    ("left_knee", "left_ankle", "left_tibia"),
    ("right_hip", "right_knee", "right_femur"),
    ("right_knee", "right_ankle", "right_tibia"),
    ("left_shoulder", "right_shoulder", "shoulder_width"),
    ("left_hip", "right_hip", "hip_width"),
    ("pelvis_center", "shoulder_center", "torso"),
)
REQUIRED_BODY_FIT_FIELDS = {
    "frame_index",
    "timestamp_pts_seconds",
    "joint_names",
    "keypoints_3d",
    "valid_mask",
    "confidence",
    "evidence_type",
    "triangulated_valid",
    "triangulated_quality",
    "shape_params_consensus",
    "scale_params_consensus",
    "body_pose_params_consensus",
    "body_pose_prior_view_count",
    "s0_names",
    "s0",
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
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--max-time-gap-seconds", type=float, default=0.050)
    parser.add_argument("--minimum-alignment-joints", type=int, default=8)
    parser.add_argument("--geometry-weight", type=float, default=8.0)
    parser.add_argument("--sam-weight-per-view", type=float, default=0.25)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "phase9_body_fit.json",
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


def body_fit_dependency_signature(args: argparse.Namespace, sequence: str) -> str:
    """Bind a body fit to its geometry, SAM priors, config, and parameters."""
    triangulation = args.triangulation_root.resolve() / sequence
    files: list[tuple[str, Path]] = [
        ("triangulation/canonical_3d.npz", triangulation / "canonical_3d.npz"),
        ("triangulation/metadata.json", triangulation / "metadata.json"),
        ("config/phase9_body_fit.json", args.gate_config.resolve()),
    ]
    for camera in CAMERAS:
        prior = args.sam_prior_root.resolve() / sequence / camera
        files.extend(
            (
                (f"sam/{camera}/sam_body_prior.npz", prior / "sam_body_prior.npz"),
                (f"sam/{camera}/metadata.json", prior / "metadata.json"),
            )
        )
    inventory: list[tuple[str, int, int, int]] = []
    for label, path in files:
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"body-fit dependency is missing or symlinked: {label}")
        stat = path.stat()
        inventory.append((label, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    payload = {
        "files": inventory,
        "parameters": {
            "max_time_gap_seconds": args.max_time_gap_seconds,
            "minimum_alignment_joints": args.minimum_alignment_joints,
            "geometry_weight": args.geometry_weight,
            "sam_weight_per_view": args.sam_weight_per_view,
            "temporal_weight": args.temporal_weight,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_existing_body_fit(
    output_dir: Path,
    sequence: str,
    timestamps: np.ndarray,
    names: list[str],
    triangulated_valid: np.ndarray,
    triangulated_quality: np.ndarray,
    gate_config: dict[str, Any],
    dependency_signature: str | None,
) -> tuple[bool, dict[str, Any] | None]:
    """Validate the body-fit schema/gate and optional exact source binding."""
    archive_path = output_dir / "body_fit.npz"
    metadata_path = output_dir / "metadata.json"
    frames_path = output_dir / "frames.csv"
    if not archive_path.is_file() or not metadata_path.is_file() or not frames_path.is_file():
        return False, None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        qa = metadata["qa"]
        if (
            metadata.get("stage") != "SEQUENCE_LEVEL_CANONICAL_BODY_FIT"
            or metadata.get("sequence") != sequence
            or (
                dependency_signature is not None
                and metadata.get("source_dependency_signature") != dependency_signature
            )
            or qa.get("status") not in {"PASS", "REVIEW_BODY_FIT_QUALITY"}
        ):
            return False, None
        with np.load(archive_path, allow_pickle=False) as payload:
            if not REQUIRED_BODY_FIT_FIELDS <= set(payload.files):
                return False, None
            frame_count = len(timestamps)
            joint_count = len(names)
            valid = payload["valid_mask"].astype(np.bool_)
            points = payload["keypoints_3d"].astype(np.float32)
            confidence = payload["confidence"].astype(np.float32)
            evidence = payload["evidence_type"].astype(np.uint8)
            if (
                points.shape != (frame_count, joint_count, 3)
                or valid.shape != (frame_count, joint_count)
                or confidence.shape != valid.shape
                or evidence.shape != valid.shape
                or not np.array_equal(
                    payload["frame_index"].astype(np.int32),
                    np.arange(frame_count, dtype=np.int32),
                )
                or not np.array_equal(
                    payload["timestamp_pts_seconds"].astype(np.float64), timestamps
                )
                or list(payload["joint_names"].astype(str)) != names
                or not np.array_equal(
                    payload["triangulated_valid"].astype(np.bool_),
                    triangulated_valid,
                )
                or not np.array_equal(
                    payload["triangulated_quality"].astype(np.float32),
                    triangulated_quality.astype(np.float32),
                    equal_nan=True,
                )
                or not np.array_equal(valid, evidence > 0)
                or np.any(evidence > 3)
                or not np.isfinite(points[valid]).all()
                or not np.isnan(points[~valid]).all()
                or not np.isfinite(confidence).all()
                or np.any((confidence < 0) | (confidence > 1))
                or not np.isfinite(payload["shape_params_consensus"]).all()
                or not np.isfinite(payload["scale_params_consensus"]).all()
            ):
                return False, None
            expected_counts = {
                "geometry_plus_prior_count": int((evidence == 2).sum()),
                "geometry_only_count": int((evidence == 1).sum()),
                "prior_only_count": int((evidence == 3).sum()),
                "missing_count": int((evidence == 0).sum()),
            }
        recomputed_status, review_reasons, fail_reasons = evaluate_fit_gate(
            qa, gate_config
        )
        if (
            int(qa.get("frame_count", -1)) != len(timestamps)
            or int(qa.get("joint_count", -1)) != len(names)
            or not bool(qa.get("finite_valid_points"))
            or not bool(qa.get("invalid_points_are_nan"))
            or not np.isclose(float(qa.get("final_valid_joint_fraction", -1)), valid.mean())
            or any(int(qa.get(key, -1)) != value for key, value in expected_counts.items())
            or qa.get("status") != recomputed_status
            or list(qa.get("review_reasons", [])) != review_reasons
            or list(qa.get("fail_reasons", [])) != fail_reasons
        ):
            return False, None
        with frames_path.open(newline="", encoding="utf-8") as handle:
            frame_rows = list(csv.DictReader(handle))
        if len(frame_rows) != len(timestamps) or any(
            int(row.get("frame_index", -1)) != index
            for index, row in enumerate(frame_rows)
        ):
            return False, None
        return True, qa
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False, None


def percentile(values: np.ndarray, q: float) -> float | None:
    selected = values[np.isfinite(values)]
    return float(np.percentile(selected, q)) if len(selected) else None


def evaluate_fit_gate(
    qa: dict[str, Any], config: dict[str, Any]
) -> tuple[str, list[str], list[str]]:
    review = config["review_if_any"]
    fail = config["fail_if_any"]
    review_reasons: list[str] = []
    fail_reasons: list[str] = []
    final_fraction = float(qa["final_valid_joint_fraction"])
    alignment_fraction = float(qa["alignment_success_fraction"])
    prior_only_fraction = float(qa["prior_only_joint_fraction"])
    displacement = qa["observation_displacement_p95_normalized"]
    bone_cv = qa["median_bone_length_cv"]
    reference = qa["anthropometry"]["reference_length_sequence_gauge"]
    if not qa["finite_valid_points"] or not qa["invalid_points_are_nan"]:
        fail_reasons.append("SCHEMA_OR_FINITE_CONTRACT")
    if reference is None or not np.isfinite(reference) or reference <= 0:
        fail_reasons.append("INVALID_ANTHROPOMETRIC_REFERENCE")
    if final_fraction < float(fail["final_valid_joint_fraction_below"]):
        fail_reasons.append("FINAL_VALID_JOINT_FRACTION")
    if displacement is None or not np.isfinite(displacement):
        fail_reasons.append("MISSING_OBSERVATION_DISPLACEMENT")
    elif displacement > float(
        fail["observation_displacement_p95_normalized_above"]
    ):
        fail_reasons.append("OBSERVATION_DISPLACEMENT_P95")
    if fail_reasons:
        return "FAIL_BODY_FIT_QUALITY", review_reasons, fail_reasons
    if final_fraction < float(review["final_valid_joint_fraction_below"]):
        review_reasons.append("FINAL_VALID_JOINT_FRACTION")
    if alignment_fraction < float(review["alignment_success_fraction_below"]):
        review_reasons.append("ALIGNMENT_SUCCESS_FRACTION")
    if displacement > float(
        review["observation_displacement_p95_normalized_above"]
    ):
        review_reasons.append("OBSERVATION_DISPLACEMENT_P95")
    if prior_only_fraction > float(review["prior_only_joint_fraction_above"]):
        review_reasons.append("PRIOR_ONLY_JOINT_FRACTION")
    if bone_cv is None or not np.isfinite(bone_cv):
        review_reasons.append("MISSING_BONE_LENGTH_CV")
    elif bone_cv > float(review["median_bone_length_cv_above"]):
        review_reasons.append("MEDIAN_BONE_LENGTH_CV")
    if review.get("camera_status_not_pass") and qa["triangulation_camera_status"] != "PASS":
        review_reasons.append("CAMERA_UNCERTAINTY")
    return (
        "REVIEW_BODY_FIT_QUALITY" if review_reasons else "PASS",
        review_reasons,
        fail_reasons,
    )


def weighted_similarity(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    iterations: int = 4,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Robustly estimate target ~= scale * R @ source + translation."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    base = np.asarray(weights, dtype=np.float64).copy()
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("similarity inputs must be matching Nx3 arrays")
    if len(source) < 3 or len(base) != len(source):
        raise ValueError("similarity needs at least three weighted points")
    robust = np.clip(base, 1e-6, None)
    for _ in range(iterations):
        total = robust.sum()
        source_mean = np.sum(source * robust[:, None], axis=0) / total
        target_mean = np.sum(target * robust[:, None], axis=0) / total
        x = source - source_mean
        y = target - target_mean
        covariance = (y * robust[:, None]).T @ x / total
        u, singular, vt = np.linalg.svd(covariance)
        correction = np.ones(3)
        correction[-1] = np.sign(np.linalg.det(u @ vt))
        rotation = u @ np.diag(correction) @ vt
        variance = np.sum(robust * np.sum(x * x, axis=1)) / total
        scale = float(np.sum(singular * correction) / max(variance, 1e-12))
        translation = target_mean - scale * (rotation @ source_mean)
        predicted = scale * (source @ rotation.T) + translation
        residual = np.linalg.norm(predicted - target, axis=1)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        huber = max(median + 2.5 * 1.4826 * mad, 1e-6)
        robust = base * np.minimum(1.0, huber / np.maximum(residual, 1e-12))
    return scale, rotation, translation, residual


def second_difference_product(values: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values)
    if len(values) < 3:
        return output
    difference = values[:-2] - 2.0 * values[1:-1] + values[2:]
    output[:-2] += difference
    output[1:-1] -= 2.0 * difference
    output[2:] += difference
    return output


def smooth_track(
    observations: np.ndarray,
    weights: np.ndarray,
    temporal_weight: float,
    max_iterations: int = 200,
) -> np.ndarray:
    """Solve weighted second-difference smoothing with matrix-free CG."""
    observations = np.asarray(observations, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = weights > 0
    if valid.sum() < 2 or temporal_weight <= 0 or len(observations) < 3:
        return observations.copy()
    seed = observations.copy()
    for axis in range(seed.shape[1]):
        seed[:, axis] = np.interp(
            np.arange(len(seed)), np.flatnonzero(valid), observations[valid, axis]
        )

    def apply(value: np.ndarray) -> np.ndarray:
        return weights[:, None] * value + temporal_weight * second_difference_product(value)

    rhs = weights[:, None] * np.nan_to_num(observations, nan=0.0)
    estimate = seed
    residual = rhs - apply(estimate)
    direction = residual.copy()
    residual_norm = np.sum(residual * residual, axis=0)
    initial = np.maximum(residual_norm, 1e-24)
    for _ in range(max_iterations):
        applied = apply(direction)
        denominator = np.sum(direction * applied, axis=0)
        step = np.divide(
            residual_norm,
            denominator,
            out=np.zeros_like(residual_norm),
            where=np.abs(denominator) > 1e-24,
        )
        estimate += direction * step
        residual -= applied * step
        next_norm = np.sum(residual * residual, axis=0)
        if np.all(next_norm <= initial * 1e-12):
            break
        beta = np.divide(
            next_norm,
            residual_norm,
            out=np.zeros_like(next_norm),
            where=residual_norm > 1e-24,
        )
        direction = residual + direction * beta
        residual_norm = next_norm
    return estimate


def nearest_indices(
    reference: np.ndarray, source: np.ndarray, maximum_gap: float
) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(len(reference), -1, dtype=np.int32)
    error = np.full(len(reference), np.nan, dtype=np.float32)
    positions = np.searchsorted(source, reference)
    for index, position in enumerate(positions):
        candidates = [item for item in (position - 1, position) if 0 <= item < len(source)]
        if not candidates:
            continue
        selected = min(candidates, key=lambda item: abs(source[item] - reference[index]))
        delta = abs(float(source[selected] - reference[index]))
        if delta <= maximum_gap:
            result[index] = selected
            error[index] = delta
    return result, error


def anthropometric_descriptor(
    points: np.ndarray, valid: np.ndarray, names: list[str]
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    medians: dict[str, float] = {}
    cvs: dict[str, float] = {}
    for start_name, end_name, label in BONES:
        start, end = names.index(start_name), names.index(end_name)
        usable = valid[:, start] & valid[:, end]
        lengths = np.linalg.norm(points[:, start] - points[:, end], axis=1)[usable]
        if not len(lengths):
            medians[label] = np.nan
            cvs[label] = np.nan
            continue
        medians[label] = float(np.median(lengths))
        cvs[label] = float(np.std(lengths) / max(np.mean(lengths), 1e-12))
    reference = np.nanmean([medians["left_femur"], medians["right_femur"]])
    labels = [label for _, _, label in BONES]
    descriptor = np.asarray([medians[label] / reference for label in labels], dtype=np.float32)
    return descriptor, labels, {
        "reference": "mean sequence-median left/right femur length",
        "reference_length_sequence_gauge": float(reference),
        "bone_length_medians_sequence_gauge": medians,
        "bone_length_cv": cvs,
    }


def load_sam_camera(path: Path, expected_names: list[str]) -> dict[str, np.ndarray]:
    with np.load(path / "sam_body_prior.npz", allow_pickle=False) as payload:
        result = {key: payload[key].copy() for key in payload.files}
    if list(result["canonical_joint_names"].astype(str)) != expected_names:
        raise RuntimeError(f"canonical joint convention differs: {path}")
    return result


def fit_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    gate_config = json.loads(args.gate_config.resolve().read_text(encoding="utf-8"))
    dependency_signature = body_fit_dependency_signature(args, sequence)
    triangulation_dir = args.triangulation_root.resolve() / sequence
    with np.load(triangulation_dir / "canonical_3d.npz", allow_pickle=False) as payload:
        timestamps = payload["timestamp_pts_seconds"].astype(np.float64)
        names = list(payload["joint_names"].astype(str))
        triangulated = payload["keypoints_3d"].astype(np.float64)
        triangulated_valid = payload["valid_mask"].astype(np.bool_)
        triangulated_quality = payload["quality_score"].astype(np.float64)
    triangulation_metadata = json.loads(
        (triangulation_dir / "metadata.json").read_text(encoding="utf-8")
    )
    output_dir = args.output_root.resolve() / sequence
    existing_valid, existing_qa = validate_existing_body_fit(
        output_dir,
        sequence,
        timestamps,
        names,
        triangulated_valid,
        triangulated_quality,
        gate_config,
        dependency_signature,
    )
    if existing_valid and existing_qa is not None:
        return {**existing_qa, "resume_skipped": True}
    priors = {
        camera: load_sam_camera(
            args.sam_prior_root.resolve() / sequence / camera, names
        )
        for camera in CAMERAS
    }
    frame_count, joint_count = triangulated_valid.shape
    core = np.asarray([names.index(name) for name in CORE_ALIGNMENT_JOINTS], dtype=np.int32)
    aligned = np.full((frame_count, len(CAMERAS), joint_count, 3), np.nan, dtype=np.float64)
    aligned_valid = np.zeros((frame_count, len(CAMERAS)), dtype=np.bool_)
    alignment_scale = np.full((frame_count, len(CAMERAS)), np.nan, dtype=np.float32)
    alignment_residual = np.full((frame_count, len(CAMERAS)), np.nan, dtype=np.float32)
    time_error_ms = np.full((frame_count, len(CAMERAS)), np.nan, dtype=np.float32)
    prior_confidence = np.zeros((frame_count, len(CAMERAS)), dtype=np.float64)
    matched_indices = np.full((frame_count, len(CAMERAS)), -1, dtype=np.int32)

    for camera_index, camera in enumerate(CAMERAS):
        prior = priors[camera]
        matched, timing_error = nearest_indices(
            timestamps,
            prior["timestamp_pts_seconds"].astype(np.float64),
            args.max_time_gap_seconds,
        )
        matched_indices[:, camera_index] = matched
        time_error_ms[:, camera_index] = timing_error * 1000.0
        for frame, prior_frame in enumerate(matched):
            if prior_frame < 0 or not bool(prior["accepted_prior"][prior_frame]):
                continue
            available = triangulated_valid[frame, core]
            indices = core[available]
            if len(indices) < args.minimum_alignment_joints:
                continue
            source = prior["canonical_local_3d"][prior_frame, indices]
            target = triangulated[frame, indices]
            finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
            if finite.sum() < args.minimum_alignment_joints:
                continue
            try:
                scale, rotation, translation, residual = weighted_similarity(
                    source[finite],
                    target[finite],
                    np.clip(triangulated_quality[frame, indices][finite], 0.05, 1.0),
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if not np.isfinite(scale) or scale <= 0:
                continue
            points = prior["canonical_local_3d"][prior_frame].astype(np.float64)
            aligned[frame, camera_index] = scale * (points @ rotation.T) + translation
            aligned_valid[frame, camera_index] = np.isfinite(
                aligned[frame, camera_index]
            ).all()
            alignment_scale[frame, camera_index] = scale
            alignment_residual[frame, camera_index] = np.median(residual)
            confidence = float(prior["target_selection_confidence"][prior_frame])
            if bool(prior["occlusion_risk"][prior_frame]):
                confidence *= 0.5
            prior_confidence[frame, camera_index] = confidence

    prior_joint_valid = np.isfinite(aligned).all(axis=-1) & aligned_valid[:, :, None]
    prior_view_count = prior_joint_valid.sum(axis=1).astype(np.uint8)
    prior_consensus = np.nanmedian(aligned, axis=1)
    measurement = np.full_like(triangulated, np.nan)
    measurement_weight = np.zeros((frame_count, joint_count), dtype=np.float64)
    evidence_type = np.zeros((frame_count, joint_count), dtype=np.uint8)
    for frame in range(frame_count):
        for joint in range(joint_count):
            views = prior_joint_valid[frame, :, joint]
            prior_weight = args.sam_weight_per_view * float(
                prior_confidence[frame, views].sum()
            )
            if triangulated_valid[frame, joint]:
                geometry_weight = args.geometry_weight * max(
                    float(triangulated_quality[frame, joint]), 0.05
                )
                numerator = geometry_weight * triangulated[frame, joint]
                total = geometry_weight
                if views.any():
                    numerator += prior_weight * prior_consensus[frame, joint]
                    total += prior_weight
                    evidence_type[frame, joint] = 2
                else:
                    evidence_type[frame, joint] = 1
                measurement[frame, joint] = numerator / total
                measurement_weight[frame, joint] = total
            elif views.sum() >= 2:
                measurement[frame, joint] = prior_consensus[frame, joint]
                measurement_weight[frame, joint] = max(prior_weight, 1e-3)
                evidence_type[frame, joint] = 3

    fitted = np.full_like(measurement, np.nan)
    final_valid = evidence_type > 0
    for joint in range(joint_count):
        track = smooth_track(
            measurement[:, joint],
            measurement_weight[:, joint],
            args.temporal_weight,
        )
        fitted[final_valid[:, joint], joint] = track[final_valid[:, joint]]

    observation_residual = np.full((frame_count, joint_count), np.nan, dtype=np.float32)
    observed = final_valid & triangulated_valid
    observation_residual[observed] = np.linalg.norm(
        fitted[observed] - triangulated[observed], axis=1
    ).astype(np.float32)
    prior_residual = np.linalg.norm(aligned - fitted[:, None], axis=-1)
    prior_residual[~prior_joint_valid] = np.nan
    final_confidence = np.zeros((frame_count, joint_count), dtype=np.float32)
    final_confidence[triangulated_valid] = np.clip(
        triangulated_quality[triangulated_valid], 0.0, 1.0
    )
    prior_only = evidence_type == 3
    final_confidence[prior_only] = np.clip(
        0.15 * prior_view_count[prior_only] / len(CAMERAS), 0.0, 0.15
    )

    shape_samples = []
    scale_samples = []
    pose_consensus = np.full((frame_count, 133), np.nan, dtype=np.float32)
    pose_view_count = np.zeros(frame_count, dtype=np.uint8)
    for camera_index, camera in enumerate(CAMERAS):
        prior = priors[camera]
        accepted = prior["accepted_prior"].astype(np.bool_)
        shape_samples.extend(prior["shape_params"][accepted])
        scale_samples.extend(prior["scale_params"][accepted])
    for frame in range(frame_count):
        samples = []
        for camera_index, camera in enumerate(CAMERAS):
            prior_frame = matched_indices[frame, camera_index]
            if prior_frame >= 0 and bool(priors[camera]["accepted_prior"][prior_frame]):
                samples.append(priors[camera]["body_pose_params"][prior_frame])
        if samples:
            pose_consensus[frame] = np.median(np.asarray(samples), axis=0)
            pose_view_count[frame] = len(samples)
    shape_consensus = np.median(np.asarray(shape_samples), axis=0).astype(np.float32)
    scale_consensus = np.median(np.asarray(scale_samples), axis=0).astype(np.float32)
    s0, s0_names, anthropometry = anthropometric_descriptor(fitted, final_valid, names)
    reference_length = anthropometry["reference_length_sequence_gauge"]
    normalized_observation = observation_residual / max(reference_length, 1e-12)

    atomic_npz(
        output_dir / "body_fit.npz",
        frame_index=np.arange(frame_count, dtype=np.int32),
        timestamp_pts_seconds=timestamps,
        joint_names=np.asarray(names),
        keypoints_3d=fitted.astype(np.float32),
        valid_mask=final_valid,
        confidence=final_confidence,
        evidence_type=evidence_type,
        triangulated_valid=triangulated_valid,
        triangulated_quality=triangulated_quality.astype(np.float32),
        prior_view_count=prior_view_count,
        observation_residual_sequence_gauge=observation_residual,
        aligned_prior_residual_sequence_gauge=prior_residual.astype(np.float32),
        alignment_scale=alignment_scale,
        alignment_residual_sequence_gauge=alignment_residual,
        sam_time_error_ms=time_error_ms,
        sam_source_frame_index=matched_indices,
        shape_params_consensus=shape_consensus,
        scale_params_consensus=scale_consensus,
        body_pose_params_consensus=pose_consensus,
        body_pose_prior_view_count=pose_view_count,
        s0_names=np.asarray(s0_names),
        s0=s0,
    )
    finite_valid = bool(np.isfinite(fitted[final_valid]).all())
    invalid_nan = bool(np.isnan(fitted[~final_valid]).all())
    camera_status = triangulation_metadata["qa"]["pose_camera_consistency_status"]
    qa = {
        "sequence": sequence,
        "frame_count": frame_count,
        "joint_count": joint_count,
        "triangulated_joint_fraction": float(triangulated_valid.mean()),
        "final_valid_joint_fraction": float(final_valid.mean()),
        "alignment_success_fraction": float(aligned_valid.mean()),
        "geometry_plus_prior_count": int((evidence_type == 2).sum()),
        "geometry_only_count": int((evidence_type == 1).sum()),
        "prior_only_count": int(prior_only.sum()),
        "prior_only_joint_fraction": float(prior_only.mean()),
        "missing_count": int((evidence_type == 0).sum()),
        "alignment_success_count": int(aligned_valid.sum()),
        "alignment_attempt_count": int(frame_count * len(CAMERAS)),
        "alignment_residual_median_normalized": (
            percentile(alignment_residual / max(reference_length, 1e-12), 50)
        ),
        "observation_displacement_median_normalized": percentile(
            normalized_observation, 50
        ),
        "observation_displacement_p95_normalized": percentile(
            normalized_observation, 95
        ),
        "finite_valid_points": finite_valid,
        "invalid_points_are_nan": invalid_nan,
        "triangulation_camera_status": camera_status,
        "anthropometry": anthropometry,
    }
    finite_bone_cv = np.asarray(
        [value for value in anthropometry["bone_length_cv"].values() if np.isfinite(value)],
        dtype=np.float64,
    )
    qa["median_bone_length_cv"] = (
        float(np.median(finite_bone_cv)) if len(finite_bone_cv) else None
    )
    status, review_reasons, fail_reasons = evaluate_fit_gate(qa, gate_config)
    qa["status"] = status
    qa["review_reasons"] = review_reasons
    qa["fail_reasons"] = fail_reasons
    qa["quality_gate_config"] = gate_config
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": sequence,
        "stage": "SEQUENCE_LEVEL_CANONICAL_BODY_FIT",
        "not_ground_truth": True,
        "world_gauge": "sequence-local arbitrary scale inherited from triangulation",
        "fitting_stages": [
            "timestamp-aware triangulated geometry anchor",
            "robust per-view/per-frame similarity alignment of MHR canonical prior",
            "weak correlated-prior fusion; geometry remains dominant",
            "weighted second-difference temporal fit",
            "sequence-level shape/scale parameter consensus and S0",
        ],
        "evidence_type_codes": {
            "0": "MISSING",
            "1": "TRIANGULATED_GEOMETRY_ONLY",
            "2": "TRIANGULATED_GEOMETRY_PLUS_ALIGNED_SAM_PRIOR",
            "3": "ALIGNED_SAM_PRIOR_ONLY_AT_LEAST_TWO_VIEWS",
        },
        "correlated_error_warning": "Sapiens2 and SAM priors are learned and not independent GT sources",
        "parameters": {
            "max_time_gap_seconds": args.max_time_gap_seconds,
            "minimum_alignment_joints": args.minimum_alignment_joints,
            "geometry_weight": args.geometry_weight,
            "sam_weight_per_view": args.sam_weight_per_view,
            "temporal_weight": args.temporal_weight,
            "core_alignment_joints": CORE_ALIGNMENT_JOINTS,
            "quality_gate_config": gate_config,
        },
        "source_triangulation_metadata": triangulation_metadata,
        "source_dependency_signature": dependency_signature,
        "qa": qa,
    }
    atomic_text(
        output_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    rows = [
        {
            "frame_index": frame,
            "timestamp_pts_seconds": f"{timestamps[frame]:.9f}",
            "valid_joints": int(final_valid[frame].sum()),
            "prior_only_joints": int(prior_only[frame].sum()),
            "missing_joints": int((evidence_type[frame] == 0).sum()),
            "sam_alignment_views": int(aligned_valid[frame].sum()),
        }
        for frame in range(frame_count)
    ]
    atomic_csv(output_dir / "frames.csv", rows)
    return {**qa, "resume_skipped": False}


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.max_time_gap_seconds <= 0
        or args.minimum_alignment_joints < 3
        or args.geometry_weight <= 0
        or args.sam_weight_per_view < 0
        or args.temporal_weight < 0
    ):
        raise RuntimeError("invalid fitting parameter")
    rows = []
    for sequence in args.sequences:
        row = fit_sequence(args, sequence)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence_count": len(rows),
        "frame_count": sum(row["frame_count"] for row in rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "review_count": sum(row["status"].startswith("REVIEW") for row in rows),
        "fail_count": sum(row["status"].startswith("FAIL") for row in rows),
        "status": "PASS_OR_REVIEW" if not any(
            row["status"].startswith("FAIL") for row in rows
        ) else "FAIL",
    }
    atomic_csv(args.runtime_dir.resolve() / "body_fit_qa.csv", rows)
    atomic_text(
        args.runtime_dir.resolve() / "body_fit_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS_OR_REVIEW" else 2


if __name__ == "__main__":
    raise SystemExit(main())
