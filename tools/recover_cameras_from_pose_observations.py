#!/usr/bin/env python3
"""Build a held-out-validated, pose-observation-conditioned camera candidate.

This tool never overwrites Phase 5 background BA.  It is a recovery path for a
sequence whose frozen cameras fail the declared foreground triangulation gate.
The result is not independent calibration evidence and remains REVIEW.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from tools.triangulate_sapiens2 import (
        CAMERAS,
        PROJECT_ROOT,
        corrected_target_time,
        first_image_shape,
        frame_dir,
        interpolate_observations,
        load_canonical,
        percentile,
        pose_camera_consistency_status,
        sequence_vggt_metadata,
        temporal_models,
        triangulate_joint,
    )
except ModuleNotFoundError:  # Direct execution adds tools/, rather than the repo root.
    from triangulate_sapiens2 import (
        CAMERAS,
        PROJECT_ROOT,
        corrected_target_time,
        first_image_shape,
        frame_dir,
        interpolate_observations,
        load_canonical,
        percentile,
        pose_camera_consistency_status,
        sequence_vggt_metadata,
        temporal_models,
        triangulate_joint,
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
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument(
        "--canonical-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sapiens2_canonical_joints.json",
    )
    parser.add_argument("--min-confidence", type=float, default=0.30)
    parser.add_argument("--essential-threshold-px", type=float, default=3.0)
    parser.add_argument("--pnp-threshold-px", type=float, default=8.0)
    parser.add_argument("--huber-scale-px", type=float, default=10.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=20260811)
    parser.add_argument("--max-bracket-gap-seconds", type=float, default=0.050)
    return parser


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def heldout_split(
    frame_count: int, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, broadly distributed and disjoint frame masks."""
    if frame_count < 10 or not 0.0 < fraction < 0.5:
        raise ValueError("need >=10 frames and heldout fraction in (0, 0.5)")
    rng = np.random.default_rng(seed)
    heldout = np.zeros(frame_count, dtype=np.bool_)
    # One shuffled choice per small temporal stratum avoids a single-pose holdout.
    stride = max(5, int(round(1.0 / fraction)))
    for start in range(0, frame_count, stride):
        stop = min(start + stride, frame_count)
        heldout[int(rng.integers(start, stop))] = True
    fit = ~heldout
    return fit, heldout


def scale_intrinsics(
    camera_payload: dict[str, Any],
    image_shapes: dict[str, tuple[int, int]],
    model_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model_height, model_width = model_hw
    source = {}
    working = []
    for camera in CAMERAS:
        intrinsic = np.asarray(
            camera_payload["cameras"][camera]["intrinsic"], dtype=np.float64
        )
        source[camera] = intrinsic.copy()
        height, width = image_shapes[camera]
        intrinsic[0] *= width / model_width
        intrinsic[1] *= height / model_height
        working.append(intrinsic)
    return np.stack(working), source


def load_aligned_canonical(
    dataset_root: Path,
    pose_root: Path,
    sequence: str,
    canonical_config: Path,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, tuple[int, int]]]:
    poses = {}
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
            poses[camera] = {key: payload[key].copy() for key in payload.files}
        image_shapes[camera] = first_image_shape(frame_dir(dataset_root, sequence, camera))
    assert source_names is not None
    canonical = load_canonical(canonical_config, source_names)
    direct = np.asarray(
        [int(row["source_index"]) for row in canonical["direct"]], dtype=np.int32
    )
    names = [str(row["canonical"]) for row in canonical["direct"]]
    reference_time = poses["cam1"]["timestamp_pts_seconds"].astype(np.float64)
    xy = np.full((len(reference_time), 3, len(direct), 2), np.nan, dtype=np.float32)
    confidence = np.full((len(reference_time), 3, len(direct)), np.nan, dtype=np.float32)
    models = temporal_models(dataset_root / "reports" / "temporal_alignment", sequence)
    for camera_index, camera in enumerate(CAMERAS):
        payload = poses[camera]
        targets = reference_time
        if camera != "cam1":
            model = models[f"cam1-{camera}"]
            targets = np.asarray(
                [corrected_target_time(value, model) for value in reference_time]
            )
        aligned = interpolate_observations(
            payload["timestamp_pts_seconds"].astype(np.float64),
            payload["keypoints_xy"].astype(np.float32),
            payload["confidence"].astype(np.float32),
            payload["target_present"].astype(np.bool_),
            targets,
            max_gap,
        )
        xy[:, camera_index] = aligned["xy"][:, direct]
        confidence[:, camera_index] = aligned["confidence"][:, direct]
    return xy, confidence, reference_time, names, image_shapes


def flatten_pair(
    xy: np.ndarray,
    confidence: np.ndarray,
    frame_mask: np.ndarray,
    first: int,
    second: int,
    min_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    usable = (
        frame_mask[:, None]
        & np.isfinite(xy[:, first]).all(axis=-1)
        & np.isfinite(xy[:, second]).all(axis=-1)
        & (confidence[:, first] >= min_confidence)
        & (confidence[:, second] >= min_confidence)
    )
    frame_joint = np.argwhere(usable)
    return xy[:, first][usable], xy[:, second][usable], frame_joint


def normalized_points(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    return cv2.undistortPoints(
        points.astype(np.float64).reshape(-1, 1, 2), intrinsic, None
    ).reshape(-1, 2)


def estimate_relative_pose(
    points_a: np.ndarray,
    points_b: np.ndarray,
    intrinsic_a: np.ndarray,
    intrinsic_b: np.ndarray,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm_a = normalized_points(points_a, intrinsic_a)
    norm_b = normalized_points(points_b, intrinsic_b)
    normalized_threshold = threshold_px / float(
        np.mean([intrinsic_a[0, 0], intrinsic_a[1, 1], intrinsic_b[0, 0], intrinsic_b[1, 1]])
    )
    essential, mask = cv2.findEssentialMat(
        norm_a,
        norm_b,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.9999,
        threshold=normalized_threshold,
    )
    if essential is None or mask is None:
        raise RuntimeError("essential matrix estimation failed")
    if essential.shape != (3, 3):
        essential = essential[:3]
    _, rotation, translation, pose_mask = cv2.recoverPose(
        essential, norm_a, norm_b, np.eye(3), mask=mask
    )
    inlier = pose_mask.reshape(-1) > 0
    if int(inlier.sum()) < 8:
        raise RuntimeError("insufficient essential/recoverPose inliers")
    return rotation, translation.reshape(3), inlier


def triangulate_pair_normalized(
    points_a: np.ndarray,
    points_b: np.ndarray,
    intrinsic_a: np.ndarray,
    intrinsic_b: np.ndarray,
    rotation_b: np.ndarray,
    translation_b: np.ndarray,
) -> np.ndarray:
    norm_a = normalized_points(points_a, intrinsic_a)
    norm_b = normalized_points(points_b, intrinsic_b)
    projection_a = np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = np.hstack([rotation_b, translation_b.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_a, projection_b, norm_a.T, norm_b.T
    )
    return (homogeneous[:3] / homogeneous[3:4]).T


def estimate_third_camera(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsic: np.ndarray,
    threshold_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float64),
        image_points.astype(np.float64),
        intrinsic,
        None,
        iterationsCount=2000,
        reprojectionError=threshold_px,
        confidence=0.9999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < 8:
        raise RuntimeError("third-camera PnP failed or has insufficient inliers")
    inlier = inliers.reshape(-1)
    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_points[inlier].astype(np.float64),
        image_points[inlier].astype(np.float64),
        intrinsic,
        None,
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    mask = np.zeros(len(object_points), dtype=np.bool_)
    mask[inlier] = True
    return rotation, translation.reshape(3), mask


def recover_topology(
    xy: np.ndarray,
    confidence: np.ndarray,
    fit_mask: np.ndarray,
    intrinsics: np.ndarray,
    first: int,
    second: int,
    third: int,
    min_confidence: float,
    essential_threshold_px: float,
    pnp_threshold_px: float,
) -> dict[str, Any]:
    """Recover three cameras with one essential pair and one tied-scale PnP."""
    points_first, points_second, frame_joint = flatten_pair(
        xy, confidence, fit_mask, first, second, min_confidence
    )
    rotation_second, translation_second, essential_inlier = estimate_relative_pose(
        points_first,
        points_second,
        intrinsics[first],
        intrinsics[second],
        essential_threshold_px,
    )
    fit_frame_joint = frame_joint[essential_inlier]
    object_points = triangulate_pair_normalized(
        points_first[essential_inlier],
        points_second[essential_inlier],
        intrinsics[first],
        intrinsics[second],
        rotation_second,
        translation_second,
    )
    points_third = xy[
        fit_frame_joint[:, 0], third, fit_frame_joint[:, 1]
    ].astype(np.float64)
    confidence_third = confidence[
        fit_frame_joint[:, 0], third, fit_frame_joint[:, 1]
    ]
    finite_third = (
        np.isfinite(points_third).all(axis=-1)
        & (confidence_third >= min_confidence)
        & np.isfinite(object_points).all(axis=-1)
    )
    object_points = object_points[finite_third]
    points_third = points_third[finite_third]
    rotation_third, translation_third, pnp_inlier = estimate_third_camera(
        object_points,
        points_third,
        intrinsics[third],
        pnp_threshold_px,
    )
    anchor_extrinsics = np.repeat(np.eye(4)[None], 3, axis=0)
    anchor_extrinsics[second, :3, :3] = rotation_second
    anchor_extrinsics[second, :3, 3] = translation_second
    anchor_extrinsics[third, :3, :3] = rotation_third
    anchor_extrinsics[third, :3, 3] = translation_third
    # Express every pose in the cam1 identity gauge, even when another view was
    # the essential-pair anchor.  This changes coordinates, not reprojection.
    cam1_inverse = np.linalg.inv(anchor_extrinsics[0])
    extrinsics = np.asarray(
        [anchor_extrinsics[index] @ cam1_inverse for index in range(3)]
    )
    return {
        "topology": f"{CAMERAS[first]}-{CAMERAS[second]}_essential__{CAMERAS[third]}_pnp",
        "extrinsics": extrinsics,
        "essential_correspondence_count": len(points_first),
        "essential_inlier_count": int(essential_inlier.sum()),
        "pnp_correspondence_count": len(object_points),
        "pnp_inlier_count": int(pnp_inlier.sum()),
    }


def extrinsics_from_camera_payload(payload: dict[str, Any]) -> np.ndarray:
    result = []
    for camera in CAMERAS:
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3] = np.asarray(
            payload["cameras"][camera]["extrinsic_world_to_camera"], dtype=np.float64
        )
        result.append(extrinsic)
    return np.stack(result)


def reprojection_metrics(
    xy: np.ndarray,
    confidence: np.ndarray,
    frame_mask: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    min_confidence: float,
    huber_scale_px: float,
) -> dict[str, Any]:
    projections = np.asarray(
        [intrinsics[index] @ extrinsics[index, :3] for index in range(3)]
    )
    errors = []
    valid_joint_count = 0
    for frame in np.flatnonzero(frame_mask):
        for joint in range(xy.shape[2]):
            result = triangulate_joint(
                xy[frame, :, joint],
                confidence[frame, :, joint],
                projections,
                intrinsics,
                extrinsics[:, :3, :3],
                extrinsics,
                min_confidence,
                huber_scale_px,
            )
            if not result["valid"]:
                continue
            valid_joint_count += 1
            errors.extend(
                result["reprojection"][np.isfinite(result["reprojection"])].tolist()
            )
    array = np.asarray(errors, dtype=np.float64)
    return {
        "frame_count": int(frame_mask.sum()),
        "valid_joint_count": valid_joint_count,
        "reprojection_observation_count": len(array),
        "median_px": percentile(array, 50),
        "p90_px": percentile(array, 90),
        "p95_px": percentile(array, 95),
    }


def recovery_gate(
    current_status: str,
    fit_metrics: dict[str, Any],
    heldout_current: dict[str, Any],
    heldout_recovered: dict[str, Any],
    essential_inliers: int,
    pnp_inliers: int,
    huber_scale_px: float,
) -> tuple[bool, list[str]]:
    reasons = []
    recovered_status = pose_camera_consistency_status(
        heldout_recovered["median_px"],
        heldout_recovered["p90_px"],
        "REVIEW",
        huber_scale_px,
    )
    if current_status != "NO_GO_TRIANGULATION":
        reasons.append("source sequence is not NO_GO; recovery is not authorized")
    if essential_inliers < 100 or pnp_inliers < 100:
        reasons.append("insufficient robust fit support")
    if heldout_recovered["valid_joint_count"] < 100:
        reasons.append("insufficient held-out triangulation support")
    if recovered_status == "NO_GO_TRIANGULATION":
        reasons.append("held-out recovery remains NO_GO")
    for key in ("median_px", "p90_px"):
        recovered = heldout_recovered[key]
        current = heldout_current[key]
        if recovered is None or current is None or recovered >= current:
            reasons.append(f"held-out {key} did not improve")
    fit_median = fit_metrics["median_px"]
    fit_p90 = fit_metrics["p90_px"]
    held_median = heldout_recovered["median_px"]
    held_p90 = heldout_recovered["p90_px"]
    if (
        fit_median is None
        or held_median is None
        or held_median > max(1.5 * fit_median, fit_median + 5.0)
    ):
        reasons.append("held-out median indicates temporal overfit")
    if (
        fit_p90 is None
        or held_p90 is None
        or held_p90 > max(1.5 * fit_p90, fit_p90 + 20.0)
    ):
        reasons.append("held-out p90 indicates temporal overfit")
    return not reasons, reasons


def camera_record(
    original: dict[str, Any], rotation: np.ndarray, translation: np.ndarray
) -> dict[str, Any]:
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = translation
    camera_to_world = np.linalg.inv(extrinsic)
    return {
        "intrinsic": original["intrinsic"],
        "extrinsic_world_to_camera": extrinsic[:3].tolist(),
        "camera_to_world": camera_to_world.tolist(),
        "camera_center_world": camera_to_world[:3, 3].tolist(),
        "rotation_convention": "OpenCV world-to-camera; Xc=R*Xw+t",
        "source": "SAPIENS2_2D_OBSERVATION_CONDITIONED",
    }


def run_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    pose_root = args.pose_root.resolve()
    camera_root = args.camera_root.resolve()
    output_dir = args.output_root.resolve() / sequence
    xy, confidence, timestamps, joint_names, image_shapes = load_aligned_canonical(
        dataset_root,
        pose_root,
        sequence,
        args.canonical_config.resolve(),
        args.max_bracket_gap_seconds,
    )
    original_cameras = json.loads(
        (camera_root / sequence / "cameras_refined.json").read_text(encoding="utf-8")
    )
    original_validation = json.loads(
        (camera_root / sequence / "validation.json").read_text(encoding="utf-8")
    )
    vggt = sequence_vggt_metadata(dataset_root, sequence)
    model_hw = (
        int(vggt["sequence_status"]["model_height"]),
        int(vggt["sequence_status"]["model_width"]),
    )
    intrinsics, _ = scale_intrinsics(original_cameras, image_shapes, model_hw)
    fit_mask, heldout_mask = heldout_split(
        len(timestamps), args.heldout_fraction, args.split_seed
    )
    current_extrinsics = extrinsics_from_camera_payload(original_cameras)
    current_fit = reprojection_metrics(
        xy, confidence, fit_mask, intrinsics, current_extrinsics,
        args.min_confidence, args.huber_scale_px,
    )
    current_heldout = reprojection_metrics(
        xy, confidence, heldout_mask, intrinsics, current_extrinsics,
        args.min_confidence, args.huber_scale_px,
    )
    candidates = []
    candidate_errors = []
    for first, second, third in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        topology = f"{CAMERAS[first]}-{CAMERAS[second]}_essential__{CAMERAS[third]}_pnp"
        try:
            candidate = recover_topology(
                xy,
                confidence,
                fit_mask,
                intrinsics,
                first,
                second,
                third,
                args.min_confidence,
                args.essential_threshold_px,
                args.pnp_threshold_px,
            )
            candidate["fit"] = reprojection_metrics(
                xy, confidence, fit_mask, intrinsics, candidate["extrinsics"],
                args.min_confidence, args.huber_scale_px,
            )
            candidate["heldout"] = reprojection_metrics(
                xy, confidence, heldout_mask, intrinsics, candidate["extrinsics"],
                args.min_confidence, args.huber_scale_px,
            )
            candidates.append(candidate)
        except (RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
            candidate_errors.append({"topology": topology, "error": str(error)})
    if not candidates:
        raise RuntimeError(f"every recovery topology failed: {candidate_errors}")
    selected = min(
        candidates,
        key=lambda item: (
            float("inf") if item["fit"]["p90_px"] is None else item["fit"]["p90_px"],
            float("inf") if item["fit"]["median_px"] is None else item["fit"]["median_px"],
        ),
    )
    recovered_extrinsics = selected["extrinsics"]
    recovered_fit = selected["fit"]
    recovered_heldout = selected["heldout"]
    current_status = pose_camera_consistency_status(
        current_fit["median_px"], current_fit["p90_px"],
        original_validation["ba_acceptance_status"], args.huber_scale_px,
    )
    accepted, reasons = recovery_gate(
        current_status,
        recovered_fit,
        current_heldout,
        recovered_heldout,
        selected["essential_inlier_count"],
        selected["pnp_inlier_count"],
        args.huber_scale_px,
    )
    recovered_status = pose_camera_consistency_status(
        recovered_heldout["median_px"], recovered_heldout["p90_px"],
        "REVIEW", args.huber_scale_px,
    )
    camera_payload = {
        "schema_version": 1,
        "sequence": sequence,
        "camera_source": "SAPIENS2_2D_OBSERVATION_CONDITIONED",
        "coordinate_convention": "OpenCV world-to-camera",
        "shared_pose_constraint": "one fixed recovered pose per physical camera; cam1 identity gauge",
        "initialization_only": False,
        "not_metric": True,
        "not_independent_calibration": True,
        "source_phase5_geometry": str(camera_root / sequence / "cameras_refined.json"),
        "cameras": {
            "cam1": camera_record(original_cameras["cameras"]["cam1"], np.eye(3), np.zeros(3)),
            "cam2": camera_record(
                original_cameras["cameras"]["cam2"],
                recovered_extrinsics[1, :3, :3],
                recovered_extrinsics[1, :3, 3],
            ),
            "cam3": camera_record(
                original_cameras["cameras"]["cam3"],
                recovered_extrinsics[2, :3, :3],
                recovered_extrinsics[2, :3, 3],
            ),
        },
    }
    validation = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": sequence,
        "status": "PASS" if accepted else "FAIL",
        "ba_acceptance_status": "REVIEW" if accepted else "FAIL",
        "observation_conditioned_acceptance_status": (
            "REVIEW_OBSERVATION_CONDITIONED" if accepted else "NO_GO_RECOVERY"
        ),
        "not_ground_truth": True,
        "not_independent_camera_validation": True,
        "eligible_for_triangulation": accepted,
        "source_pose": "Sapiens2-5B target-only official flip-test",
        "source_camera_status": original_validation["ba_acceptance_status"],
        "source_pose_camera_status": current_status,
        "recovered_pose_camera_status_heldout": recovered_status,
        "fit_frame_count": int(fit_mask.sum()),
        "heldout_frame_count": int(heldout_mask.sum()),
        "fit_heldout_overlap_count": int(np.count_nonzero(fit_mask & heldout_mask)),
        "selected_topology": selected["topology"],
        "essential_correspondence_count": selected["essential_correspondence_count"],
        "essential_inlier_count": selected["essential_inlier_count"],
        "pnp_correspondence_count": selected["pnp_correspondence_count"],
        "pnp_inlier_count": selected["pnp_inlier_count"],
        "candidate_metrics": [
            {
                key: value
                for key, value in candidate.items()
                if key != "extrinsics"
            }
            for candidate in candidates
        ],
        "candidate_errors": candidate_errors,
        "current_fit": current_fit,
        "current_heldout": current_heldout,
        "recovered_fit": recovered_fit,
        "recovered_heldout": recovered_heldout,
        "rejection_reasons": reasons,
        "parameters": {
            "canonical_joint_names": joint_names,
            "min_confidence": args.min_confidence,
            "essential_threshold_px": args.essential_threshold_px,
            "pnp_threshold_px": args.pnp_threshold_px,
            "huber_scale_px": args.huber_scale_px,
            "heldout_fraction": args.heldout_fraction,
            "split_seed": args.split_seed,
            "timestamp_aware": True,
            "rgb_interpolation_performed": False,
        },
    }
    atomic_text(
        output_dir / "cameras_refined.json",
        json.dumps(camera_payload, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_text(
        output_dir / "validation.json",
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
    )
    return validation


def main() -> int:
    args = build_parser().parse_args()
    rows = []
    for sequence in args.sequences:
        result = run_sequence(args, sequence)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence_count": len(rows),
        "accepted_count": sum(row["eligible_for_triangulation"] for row in rows),
        "rejected_count": sum(not row["eligible_for_triangulation"] for row in rows),
        "status": "PASS" if all(row["eligible_for_triangulation"] for row in rows) else "REVIEW",
        "not_independent_camera_validation": True,
    }
    atomic_text(
        args.output_root.resolve() / "recovery_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
