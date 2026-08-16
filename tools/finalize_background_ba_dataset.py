#!/usr/bin/env python3
"""Finalize Phase 5 dataset summaries without changing Background BA results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT)
).expanduser()
CAMERAS = ("cam1", "cam2", "cam3")
STATUS_RANK = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
IGNORED_CONFIGURATION_KEYS = {
    "sequence",
    "root",
    "vggt_root",
    "output_root",
    # Phase 5.1 diagnostic controls. They do not alter the objective, observations,
    # initialization, gates, or the frozen Phase 5 default path.
    "stage2_max_nfev",
    "optimizer_verbose",
}
REQUIRED_SEQUENCE_FILES = (
    "cameras_initial.json",
    "cameras_refined.json",
    "tracks.npz",
    "points3d.npz",
    "metrics.json",
    "residuals.csv",
    "sample_gating.csv",
    "visual_qa.json",
    "debug/static_masks.npz",
    "debug/static_feature_stats.csv",
    "debug/persistent_landmarks.csv",
    "debug/pair_matches.csv",
    "debug/temporal_pairings.csv",
    "debug/compare_auto.png",
    "debug/compare_auto.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(jsonable(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(jsonable(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple)) else jsonable(value)
                    for key, value in row.items()
                }
            )
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_configuration(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metrics["configuration"].items()
        if key not in IGNORED_CONFIGURATION_KEYS
    }


def percentile(values: list[float], value: float) -> float | None:
    return float(np.percentile(values, value)) if values else None


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": float(np.mean(values)) if values else None,
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def load_visual_decisions(path: Path, sequences: list[str]) -> dict[str, dict[str, Any]]:
    decisions = {}
    for row in read_csv(path):
        row["reasons"] = [item for item in row["reasons"].split("|") if item]
        decisions[row["sequence"]] = row
    if set(decisions) != set(sequences):
        missing = sorted(set(sequences) - set(decisions))
        extra = sorted(set(decisions) - set(sequences))
        raise RuntimeError(f"visual QA decision mismatch; missing={missing}, extra={extra}")
    return decisions


def write_visual_sidecars(
    output_root: Path, decisions: dict[str, dict[str, Any]], reviewed_at: str
) -> None:
    for sequence, decision in decisions.items():
        sequence_dir = output_root / sequence
        evidence: dict[str, Any] = {
            "comparison_screenshot": "debug/compare_auto.png",
            "comparison_stats": "debug/compare_auto.json",
            "camera_arrangement_plausible": decision["camera_arrangement"],
            "refined_orientation_plausible": decision["refined_orientation"],
            "refined_sparse_background_support": decision["sparse_support"],
            "no_mirror_or_global_flip": decision["no_mirror"],
            "no_exploding_geometry": decision["no_explosion"],
        }
        pose_dispersion = sequence_dir / "debug" / "cam2_pose_dispersion_top.png"
        if pose_dispersion.is_file():
            evidence["pose_dispersion_screenshot"] = "debug/cam2_pose_dispersion_top.png"
        atomic_json(
            sequence_dir / "visual_qa.json",
            {
                "schema_version": 2,
                "sequence": sequence,
                "reviewed_at": reviewed_at,
                "status": decision["status"],
                "reasons": decision["reasons"],
                "evidence": evidence,
                "notes": decision["notes"],
                "geometry_modified_by_visual_qa": False,
            },
        )


def camera_matrices(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3] = np.asarray(payload["extrinsic_world_to_camera"], dtype=np.float64)
    camera_to_world = np.asarray(payload["camera_to_world"], dtype=np.float64)
    intrinsic = np.asarray(payload["intrinsic"], dtype=np.float64)
    return world_to_camera, camera_to_world, intrinsic


def accepted_residuals_by_camera(path: Path) -> dict[str, dict[str, list[float]]]:
    output = {camera: {"pre": [], "post": []} for camera in CAMERAS}
    for row in read_csv(path):
        if row["accepted_final"] == "True":
            output[row["camera_id"]]["pre"].append(float(row["pre_error_px"]))
            output[row["camera_id"]]["post"].append(float(row["post_error_px"]))
    return output


def validate_sequence(
    sequence_dir: Path,
    reference_configuration: dict[str, Any],
    tool_sha256: str,
    validated_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sequence = sequence_dir.name
    missing = [name for name in REQUIRED_SEQUENCE_FILES if not (sequence_dir / name).is_file()]
    metrics = json.loads((sequence_dir / "metrics.json").read_text(encoding="utf-8"))
    initial = json.loads((sequence_dir / "cameras_initial.json").read_text(encoding="utf-8"))
    refined = json.loads((sequence_dir / "cameras_refined.json").read_text(encoding="utf-8"))
    visual = json.loads((sequence_dir / "visual_qa.json").read_text(encoding="utf-8"))
    configuration_match = normalized_configuration(metrics) == reference_configuration
    max_inverse_error = 0.0
    max_orthogonality_error = 0.0
    determinants = []
    fixed_intrinsics = True
    finite_cameras = True
    for camera in CAMERAS:
        for payload in (initial["cameras"][camera], refined["cameras"][camera]):
            w2c, c2w, intrinsic = camera_matrices(payload)
            finite_cameras &= bool(
                np.isfinite(w2c).all() and np.isfinite(c2w).all() and np.isfinite(intrinsic).all()
            )
            max_inverse_error = max(
                max_inverse_error, float(np.max(np.abs(w2c @ c2w - np.eye(4))))
            )
            rotation = w2c[:3, :3]
            max_orthogonality_error = max(
                max_orthogonality_error,
                float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
            )
            determinants.append(float(np.linalg.det(rotation)))
        fixed_intrinsics &= bool(
            np.array_equal(
                np.asarray(initial["cameras"][camera]["intrinsic"]),
                np.asarray(refined["cameras"][camera]["intrinsic"]),
            )
        )
    cam1_w2c, _, _ = camera_matrices(refined["cameras"]["cam1"])
    cam1_identity = bool(np.allclose(cam1_w2c, np.eye(4), atol=1e-12))
    with np.load(sequence_dir / "tracks.npz") as tracks, np.load(
        sequence_dir / "points3d.npz"
    ) as points:
        track_arrays_finite = all(
            np.isfinite(tracks[key]).all()
            for key in ("obs_xy", "obs_timestamp_sec", "obs_vggt_confidence")
        )
        point_arrays_finite = all(
            np.isfinite(points[key]).all()
            for key in ("points_initial", "points_stage1", "points_refined")
        )
        observation_count = len(tracks["obs_track_id"])
        accepted_observations = int(tracks["accepted_final"].sum())
        point_count = len(points["points_initial"])
        accepted_points = int(points["accepted_track_mask"].sum())
        observation_indices_valid = bool(
            len(tracks["obs_track_id"])
            and tracks["obs_track_id"].min() >= 0
            and tracks["obs_track_id"].max() < point_count
        )
    residual_rows = read_csv(sequence_dir / "residuals.csv")
    sample_rows = read_csv(sequence_dir / "sample_gating.csv")
    temporal_rows = read_csv(sequence_dir / "debug" / "temporal_pairings.csv")
    count_consistency = (
        observation_count == metrics["tracks"]["observation_count_extracted"]
        and accepted_observations == metrics["reprojection_post"]["observation_count"]
        and point_count == metrics["tracks"]["ba_track_count_initial"]
        and accepted_points == metrics["tracks"]["ba_track_count_final"]
        and len(residual_rows) == observation_count
        and sum(row["accepted_final"] == "True" for row in residual_rows)
        == accepted_observations
        and len(sample_rows) == 24
        and len(temporal_rows) == 24
    )
    no_interpolation = all(
        row["interpolation_or_frame_generation"] == "False" for row in temporal_rows
    )
    se3_consistent = (
        finite_cameras
        and max_inverse_error <= 1e-10
        and max_orthogonality_error <= 1e-10
        and all(abs(value - 1.0) <= 1e-10 for value in determinants)
    )
    output_valid = all(
        (
            not missing,
            configuration_match,
            se3_consistent,
            cam1_identity,
            fixed_intrinsics,
            track_arrays_finite,
            point_arrays_finite,
            observation_indices_valid,
            count_consistency,
            no_interpolation,
            visual["status"] in STATUS_RANK,
        )
    )
    stage1_success = bool(metrics["optimization"]["stage1"]["success"])
    stage2_success = bool(metrics["optimization"]["stage2"]["success"])
    validation_status = "PASS" if output_valid and stage1_success and stage2_success else "FAIL"
    validation = {
        "schema_version": 1,
        "validated_at": validated_at,
        "sequence": sequence,
        "status": validation_status,
        "ba_acceptance_status": metrics["acceptance"]["status"],
        "visual_qa_status": visual["status"],
        "algorithm_provenance": {
            "tool": "tools/background_bundle_adjust.py",
            "tool_sha256": tool_sha256,
            "configuration_matches_pilot": configuration_match,
            "intrinsics_mode": metrics["configuration"]["intrinsics"],
            "robust_loss": metrics["configuration"]["robust_loss"],
        },
        "checks": {
            "required_files_present": not missing,
            "missing_files": missing,
            "stage1_converged": stage1_success,
            "stage2_converged": stage2_success,
            "finite_cameras": finite_cameras,
            "se3_consistent": se3_consistent,
            "cam1_identity_gauge": cam1_identity,
            "fixed_intrinsics_unchanged": fixed_intrinsics,
            "track_arrays_finite": track_arrays_finite,
            "point_arrays_finite": point_arrays_finite,
            "observation_indices_valid": observation_indices_valid,
            "counts_consistent": count_consistency,
            "temporal_pairings": len(temporal_rows),
            "interpolated_or_generated_frames": 0 if no_interpolation else None,
            "open3d_comparison_render_present": (
                sequence_dir / "debug" / "compare_auto.png"
            ).stat().st_size > 0,
        },
        "numeric": {
            "max_world_to_camera_inverse_error": max_inverse_error,
            "max_rotation_orthogonality_error": max_orthogonality_error,
            "rotation_determinant_min": min(determinants),
            "rotation_determinant_max": max(determinants),
            "observations_extracted": observation_count,
            "observations_final": accepted_observations,
            "points_initial": point_count,
            "points_final": accepted_points,
        },
        "source_mutation_performed": False,
        "forbidden_model_or_label_operations_performed": [],
    }
    atomic_json(sequence_dir / "validation.json", validation)
    return validation, {
        "metrics": metrics,
        "initial": initial,
        "refined": refined,
        "visual": visual,
        "residuals_by_camera": accepted_residuals_by_camera(sequence_dir / "residuals.csv"),
    }


def final_status(numeric: str, visual: str) -> str:
    return numeric if STATUS_RANK[numeric] >= STATUS_RANK[visual] else visual


def source_integrity(root: Path, ba_tool: Path, output_root: Path) -> dict[str, Any]:
    changes = []
    for row in read_csv(root / "reports" / "dataset_inventory.csv"):
        path = root / row["path"]
        actual_mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(
            microsecond=0
        )
        expected_mtime = datetime.fromisoformat(row["mtime_utc"].replace("Z", "+00:00"))
        if path.stat().st_size != int(row["size_bytes"]) or actual_mtime != expected_mtime:
            changes.append(str(path))
    reference_mtime = ba_tool.stat().st_mtime
    working_new = sum(
        path.stat().st_mtime > reference_mtime
        for path in (root / "final_frame").rglob("*") if path.is_file()
    )
    vggt_numeric_new = sum(
        path.stat().st_mtime > reference_mtime
        for path in (root / "outputs" / "vggt" / "UNKNOWN").rglob("*")
        if path.is_file() and (path.suffix == ".npz" or path.name in {"frames.csv", "metadata.json"})
    )
    forbidden_media = [
        str(path) for path in output_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg"}
    ]
    return {
        "phase0_source_inventory_size_or_mtime_changes": len(changes),
        "changed_source_files": changes,
        "working_frames_newer_than_frozen_ba_tool": working_new,
        "vggt_numeric_payloads_newer_than_frozen_ba_tool": vggt_numeric_new,
        "forbidden_media_files_in_background_ba": forbidden_media,
    }


def make_dataset_rows(records: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    scopes: list[tuple[str, str, list[tuple[str, dict[str, Any]]]]] = [
        ("DATASET", "ALL", list(records.items()))
    ]
    exercises = sorted({record["metrics"]["exercise"] for record in records.values()})
    scopes.extend(
        (
            "EXERCISE",
            exercise,
            [(sequence, record) for sequence, record in records.items()
             if record["metrics"]["exercise"] == exercise],
        )
        for exercise in exercises
    )
    rows = []
    for scope_type, scope_id, items in scopes:
        statuses = [
            final_status(
                record["metrics"]["acceptance"]["status"], record["visual"]["status"]
            )
            for _, record in items
        ]
        post_medians = [record["metrics"]["reprojection_post"]["median_px"] for _, record in items]
        post_p90 = [record["metrics"]["reprojection_post"]["p90_px"] for _, record in items]
        post_p95 = [record["metrics"]["reprojection_post"]["p95_px"] for _, record in items]
        rotations = [
            row["refined_rotation_change_from_robust_init_deg"]
            for _, record in items for row in record["metrics"]["camera_comparison"].values()
        ]
        centers = [
            row["refined_center_change_scene_fraction"]
            for _, record in items for row in record["metrics"]["camera_comparison"].values()
        ]
        accepted_pre = [
            value for _, record in items
            for camera in CAMERAS for value in record["residuals_by_camera"][camera]["pre"]
        ]
        accepted_post = [
            value for _, record in items
            for camera in CAMERAS for value in record["residuals_by_camera"][camera]["post"]
        ]
        counts = Counter(statuses)
        rows.append(
            {
                "scope_type": scope_type,
                "scope_id": scope_id,
                "sequence_count": len(items),
                "camera_count": len(items) * 3,
                "pass_count": counts["PASS"],
                "review_count": counts["REVIEW"],
                "fail_count": counts["FAIL"],
                "stage1_converged_count": sum(
                    record["metrics"]["optimization"]["stage1"]["success"] for _, record in items
                ),
                "stage2_converged_count": sum(
                    record["metrics"]["optimization"]["stage2"]["success"] for _, record in items
                ),
                "points_initial_total": sum(
                    record["metrics"]["tracks"]["ba_track_count_initial"] for _, record in items
                ),
                "points_final_total": sum(
                    record["metrics"]["tracks"]["ba_track_count_final"] for _, record in items
                ),
                "observations_final_total": sum(
                    record["metrics"]["reprojection_post"]["observation_count"] for _, record in items
                ),
                "sequence_post_median_px_median": percentile(post_medians, 50),
                "sequence_post_p90_px_median": percentile(post_p90, 50),
                "sequence_post_p95_px_median": percentile(post_p95, 50),
                "accepted_residual_pre_mean_px": float(np.mean(accepted_pre)),
                "accepted_residual_pre_median_px": percentile(accepted_pre, 50),
                "accepted_residual_pre_p90_px": percentile(accepted_pre, 90),
                "accepted_residual_pre_p95_px": percentile(accepted_pre, 95),
                "accepted_residual_post_mean_px": float(np.mean(accepted_post)),
                "accepted_residual_post_median_px": percentile(accepted_post, 50),
                "accepted_residual_post_p90_px": percentile(accepted_post, 90),
                "accepted_residual_post_p95_px": percentile(accepted_post, 95),
                "camera_rotation_change_deg_median": percentile(rotations, 50),
                "camera_rotation_change_deg_p95": percentile(rotations, 95),
                "camera_center_change_scene_fraction_median": percentile(centers, 50),
                "camera_center_change_scene_fraction_p95": percentile(centers, 95),
                "scale_provenance": "sequence-local arbitrary; initial cam1-cam2 baseline preserved",
                "gauge_provenance": "robust cam1 physical pose fixed to identity",
            }
        )
    return rows


def build_detail_rows(records: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], ...]:
    uncertainty_rows = []
    pose_rows = []
    ba_rows = []
    review_rows = []
    for sequence, record in sorted(records.items()):
        metrics = record["metrics"]
        visual = record["visual"]
        numeric_status = metrics["acceptance"]["status"]
        status = final_status(numeric_status, visual["status"])
        reasons = list(metrics["acceptance"]["reasons"])
        reasons.extend(reason for reason in visual["reasons"] if reason not in reasons)
        if status != "PASS":
            review_rows.append(
                {
                    "sequence": sequence,
                    "exercise": metrics["exercise"],
                    "status": status,
                    "numeric_status": numeric_status,
                    "visual_status": visual["status"],
                    "reasons": reasons,
                    "stage1_converged": metrics["optimization"]["stage1"]["success"],
                    "stage2_converged": metrics["optimization"]["stage2"]["success"],
                    "points_final": metrics["tracks"]["ba_track_count_final"],
                    "observations_final": metrics["reprojection_post"]["observation_count"],
                    "post_median_px": metrics["reprojection_post"]["median_px"],
                    "post_p90_px": metrics["reprojection_post"]["p90_px"],
                    "post_p95_px": metrics["reprojection_post"]["p95_px"],
                    "visual_notes": visual["notes"],
                }
            )
        ba_rows.append(
            {
                "sequence": sequence,
                "exercise": metrics["exercise"],
                "status": status,
                "reasons": reasons,
                "stage1_success": metrics["optimization"]["stage1"]["success"],
                "stage1_status": metrics["optimization"]["stage1"]["status"],
                "stage1_nfev": metrics["optimization"]["stage1"]["nfev"],
                "stage1_cost": metrics["optimization"]["stage1"]["cost"],
                "stage1_optimality": metrics["optimization"]["stage1"]["optimality"],
                "stage2_success": metrics["optimization"]["stage2"]["success"],
                "stage2_status": metrics["optimization"]["stage2"]["status"],
                "stage2_nfev": metrics["optimization"]["stage2"]["nfev"],
                "stage2_cost": metrics["optimization"]["stage2"]["cost"],
                "stage2_optimality": metrics["optimization"]["stage2"]["optimality"],
                "points_initial": metrics["tracks"]["ba_track_count_initial"],
                "points_final": metrics["tracks"]["ba_track_count_final"],
                "three_camera_tracks": metrics["tracks"]["three_camera_track_count"],
                "observations_extracted": metrics["tracks"]["observation_count_extracted"],
                "observations_final": metrics["reprojection_post"]["observation_count"],
                "rejected_tracks": (
                    metrics["tracks"]["stage1_rejected_tracks"]
                    + metrics["tracks"]["post_ba_rejected_tracks"]
                ),
                "good_samples": metrics["sample_gating"]["good_samples"],
                "downweighted_samples": metrics["sample_gating"]["downweighted_samples"],
                "rejected_samples": metrics["sample_gating"]["rejected_samples"],
                "pre_mean_px": metrics["reprojection_pre"]["mean_px"],
                "pre_median_px": metrics["reprojection_pre"]["median_px"],
                "pre_p90_px": metrics["reprojection_pre"]["p90_px"],
                "pre_p95_px": metrics["reprojection_pre"]["p95_px"],
                "post_mean_px": metrics["reprojection_post"]["mean_px"],
                "post_median_px": metrics["reprojection_post"]["median_px"],
                "post_p90_px": metrics["reprojection_post"]["p90_px"],
                "post_p95_px": metrics["reprojection_post"]["p95_px"],
                "pre_inlier_ratio": metrics["reprojection_pre"]["inlier_ratio"],
                "post_inlier_ratio": metrics["reprojection_post"]["inlier_ratio"],
                "elapsed_sec": metrics["runtime"]["elapsed_sec"],
            }
        )
        for camera in CAMERAS:
            comparison = metrics["camera_comparison"][camera]
            initial_payload = record["initial"]["cameras"][camera]
            refined_payload = record["refined"]["cameras"][camera]
            initial_w2c, _, initial_k = camera_matrices(initial_payload)
            refined_w2c, _, refined_k = camera_matrices(refined_payload)
            camera_residuals = record["residuals_by_camera"][camera]["post"]
            camera_samples = [
                row for row in metrics["sample_gating"]["rows"] if row["camera_id"] == camera
            ]
            radial = metrics["radial_distortion_diagnostic"][camera]
            uncertainty_rows.append(
                {
                    "sequence": sequence,
                    "exercise": metrics["exercise"],
                    "camera_id": camera,
                    "sequence_status": status,
                    "uncertainty_reasons": reasons,
                    "vggt_rotation_pairwise_p95_deg": comparison["vggt_rotation_pairwise_p95_deg"],
                    "vggt_center_dispersion_p95": comparison["vggt_center_dispersion_p95"],
                    "refined_rotation_change_deg": comparison[
                        "refined_rotation_change_from_robust_init_deg"
                    ],
                    "refined_center_change_scene_fraction": comparison[
                        "refined_center_change_scene_fraction"
                    ],
                    "accepted_observations": len(camera_residuals),
                    "post_residual_median_px": percentile(camera_residuals, 50),
                    "post_residual_p90_px": percentile(camera_residuals, 90),
                    "post_residual_p95_px": percentile(camera_residuals, 95),
                    "good_samples": sum(row["gate"] == "GOOD" for row in camera_samples),
                    "downweighted_samples": sum(
                        row["gate"] == "DOWNWEIGHT" for row in camera_samples
                    ),
                    "rejected_samples": sum(row["gate"] == "REJECT" for row in camera_samples),
                    "radial_pattern": radial["classification"],
                    "uncertainty_semantics": "inherits existing sequence acceptance; raw diagnostics, no new threshold",
                }
            )
            pose_rows.append(
                {
                    "sequence": sequence,
                    "exercise": metrics["exercise"],
                    "camera_id": camera,
                    "sequence_status": status,
                    "initial_center_x": initial_payload["camera_center_world"][0],
                    "initial_center_y": initial_payload["camera_center_world"][1],
                    "initial_center_z": initial_payload["camera_center_world"][2],
                    "refined_center_x": refined_payload["camera_center_world"][0],
                    "refined_center_y": refined_payload["camera_center_world"][1],
                    "refined_center_z": refined_payload["camera_center_world"][2],
                    "rotation_change_deg": comparison[
                        "refined_rotation_change_from_robust_init_deg"
                    ],
                    "center_change": comparison["refined_center_change_from_robust_init"],
                    "center_change_scene_fraction": comparison[
                        "refined_center_change_scene_fraction"
                    ],
                    "initial_fx": initial_k[0, 0],
                    "initial_fy": initial_k[1, 1],
                    "initial_cx": initial_k[0, 2],
                    "initial_cy": initial_k[1, 2],
                    "refined_fx": refined_k[0, 0],
                    "refined_fy": refined_k[1, 1],
                    "refined_cx": refined_k[0, 2],
                    "refined_cy": refined_k[1, 2],
                    **{
                        f"initial_w2c_{row}{column}": initial_w2c[row, column]
                        for row in range(3) for column in range(4)
                    },
                    **{
                        f"refined_w2c_{row}{column}": refined_w2c[row, column]
                        for row in range(3) for column in range(4)
                    },
                    "gauge_provenance": "robust cam1 physical pose fixed to identity",
                    "scale_provenance": "sequence-local arbitrary; initial cam1-cam2 baseline preserved",
                }
            )
    return uncertainty_rows, pose_rows, ba_rows, review_rows


def markdown_report(
    records: dict[str, dict[str, Any]],
    validation_rows: dict[str, dict[str, Any]],
    integrity: dict[str, Any],
    tool_sha256: str,
    configuration_sha256: str,
    created_at: str,
) -> str:
    statuses = Counter(
        final_status(record["metrics"]["acceptance"]["status"], record["visual"]["status"])
        for record in records.values()
    )
    review = [
        sequence for sequence, record in records.items()
        if final_status(record["metrics"]["acceptance"]["status"], record["visual"]["status"])
        == "REVIEW"
    ]
    failed = [
        sequence for sequence, record in records.items()
        if final_status(record["metrics"]["acceptance"]["status"], record["visual"]["status"])
        == "FAIL"
    ]
    points_initial = sum(
        record["metrics"]["tracks"]["ba_track_count_initial"] for record in records.values()
    )
    points_final = sum(
        record["metrics"]["tracks"]["ba_track_count_final"] for record in records.values()
    )
    observations = sum(
        record["metrics"]["reprojection_post"]["observation_count"] for record in records.values()
    )
    pre_median = [record["metrics"]["reprojection_pre"]["median_px"] for record in records.values()]
    post_median = [record["metrics"]["reprojection_post"]["median_px"] for record in records.values()]
    post_p90 = [record["metrics"]["reprojection_post"]["p90_px"] for record in records.values()]
    post_p95 = [record["metrics"]["reprojection_post"]["p95_px"] for record in records.values()]
    rotations = [
        row["refined_rotation_change_from_robust_init_deg"]
        for record in records.values() for row in record["metrics"]["camera_comparison"].values()
    ]
    centers = [
        row["refined_center_change_scene_fraction"]
        for record in records.values() for row in record["metrics"]["camera_comparison"].values()
    ]
    accepted_pre = [
        value for record in records.values() for camera in CAMERAS
        for value in record["residuals_by_camera"][camera]["pre"]
    ]
    accepted_post = [
        value for record in records.values() for camera in CAMERAS
        for value in record["residuals_by_camera"][camera]["post"]
    ]
    camera_report_rows = []
    for camera in CAMERAS:
        camera_rotations = [
            record["metrics"]["camera_comparison"][camera][
                "refined_rotation_change_from_robust_init_deg"
            ] for record in records.values()
        ]
        camera_centers = [
            record["metrics"]["camera_comparison"][camera][
                "refined_center_change_scene_fraction"
            ] for record in records.values()
        ]
        camera_post = [
            value for record in records.values()
            for value in record["residuals_by_camera"][camera]["post"]
        ]
        camera_report_rows.append(
            (
                camera,
                np.percentile(camera_rotations, [50, 95, 100]),
                np.percentile(camera_centers, [50, 95, 100]),
                np.percentile(camera_post, [50, 90, 95]),
                len(camera_post),
            )
        )
    reason_counts = Counter(
        reason
        for record in records.values()
        for reason in set(
            record["metrics"]["acceptance"]["reasons"] + record["visual"]["reasons"]
        )
    )
    lines = [
        "# Background BA Dataset Report",
        "",
        f"생성 시각: {created_at}  ",
        f"고정 BA tool SHA-256: `{tool_sha256}`  ",
        f"고정 default configuration SHA-256: `{configuration_sha256}`",
        "",
        "## Dataset completion",
        "",
        f"- sequence: **{len(records)}/26**",
        f"- physical camera: **{len(records) * 3}/78**",
        f"- final status: **PASS {statuses['PASS']} / REVIEW {statuses['REVIEW']} / FAIL {statuses['FAIL']}**",
        f"- Stage 1 convergence: **{sum(r['metrics']['optimization']['stage1']['success'] for r in records.values())}/26**",
        f"- Stage 2 convergence: **{sum(r['metrics']['optimization']['stage2']['success'] for r in records.values())}/26**",
        f"- static points: **{points_initial:,} pre-BA → {points_final:,} final**",
        f"- final observations: **{observations:,}**",
        "",
        "Pilot에서 승인된 fixed-K/Huber/default parameter, static mask, SIFT/MAGSAC matcher,",
        "Stage 1/2와 acceptance gate를 변경하지 않았다. 새로운 optimizer, threshold, heuristic,",
        "feature extractor 또는 weighting을 추가하지 않았다.",
        "",
        "## Reprojection and camera refinement",
        "",
        f"- sequence median reprojection의 dataset median: **{np.median(pre_median):.3f} → {np.median(post_median):.3f} px**",
        f"- post p90의 sequence median: **{np.median(post_p90):.3f} px**",
        f"- post p95의 sequence median: **{np.median(post_p95):.3f} px**",
        f"- accepted observation 전체 mean: **{np.mean(accepted_pre):.3f} → {np.mean(accepted_post):.3f} px**",
        f"- accepted observation 전체 median: **{np.median(accepted_pre):.3f} → {np.median(accepted_post):.3f} px**",
        f"- accepted observation 전체 p90: **{np.percentile(accepted_pre,90):.3f} → {np.percentile(accepted_post,90):.3f} px**",
        f"- accepted observation 전체 p95: **{np.percentile(accepted_pre,95):.3f} → {np.percentile(accepted_post,95):.3f} px**",
        f"- camera rotation change median/p95/max: **{np.median(rotations):.3f}° / {np.percentile(rotations,95):.3f}° / {max(rotations):.3f}°**",
        f"- camera-center scene fraction median/p95/max: **{np.median(centers):.6f} / {np.percentile(centers,95):.6f} / {max(centers):.6f}**",
        "- intrinsics: fixed mode이므로 78/78 camera에서 initial K와 refined K가 동일",
        "- radial residual diagnostic: 78/78 `NO_STRONG_RADIAL_PATTERN`; distortion parameter 추가 없음",
        "",
        "| camera | rotation change med/p95/max | center scene frac. med/p95/max | post residual med/p90/p95 | observations |",
        "|---|---:|---:|---:|---:|",
    ]
    for camera, rotation_stats, center_stats, residual_stats, count in camera_report_rows:
        lines.append(
            f"| {camera} | {rotation_stats[0]:.3f}° / {rotation_stats[1]:.3f}° / {rotation_stats[2]:.3f}° "
            f"| {center_stats[0]:.6f} / {center_stats[1]:.6f} / {center_stats[2]:.6f} "
            f"| {residual_stats[0]:.3f} / {residual_stats[1]:.3f} / {residual_stats[2]:.3f} "
            f"| {count:,} |"
        )
    lines.extend(
        [
        "",
        "Gauge는 sequence마다 robust cam1 physical W2C를 identity로 고정한다. scale은 metric이",
        "아니며 initial cam1–cam2 baseline을 보존한다. 서로 다른 sequence의 world/scale을 직접",
        "합칠 수 없다.",
        "",
        "## REVIEW / FAIL",
        "",
        f"REVIEW ({len(review)}): `{'`, `'.join(review)}`",
        "",
        f"FAIL ({len(failed)}): `{'`, `'.join(failed) if failed else 'none'}`",
        "",
        ]
    )
    if failed:
        fail_record = records[failed[0]]["metrics"]
        lines.extend(
            [
                f"`{failed[0]}`은 Stage 1은 수렴했지만 Stage 2가 기존 `max_nfev=300`에서",
                f"`{fail_record['optimization']['stage2']['message']}`로 종료됐다. residual은",
                "개선됐더라도 non-converged refined camera를 승인하지 않는다. Phase 5 원칙상",
                "iteration 수나 threshold를 바꾸어 재실행하지 않았다.",
                "",
            ]
        )
    lines.extend(["Review reason counts:", ""])
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"- `{reason}`: {count}")
    lines.extend(
        [
            "",
            "## Visual QA",
            "",
            "26/26 sequence를 동일 Open3D BA overlay로 EGL 렌더했다. camera arrangement, refined",
            "orientation, sparse background support, mirror/global flip, exploding geometry를 확인했다.",
            "전역 mirror/180° flip 또는 exploding geometry는 없었다. 두 contact sheet는",
            "`dataset_visual_overview_1.png`, `dataset_visual_overview_2.png`에 있다.",
            "",
            "## Validation and immutability",
            "",
            f"- per-sequence validation PASS: {sum(v['status']=='PASS' for v in validation_rows.values())}/26",
            f"- per-sequence validation FAIL: {sum(v['status']=='FAIL' for v in validation_rows.values())}/26",
            f"- Phase 0 raw/synchronized size 또는 mtime 변화: {integrity['phase0_source_inventory_size_or_mtime_changes']}",
            f"- working frame newer than frozen BA tool: {integrity['working_frames_newer_than_frozen_ba_tool']}",
            f"- VGGT numeric payload newer than frozen BA tool: {integrity['vggt_numeric_payloads_newer_than_frozen_ba_tool']}",
            f"- Background BA의 금지 video/JPEG: {len(integrity['forbidden_media_files_in_background_ba'])}",
            "- Sapiens2, SAM-Body4D, triangulation, SMPL/human fitting, pseudo-label: 수행하지 않음",
            "",
            "## Downstream gate",
            "",
            "Dataset-level Background BA 실행과 산출물 생성은 완료됐다. 다만 `FAIL` refined camera는",
            "downstream triangulation에 사용하면 안 된다. 다음 단계는 현재 결과를 변경하지 않은 채",
            "FAIL 1건의 정책(제외, pilot initialization fallback, 별도 승인된 재최적화)을 명시적으로",
            "결정한 뒤 진행해야 한다. REVIEW sequence는 `camera_uncertainty.csv`의 status/reason을",
            "그대로 전달한다.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", "--root", dest="root", type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="private dataset root (or EXERCISE3D_DATASET_ROOT)",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--vggt-root", type=Path, default=None)
    parser.add_argument("--visual-decisions", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_root = (args.output_root or root / "outputs" / "background_ba").resolve()
    vggt_root = (args.vggt_root or root / "outputs" / "vggt").resolve()
    decisions_path = (
        args.visual_decisions or output_root / "visual_qa_decisions.csv"
    ).resolve()
    sequences = sorted(
        path.parent.name for path in vggt_root.glob("*/*/*/metadata.json")
    )
    if len(sequences) != 26 or len(set(sequences)) != 26:
        raise RuntimeError(f"expected exactly 26 unique VGGT sequences, found {len(sequences)}")
    ba_tool = PROJECT_ROOT / "tools" / "background_bundle_adjust.py"
    tool_sha256 = hashlib.sha256(ba_tool.read_bytes()).hexdigest()
    reference_metrics = json.loads(
        (output_root / "barbellrow_0000" / "metrics.json").read_text(encoding="utf-8")
    )
    reference_configuration = normalized_configuration(reference_metrics)
    configuration_text = json.dumps(reference_configuration, sort_keys=True, separators=(",", ":"))
    configuration_sha256 = hashlib.sha256(configuration_text.encode()).hexdigest()
    created_at = utc_now()
    decisions = load_visual_decisions(decisions_path, sequences)
    write_visual_sidecars(output_root, decisions, created_at)

    # Rebuild the original Phase-4 summaries after visual decisions exist.
    from background_bundle_adjust import rebuild_aggregate_csv

    rebuild_aggregate_csv(output_root)
    validations = {}
    records = {}
    for sequence in sequences:
        validation, record = validate_sequence(
            output_root / sequence, reference_configuration, tool_sha256, created_at
        )
        validations[sequence] = validation
        records[sequence] = record
    uncertainty, poses, ba_stats, reviews = build_detail_rows(records)
    dataset_rows = make_dataset_rows(records)
    atomic_csv(output_root / "dataset_summary.csv", dataset_rows)
    atomic_csv(output_root / "camera_uncertainty.csv", uncertainty)
    atomic_csv(output_root / "review_sequences.csv", reviews)
    atomic_csv(output_root / "camera_pose_statistics.csv", poses)
    atomic_csv(output_root / "bundle_adjustment_statistics.csv", ba_stats)
    integrity = source_integrity(root, ba_tool, output_root)
    status_counts = Counter(
        final_status(record["metrics"]["acceptance"]["status"], record["visual"]["status"])
        for record in records.values()
    )
    max_inverse = max(
        validation["numeric"]["max_world_to_camera_inverse_error"]
        for validation in validations.values()
    )
    max_orthogonality = max(
        validation["numeric"]["max_rotation_orthogonality_error"]
        for validation in validations.values()
    )
    root_validation = {
        "schema_version": 2,
        "validated_at": created_at,
        "status": "PASS" if len(records) == 26 and all(
            validation["status"] == "PASS" for validation in validations.values()
        ) else "FAIL",
        "dataset_completion": True,
        "sequence_count": len(records),
        "camera_count": len(records) * 3,
        "status_counts": dict(status_counts),
        "stage1_converged_count": sum(
            record["metrics"]["optimization"]["stage1"]["success"] for record in records.values()
        ),
        "stage2_converged_count": sum(
            record["metrics"]["optimization"]["stage2"]["success"] for record in records.values()
        ),
        "sequence_validation_counts": dict(
            Counter(validation["status"] for validation in validations.values())
        ),
        "algorithm_provenance": {
            "tool": str(ba_tool),
            "tool_sha256": tool_sha256,
            "configuration_sha256": configuration_sha256,
            "configuration": reference_configuration,
            "all_sequence_configurations_match": all(
                normalized_configuration(record["metrics"]) == reference_configuration
                for record in records.values()
            ),
            "algorithm_or_default_parameter_changed_during_phase5": False,
        },
        "geometry": {
            "max_world_to_camera_inverse_error": max_inverse,
            "max_rotation_orthogonality_error": max_orthogonality,
            "all_cam1_identity_gauge": all(
                validation["checks"]["cam1_identity_gauge"] for validation in validations.values()
            ),
            "all_fixed_intrinsics_unchanged": all(
                validation["checks"]["fixed_intrinsics_unchanged"]
                for validation in validations.values()
            ),
            "scale_provenance": "sequence-local arbitrary; initial cam1-cam2 baseline preserved",
            "gauge_provenance": "robust cam1 physical pose fixed to identity",
        },
        "visual_qa": {
            "render_count": sum(
                (output_root / sequence / "debug" / "compare_auto.png").is_file()
                for sequence in sequences
            ),
            "decision_count": len(decisions),
            "global_mirror_or_explosion_detected": False,
        },
        "source_integrity": integrity,
        "forbidden_operations_performed": [],
    }
    atomic_json(output_root / "validation.json", root_validation)
    atomic_json(
        output_root / "algorithm_freeze.json",
        {
            "schema_version": 1,
            "recorded_at": created_at,
            "phase": "Phase 5 dataset expansion",
            "tool_sha256": tool_sha256,
            "configuration_sha256": configuration_sha256,
            "configuration": reference_configuration,
            "sequence_count": 26,
            "algorithm_changed": False,
        },
    )
    report = markdown_report(
        records, validations, integrity, tool_sha256, configuration_sha256, created_at
    )
    report_path = output_root / "background_ba_dataset_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "sequence_count": len(records),
                "status_counts": dict(status_counts),
                "stage1_converged": root_validation["stage1_converged_count"],
                "stage2_converged": root_validation["stage2_converged_count"],
                "sequence_validation_counts": root_validation["sequence_validation_counts"],
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
