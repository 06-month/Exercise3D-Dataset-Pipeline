#!/usr/bin/env python3
"""Build non-calibrated frame/sequence quality vectors from frozen evidence."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("cam1", "cam2", "cam3")
FRAME_STATUS_CODES = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
TARGET_STATUS_CODES = {
    "UNMAPPED": 0,
    "TARGET": 1,
    "TARGET_AMBIGUOUS": 2,
    "NO_TARGET": 3,
    "OTHER": 4,
}
SAM_STATUS_CODES = {"UNMAPPED": 0, "ACCEPTED": 1, "REJECTED_OR_INVALID": 2}
QUALITY_FLAG_BITS = {
    "TARGET_VIEW_MISSING_OR_ABSTAINED": 1 << 0,
    "IDENTITY_RISK": 1 << 1,
    "OCCLUSION_RISK": 1 << 2,
    "SAM_PRIOR_REJECTED_OR_INVALID": 1 << 3,
    "TRIANGULATION_JOINT_MISSING": 1 << 4,
    "PRIOR_ONLY_JOINT_USED": 1 << 5,
    "BODY_JOINT_MISSING": 1 << 6,
    "MODE_C_REVIEW_CANDIDATE": 1 << 7,
    "SEQUENCE_CAMERA_REVIEW": 1 << 8,
}
REQUIRED_QUALITY_FIELDS = {
    "frame_index",
    "timestamp_pts_seconds",
    "frame_status_code",
    "quality_flag_bits",
    "target_status_code",
    "target_mapped",
    "target_selection_confidence",
    "identity_risk",
    "detector_duplicate_count",
    "possible_reflection_count",
    "pose_valid_joint_fraction",
    "sam_status_code",
    "sam_occlusion_risk",
    "sam_failure_reason",
    "sam_time_error_ms",
    "canonical_triangulation_valid_fraction",
    "canonical_triangulation_quality_median",
    "triangulation_reprojection_median_px",
    "triangulation_ray_angle_median_deg",
    "body_valid_joint_fraction",
    "body_confidence_median",
    "body_evidence_fraction",
    "body_observation_residual_normalized",
    "sam_alignment_residual_normalized",
    "mode_c_review_candidate",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty sequence list")
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object JSON: {path}")
    return value


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finite_row_median(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("finite_row_median expects at least two dimensions")
    flattened = array.reshape(len(array), -1)
    result = np.full(len(array), np.nan, dtype=np.float64)
    finite_rows = np.isfinite(flattened).any(axis=1)
    if finite_rows.any():
        result[finite_rows] = np.nanmedian(flattened[finite_rows], axis=1)
    return result


def source_positions(frame_index: np.ndarray, requested: np.ndarray) -> np.ndarray:
    lookup = {int(value): index for index, value in enumerate(frame_index)}
    return np.asarray([lookup.get(int(value), -1) for value in requested], dtype=np.int32)


def mapped_values(
    values: np.ndarray, positions: np.ndarray, fill: Any
) -> tuple[np.ndarray, np.ndarray]:
    valid = (positions >= 0) & (positions < len(values))
    shape = (len(positions),) + values.shape[1:]
    result = np.full(shape, fill, dtype=values.dtype)
    result[valid] = values[positions[valid]]
    return result, valid


def finite_nan_contract(points: np.ndarray, valid: np.ndarray) -> bool:
    return bool(np.isfinite(points[valid]).all() and np.isnan(points[~valid]).all())


def mode_c_reference_mask(metadata: dict[str, Any], frame_count: int) -> np.ndarray:
    result = np.zeros((frame_count, len(CAMERAS)), dtype=np.bool_)
    cameras = {str(row.get("camera")): row for row in metadata.get("cameras", [])}
    for camera_index, camera in enumerate(CAMERAS):
        row = cameras.get(camera, {})
        for clip in row.get("clips_reference_timeline", []):
            start = max(0, int(clip["start_frame_index"]))
            stop = min(frame_count, int(clip["end_frame_index"]) + 1)
            result[start:stop, camera_index] = True
    return result


def validate_quality_output(
    output_root: Path, body_path: Path
) -> tuple[bool, list[str], dict[str, Any] | None]:
    reasons: list[str] = []
    metadata_path = output_root / "metadata.json"
    vector_path = output_root / "quality_vector.npz"
    try:
        metadata = read_json(metadata_path)
        quality = read_npz(vector_path)
        body = read_npz(body_path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False, ["missing_or_invalid_quality_output"], None
    missing = sorted(REQUIRED_QUALITY_FIELDS - set(quality))
    if missing:
        reasons.extend(f"missing_field:{key}" for key in missing)
        return False, reasons, metadata
    frame_count = len(body["frame_index"])
    if not np.array_equal(quality["frame_index"], body["frame_index"]):
        reasons.append("frame_index_mismatch")
    if not np.array_equal(
        quality["timestamp_pts_seconds"], body["timestamp_pts_seconds"]
    ):
        reasons.append("timestamp_mismatch")
    expected_shapes = {
        "frame_status_code": (frame_count,),
        "quality_flag_bits": (frame_count,),
        "target_status_code": (frame_count, 3),
        "target_mapped": (frame_count, 3),
        "pose_valid_joint_fraction": (frame_count, 3),
        "sam_status_code": (frame_count, 3),
        "body_evidence_fraction": (frame_count, 4),
        "sam_alignment_residual_normalized": (frame_count, 3),
        "mode_c_review_candidate": (frame_count, 3),
    }
    for key, shape in expected_shapes.items():
        if quality[key].shape != shape:
            reasons.append(f"shape_mismatch:{key}")
    if not np.isin(
        quality["frame_status_code"], list(FRAME_STATUS_CODES.values())
    ).all():
        reasons.append("frame_status_code_invalid")
    if not np.isfinite(quality["body_valid_joint_fraction"]).all():
        reasons.append("body_valid_fraction_nonfinite")
    qa = metadata.get("qa", {})
    try:
        metadata_frame_count = int(qa.get("frame_count", -1))
    except (TypeError, ValueError):
        metadata_frame_count = -1
    if metadata_frame_count != frame_count:
        reasons.append("metadata_frame_count_mismatch")
    status_counts = {
        "pass_frame_count": int(
            (quality["frame_status_code"] == FRAME_STATUS_CODES["PASS"]).sum()
        ),
        "review_frame_count": int(
            (quality["frame_status_code"] == FRAME_STATUS_CODES["REVIEW"]).sum()
        ),
        "fail_frame_count": int(
            (quality["frame_status_code"] == FRAME_STATUS_CODES["FAIL"]).sum()
        ),
    }
    for key, value in status_counts.items():
        try:
            metadata_value = int(qa.get(key, -1))
        except (TypeError, ValueError):
            metadata_value = -1
        if metadata_value != value:
            reasons.append(f"metadata_{key}_mismatch")
    return not reasons, reasons, metadata


def compute_quality_vectors(
    selections: dict[str, dict[str, np.ndarray]],
    poses: dict[str, dict[str, np.ndarray]],
    sam_priors: dict[str, dict[str, np.ndarray]],
    triangulated: dict[str, np.ndarray],
    canonical: dict[str, np.ndarray],
    body: dict[str, np.ndarray],
    triangulation_metadata: dict[str, Any],
    body_metadata: dict[str, Any],
    mode_c_metadata: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frame_index = body["frame_index"].astype(np.int32)
    timestamp = body["timestamp_pts_seconds"].astype(np.float64)
    frame_count = len(frame_index)
    if body["sam_source_frame_index"].shape != (frame_count, len(CAMERAS)):
        raise RuntimeError("body sam_source_frame_index shape mismatch")
    for payload, label in ((canonical, "canonical"), (triangulated, "triangulated")):
        if not np.array_equal(payload["frame_index"], frame_index):
            raise RuntimeError(f"{label} frame index mismatch")
        if not np.array_equal(payload["timestamp_pts_seconds"], timestamp):
            raise RuntimeError(f"{label} timestamp mismatch")
    if not np.array_equal(canonical["joint_names"], body["joint_names"]):
        raise RuntimeError("canonical/body joint convention mismatch")
    if not finite_nan_contract(body["keypoints_3d"], body["valid_mask"]):
        raise RuntimeError("body finite/NaN contract failure")
    if not finite_nan_contract(canonical["keypoints_3d"], canonical["valid_mask"]):
        raise RuntimeError("canonical finite/NaN contract failure")

    target_status_code = np.zeros((frame_count, 3), dtype=np.uint8)
    target_selection_confidence = np.full((frame_count, 3), np.nan, dtype=np.float32)
    target_mapped = np.zeros((frame_count, 3), dtype=np.bool_)
    identity_risk = np.zeros((frame_count, 3), dtype=np.bool_)
    detector_duplicate_count = np.zeros((frame_count, 3), dtype=np.int16)
    possible_reflection_count = np.zeros((frame_count, 3), dtype=np.int16)
    pose_valid_joint_fraction = np.full((frame_count, 3), np.nan, dtype=np.float32)
    sam_status_code = np.zeros((frame_count, 3), dtype=np.uint8)
    occlusion_risk = np.zeros((frame_count, 3), dtype=np.bool_)
    sam_failure = np.full((frame_count, 3), "", dtype="<U48")

    requested = body["sam_source_frame_index"].astype(np.int32)
    for camera_index, camera in enumerate(CAMERAS):
        selection = selections[camera]
        pose = poses[camera]
        sam = sam_priors[camera]
        selected_positions = source_positions(selection["frame_index"], requested[:, camera_index])
        pose_positions = source_positions(pose["frame_index"], requested[:, camera_index])
        sam_positions = source_positions(sam["source_frame_index"], requested[:, camera_index])

        statuses, selection_mapped = mapped_values(
            selection["target_status"].astype("<U20"), selected_positions, ""
        )
        ambiguous, _ = mapped_values(
            selection["target_ambiguous"].astype(np.bool_), selected_positions, False
        )
        no_target, _ = mapped_values(
            selection["no_target"].astype(np.bool_), selected_positions, False
        )
        confidence, _ = mapped_values(
            selection["target_selection_confidence"].astype(np.float32),
            selected_positions,
            np.nan,
        )
        risk = np.zeros(frame_count, dtype=np.bool_)
        for key in (
            "identity_switch_risk",
            "global_track_ambiguity",
            "association_ambiguity",
            "target_fragmentation_risk",
        ):
            if key in selection:
                mapped, _ = mapped_values(
                    selection[key].astype(np.bool_), selected_positions, False
                )
                risk |= mapped
        duplicates, _ = mapped_values(
            selection.get(
                "detector_duplicate_count", np.zeros(len(selection["frame_index"]), dtype=np.int16)
            ).astype(np.int16),
            selected_positions,
            0,
        )
        reflections, _ = mapped_values(
            selection.get(
                "possible_reflection_count", np.zeros(len(selection["frame_index"]), dtype=np.int16)
            ).astype(np.int16),
            selected_positions,
            0,
        )
        codes = np.full(frame_count, TARGET_STATUS_CODES["OTHER"], dtype=np.uint8)
        codes[~selection_mapped] = TARGET_STATUS_CODES["UNMAPPED"]
        codes[selection_mapped & (statuses == "TARGET")] = TARGET_STATUS_CODES["TARGET"]
        codes[selection_mapped & ambiguous] = TARGET_STATUS_CODES["TARGET_AMBIGUOUS"]
        codes[selection_mapped & no_target] = TARGET_STATUS_CODES["NO_TARGET"]

        pose_valid, pose_mapped = mapped_values(
            pose["valid_mask"].astype(np.bool_), pose_positions, False
        )
        pose_fraction = pose_valid.mean(axis=1).astype(np.float32)
        pose_fraction[~pose_mapped] = np.nan

        sam_accepted, sam_mapped = mapped_values(
            sam["accepted_prior"].astype(np.bool_), sam_positions, False
        )
        sam_output_valid, _ = mapped_values(
            sam["output_valid"].astype(np.bool_), sam_positions, False
        )
        sam_occlusion, _ = mapped_values(
            sam["occlusion_risk"].astype(np.bool_), sam_positions, False
        )
        failures, _ = mapped_values(
            sam["failure_reason"].astype("<U48"), sam_positions, ""
        )
        sam_codes = np.full(
            frame_count, SAM_STATUS_CODES["REJECTED_OR_INVALID"], dtype=np.uint8
        )
        sam_codes[~sam_mapped] = SAM_STATUS_CODES["UNMAPPED"]
        sam_codes[sam_mapped & sam_accepted & sam_output_valid] = SAM_STATUS_CODES["ACCEPTED"]

        target_status_code[:, camera_index] = codes
        target_selection_confidence[:, camera_index] = confidence
        target_mapped[:, camera_index] = selection_mapped
        identity_risk[:, camera_index] = risk
        detector_duplicate_count[:, camera_index] = duplicates
        possible_reflection_count[:, camera_index] = reflections
        pose_valid_joint_fraction[:, camera_index] = pose_fraction
        sam_status_code[:, camera_index] = sam_codes
        occlusion_risk[:, camera_index] = sam_occlusion
        sam_failure[:, camera_index] = failures

    target_accepted = target_status_code == TARGET_STATUS_CODES["TARGET"]
    sam_accepted = sam_status_code == SAM_STATUS_CODES["ACCEPTED"]
    canonical_valid_fraction = canonical["valid_mask"].mean(axis=1).astype(np.float32)
    body_valid_fraction = body["valid_mask"].mean(axis=1).astype(np.float32)
    triangulation_quality_median = finite_row_median(canonical["quality_score"]).astype(np.float32)
    reprojection_median_px = finite_row_median(
        triangulated["per_view_reprojection_px"]
    ).astype(np.float32)
    ray_angle_median_deg = finite_row_median(
        triangulated["min_ray_angle_deg"]
    ).astype(np.float32)
    body_confidence_median = finite_row_median(body["confidence"]).astype(np.float32)
    evidence = body["evidence_type"]
    evidence_fraction = np.stack(
        [(evidence == code).mean(axis=1) for code in range(4)], axis=1
    ).astype(np.float32)
    reference_length = float(
        body_metadata["qa"]["anthropometry"]["reference_length_sequence_gauge"]
    )
    if not np.isfinite(reference_length) or reference_length <= 0:
        raise RuntimeError("invalid anthropometric reference")
    observation_residual_normalized = (
        finite_row_median(body["observation_residual_sequence_gauge"]) / reference_length
    ).astype(np.float32)
    alignment_residual_normalized = (
        body["alignment_residual_sequence_gauge"].astype(np.float64) / reference_length
    ).astype(np.float32)
    mode_c_candidate = mode_c_reference_mask(mode_c_metadata, frame_count)

    flags = np.zeros(frame_count, dtype=np.uint32)

    def set_flag(name: str, condition: np.ndarray | bool) -> None:
        nonlocal flags
        flags[np.asarray(condition, dtype=np.bool_)] |= np.uint32(QUALITY_FLAG_BITS[name])

    set_flag("TARGET_VIEW_MISSING_OR_ABSTAINED", target_accepted.sum(axis=1) < 3)
    set_flag("IDENTITY_RISK", identity_risk.any(axis=1))
    set_flag("OCCLUSION_RISK", occlusion_risk.any(axis=1))
    set_flag("SAM_PRIOR_REJECTED_OR_INVALID", sam_accepted.sum(axis=1) < 3)
    set_flag("TRIANGULATION_JOINT_MISSING", canonical_valid_fraction < 1.0)
    set_flag("PRIOR_ONLY_JOINT_USED", (evidence == 3).any(axis=1))
    set_flag("BODY_JOINT_MISSING", body_valid_fraction < 1.0)
    set_flag("MODE_C_REVIEW_CANDIDATE", mode_c_candidate.any(axis=1))
    camera_status = str(triangulation_metadata["qa"]["camera_acceptance"])
    if camera_status != "PASS":
        flags |= np.uint32(QUALITY_FLAG_BITS["SEQUENCE_CAMERA_REVIEW"])
    frame_status = np.where(
        flags == 0, FRAME_STATUS_CODES["PASS"], FRAME_STATUS_CODES["REVIEW"]
    ).astype(np.uint8)

    body_status = str(body_metadata["qa"]["status"])
    mode_c_status = str(mode_c_metadata["status"])
    triangulation_status = str(triangulation_metadata["qa"]["quality_status"])
    sequence_status = (
        "FAIL"
        if body_status.startswith("FAIL") or triangulation_status.startswith("FAIL")
        else "REVIEW"
        if body_status.startswith("REVIEW")
        or triangulation_status.startswith("REVIEW")
        or mode_c_status == "REVIEW_MODE_C_CANDIDATE"
        or np.any(frame_status == FRAME_STATUS_CODES["REVIEW"])
        else "PASS"
    )
    flag_counts = {
        name: int(((flags & np.uint32(bit)) != 0).sum())
        for name, bit in QUALITY_FLAG_BITS.items()
    }
    arrays = {
        "frame_index": frame_index,
        "timestamp_pts_seconds": timestamp,
        "frame_status_code": frame_status,
        "quality_flag_bits": flags,
        "target_status_code": target_status_code,
        "target_mapped": target_mapped,
        "target_selection_confidence": target_selection_confidence,
        "identity_risk": identity_risk,
        "detector_duplicate_count": detector_duplicate_count,
        "possible_reflection_count": possible_reflection_count,
        "pose_valid_joint_fraction": pose_valid_joint_fraction,
        "sam_status_code": sam_status_code,
        "sam_occlusion_risk": occlusion_risk,
        "sam_failure_reason": sam_failure,
        "sam_time_error_ms": body["sam_time_error_ms"].astype(np.float32),
        "canonical_triangulation_valid_fraction": canonical_valid_fraction,
        "canonical_triangulation_quality_median": triangulation_quality_median,
        "triangulation_reprojection_median_px": reprojection_median_px,
        "triangulation_ray_angle_median_deg": ray_angle_median_deg,
        "body_valid_joint_fraction": body_valid_fraction,
        "body_confidence_median": body_confidence_median,
        "body_evidence_fraction": evidence_fraction,
        "body_observation_residual_normalized": observation_residual_normalized,
        "sam_alignment_residual_normalized": alignment_residual_normalized,
        "mode_c_review_candidate": mode_c_candidate,
    }
    metadata = {
        "schema_version": 1,
        "stage": "PHASE11_PSEUDOLABEL_QUALITY_CONTROL",
        "not_ground_truth": True,
        "not_calibrated_probability": True,
        "scalar_quality_score_defined": False,
        "policy": (
            "Preserve source-specific evidence and categorical reasons; do not collapse correlated "
            "learned signals into a claimed accuracy probability."
        ),
        "frame_status_codes": FRAME_STATUS_CODES,
        "target_status_codes": TARGET_STATUS_CODES,
        "sam_status_codes": SAM_STATUS_CODES,
        "quality_flag_bits": QUALITY_FLAG_BITS,
        "body_evidence_type_codes": body_metadata.get("evidence_type_codes", {}),
        "source_status": {
            "triangulation": triangulation_status,
            "body_fit": body_status,
            "mode_c_assessment": mode_c_status,
            "camera": camera_status,
        },
        "qa": {
            "frame_count": frame_count,
            "pass_frame_count": int((frame_status == FRAME_STATUS_CODES["PASS"]).sum()),
            "review_frame_count": int((frame_status == FRAME_STATUS_CODES["REVIEW"]).sum()),
            "fail_frame_count": int((frame_status == FRAME_STATUS_CODES["FAIL"]).sum()),
            "quality_flag_frame_counts": flag_counts,
            "target_abstention_or_unmapped_view_count": int((~target_accepted).sum()),
            "sam_rejected_or_unmapped_view_count": int((~sam_accepted).sum()),
            "sequence_status": sequence_status,
        },
    }
    return arrays, metadata


def build_sequence_quality(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    output = args.output_root.resolve() / sequence
    body_path = args.body_fit_root.resolve() / sequence / "body_fit.npz"
    complete, _, existing = validate_quality_output(output, body_path)
    if complete and existing is not None:
        return {**existing, "resume_skipped": True}
    selections = {}
    poses = {}
    sam_priors = {}
    for camera in CAMERAS:
        selections[camera] = read_npz(
            args.selection_root.resolve() / sequence / camera / "target_selection.npz"
        )
        poses[camera] = read_npz(
            args.pose_root.resolve() / sequence / camera / "poses_2d.npz"
        )
        sam_priors[camera] = read_npz(
            args.sam_prior_root.resolve() / sequence / camera / "sam_body_prior.npz"
        )
    triangulation_root = args.triangulation_root.resolve() / sequence
    body_root = args.body_fit_root.resolve() / sequence
    triangulated = read_npz(triangulation_root / "triangulated_3d.npz")
    canonical = read_npz(triangulation_root / "canonical_3d.npz")
    body = read_npz(body_root / "body_fit.npz")
    arrays, metadata = compute_quality_vectors(
        selections,
        poses,
        sam_priors,
        triangulated,
        canonical,
        body,
        read_json(triangulation_root / "metadata.json"),
        read_json(body_root / "metadata.json"),
        read_json(
            args.sam_mode_c_review_root.resolve()
            / sequence
            / "mode_c_escalation.json"
        ),
    )
    metadata = {"created_at_utc": utc_now(), "sequence": sequence, **metadata}
    atomic_npz(output / "quality_vector.npz", arrays)
    atomic_json(output / "metadata.json", metadata)
    complete, reasons, _ = validate_quality_output(output, body_path)
    if not complete:
        raise RuntimeError("quality output validation failed: " + ";".join(reasons))
    return {**metadata, "resume_skipped": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--pose-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--sam-mode-c-review-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    return parser


def summarize_quality_outputs(output_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(output_root.glob("*/metadata.json")):
        try:
            metadata = read_json(path)
            if metadata.get("stage") != "PHASE11_PSEUDOLABEL_QUALITY_CONTROL":
                continue
            sequence = str(metadata["sequence"])
            status = str(metadata["qa"]["sequence_status"])
            frame_count = int(metadata["qa"]["frame_count"])
            if status not in FRAME_STATUS_CODES:
                continue
            rows.append(
                {"sequence": sequence, "status": status, "frame_count": frame_count}
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence_count": len(rows),
        "frame_count": sum(row["frame_count"] for row in rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "review_count": sum(row["status"] == "REVIEW" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "sequences": rows,
        "status": "FAIL" if any(row["status"] == "FAIL" for row in rows) else "PASS",
    }


def main() -> int:
    args = build_parser().parse_args()
    rows = []
    for sequence in args.sequences:
        result = build_sequence_quality(args, sequence)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = summarize_quality_outputs(args.output_root.resolve())
    atomic_json(args.output_root.resolve() / "quality_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
