#!/usr/bin/env python3
"""Compute MPJPE, N-MPJPE and PA-MPJPE for a prepared Fit3D evaluation pair."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-npz", type=Path, required=True)
    parser.add_argument("--ground-truth-npz", type=Path, required=True)
    parser.add_argument("--prediction-key", default="keypoints_3d")
    parser.add_argument("--ground-truth-key", default="keypoints_3d")
    parser.add_argument("--valid-key", default="valid_mask")
    parser.add_argument("--root-joint-index", type=int, required=True)
    parser.add_argument(
        "--input-unit-to-millimeters",
        type=float,
        default=1000.0,
        help="1000 for meter inputs, 1 for millimeter inputs",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def root_center(points: np.ndarray, root_index: int) -> np.ndarray:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("pose array must be FxJx3")
    if not 0 <= root_index < points.shape[1]:
        raise ValueError("root joint index is outside joint axis")
    return points - points[:, root_index : root_index + 1]


def scale_align_frame(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    denominator = float(np.sum(prediction * prediction))
    if denominator <= 1e-12:
        raise ValueError("degenerate prediction for scale alignment")
    scale = float(np.sum(prediction * target) / denominator)
    return prediction * scale


def procrustes_align_frame(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(prediction) < 3:
        raise ValueError("Procrustes alignment needs at least three joints")
    prediction_mean = prediction.mean(axis=0)
    target_mean = target.mean(axis=0)
    source = prediction - prediction_mean
    destination = target - target_mean
    covariance = source.T @ destination
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.ones(3)
    correction[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag(correction) @ vt
    denominator = float(np.sum(source * source))
    if denominator <= 1e-12:
        raise ValueError("degenerate prediction for Procrustes alignment")
    scale = float(np.sum(singular * correction) / denominator)
    return scale * (source @ rotation) + target_mean


def evaluate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    root_index: int,
    unit_to_mm: float,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    if prediction.shape != target.shape or prediction.shape[:2] != valid.shape:
        raise ValueError("prediction, target and valid shapes differ")
    if unit_to_mm <= 0:
        raise ValueError("unit conversion must be positive")
    finite = np.isfinite(prediction).all(axis=-1) & np.isfinite(target).all(axis=-1)
    root_usable = valid[:, root_index : root_index + 1] & finite[:, root_index : root_index + 1]
    usable = valid & finite & root_usable
    prediction_centered = root_center(prediction, root_index)
    target_centered = root_center(target, root_index)
    mpjpe_errors = []
    n_errors = []
    pa_errors = []
    frame_metrics = []
    for frame in range(len(prediction)):
        mask = usable[frame]
        if mask.sum() < 3:
            continue
        predicted = prediction_centered[frame, mask]
        actual = target_centered[frame, mask]
        mpjpe = np.linalg.norm(predicted - actual, axis=1)
        scaled = scale_align_frame(predicted, actual)
        n_mpjpe = np.linalg.norm(scaled - actual, axis=1)
        aligned = procrustes_align_frame(predicted, actual)
        pa_mpjpe = np.linalg.norm(aligned - actual, axis=1)
        mpjpe_errors.extend(mpjpe)
        n_errors.extend(n_mpjpe)
        pa_errors.extend(pa_mpjpe)
        frame_metrics.append(
            {
                "frame_index": frame,
                "joint_count": int(mask.sum()),
                "mpjpe_mm": float(np.mean(mpjpe) * unit_to_mm),
                "n_mpjpe_mm": float(np.mean(n_mpjpe) * unit_to_mm),
                "pa_mpjpe_mm": float(np.mean(pa_mpjpe) * unit_to_mm),
            }
        )
    if not frame_metrics:
        raise RuntimeError("no frame has enough valid joints for evaluation")
    return {
        "evaluated_frame_count": len(frame_metrics),
        "evaluated_joint_observation_count": len(mpjpe_errors),
        "mpjpe_mm": float(np.mean(mpjpe_errors) * unit_to_mm),
        "n_mpjpe_mm": float(np.mean(n_errors) * unit_to_mm),
        "pa_mpjpe_mm": float(np.mean(pa_errors) * unit_to_mm),
        "root_alignment": "prediction and GT root joint subtracted per frame",
        "n_mpjpe_alignment": "per-frame scale only after root alignment",
        "pa_mpjpe_alignment": "per-frame similarity Procrustes",
        "frames": frame_metrics,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = build_parser().parse_args()
    with np.load(args.prediction_npz.resolve(), allow_pickle=False) as payload:
        prediction = payload[args.prediction_key].copy()
        prediction_valid = (
            payload[args.valid_key].copy()
            if args.valid_key in payload.files
            else np.isfinite(prediction).all(axis=-1)
        )
    with np.load(args.ground_truth_npz.resolve(), allow_pickle=False) as payload:
        target = payload[args.ground_truth_key].copy()
        target_valid = (
            payload[args.valid_key].copy()
            if args.valid_key in payload.files
            else np.isfinite(target).all(axis=-1)
        )
    result = evaluate_metrics(
        prediction,
        target,
        prediction_valid & target_valid,
        args.root_joint_index,
        args.input_unit_to_millimeters,
    )
    result.update(
        {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "not_executed_on_fit3d_by_implementation_alone": False,
        }
    )
    atomic_json(args.output_json.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
