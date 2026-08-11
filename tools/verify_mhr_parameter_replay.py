#!/usr/bin/env python3
"""Verify that compact MHR parameters exactly reproduce saved numeric output."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mhr-model", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--maximum-absolute-error", type=float, default=1e-5)
    return parser


def maximum_absolute(reference: np.ndarray, actual: np.ndarray) -> float:
    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("replay comparison contains non-finite values")
    return float(np.max(np.abs(reference.astype(np.float64) - actual.astype(np.float64))))


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = build_parser().parse_args()
    if args.maximum_absolute_error <= 0:
        raise RuntimeError("maximum absolute error must be positive")
    import torch

    with np.load(args.sample_npz.resolve(), allow_pickle=False) as sample:
        required = {
            "shape_params",
            "expr_params",
            "mhr_model_params",
            "pred_keypoints_3d",
            "pred_vertices",
            "pred_joint_coords",
        }
        missing = required - set(sample.files)
        if missing:
            raise RuntimeError(f"sample is missing MHR replay fields: {sorted(missing)}")
        arrays = {key: sample[key].copy() for key in required}
    state = torch.load(
        args.checkpoint.resolve(),
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    mapping = state["head_pose.keypoint_mapping"]
    model = torch.jit.load(str(args.mhr_model.resolve()), map_location="cpu").eval()
    with torch.inference_mode():
        vertices, skeleton = model(
            torch.from_numpy(arrays["shape_params"])[None],
            torch.from_numpy(arrays["mhr_model_params"])[None],
            torch.from_numpy(arrays["expr_params"])[None],
        )
        joint_coordinates = skeleton[..., :3]
        combined = torch.cat([vertices, joint_coordinates], dim=1) / 100.0
        keypoints = torch.einsum("kn,bnc->bkc", mapping, combined)
        keypoints[:, :, 1:3] *= -1
        replay_vertices = vertices / 100.0
        replay_vertices[:, :, 1:3] *= -1
        replay_joints = joint_coordinates / 100.0
        replay_joints[:, :, 1:3] *= -1
    errors = {
        "keypoints_max_abs_m": maximum_absolute(
            arrays["pred_keypoints_3d"], keypoints[0, :70].cpu().numpy()
        ),
        "vertices_max_abs_m": maximum_absolute(
            arrays["pred_vertices"], replay_vertices[0].cpu().numpy()
        ),
        "joint_coordinates_max_abs_m": maximum_absolute(
            arrays["pred_joint_coords"], replay_joints[0].cpu().numpy()
        ),
    }
    status = (
        "PASS"
        if max(errors.values()) <= args.maximum_absolute_error
        else "FAIL_NUMERICAL_REPLAY"
    )
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "threshold_max_abs_m": args.maximum_absolute_error,
        "sample_name": args.sample_npz.name,
        "model_parameter_count": int(arrays["mhr_model_params"].size),
        "keypoint_mapping_shape": list(mapping.shape),
        "mhr_vertex_count": int(vertices.shape[1]),
        "mhr_joint_count": int(joint_coordinates.shape[1]),
        **errors,
        "semantics": "serialization replay only; not a model accuracy or GT validation",
    }
    atomic_json(args.output_json.resolve(), result)
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
