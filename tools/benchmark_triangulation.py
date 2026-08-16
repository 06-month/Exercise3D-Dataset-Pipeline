#!/usr/bin/env python3
"""Synthetic throughput benchmark for future timestamp-aware 3-view DLT."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=2000)
    parser.add_argument("--joints", type=int, default=308)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def project(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)], axis=-1
    )
    projected = homogeneous @ matrix.T
    return projected[..., :2] / projected[..., 2:3]


def triangulate(observations: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    frames, cameras, joints, _ = observations.shape
    rows = []
    for camera in range(cameras):
        x = observations[:, camera, :, 0]
        y = observations[:, camera, :, 1]
        matrix = matrices[camera]
        rows.append(x[..., None] * matrix[2] - matrix[0])
        rows.append(y[..., None] * matrix[2] - matrix[1])
    system = np.stack(rows, axis=-2).reshape(frames * joints, 2 * cameras, 4)
    _, _, vh = np.linalg.svd(system, full_matrices=False)
    homogeneous = vh[:, -1]
    xyz = homogeneous[:, :3] / homogeneous[:, 3:4]
    return xyz.reshape(frames, joints, 3)


def main() -> int:
    args = parse_args()
    if args.frames < 1 or args.joints < 1 or args.repeats < 1:
        raise RuntimeError("frames, joints, and repeats must be positive")
    rng = np.random.default_rng(20260815)
    intrinsics = np.array(
        [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    centers = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.2], [-1.5, 0.2, 1.0]])
    matrices = []
    for center in centers:
        extrinsic = np.concatenate([np.eye(3), -center[:, None]], axis=1)
        matrices.append(intrinsics @ extrinsic)
    matrices_array = np.stack(matrices)
    truth = rng.normal(size=(args.frames, args.joints, 3))
    truth[..., 2] = np.abs(truth[..., 2]) + 5.0
    observations = np.stack(
        [project(truth, matrix) for matrix in matrices_array], axis=1
    )
    observations += rng.normal(scale=0.5, size=observations.shape)

    triangulate(observations[: min(10, args.frames)], matrices_array)
    latencies = []
    reconstruction = None
    for _ in range(args.repeats):
        started = time.perf_counter()
        reconstruction = triangulate(observations, matrices_array)
        latencies.append(time.perf_counter() - started)
    assert reconstruction is not None
    finite = bool(np.isfinite(reconstruction).all())
    rmse = float(np.sqrt(np.mean((reconstruction - truth) ** 2)))
    median = float(np.median(latencies))
    row = {
        "status": "PASS" if finite else "FAIL",
        "benchmark_kind": "synthetic_vectorized_3view_dlt_no_ransac",
        "frames": args.frames,
        "joints_per_frame": args.joints,
        "observations_per_joint": 3,
        "repeats": args.repeats,
        "latency_median_seconds": median,
        "frames_per_second": args.frames / median,
        "seconds_per_frame": median / args.frames,
        "finite": finite,
        "reconstruction_rmse_world_units": rmse,
        "full_dataset_estimate_seconds": 65595 * median / args.frames,
        "scope_note": "DLT core only; excludes timestamp pairing, robust gating, and I/O",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    os.replace(temporary, args.output)
    print(row)
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
