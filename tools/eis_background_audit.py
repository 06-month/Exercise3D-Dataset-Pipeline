#!/usr/bin/env python3
"""Audit encoded camera stability using static-background feature tracks.

The audit is deliberately read-only with respect to dataset assets.  It decodes
``synced_video/*/*/cam?.mp4`` and writes metrics/figures below
``reports/eis_audit``.  It does not generate corrected or warped video.

OpenCV is required for forward/backward Lucas-Kanade tracking and RANSAC model
fitting (``pip install opencv-python-headless``).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only on missing dependency
    raise SystemExit(
        "OpenCV is required. Install with: pip install opencv-python-headless"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT)
).expanduser()
SET_RE = re.compile(r"^(?P<exercise>[a-z]+)_(?P<take>\d{4})$")
CAM_RE = re.compile(r"^cam(?P<camera>[123])\.mp4$")

# Thresholds are normalized by the decoded proxy diagonal.  Recommendations
# additionally require repetition, model support, and within-sequence view
# comparison; they are never based on one pixel cutoff or a single frame pair.
THRESHOLDS = {
    "min_successful_local_pairs": 20,
    "min_success_fraction": 0.65,
    "min_median_tracks": 30,
    "min_median_homography_inlier_ratio": 0.55,
    "global_motion_effect_norm": 0.0015,
    "long_baseline_effect_norm": 0.0030,
    "spatial_residual_effect_norm": 0.0010,
    "minimum_repeat_fraction": 0.25,
    "global_model_explained_ratio": 0.60,
    "within_sequence_outlier_ratio": 1.50,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def round_or_none(value: Any, digits: int = 9) -> float | None:
    parsed = finite_float(value)
    return round(parsed, digits) if parsed is not None else None


def pct(values: Iterable[float | None], percentile: float) -> float | None:
    data = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if data.size == 0:
        return None
    return float(np.percentile(data, percentile))


def median(values: Iterable[float | None]) -> float | None:
    return pct(values, 50)


def mean(values: Iterable[float | None]) -> float | None:
    data = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(data)) if data else None


def ffprobe_packet_pts(path: Path) -> list[float]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts_time",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "ffprobe packet PTS scan failed")
    timestamps: list[float] = []
    for line in completed.stdout.splitlines():
        token = line.strip().split(",", 1)[0]
        value = finite_float(token)
        if value is not None:
            timestamps.append(value)
    return sorted(timestamps)


def resize_gray(frame: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def decode_grayscale(path: Path, max_dimension: int) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
    timestamps = ffprobe_packet_pts(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open video")
    nominal_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(resize_gray(frame, max_dimension))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("no video frames decoded")
    if len(timestamps) != len(frames):
        # Packet count should equal presentation-frame count for these H.264
        # files.  A small mismatch can occur for malformed containers; keep a
        # clearly marked nominal fallback rather than silently shifting PTS.
        if nominal_fps <= 0:
            raise RuntimeError(
                f"decoded frame / packet PTS mismatch ({len(frames)} vs {len(timestamps)})"
            )
        timestamps = [index / nominal_fps for index in range(len(frames))]
        pts_source = "NOMINAL_FPS_FALLBACK"
    else:
        pts_source = "ALL_VIDEO_PACKET_PTS"
    return frames, timestamps, {
        "source_width": source_width,
        "source_height": source_height,
        "proxy_width": int(frames[0].shape[1]),
        "proxy_height": int(frames[0].shape[0]),
        "nominal_fps": nominal_fps,
        "decoded_frame_count": len(frames),
        "pts_source": pts_source,
    }


def background_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    # Remove the central human/equipment region.  Keep image borders and four
    # corners, where tripod-fixed scene structure is most likely to remain.
    x0, x1 = round(width * 0.22), round(width * 0.78)
    y0, y1 = round(height * 0.18), round(height * 0.82)
    mask[y0:y1, x0:x1] = 0
    # Exclude the very edge, where codec padding and interpolation can create
    # artificial tracks.
    edge = max(2, round(min(height, width) * 0.01))
    mask[:edge, :] = 0
    mask[-edge:, :] = 0
    mask[:, :edge] = 0
    mask[:, -edge:] = 0
    return mask


def temporal_median_background(frames: list[np.ndarray], sample_count: int = 41) -> np.ndarray:
    """Return a robust reference without retaining another full video copy."""
    indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames)), dtype=int)
    stack = np.stack([frames[int(index)] for index in indices], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def dynamic_foreground_mask(frame: np.ndarray, temporal_background: np.ndarray) -> np.ndarray:
    """Identify broad foreground regions while preserving thin shifted edges.

    A global/EIS displacement creates mostly thin difference contours around
    background edges.  People and equipment create broad connected regions.
    Morphological opening and component-area filtering therefore remove the
    former from this *exclusion* mask while retaining the latter.
    """
    difference = cv2.absdiff(frame, temporal_background)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)
    binary = (difference >= 18).astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    minimum_area = max(64, round(frame.size * 0.001))
    foreground = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            foreground[labels == label] = 255
    return cv2.dilate(
        foreground,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )


def motion_energy(frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) < 2:
        return np.empty(0, dtype=np.float64)
    target_max = 160
    tiny: list[np.ndarray] = []
    for frame in frames:
        height, width = frame.shape
        scale = min(1.0, target_max / max(height, width))
        tiny.append(
            cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        )
    return np.asarray(
        [
            float(np.mean(cv2.absdiff(previous, current), dtype=np.float64))
            for previous, current in zip(tiny, tiny[1:])
        ],
        dtype=np.float64,
    )


def separated_extrema(
    energies: np.ndarray,
    ascending: bool,
    count: int,
    separation: int,
    valid_start: int,
    valid_end: int,
) -> list[int]:
    # Energy index j represents the pair (j -> j+1), so center frame is j+1.
    candidates = np.argsort(energies)
    if not ascending:
        candidates = candidates[::-1]
    selected: list[int] = []
    for energy_index in candidates:
        center = int(energy_index) + 1
        if center < valid_start or center > valid_end:
            continue
        if all(abs(center - previous) >= separation for previous in selected):
            selected.append(center)
            if len(selected) >= count:
                break
    return selected


def nearest_index(timestamps: list[float], target: float) -> int:
    array = np.asarray(timestamps)
    return int(np.argmin(np.abs(array - target)))


def select_pairs(
    timestamps: list[float],
    energies: np.ndarray,
    window_radius: int,
) -> tuple[list[dict[str, Any]], list[int], int, tuple[float, float]]:
    frame_count = len(timestamps)
    valid_start = max(window_radius + 1, 2)
    valid_end = min(frame_count - window_radius - 1, frame_count - 2)
    duration = max(timestamps[-1] - timestamps[0], 1e-9)
    coverage_centers = [
        nearest_index(timestamps, timestamps[0] + duration * fraction)
        for fraction in (0.10, 0.50, 0.90)
    ]
    separation = max(window_radius * 3, 15)
    low_centers = separated_extrema(
        energies, True, 3, separation, valid_start, valid_end
    )
    high_centers = separated_extrema(
        energies, False, 3, separation, valid_start, valid_end
    )
    reasons_by_pair: dict[int, set[str]] = defaultdict(set)
    for label, centers in (
        ("coverage", coverage_centers),
        ("static_window", low_centers),
        ("fast_window", high_centers),
    ):
        for center in centers:
            for pair_end in range(center - window_radius + 1, center + window_radius + 1):
                if 1 <= pair_end < frame_count:
                    reasons_by_pair[pair_end].add(label)

    q25 = float(np.percentile(energies, 25)) if energies.size else 0.0
    q75 = float(np.percentile(energies, 75)) if energies.size else 0.0
    pairs: list[dict[str, Any]] = []
    for pair_end in sorted(reasons_by_pair):
        energy = float(energies[pair_end - 1])
        if energy <= q25:
            motion_class = "STATIC"
        elif energy >= q75:
            motion_class = "FAST"
        else:
            motion_class = "MODERATE"
        fraction = (timestamps[pair_end] - timestamps[0]) / duration
        temporal_section = "EARLY" if fraction < 1 / 3 else "MIDDLE" if fraction < 2 / 3 else "LATE"
        pairs.append(
            {
                "start_index": pair_end - 1,
                "end_index": pair_end,
                "selection_reason": "+".join(sorted(reasons_by_pair[pair_end])),
                "motion_class": motion_class,
                "temporal_section": temporal_section,
                "motion_energy": energy,
            }
        )
    reference_index = low_centers[0] if low_centers else coverage_centers[0]
    long_targets = sorted(
        {
            center
            for center in coverage_centers + low_centers + high_centers
            if center != reference_index
        }
    )
    return pairs, long_targets, reference_index, (q25, q75)


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def analyze_pair(
    previous: np.ndarray,
    current: np.ndarray,
    *,
    long_baseline: bool = False,
    include_debug: bool = False,
    foreground_previous: np.ndarray | None = None,
    foreground_current: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    height, width = previous.shape
    diagonal = math.hypot(width, height)
    mask = background_mask(previous.shape)
    foreground_union = np.zeros_like(mask)
    if foreground_previous is not None:
        foreground_union = cv2.bitwise_or(foreground_union, foreground_previous)
    if foreground_current is not None:
        foreground_union = cv2.bitwise_or(foreground_union, foreground_current)
    mask[foreground_union > 0] = 0
    base_background_support = int(np.sum(background_mask(previous.shape) > 0))
    retained_background_support = int(np.sum(mask > 0))
    retained_fraction = (
        retained_background_support / base_background_support if base_background_support else 0.0
    )
    points0 = cv2.goodFeaturesToTrack(
        previous,
        mask=mask,
        maxCorners=700,
        qualityLevel=0.008,
        minDistance=5,
        blockSize=7,
        useHarrisDetector=False,
    )
    if points0 is None or len(points0) < 12:
        return {
            "status": "INSUFFICIENT_FEATURES",
            "detected_features": 0 if points0 is None else len(points0),
            "retained_background_mask_fraction": round_or_none(retained_fraction, 6),
        }, None

    lk_window = (31, 31) if long_baseline else (21, 21)
    max_level = 4 if long_baseline else 3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01)
    points1, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points0,
        None,
        winSize=lk_window,
        maxLevel=max_level,
        criteria=criteria,
    )
    if points1 is None or status_forward is None:
        return {"status": "LK_FORWARD_FAILED", "detected_features": len(points0)}, None
    points0_back, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        current,
        previous,
        points1,
        None,
        winSize=lk_window,
        maxLevel=max_level,
        criteria=criteria,
    )
    if points0_back is None or status_backward is None:
        return {"status": "LK_BACKWARD_FAILED", "detected_features": len(points0)}, None

    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)
    p0_back = points0_back.reshape(-1, 2)
    forward_ok = status_forward.reshape(-1).astype(bool)
    backward_ok = status_backward.reshape(-1).astype(bool)
    fb_error = np.linalg.norm(p0 - p0_back, axis=1)
    fb_limit = 1.25 if long_baseline else 0.75
    inside = (
        (p1[:, 0] >= 0)
        & (p1[:, 0] < width)
        & (p1[:, 1] >= 0)
        & (p1[:, 1] < height)
    )
    keep = forward_ok & backward_ok & inside & np.isfinite(fb_error) & (fb_error <= fb_limit)
    p0 = p0[keep]
    p1 = p1[keep]
    fb_error_kept = fb_error[keep]
    if len(p0) < 12:
        return {
            "status": "INSUFFICIENT_FB_TRACKS",
            "detected_features": len(points0),
            "fb_tracks": len(p0),
        }, None

    reprojection_threshold = 1.75 if long_baseline else 1.25
    homography, homography_mask = cv2.findHomography(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
        maxIters=3000,
        confidence=0.995,
    )
    affine, affine_mask = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    if homography is None or homography_mask is None:
        return {
            "status": "HOMOGRAPHY_FAILED",
            "detected_features": len(points0),
            "fb_tracks": len(p0),
        }, None
    inliers = homography_mask.reshape(-1).astype(bool)
    if int(np.sum(inliers)) < 10:
        return {
            "status": "INSUFFICIENT_RANSAC_INLIERS",
            "detected_features": len(points0),
            "fb_tracks": len(p0),
            "homography_inliers": int(np.sum(inliers)),
        }, None

    flow = p1 - p0
    displacement = np.linalg.norm(flow, axis=1)
    displacement_inliers = displacement[inliers]
    median_flow = np.median(flow[inliers], axis=0)
    predicted = transform_points(p0, homography)
    residual = np.linalg.norm(p1 - predicted, axis=1)
    residual_inliers = residual[inliers]

    grid_values: list[float] = []
    grid_support = 0
    for row in range(4):
        for column in range(4):
            cell = (
                inliers
                & (p0[:, 0] >= width * column / 4)
                & (p0[:, 0] < width * (column + 1) / 4)
                & (p0[:, 1] >= height * row / 4)
                & (p0[:, 1] < height * (row + 1) / 4)
            )
            if int(np.sum(cell)) >= 3:
                grid_values.append(float(np.median(residual[cell])))
                grid_support += 1
    grid_spread = max(grid_values) - min(grid_values) if len(grid_values) >= 2 else None

    translation_x = translation_y = rotation_deg = scale_delta = None
    affine_inlier_ratio = None
    if affine is not None:
        translation_x = float(affine[0, 2])
        translation_y = float(affine[1, 2])
        rotation_deg = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
        scale = math.hypot(float(affine[0, 0]), float(affine[1, 0]))
        scale_delta = scale - 1.0
        if affine_mask is not None:
            affine_inlier_ratio = float(np.mean(affine_mask.reshape(-1).astype(bool)))

    result = {
        "status": "OK",
        "retained_background_mask_fraction": round_or_none(retained_fraction, 6),
        "detected_features": int(len(points0)),
        "fb_tracks": int(len(p0)),
        "fb_error_median_px": round_or_none(float(np.median(fb_error_kept)), 6),
        "homography_inliers": int(np.sum(inliers)),
        "homography_inlier_ratio": round_or_none(float(np.mean(inliers)), 6),
        "affine_inlier_ratio": round_or_none(affine_inlier_ratio, 6),
        "raw_displacement_median_px": round_or_none(float(np.median(displacement_inliers)), 6),
        "raw_displacement_p90_px": round_or_none(float(np.percentile(displacement_inliers, 90)), 6),
        "raw_displacement_p95_px": round_or_none(float(np.percentile(displacement_inliers, 95)), 6),
        "raw_displacement_median_norm": round_or_none(float(np.median(displacement_inliers)) / diagonal),
        "raw_displacement_p95_norm": round_or_none(float(np.percentile(displacement_inliers, 95)) / diagonal),
        "median_flow_x_px": round_or_none(float(median_flow[0]), 6),
        "median_flow_y_px": round_or_none(float(median_flow[1]), 6),
        "median_global_flow_norm": round_or_none(float(np.linalg.norm(median_flow)) / diagonal),
        "affine_translation_x_px": round_or_none(translation_x, 6),
        "affine_translation_y_px": round_or_none(translation_y, 6),
        "affine_translation_norm": round_or_none(
            math.hypot(translation_x, translation_y) / diagonal
            if translation_x is not None and translation_y is not None
            else None
        ),
        "affine_rotation_deg": round_or_none(rotation_deg, 6),
        "affine_scale_delta": round_or_none(scale_delta, 9),
        "homography_residual_median_px": round_or_none(float(np.median(residual_inliers)), 6),
        "homography_residual_p95_px": round_or_none(float(np.percentile(residual_inliers, 95)), 6),
        "homography_residual_median_norm": round_or_none(float(np.median(residual_inliers)) / diagonal),
        "homography_residual_p95_norm": round_or_none(float(np.percentile(residual_inliers, 95)) / diagonal),
        "grid_supported_cells": grid_support,
        "grid_residual_spread_px": round_or_none(grid_spread, 6),
        "grid_residual_spread_norm": round_or_none(grid_spread / diagonal if grid_spread is not None else None),
        "homography_perspective_x_scaled": round_or_none(float(homography[2, 0]) * width, 9),
        "homography_perspective_y_scaled": round_or_none(float(homography[2, 1]) * height, 9),
    }
    debug = None
    if include_debug:
        debug = {
            "p0": p0,
            "p1": p1,
            "inliers": inliers,
            "homography": homography,
        }
    return result, debug


def discover_videos(root: Path, only: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    filters = [item.lower() for item in only]
    for path in sorted((root / "synced_video").glob("*/*/cam?.mp4")):
        set_match = SET_RE.match(path.parent.name)
        cam_match = CAM_RE.match(path.name)
        if not set_match or not cam_match:
            continue
        camera = int(cam_match.group("camera"))
        set_id = path.parent.name
        camera_id = f"cam{camera}"
        searchable = {set_id.lower(), camera_id.lower(), f"{set_id}/{camera_id}".lower(), relpath(path, root).lower()}
        if filters and not any(any(token in candidate for candidate in searchable) for token in filters):
            continue
        specs.append(
            {
                "absolute_path": str(path),
                "path": relpath(path, root),
                "exercise": set_match.group("exercise"),
                "take": set_match.group("take"),
                "set_id": set_id,
                "camera_id": camera_id,
            }
        )
    return specs


def audit_video(spec: dict[str, Any], max_dimension: int, window_radius: int) -> dict[str, Any]:
    path = Path(spec["absolute_path"])
    frames, timestamps, decode = decode_grayscale(path, max_dimension)
    energies = motion_energy(frames)
    pair_specs, long_targets, reference_index, energy_quantiles = select_pairs(
        timestamps, energies, window_radius
    )
    temporal_background = temporal_median_background(frames)
    needed_indices = {
        index
        for pair in pair_specs
        for index in (pair["start_index"], pair["end_index"])
    }
    needed_indices.update(long_targets)
    needed_indices.add(reference_index)
    foreground_masks = {
        index: dynamic_foreground_mask(frames[index], temporal_background)
        for index in needed_indices
    }
    local_rows: list[dict[str, Any]] = []
    for pair in pair_specs:
        result, _ = analyze_pair(
            frames[pair["start_index"]],
            frames[pair["end_index"]],
            foreground_previous=foreground_masks[pair["start_index"]],
            foreground_current=foreground_masks[pair["end_index"]],
        )
        local_rows.append(
            {
                **{key: spec[key] for key in ("path", "exercise", "take", "set_id", "camera_id")},
                "pair_type": "NATIVE_ADJACENT",
                "start_index": pair["start_index"],
                "end_index": pair["end_index"],
                "start_time_sec": round_or_none(timestamps[pair["start_index"]], 6),
                "end_time_sec": round_or_none(timestamps[pair["end_index"]], 6),
                "delta_time_sec": round_or_none(
                    timestamps[pair["end_index"]] - timestamps[pair["start_index"]], 6
                ),
                "selection_reason": pair["selection_reason"],
                "motion_class": pair["motion_class"],
                "temporal_section": pair["temporal_section"],
                "motion_energy": round_or_none(pair["motion_energy"], 6),
                **result,
            }
        )

    long_rows: list[dict[str, Any]] = []
    for target_index in long_targets:
        result, _ = analyze_pair(
            frames[reference_index],
            frames[target_index],
            long_baseline=True,
            foreground_previous=foreground_masks[reference_index],
            foreground_current=foreground_masks[target_index],
        )
        long_rows.append(
            {
                **{key: spec[key] for key in ("path", "exercise", "take", "set_id", "camera_id")},
                "pair_type": "LONG_BASELINE",
                "start_index": reference_index,
                "end_index": target_index,
                "start_time_sec": round_or_none(timestamps[reference_index], 6),
                "end_time_sec": round_or_none(timestamps[target_index], 6),
                "delta_time_sec": round_or_none(
                    abs(timestamps[target_index] - timestamps[reference_index]), 6
                ),
                "selection_reason": "reference_to_sample_center",
                "motion_class": "MIXED",
                "temporal_section": "CROSS_SECTION",
                "motion_energy": None,
                **result,
            }
        )

    return {
        "spec": {key: value for key, value in spec.items() if key != "absolute_path"},
        "decode": decode,
        "duration_sec": round_or_none(timestamps[-1] - timestamps[0], 6),
        "first_pts_sec": round_or_none(timestamps[0], 6),
        "last_pts_sec": round_or_none(timestamps[-1], 6),
        "motion_energy_q25": round_or_none(energy_quantiles[0], 6),
        "motion_energy_q75": round_or_none(energy_quantiles[1], 6),
        "reference_frame_index": reference_index,
        "reference_time_sec": round_or_none(timestamps[reference_index], 6),
        "local_rows": local_rows,
        "long_rows": long_rows,
    }


def aggregate_video(result: dict[str, Any]) -> dict[str, Any]:
    spec = result["spec"]
    decode = result["decode"]
    local_all = result["local_rows"]
    long_all = result["long_rows"]
    local = [row for row in local_all if row["status"] == "OK"]
    long = [row for row in long_all if row["status"] == "OK"]
    static = [row for row in local if row["motion_class"] == "STATIC"]
    fast = [row for row in local if row["motion_class"] == "FAST"]
    raw_effect = THRESHOLDS["global_motion_effect_norm"]
    spatial_effect = THRESHOLDS["spatial_residual_effect_norm"]
    global_repeat = mean(
        [1.0 if (row.get("raw_displacement_p95_norm") or 0.0) > raw_effect else 0.0 for row in local]
    )
    spatial_repeat = mean(
        [
            1.0
            if (row.get("homography_residual_p95_norm") or 0.0) > spatial_effect
            and (row.get("grid_supported_cells") or 0) >= 4
            else 0.0
            for row in local
        ]
    )
    return {
        **spec,
        **decode,
        "duration_sec": result["duration_sec"],
        "first_pts_sec": result["first_pts_sec"],
        "last_pts_sec": result["last_pts_sec"],
        "reference_frame_index": result["reference_frame_index"],
        "reference_time_sec": result["reference_time_sec"],
        "motion_energy_q25": result["motion_energy_q25"],
        "motion_energy_q75": result["motion_energy_q75"],
        "local_pairs_requested": len(local_all),
        "local_pairs_successful": len(local),
        "local_success_fraction": round_or_none(len(local) / len(local_all) if local_all else 0.0, 6),
        "long_pairs_requested": len(long_all),
        "long_pairs_successful": len(long),
        "long_success_fraction": round_or_none(len(long) / len(long_all) if long_all else 0.0, 6),
        "median_fb_tracks": round_or_none(median(row.get("fb_tracks") for row in local), 3),
        "median_retained_background_mask_fraction": round_or_none(
            median(row.get("retained_background_mask_fraction") for row in local), 6
        ),
        "median_homography_inlier_ratio": round_or_none(
            median(row.get("homography_inlier_ratio") for row in local), 6
        ),
        "p10_homography_inlier_ratio": round_or_none(
            pct((row.get("homography_inlier_ratio") for row in local), 10), 6
        ),
        "raw_displacement_median_norm": round_or_none(
            median(row.get("raw_displacement_median_norm") for row in local)
        ),
        "raw_displacement_p90_norm": round_or_none(
            pct((row.get("raw_displacement_p95_norm") for row in local), 90)
        ),
        "raw_displacement_p95_norm": round_or_none(
            pct((row.get("raw_displacement_p95_norm") for row in local), 95)
        ),
        "global_motion_repeat_fraction": round_or_none(global_repeat, 6),
        "affine_translation_p95_norm": round_or_none(
            pct((row.get("affine_translation_norm") for row in local), 95)
        ),
        "affine_abs_rotation_p95_deg": round_or_none(
            pct((abs(row["affine_rotation_deg"]) if row.get("affine_rotation_deg") is not None else None for row in local), 95),
            6,
        ),
        "affine_abs_scale_delta_p95": round_or_none(
            pct((abs(row["affine_scale_delta"]) if row.get("affine_scale_delta") is not None else None for row in local), 95)
        ),
        "homography_residual_median_norm": round_or_none(
            median(row.get("homography_residual_median_norm") for row in local)
        ),
        "homography_residual_p95_norm": round_or_none(
            pct((row.get("homography_residual_p95_norm") for row in local), 95)
        ),
        "spatial_residual_repeat_fraction": round_or_none(spatial_repeat, 6),
        "grid_residual_spread_p95_norm": round_or_none(
            pct((row.get("grid_residual_spread_norm") for row in local), 95)
        ),
        "median_grid_supported_cells": round_or_none(
            median(row.get("grid_supported_cells") for row in local), 3
        ),
        "static_raw_displacement_p95_norm": round_or_none(
            pct((row.get("raw_displacement_p95_norm") for row in static), 95)
        ),
        "fast_raw_displacement_p95_norm": round_or_none(
            pct((row.get("raw_displacement_p95_norm") for row in fast), 95)
        ),
        "static_residual_p95_norm": round_or_none(
            pct((row.get("homography_residual_p95_norm") for row in static), 95)
        ),
        "fast_residual_p95_norm": round_or_none(
            pct((row.get("homography_residual_p95_norm") for row in fast), 95)
        ),
        "long_raw_displacement_p95_norm": round_or_none(
            pct((row.get("raw_displacement_p95_norm") for row in long), 95)
        ),
        "long_residual_p95_norm": round_or_none(
            pct((row.get("homography_residual_p95_norm") for row in long), 95)
        ),
        "failure_status_counts": dict(
            sorted(
                {
                    status: sum(row["status"] == status for row in local_all + long_all)
                    for status in {row["status"] for row in local_all + long_all if row["status"] != "OK"}
                }.items()
            )
        ),
    }


def add_view_comparison(rows: list[dict[str, Any]]) -> None:
    by_set: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_set[row["set_id"]].append(row)
    for set_rows in by_set.values():
        for row in set_rows:
            peers = [peer for peer in set_rows if peer is not row]
            for metric, output_name in (
                ("raw_displacement_p95_norm", "sequence_relative_raw_p95_ratio"),
                ("homography_residual_p95_norm", "sequence_relative_residual_p95_ratio"),
                ("local_success_fraction", "sequence_relative_support_ratio"),
            ):
                own = row.get(metric)
                peer_value = median(peer.get(metric) for peer in peers)
                if own is None or peer_value is None:
                    ratio = None
                elif abs(peer_value) < 1e-12:
                    ratio = 1.0 if abs(own) < 1e-12 else None
                else:
                    ratio = own / peer_value
                row[output_name] = round_or_none(ratio, 6)


def classify(row: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    local_success = row.get("local_pairs_successful") or 0
    success_fraction = row.get("local_success_fraction") or 0.0
    tracks = row.get("median_fb_tracks") or 0.0
    inlier_ratio = row.get("median_homography_inlier_ratio") or 0.0
    support_ratio = row.get("sequence_relative_support_ratio")
    if local_success < THRESHOLDS["min_successful_local_pairs"]:
        return "INSUFFICIENT_EVIDENCE", [f"only {local_success} successful native-adjacent pairs"]
    if (
        success_fraction < THRESHOLDS["min_success_fraction"]
        or tracks < THRESHOLDS["min_median_tracks"]
        or inlier_ratio < THRESHOLDS["min_median_homography_inlier_ratio"]
    ):
        reasons.append(
            f"weak background support: success={success_fraction:.2f}, tracks={tracks:.0f}, inlier={inlier_ratio:.2f}"
        )
        if support_ratio is not None:
            reasons.append(f"within-sequence support ratio={support_ratio:.2f}")
        return "UNRELIABLE_VIEW", reasons

    raw_p95 = row.get("raw_displacement_p95_norm") or 0.0
    residual_p95 = row.get("homography_residual_p95_norm") or 0.0
    long_raw = row.get("long_raw_displacement_p95_norm") or 0.0
    global_repeat = row.get("global_motion_repeat_fraction") or 0.0
    spatial_repeat = row.get("spatial_residual_repeat_fraction") or 0.0
    raw_relative = row.get("sequence_relative_raw_p95_ratio")
    residual_relative = row.get("sequence_relative_residual_p95_ratio")
    explained_ratio = residual_p95 / raw_p95 if raw_p95 > 1e-12 else 0.0

    spatial_evidence = (
        residual_p95 > THRESHOLDS["spatial_residual_effect_norm"]
        and spatial_repeat >= THRESHOLDS["minimum_repeat_fraction"]
        and (residual_relative is None or residual_relative >= THRESHOLDS["within_sequence_outlier_ratio"] or residual_p95 > 2 * THRESHOLDS["spatial_residual_effect_norm"])
    )
    if spatial_evidence:
        reasons.extend(
            [
                f"homography residual p95={residual_p95:.6f} diagonal",
                f"spatial effect repeats in {spatial_repeat:.0%} of sampled pairs",
                f"within-sequence residual ratio={residual_relative:.2f}" if residual_relative is not None else "within-sequence residual ratio unavailable",
            ]
        )
        return "SPATIAL_WARP_SUSPECTED", reasons

    global_evidence = (
        (
            raw_p95 > THRESHOLDS["global_motion_effect_norm"]
            or long_raw > THRESHOLDS["long_baseline_effect_norm"]
        )
        and global_repeat >= THRESHOLDS["minimum_repeat_fraction"]
        and explained_ratio <= THRESHOLDS["global_model_explained_ratio"]
        and (raw_relative is None or raw_relative >= 1.0 or long_raw > 2 * THRESHOLDS["long_baseline_effect_norm"])
    )
    if global_evidence:
        reasons.extend(
            [
                f"raw background p95={raw_p95:.6f}, long-baseline p95={long_raw:.6f} diagonal",
                f"global effect repeats in {global_repeat:.0%} of sampled pairs",
                f"homography residual/raw ratio={explained_ratio:.2f}",
                f"within-sequence raw ratio={raw_relative:.2f}" if raw_relative is not None else "within-sequence raw ratio unavailable",
            ]
        )
        return "GLOBAL_WARP_CORRECTION_CANDIDATE", reasons

    reasons.extend(
        [
            f"supported pairs={local_success}, median tracks={tracks:.0f}, median inlier={inlier_ratio:.2f}",
            f"raw p95={raw_p95:.6f}, residual p95={residual_p95:.6f}, long p95={long_raw:.6f} diagonal",
            f"repeat fractions global={global_repeat:.0%}, spatial={spatial_repeat:.0%}",
        ]
    )
    return "FIXED_CAMERA_OK", reasons


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    if preferred_fields:
        fields.extend(field for field in preferred_fields if any(field in row for row in rows))
    else:
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_time_plot(row: dict[str, Any], local_rows: list[dict[str, Any]]) -> str:
    ok = [pair for pair in local_rows if pair["status"] == "OK"]
    width, height = 900, 420
    left, right, top, bottom = 70, 25, 55, 55
    plot_width = width - left - right
    plot_height = height - top - bottom
    times = [pair["end_time_sec"] for pair in ok]
    raw = [(pair.get("raw_displacement_p95_norm") or 0.0) * 1000 for pair in ok]
    residual = [(pair.get("homography_residual_p95_norm") or 0.0) * 1000 for pair in ok]
    x_min = min(times) if times else 0.0
    x_max = max(times) if times else max(row.get("duration_sec") or 1.0, 1.0)
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_max = max(raw + residual + [THRESHOLDS["global_motion_effect_norm"] * 1000, 0.1]) * 1.15

    def xy(time_value: float, metric_value: float) -> tuple[float, float]:
        x = left + (time_value - x_min) / (x_max - x_min) * plot_width
        y = top + plot_height - metric_value / y_max * plot_height
        return x, y

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">{xml_escape(row["set_id"] + " / " + row["camera_id"] + " — " + row["recommendation"])}</text>',
        f'<text x="{left}" y="44" font-family="sans-serif" font-size="11" fill="#444">PTS seconds; displacement in 10⁻³ of proxy diagonal</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
    ]
    for tick in range(6):
        y_value = y_max * tick / 5
        y = top + plot_height - tick / 5 * plot_height
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{y_value:.2f}</text>')
    for tick in range(6):
        value = x_min + (x_max - x_min) * tick / 5
        x = left + plot_width * tick / 5
        lines.append(f'<text x="{x:.1f}" y="{top + plot_height + 20}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.1f}</text>')
    threshold_y = xy(x_min, THRESHOLDS["global_motion_effect_norm"] * 1000)[1]
    lines.append(f'<line x1="{left}" y1="{threshold_y:.1f}" x2="{left + plot_width}" y2="{threshold_y:.1f}" stroke="#999" stroke-dasharray="4 4"/>')
    for values, color, label in ((raw, "#1565c0", "raw p95"), (residual, "#d84315", "homography residual p95")):
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(t, value) for t, value in zip(times, values)))
        if points:
            lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}"/>')
        legend_x = left + (0 if color == "#1565c0" else 180)
        lines.append(f'<line x1="{legend_x}" y1="{height - 17}" x2="{legend_x + 24}" y2="{height - 17}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{legend_x + 30}" y="{height - 13}" font-family="sans-serif" font-size="11">{label}</text>')
    for pair in ok:
        color = {"STATIC": "#2e7d32", "FAST": "#c62828", "MODERATE": "#757575"}[pair["motion_class"]]
        x, y = xy(pair["end_time_sec"], (pair.get("raw_displacement_p95_norm") or 0.0) * 1000)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{color}"/>')
    lines.append("</svg>")
    return "\n".join(lines)


def recommendation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(
            {
                recommendation: sum(row["recommendation"] == recommendation for row in rows)
                for recommendation in {row["recommendation"] for row in rows}
            }.items()
        )
    )


def metric_distribution(rows: list[dict[str, Any]], metric: str) -> dict[str, float | None]:
    values = [finite_float(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    return {
        "min": round_or_none(min(values) if values else None),
        "median": round_or_none(median(values)),
        "p95": round_or_none(pct(values, 95)),
        "max": round_or_none(max(values) if values else None),
    }


def dataset_statistics(
    rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics = [
        "local_success_fraction",
        "median_fb_tracks",
        "median_retained_background_mask_fraction",
        "median_homography_inlier_ratio",
        "raw_displacement_p95_norm",
        "homography_residual_p95_norm",
        "long_raw_displacement_p95_norm",
        "long_residual_p95_norm",
        "global_motion_repeat_fraction",
        "spatial_residual_repeat_fraction",
    ]
    by_camera: dict[str, Any] = {}
    for camera_id in sorted({row["camera_id"] for row in rows}):
        camera_rows = [row for row in rows if row["camera_id"] == camera_id]
        by_camera[camera_id] = {
            metric: metric_distribution(camera_rows, metric) for metric in metrics
        }
    status_counts: dict[str, int] = {}
    for pair in pair_rows:
        key = f"{pair['pair_type']}:{pair['status']}"
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "video_count": len(rows),
        "pair_status_counts": dict(sorted(status_counts.items())),
        "overall": {metric: metric_distribution(rows, metric) for metric in metrics},
        "by_camera": by_camera,
    }


def write_summary_markdown(
    path: Path,
    generated_at: str,
    rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    errors: list[str],
) -> None:
    counts = recommendation_counts(rows)
    statistics_summary = dataset_statistics(rows, pair_rows)
    overall = statistics_summary["overall"]
    lines = [
        "# Exercise3D EIS/OIS and background-stability audit",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This audit decoded synchronized camera videos but did not modify or warp any source/derivative asset.",
        "",
        "## Outcome",
        "",
    ]
    for recommendation, count in counts.items():
        lines.append(f"- `{recommendation}`: **{count}** videos")
    lines.extend(
        [
            "",
            f"Analyzed **{len(rows)}** camera videos using **{sum(pair['status'] == 'OK' for pair in pair_rows)}** successful feature-pair fits.",
            "",
            "Recommendations are screening labels, not proof that a phone enabled or disabled EIS/OIS. "
            "They describe whether the encoded projection behaves as fixed, globally varying, spatially varying, or unsupported by enough background evidence.",
            "",
            "## Method",
            "",
            "- Time coordinates come from all video packet PTS values, not frame index alone.",
            "- Frames were decoded to a bounded proxy resolution; all displacement thresholds are normalized by proxy diagonal.",
            "- Central foreground was excluded. Border/corner features were tracked with forward-backward Lucas-Kanade.",
            "- Broad moving foreground was additionally excluded using temporal-median difference components; thin shifted-background edge contours were retained.",
            "- RANSAC partial-affine and homography fits separated global motion from post-homography residual.",
            "- Samples include early/middle/late coverage, low-motion windows, high-motion windows, native-adjacent pairs, and long-baseline comparisons.",
            "- Decisions require normalized effect size, repetition, feature/model support, residual-to-raw behavior, and comparison with the other two views in the same sequence.",
            "",
            "## Aggregate evidence",
            "",
            f"- Native-adjacent support: **{statistics_summary['pair_status_counts'].get('NATIVE_ADJACENT:OK', 0)} successful pairs**, with per-video success fraction "
            f"{overall['local_success_fraction']['min']:.2f}–{overall['local_success_fraction']['max']:.2f}.",
            f"- Median tracked background features per video: **{overall['median_fb_tracks']['median']:.0f}** "
            f"(minimum {overall['median_fb_tracks']['min']:.0f}).",
            f"- Raw background displacement p95 across videos: median **{overall['raw_displacement_p95_norm']['median']:.6f}**, "
            f"dataset maximum **{overall['raw_displacement_p95_norm']['max']:.6f}** of proxy diagonal.",
            f"- Post-homography residual p95: median **{overall['homography_residual_p95_norm']['median']:.6f}**, "
            f"dataset maximum **{overall['homography_residual_p95_norm']['max']:.6f}** of proxy diagonal.",
            f"- Long-baseline raw p95 maximum: **{overall['long_raw_displacement_p95_norm']['max']:.6f}** of proxy diagonal.",
            f"- Maximum repeated-effect fractions: global **{overall['global_motion_repeat_fraction']['max']:.1%}**, "
            f"spatial **{overall['spatial_residual_repeat_fraction']['max']:.1%}** "
            f"(decision minimum {THRESHOLDS['minimum_repeat_fraction']:.0%}).",
            "",
            "## Per-video recommendations",
            "",
            "| set | camera | recommendation | local support | raw p95 (diag) | residual p95 (diag) | long p95 (diag) |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['set_id']} | {row['camera_id']} | {row['recommendation']} | "
            f"{row['local_pairs_successful']}/{row['local_pairs_requested']} | "
            f"{(row.get('raw_displacement_p95_norm') or 0):.6f} | "
            f"{(row.get('homography_residual_p95_norm') or 0):.6f} | "
            f"{(row.get('long_raw_displacement_p95_norm') or 0):.6f} |"
        )
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "`FIXED_CAMERA_OK` means no repeated material encoded-background variation was found at this audit's scale. "
            "`GLOBAL_WARP_CORRECTION_CANDIDATE` means a repeated global transform explains most motion and should be evaluated before calibration. "
            "`SPATIAL_WARP_SUSPECTED` means a single homography leaves repeated spatial residual. "
            "`UNRELIABLE_VIEW`/`INSUFFICIENT_EVIDENCE` mean the background did not support a confident conclusion.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


SUMMARY_FIELDS = [
    "set_id",
    "exercise",
    "take",
    "camera_id",
    "path",
    "recommendation",
    "recommendation_reasons",
    "local_pairs_successful",
    "local_pairs_requested",
    "local_success_fraction",
    "median_fb_tracks",
    "median_homography_inlier_ratio",
    "raw_displacement_p95_norm",
    "homography_residual_p95_norm",
    "long_raw_displacement_p95_norm",
    "global_motion_repeat_fraction",
    "spatial_residual_repeat_fraction",
    "sequence_relative_raw_p95_ratio",
    "sequence_relative_residual_p95_ratio",
]


def run_audit(
    root: Path,
    output_dir: Path,
    jobs: int,
    max_dimension: int,
    window_radius: int,
    only: list[str],
) -> int:
    generated_at = utc_now()
    specs = discover_videos(root, only)
    if not specs:
        print("No synchronized camera videos matched.", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    debug_dir = output_dir / "debug"
    figures_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"Auditing {len(specs)} synchronized camera videos with {jobs} workers...", flush=True)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        future_to_spec = {
            executor.submit(audit_video, spec, max_dimension, window_radius): spec for spec in specs
        }
        completed_count = 0
        for future in as_completed(future_to_spec):
            completed_count += 1
            spec = future_to_spec[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f"{spec['path']}: {type(exc).__name__}: {exc}")
            print(
                f"  {completed_count}/{len(specs)} {spec['set_id']}/{spec['camera_id']}",
                flush=True,
            )
    results.sort(key=lambda result: (result["spec"]["set_id"], result["spec"]["camera_id"]))
    aggregates = [aggregate_video(result) for result in results]
    add_view_comparison(aggregates)
    for row in aggregates:
        recommendation, reasons = classify(row)
        row["recommendation"] = recommendation
        row["recommendation_reasons"] = reasons

    pair_rows = [pair for result in results for pair in result["local_rows"] + result["long_rows"]]
    write_csv(output_dir / "summary.csv", aggregates, SUMMARY_FIELDS)
    write_csv(output_dir / "per_video_metrics.csv", aggregates)
    write_csv(output_dir / "pair_metrics.csv", pair_rows)
    result_by_identity = {
        (result["spec"]["set_id"], result["spec"]["camera_id"]): result for result in results
    }
    for row in aggregates:
        result = result_by_identity[(row["set_id"], row["camera_id"])]
        figure_path = figures_dir / f"{row['set_id']}_{row['camera_id']}.svg"
        figure_path.write_text(svg_time_plot(row, result["local_rows"]), encoding="utf-8")
    (figures_dir / "README.md").write_text(
        "# Figure index\n\nEach SVG plots PTS seconds against raw and post-homography background displacement. "
        "Point colors: green=low motion, red=high motion, gray=moderate motion.\n",
        encoding="utf-8",
    )
    (debug_dir / "README.md").write_text(
        "# Debug policy\n\nNo corrected/warped video is generated. Pair-level numeric diagnostics are in "
        "`../pair_metrics.csv`; SVGs are in `../figures/`. This directory is reserved for "
        "future evidence snapshots only when a flagged view needs manual adjudication.\n",
        encoding="utf-8",
    )
    write_summary_markdown(
        output_dir / "README.md", generated_at, aggregates, pair_rows, errors
    )
    payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "dataset_root": ".",
        "input_scope": "synced_video/*/*/cam?.mp4",
        "source_mutation": False,
        "correction_or_warp_outputs_created": False,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "settings": {
            "max_proxy_dimension": max_dimension,
            "local_window_radius_frames": window_radius,
            "central_foreground_exclusion": {"x": [0.22, 0.78], "y": [0.18, 0.82]},
            "dynamic_foreground_exclusion": {
                "reference": "41-frame temporal median",
                "absolute_difference_threshold_8bit": 18,
                "broad_component_minimum_fraction": 0.001,
                "note": "opening removes thin shifted-background edge contours before component filtering",
            },
            "spatial_grid": [4, 4],
            "thresholds": THRESHOLDS,
        },
        "recommendation_counts": recommendation_counts(aggregates),
        "aggregate_statistics": dataset_statistics(aggregates, pair_rows),
        "videos": aggregates,
        "errors": errors,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "videos": len(aggregates),
                "successful_pair_fits": sum(pair["status"] == "OK" for pair in pair_rows),
                "recommendations": payload["recommendation_counts"],
                "errors": errors,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if not errors else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", "--root", dest="root", type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="private dataset root (or EXERCISE3D_DATASET_ROOT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default: DATASET_ROOT/reports/eis_audit; set explicitly for read-only sources",
    )
    parser.add_argument("--jobs", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--max-dimension", type=int, default=480)
    parser.add_argument("--window-radius", type=int, default=6)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="substring filter for set, camera, or relative path (repeatable)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "reports" / "eis_audit").resolve()
    if not (root / "synced_video").is_dir():
        print(f"Missing synchronized video directory: {root / 'synced_video'}", file=sys.stderr)
        return 2
    if args.max_dimension < 240:
        print("--max-dimension must be at least 240", file=sys.stderr)
        return 2
    if args.window_radius < 2:
        print("--window-radius must be at least 2", file=sys.stderr)
        return 2
    return run_audit(
        root,
        output_dir,
        args.jobs,
        args.max_dimension,
        args.window_radius,
        args.only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
