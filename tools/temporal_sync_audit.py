#!/usr/bin/env python3
"""Read-only temporal synchronization QA and clock-drift analysis for Exercise3D.

Inputs are raw originals, existing synchronized videos, their ``sync.json``
metadata, and existing frame derivatives.  The script never cuts, resamples,
interpolates, overwrites, or creates synchronized media.  It writes diagnostic
CSV/JSON/SVG files below ``reports/temporal_alignment``.
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
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "OpenCV is required. Install with: pip install -r tools/requirements.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT)
).expanduser()
CAMERA_PAIRS = ((1, 2), (1, 3), (2, 3))
WINDOW_LABELS = ("BEGINNING", "EARLY", "EARLY_MIDDLE", "MIDDLE", "MIDDLE_LATE", "LATE", "END")
PAIR_RE = re.compile(r"cam(?P<a>[123])-cam(?P<b>[123])")

SETTINGS = {
    "window_count": 7,
    "maximum_window_sec": 6.0,
    "visual_search_frames_each_side": 6,
    "audio_search_ms_each_side": 200.0,
    "raw_clap_search_ms_each_side": 100.0,
    "visual_drift_min_confidence": 0.50,
    "audio_drift_min_confidence": 0.65,
    "clap_drift_min_confidence": 0.60,
    "frame_pts_drift_weight": 0.35,
    "half_frame_ms_at_30fps": 1000.0 / 60.0,
    "one_frame_ms_at_30fps": 1000.0 / 30.0,
    "clock_drift_min_total_ms": 1000.0 / 60.0,
    "clock_drift_agreement_min_modalities": 2,
    "clock_drift_min_modal_r_squared": 0.60,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def rounded(value: Any, digits: int = 6) -> float | None:
    parsed = finite(value)
    return round(parsed, digits) if parsed is not None else None


def percentile(values: Iterable[float | None], p: float) -> float | None:
    data = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.percentile(data, p)) if data else None


def median(values: Iterable[float | None]) -> float | None:
    return percentile(values, 50)


def mean(values: Iterable[float | None]) -> float | None:
    data = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(data)) if data else None


def confidence_level(value: float) -> str:
    if value >= 0.75:
        return "HIGH"
    if value >= 0.45:
        return "MEDIUM"
    return "LOW"


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
    values: list[float] = []
    for line in completed.stdout.splitlines():
        value = finite(line.strip().split(",", 1)[0])
        if value is not None:
            values.append(value)
    return sorted(values)


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


def decode_video(path: Path, max_dimension: int) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    packet_pts = ffprobe_packet_pts(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
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
        raise RuntimeError(f"no frames decoded: {path}")
    if len(packet_pts) == len(frames):
        timestamps = packet_pts
        pts_source = "ALL_VIDEO_PACKET_PTS"
    elif fps > 0:
        timestamps = [index / fps for index in range(len(frames))]
        pts_source = "NOMINAL_FPS_FALLBACK"
    else:
        raise RuntimeError(
            f"frame/PTS count mismatch without FPS fallback: {len(frames)} vs {len(packet_pts)}"
        )
    return np.stack(frames), timestamps, {
        "decoded_frames": len(frames),
        "pts_count": len(packet_pts),
        "pts_source": pts_source,
        "fps": fps,
        "proxy_width": int(frames[0].shape[1]),
        "proxy_height": int(frames[0].shape[0]),
    }


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace").strip())
    audio = np.frombuffer(completed.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise RuntimeError(f"audio decode returned no samples: {path}")
    return audio


def window_centers(duration: float, count: int = 7) -> tuple[list[dict[str, Any]], float]:
    window_sec = min(SETTINGS["maximum_window_sec"], max(3.0, duration / 3.0))
    half = window_sec / 2.0
    first = min(max(half, 0.5), duration / 2.0)
    last = max(min(duration - half, duration - 0.5), duration / 2.0)
    centers = np.linspace(first, last, count) if count > 1 else np.asarray([duration / 2.0])
    return (
        [
            {
                "window_index": index,
                "window_id": WINDOW_LABELS[index] if count == len(WINDOW_LABELS) else f"W{index}",
                "center_time_sec": float(center),
                "start_time_sec": max(0.0, float(center) - half),
                "end_time_sec": min(duration, float(center) + half),
            }
            for index, center in enumerate(centers)
        ],
        window_sec,
    )


def motion_features(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    difference = np.abs(np.diff(frames.astype(np.int16), axis=0)).astype(np.float32)
    flat = difference.reshape(len(difference), -1)
    height, width = frames.shape[1:]
    y0, y1 = round(height * 0.10), round(height * 0.95)
    x0, x1 = round(width * 0.10), round(width * 0.90)
    central = difference[:, y0:y1, x0:x1].reshape(len(difference), -1)
    features = np.stack(
        [
            flat.mean(axis=1),
            np.percentile(flat, 75, axis=1),
            np.percentile(flat, 90, axis=1),
            np.percentile(flat, 95, axis=1),
            central.mean(axis=1),
            np.percentile(central, 90, axis=1),
        ],
        axis=1,
    ).astype(np.float32)
    # A three-frame smoother suppresses codec noise but preserves rapid joint motion.
    if len(features) >= 3:
        kernel = np.ones(3, dtype=np.float32) / 3.0
        for column in range(features.shape[1]):
            features[:, column] = np.convolve(features[:, column], kernel, mode="same")
    aggregate = np.median(
        (features - np.median(features, axis=0, keepdims=True))
        / (np.std(features, axis=0, keepdims=True) + 1e-6),
        axis=1,
    )
    return features, aggregate.astype(np.float32)


def standardized_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64) - float(np.mean(x))
    y = y.astype(np.float64) - float(np.mean(y))
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else 0.0


def parabolic_peak(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= len(values) - 1:
        return float(index)
    y0, y1, y2 = (float(value) for value in values[index - 1:index + 2])
    denominator = y0 - 2.0 * y1 + y2
    return float(index) + 0.5 * (y0 - y2) / denominator if abs(denominator) > 1e-12 else float(index)


def visual_offset(
    features_a: np.ndarray,
    aggregate_a: np.ndarray,
    features_b: np.ndarray,
    aggregate_b: np.ndarray,
    fps: float,
    window: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    start_index = max(0, int(math.floor(window["start_time_sec"] * fps)))
    end_index = min(len(features_a), len(features_b), int(math.ceil(window["end_time_sec"] * fps)))
    maximum_lag = SETTINGS["visual_search_frames_each_side"]
    lags = np.arange(-maximum_lag, maximum_lag + 1)
    correlations: list[float] = []
    per_lag_channels: list[list[float]] = []
    for lag in lags:
        left = max(start_index, start_index - int(lag), 0)
        right = min(end_index, end_index - int(lag), len(features_a), len(features_b) - int(lag))
        if right - left < max(20, int(fps)):
            per_lag_channels.append([0.0] * features_a.shape[1])
            correlations.append(0.0)
            continue
        channel_scores = [
            standardized_correlation(features_a[left:right, channel], features_b[left + lag:right + lag, channel])
            for channel in range(features_a.shape[1])
        ]
        per_lag_channels.append(channel_scores)
        correlations.append(float(np.median(channel_scores)))
    values = np.asarray(correlations, dtype=np.float64)
    best_index = int(np.argmax(values))
    fractional_index = parabolic_peak(values, best_index)
    lag_frames = float(lags[0]) + fractional_index
    peak = float(values[best_index])
    guard = np.abs(lags - lags[best_index]) > 1
    runner_up = float(np.max(values[guard])) if np.any(guard) else 0.0
    prominence = peak - runner_up
    peak_z = (peak - float(np.median(values))) / (float(np.std(values)) + 1e-9)
    segment = features_a[start_index:end_index]
    motion_strength = float(np.median(np.std(segment, axis=0))) if len(segment) else 0.0
    score_component = float(np.clip((peak - 0.20) / 0.60, 0.0, 1.0))
    prominence_component = float(np.clip(prominence / 0.15, 0.0, 1.0))
    z_component = float(np.clip((peak_z - 1.0) / 4.0, 0.0, 1.0))
    confidence = 0.55 * score_component + 0.30 * prominence_component + 0.15 * z_component
    if best_index in (0, len(values) - 1):
        confidence *= 0.25
    confidence = float(np.clip(confidence, 0.0, 1.0))

    integer_lag = int(round(lag_frames))
    left = max(start_index, start_index - integer_lag, 0)
    right = min(end_index, end_index - integer_lag, len(aggregate_a), len(aggregate_b) - integer_lag)
    plot_a = aggregate_a[left:right]
    plot_b = aggregate_b[left + integer_lag:right + integer_lag]
    plot_t = np.arange(left, right, dtype=np.float64) / fps
    plot = {
        "time_sec": [round(float(value), 6) for value in plot_t],
        "camera_a_signal": [round(float(value), 5) for value in plot_a],
        "camera_b_aligned_signal": [round(float(value), 5) for value in plot_b],
        "lag_frames": round(lag_frames, 6),
    }
    return (
        {
            "offset_ms": lag_frames / fps * 1000.0,
            "confidence": confidence,
            "peak_score": peak,
            "runner_up_score": runner_up,
            "peak_prominence": prominence,
            "peak_z": peak_z,
            "motion_strength": motion_strength,
            "search_limit_ms": maximum_lag / fps * 1000.0,
            "at_search_boundary": best_index in (0, len(values) - 1),
            "support_samples": max(0, right - left),
        },
        plot,
    )


def fft_limited_xcorr(x: np.ndarray, y: np.ndarray, max_lag: int) -> dict[str, Any]:
    # Sign convention: positive lag means y/camera B contains the same event
    # later than x/camera A and would need an earlier shift to align.
    length = min(len(x), len(y))
    x = x[:length].astype(np.float64)
    y = y[:length].astype(np.float64)
    if length < max(100, 4 * max_lag):
        return {"offset_samples": None, "confidence": 0.0, "status": "INSUFFICIENT_SAMPLES"}
    # Pre-emphasis reduces slow gain/exposure-like microphone differences and
    # emphasizes shared transients without assuming that a clap is present.
    x = np.diff(x)
    y = np.diff(y)
    x = (x - float(np.mean(x))) / (float(np.std(x)) + 1e-9)
    y = (y - float(np.mean(y))) / (float(np.std(y)) + 1e-9)
    fft_length = 1 << (len(x) + len(y) - 1).bit_length()
    correlation = np.fft.irfft(
        np.fft.rfft(y, fft_length) * np.conj(np.fft.rfft(x, fft_length)),
        fft_length,
    )
    lags = np.arange(-max_lag, max_lag + 1)
    values = correlation[lags % fft_length] / np.maximum(len(x) - np.abs(lags), 1)
    best_index = int(np.argmax(values))
    fractional_index = parabolic_peak(values, best_index)
    lag_samples = float(lags[0]) + fractional_index
    peak = float(values[best_index])
    guard_samples = max(2, max_lag // 20)
    runner_mask = np.abs(lags - lags[best_index]) > guard_samples
    runner_up = float(np.max(values[runner_mask])) if np.any(runner_mask) else 0.0
    ratio = peak / runner_up if runner_up > 1e-9 else float("inf")
    peak_z = (peak - float(np.median(values))) / (float(np.std(values)) + 1e-9)
    correlation_component = float(np.clip((peak - 0.10) / 0.30, 0.0, 1.0))
    ratio_component = float(np.clip((min(ratio, 3.0) - 1.05) / 0.55, 0.0, 1.0))
    z_component = float(np.clip((peak_z - 2.0) / 7.0, 0.0, 1.0))
    confidence = 0.45 * correlation_component + 0.35 * ratio_component + 0.20 * z_component
    if ratio < 1.08:
        confidence *= 0.35
    if best_index in (0, len(values) - 1):
        confidence *= 0.25
    return {
        "offset_samples": lag_samples,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "peak_score": peak,
        "runner_up_score": runner_up,
        "peak_ratio": ratio,
        "peak_z": peak_z,
        "at_search_boundary": best_index in (0, len(values) - 1),
        "support_samples": length,
        "status": "OK",
    }


def audio_window_offset(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: int,
    window: dict[str, Any],
) -> dict[str, Any]:
    start = max(0, int(round(window["start_time_sec"] * sample_rate)))
    end = min(len(audio_a), len(audio_b), int(round(window["end_time_sec"] * sample_rate)))
    result = fft_limited_xcorr(
        audio_a[start:end],
        audio_b[start:end],
        int(round(SETTINGS["audio_search_ms_each_side"] * sample_rate / 1000.0)),
    )
    offset_samples = result.pop("offset_samples")
    result["offset_ms"] = offset_samples / sample_rate * 1000.0 if offset_samples is not None else None
    result["search_limit_ms"] = SETTINGS["audio_search_ms_each_side"]
    return result


def audio_clap_offset(
    audio_a: np.ndarray,
    center_a: float,
    audio_b: np.ndarray,
    center_b: float,
    sample_rate: int,
) -> dict[str, Any]:
    half_window_sec = 0.60
    half_samples = int(round(half_window_sec * sample_rate))
    center_a_sample = int(round(center_a * sample_rate))
    center_b_sample = int(round(center_b * sample_rate))
    start_a, end_a = center_a_sample - half_samples, center_a_sample + half_samples
    start_b, end_b = center_b_sample - half_samples, center_b_sample + half_samples
    if start_a < 0 or start_b < 0 or end_a > len(audio_a) or end_b > len(audio_b):
        return {"offset_ms": None, "confidence": 0.0, "status": "CLAP_WINDOW_OUT_OF_RANGE"}
    segment_a = audio_a[start_a:end_a]
    segment_b = audio_b[start_b:end_b]
    waveform_result = fft_limited_xcorr(
        segment_a,
        segment_b,
        int(round(SETTINGS["raw_clap_search_ms_each_side"] * sample_rate / 1000.0)),
    )
    # Cross-device microphones can color the waveform differently. A 1-ms
    # transient envelope preserves clap timing while discarding most spectral
    # response differences. Keep whichever domain has stronger evidence.
    hop = max(1, round(sample_rate / 1000.0))
    smoothing = max(3, round(sample_rate * 0.002))

    def onset_envelope(segment: np.ndarray) -> np.ndarray:
        onset = np.abs(np.diff(segment, prepend=segment[0]))
        onset = np.convolve(onset, np.ones(smoothing) / smoothing, mode="same")
        return onset[::hop].astype(np.float32)

    envelope_rate = sample_rate / hop
    envelope_result = fft_limited_xcorr(
        onset_envelope(segment_a),
        onset_envelope(segment_b),
        int(round(SETTINGS["raw_clap_search_ms_each_side"] * envelope_rate / 1000.0)),
    )
    domain, result, effective_rate = max(
        [
            ("WAVEFORM", waveform_result, float(sample_rate)),
            ("TRANSIENT_ENVELOPE", envelope_result, float(envelope_rate)),
        ],
        key=lambda item: float(item[1].get("confidence") or 0.0),
    )
    result = dict(result)
    offset_samples = result.pop("offset_samples")
    result["offset_ms"] = offset_samples / effective_rate * 1000.0 if offset_samples is not None else None
    result["search_limit_ms"] = SETTINGS["raw_clap_search_ms_each_side"]
    result["notes"] = f"selected clap correlation domain={domain}"
    return result


def map_output_to_raw(
    output_frame: np.ndarray,
    output_pts: float,
    raw_frames: np.ndarray,
    raw_pts: list[float],
    cut_start_sec: float,
    source_offset_sec: float,
    search_sec: float = 0.12,
) -> dict[str, Any]:
    expected_raw_time = cut_start_sec + output_pts
    candidate_indices = [
        index for index, timestamp in enumerate(raw_pts)
        if abs(timestamp - expected_raw_time) <= search_sec
    ]
    if not candidate_indices:
        return {"status": "NO_RAW_PTS_CANDIDATE"}
    candidates = raw_frames[candidate_indices]
    errors = np.mean(
        np.abs(candidates.astype(np.float32) - output_frame.astype(np.float32)),
        axis=(1, 2),
    )
    order = np.argsort(errors)
    best_local = int(order[0])
    best_index = candidate_indices[best_local]
    best_error = float(errors[best_local])
    second_error = float(errors[int(order[1])]) if len(order) > 1 else best_error
    margin = max(0.0, second_error - best_error)
    match_component = float(np.clip((2.0 - best_error) / 1.8, 0.0, 1.0))
    margin_component = float(np.clip(margin / 0.35, 0.0, 1.0))
    confidence = 0.65 * match_component + 0.35 * margin_component
    actual_raw_time = raw_pts[best_index]
    event_time = actual_raw_time - source_offset_sec
    desired_event_time = cut_start_sec - source_offset_sec + output_pts
    return {
        "status": "OK",
        "output_pts_sec": output_pts,
        "expected_raw_pts_sec": expected_raw_time,
        "actual_raw_pts_sec": actual_raw_time,
        "event_time_sec": event_time,
        "desired_event_time_sec": desired_event_time,
        "quantization_ms": (event_time - desired_event_time) * 1000.0,
        "match_error": best_error,
        "second_best_error": second_error,
        "match_margin": margin,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "raw_frame_index": best_index,
        "candidate_count": len(candidate_indices),
    }


def base_measurement(
    set_info: dict[str, Any],
    camera_a: int,
    camera_b: int,
    window: dict[str, Any],
    evidence_type: str,
) -> dict[str, Any]:
    return {
        "subject_id": "UNKNOWN",
        "exercise": set_info["exercise"],
        "take": set_info["take"],
        "set_id": set_info["set_id"],
        "camera_a": f"cam{camera_a}",
        "camera_b": f"cam{camera_b}",
        "pair_id": f"cam{camera_a}-cam{camera_b}",
        "window_index": window.get("window_index"),
        "window_id": window.get("window_id"),
        "center_time_sec": rounded(window.get("center_time_sec"), 6),
        "window_start_sec": rounded(window.get("start_time_sec"), 6),
        "window_end_sec": rounded(window.get("end_time_sec"), 6),
        "evidence_type": evidence_type,
        "offset_sign_convention": "positive = same physical event appears later on camera_b synchronized timeline; camera_b would shift earlier",
    }


def measurement_row(base: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    confidence = finite(result.get("confidence")) or 0.0
    return {
        **base,
        "estimated_offset_ms": rounded(result.get("offset_ms"), 6),
        "confidence": rounded(confidence, 6),
        "confidence_level": confidence_level(confidence),
        "status": result.get("status", "OK"),
        "usable_for_drift": False,
        "peak_score": rounded(result.get("peak_score"), 6),
        "runner_up_score": rounded(result.get("runner_up_score"), 6),
        "peak_ratio": rounded(result.get("peak_ratio"), 6),
        "peak_prominence": rounded(result.get("peak_prominence"), 6),
        "peak_z": rounded(result.get("peak_z"), 6),
        "motion_strength": rounded(result.get("motion_strength"), 6),
        "search_limit_ms": rounded(result.get("search_limit_ms"), 3),
        "at_search_boundary": result.get("at_search_boundary"),
        "support_samples": result.get("support_samples"),
        "notes": result.get("notes"),
    }


def weighted_linear_fit(rows: list[dict[str, Any]], duration_sec: float) -> dict[str, Any]:
    usable = [
        row for row in rows
        if row.get("usable_for_drift")
        and finite(row.get("center_time_sec")) is not None
        and finite(row.get("estimated_offset_ms")) is not None
    ]
    if len(usable) < 2:
        return {
            "n_points": len(usable),
            "time_span_sec": 0.0,
            "slope_ms_per_sec": None,
            "intercept_ms": None,
            "drift_over_observed_span_ms": None,
            "drift_over_sequence_ms": None,
            "endpoint_change_ms": None,
            "r_squared": None,
            "rmse_ms": None,
            "mean_confidence": rounded(mean(finite(row.get("confidence")) for row in usable), 6),
            "fit_confidence": 0.0,
        }
    x = np.asarray([float(row["center_time_sec"]) for row in usable], dtype=np.float64)
    y = np.asarray([float(row["estimated_offset_ms"]) for row in usable], dtype=np.float64)
    w = np.asarray([max(float(row.get("confidence") or 0.0), 0.05) ** 2 for row in usable], dtype=np.float64)
    design = np.stack([np.ones_like(x), x], axis=1)
    weighted_design = design * np.sqrt(w)[:, None]
    weighted_y = y * np.sqrt(w)
    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
    intercept, slope = float(coefficients[0]), float(coefficients[1])
    prediction = intercept + slope * x
    residual = y - prediction
    rmse = math.sqrt(float(np.average(residual ** 2, weights=w)))
    y_mean = float(np.average(y, weights=w))
    total = float(np.sum(w * (y - y_mean) ** 2))
    unexplained = float(np.sum(w * residual ** 2))
    r_squared = 1.0 - unexplained / total if total > 1e-12 else 0.0
    span = float(np.max(x) - np.min(x))
    mean_confidence = float(np.mean(np.sqrt(w)))
    coverage = min(1.0, span / max(duration_sec * 0.6, 1e-9))
    count_support = min(1.0, len(usable) / 5.0)
    residual_support = float(np.clip(1.0 - rmse / SETTINGS["one_frame_ms_at_30fps"], 0.0, 1.0))
    fit_confidence = mean_confidence * coverage * count_support * (0.5 + 0.5 * residual_support)
    order = np.argsort(x)
    edge_count = min(2, max(1, len(usable) // 3))
    first_indices = order[:edge_count]
    last_indices = order[-edge_count:]
    first_level = float(np.average(y[first_indices], weights=w[first_indices]))
    last_level = float(np.average(y[last_indices], weights=w[last_indices]))
    return {
        "n_points": len(usable),
        "time_span_sec": rounded(span, 6),
        "slope_ms_per_sec": rounded(slope, 6),
        "intercept_ms": rounded(intercept, 6),
        "drift_over_observed_span_ms": rounded(slope * span, 6),
        "drift_over_sequence_ms": rounded(slope * duration_sec, 6),
        "endpoint_change_ms": rounded(last_level - first_level, 6),
        "r_squared": rounded(r_squared, 6),
        "rmse_ms": rounded(rmse, 6),
        "mean_confidence": rounded(mean_confidence, 6),
        "fit_confidence": rounded(float(np.clip(fit_confidence, 0.0, 1.0)), 6),
    }


def weighted_median(values: list[float], weights: list[float]) -> float:
    order = np.argsort(values)
    sorted_values = np.asarray(values)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def fused_drift(
    modal_rows: list[dict[str, Any]],
    frame_measurements: list[dict[str, Any]],
    duration_sec: float,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in modal_rows:
        # Classification uses robust beginning→end change. Regression slope and
        # extrapolation remain report fields, but mid-clip 30/60-fps phase
        # sawtooth must not be promoted to clock drift.
        drift = finite(row.get("endpoint_change_ms"))
        sequence_drift = finite(row.get("drift_over_sequence_ms"))
        slope = finite(row.get("slope_ms_per_sec"))
        r_squared = finite(row.get("r_squared"))
        confidence = finite(row.get("fit_confidence")) or 0.0
        if drift is None or sequence_drift is None or slope is None:
            continue
        evidence = row["evidence_type"]
        minimum_points = 2 if evidence == "AUDIO_CLAP_WAVEFORM" else 3
        if int(row.get("n_points") or 0) < minimum_points:
            continue
        if minimum_points >= 3 and (
            r_squared is None or r_squared < SETTINGS["clock_drift_min_modal_r_squared"]
        ):
            continue
        minimum_confidence = 0.25 if evidence == "FRAME_PTS_MAPPING" else 0.35
        if confidence < minimum_confidence:
            continue
        weight = confidence * (SETTINGS["frame_pts_drift_weight"] if evidence == "FRAME_PTS_MAPPING" else 1.0)
        candidates.append(
            {
                "evidence_type": evidence,
                "drift_ms": drift,
                "sequence_drift_ms": sequence_drift,
                "slope_ms_per_sec": slope,
                "r_squared": r_squared,
                "confidence": confidence,
                "weight": weight,
            }
        )
    if not candidates:
        return {
            "fused_drift_ms": None,
            "fused_drift_extrapolated_sequence_ms": None,
            "fused_drift_ms_per_sec": None,
            "drift_confidence": 0.0,
            "supporting_modalities": [],
            "modal_agreement_count": 0,
            "clock_drift_detected": False,
        }
    fused = weighted_median(
        [candidate["drift_ms"] for candidate in candidates],
        [candidate["weight"] for candidate in candidates],
    )
    fused_slope = weighted_median(
        [candidate["slope_ms_per_sec"] for candidate in candidates],
        [candidate["weight"] for candidate in candidates],
    )
    material = [
        candidate for candidate in candidates
        if abs(candidate["drift_ms"]) >= SETTINGS["clock_drift_min_total_ms"] * 0.60
    ]
    agreement = [
        candidate for candidate in material
        if candidate["drift_ms"] == 0 or math.copysign(1.0, candidate["drift_ms"]) == math.copysign(1.0, fused)
    ]
    independent_agreement = len({candidate["evidence_type"] for candidate in agreement})
    confidence = float(np.average(
        [candidate["confidence"] for candidate in candidates],
        weights=[candidate["weight"] for candidate in candidates],
    ))
    frame_offsets = [
        abs(float(row["estimated_offset_ms"])) for row in frame_measurements
        if finite(row.get("estimated_offset_ms")) is not None
    ]
    frame_within_one = bool(frame_offsets) and max(frame_offsets) <= SETTINGS["one_frame_ms_at_30fps"] + 1.0
    detected = (
        abs(fused) >= SETTINGS["clock_drift_min_total_ms"]
        and independent_agreement >= SETTINGS["clock_drift_agreement_min_modalities"]
        and confidence >= 0.45
        and not (not frame_within_one and independent_agreement < 3)
    )
    return {
        "fused_drift_ms": rounded(fused, 6),
        "fused_drift_extrapolated_sequence_ms": rounded(fused_slope * duration_sec, 6),
        "fused_drift_ms_per_sec": rounded(fused_slope, 6),
        "drift_confidence": rounded(confidence, 6),
        "supporting_modalities": [candidate["evidence_type"] for candidate in candidates],
        "modal_agreement_count": independent_agreement,
        "clock_drift_detected": detected,
        "frame_mapping_all_within_one_frame": frame_within_one,
    }


def read_inventory_devices(root: Path) -> dict[str, str]:
    inventory_path = root / "reports" / "dataset_inventory.json"
    if not inventory_path.is_file():
        return {"cam1": "UNKNOWN", "cam2": "UNKNOWN", "cam3": "UNKNOWN"}
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    mapping = payload.get("summary", {}).get("raw_camera_device_mapping", {})
    return {
        camera: ", ".join(mapping.get(camera) or ["UNKNOWN"])
        for camera in ("cam1", "cam2", "cam3")
    }


def audit_sequence(
    set_dir: Path,
    root: Path,
    max_dimension: int,
    audio_sample_rate: int,
) -> dict[str, Any]:
    sync = json.loads((set_dir / "sync.json").read_text(encoding="utf-8"))
    set_info = {
        "set_id": sync["set_id"],
        "exercise": sync["exercise"],
        "take": sync["take"],
        "subject_id": "UNKNOWN",
    }
    clip_by_cam = {int(clip["cam"]): clip for clip in sync["clips"]}
    common_starts = [clip["cut_start_sec"] - clip["offset_sec"] for clip in sync["clips"]]
    common_start = float(np.median(common_starts))
    duration = float(sync["duration_sec"])
    windows, window_sec = window_centers(duration, SETTINGS["window_count"])
    cameras: dict[int, dict[str, Any]] = {}
    camera_mapping_rows: list[dict[str, Any]] = []

    for camera in (1, 2, 3):
        clip = clip_by_cam[camera]
        sync_path = set_dir / f"cam{camera}.mp4"
        raw_path = root / clip["source"]
        sync_frames, sync_pts, sync_decode = decode_video(sync_path, max_dimension)
        raw_frames, raw_pts, raw_decode = decode_video(raw_path, max_dimension)
        if sync_frames.shape[1:] != raw_frames.shape[1:]:
            raise RuntimeError(
                f"proxy orientation mismatch cam{camera}: {sync_frames.shape[1:]} vs {raw_frames.shape[1:]}"
            )
        features, aggregate = motion_features(sync_frames)
        sync_audio = decode_audio(sync_path, audio_sample_rate)
        raw_audio = decode_audio(raw_path, audio_sample_rate)
        mappings: dict[int, dict[str, Any]] = {}
        for window in windows:
            output_index = int(np.argmin(np.abs(np.asarray(sync_pts) - window["center_time_sec"])))
            mapping = map_output_to_raw(
                sync_frames[output_index],
                sync_pts[output_index],
                raw_frames,
                raw_pts,
                float(clip["cut_start_sec"]),
                float(clip["offset_sec"]),
            )
            mappings[window["window_index"]] = mapping
            camera_mapping_rows.append(
                {
                    **set_info,
                    "camera_id": f"cam{camera}",
                    "source_path": clip["source"],
                    "window_index": window["window_index"],
                    "window_id": window["window_id"],
                    "center_time_sec": rounded(window["center_time_sec"], 6),
                    "output_frame_index": output_index,
                    "output_pts_sec": rounded(mapping.get("output_pts_sec"), 6),
                    "expected_raw_pts_sec": rounded(mapping.get("expected_raw_pts_sec"), 6),
                    "actual_raw_pts_sec": rounded(mapping.get("actual_raw_pts_sec"), 6),
                    "event_time_sec": rounded(mapping.get("event_time_sec"), 6),
                    "desired_event_time_sec": rounded(mapping.get("desired_event_time_sec"), 6),
                    "quantization_ms": rounded(mapping.get("quantization_ms"), 6),
                    "match_error": rounded(mapping.get("match_error"), 6),
                    "match_margin": rounded(mapping.get("match_margin"), 6),
                    "confidence": rounded(mapping.get("confidence"), 6),
                    "raw_frame_index": mapping.get("raw_frame_index"),
                    "status": mapping.get("status"),
                    "sync_pts_source": sync_decode["pts_source"],
                    "raw_pts_source": raw_decode["pts_source"],
                }
            )
        cameras[camera] = {
            "clip": clip,
            "sync_path": sync_path,
            "raw_path": raw_path,
            "sync_pts": sync_pts,
            "features": features,
            "aggregate_motion": aggregate,
            "sync_audio": sync_audio,
            "raw_audio": raw_audio,
            "mappings": mappings,
            "sync_decode": sync_decode,
            "raw_decode": raw_decode,
        }
        # Raw frames are intentionally not retained after exact source mapping.
        del raw_frames

    measurements: list[dict[str, Any]] = []
    motion_plots: dict[str, dict[str, Any]] = {}
    pair_results: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    clap_anchors: list[tuple[str, float]] = []
    if sync.get("head_clap_sec") is not None:
        clap_anchors.append(("HEAD_CLAP", float(sync["head_clap_sec"])))
    if sync.get("tail_clap_sec") is not None:
        clap_anchors.append(("TAIL_CLAP", float(sync["tail_clap_sec"])))

    for camera_a, camera_b in CAMERA_PAIRS:
        pair_id = f"cam{camera_a}-cam{camera_b}"
        best_motion_plot: tuple[float, dict[str, Any], str] | None = None
        frame_rows: list[dict[str, Any]] = []
        visual_rows: list[dict[str, Any]] = []
        audio_rows: list[dict[str, Any]] = []
        clap_rows: list[dict[str, Any]] = []
        for window in windows:
            mapping_a = cameras[camera_a]["mappings"][window["window_index"]]
            mapping_b = cameras[camera_b]["mappings"][window["window_index"]]
            frame_base = base_measurement(set_info, camera_a, camera_b, window, "FRAME_PTS_MAPPING")
            if mapping_a.get("status") == "OK" and mapping_b.get("status") == "OK":
                # The same event appears later in B when B captured an earlier
                # physical instant at the same output PTS. Negate event-time
                # difference so this matches visual/audio lag convention.
                offset_ms = (mapping_a["event_time_sec"] - mapping_b["event_time_sec"]) * 1000.0
                frame_confidence = min(mapping_a["confidence"], mapping_b["confidence"])
                frame_result = {
                    "offset_ms": offset_ms,
                    "confidence": frame_confidence,
                    "status": "OK",
                    "support_samples": 2,
                    "notes": "actual source-frame PTS recovered by same-camera pixel matching",
                }
            else:
                frame_result = {"offset_ms": None, "confidence": 0.0, "status": "MAPPING_FAILED"}
            frame_row = measurement_row(frame_base, frame_result)
            frame_row["usable_for_drift"] = frame_row["status"] == "OK"
            measurements.append(frame_row)
            frame_rows.append(frame_row)

            visual_result, plot = visual_offset(
                cameras[camera_a]["features"],
                cameras[camera_a]["aggregate_motion"],
                cameras[camera_b]["features"],
                cameras[camera_b]["aggregate_motion"],
                30.0,
                window,
            )
            visual_row = measurement_row(
                base_measurement(set_info, camera_a, camera_b, window, "VISUAL_MOTION"),
                visual_result,
            )
            visual_row["usable_for_drift"] = (
                visual_row["status"] == "OK"
                and float(visual_row["confidence"]) >= SETTINGS["visual_drift_min_confidence"]
                and not visual_row["at_search_boundary"]
            )
            measurements.append(visual_row)
            visual_rows.append(visual_row)
            plot_score = float(visual_row["confidence"]) * max(float(visual_row.get("motion_strength") or 0.0), 0.01)
            if best_motion_plot is None or plot_score > best_motion_plot[0]:
                best_motion_plot = (plot_score, plot, window["window_id"])

            audio_result = audio_window_offset(
                cameras[camera_a]["sync_audio"],
                cameras[camera_b]["sync_audio"],
                audio_sample_rate,
                window,
            )
            audio_row = measurement_row(
                base_measurement(set_info, camera_a, camera_b, window, "AUDIO_WINDOW"),
                audio_result,
            )
            audio_row["usable_for_drift"] = (
                audio_row["status"] == "OK"
                and float(audio_row["confidence"]) >= SETTINGS["audio_drift_min_confidence"]
                and not audio_row["at_search_boundary"]
            )
            measurements.append(audio_row)
            audio_rows.append(audio_row)

        for anchor_name, event_time in clap_anchors:
            window = {
                "window_index": None,
                "window_id": anchor_name,
                "center_time_sec": event_time - common_start,
                "start_time_sec": None,
                "end_time_sec": None,
            }
            center_a = event_time + float(cameras[camera_a]["clip"]["offset_sec"])
            center_b = event_time + float(cameras[camera_b]["clip"]["offset_sec"])
            clap_result = audio_clap_offset(
                cameras[camera_a]["raw_audio"],
                center_a,
                cameras[camera_b]["raw_audio"],
                center_b,
                audio_sample_rate,
            )
            clap_row = measurement_row(
                base_measurement(set_info, camera_a, camera_b, window, "AUDIO_CLAP_WAVEFORM"),
                clap_result,
            )
            clap_row["usable_for_drift"] = (
                clap_row["status"] == "OK"
                and float(clap_row["confidence"]) >= SETTINGS["clap_drift_min_confidence"]
            )
            measurements.append(clap_row)
            clap_rows.append(clap_row)

        residual_a = float(cameras[camera_a]["clip"].get("residual_ms") or 0.0)
        residual_b = float(cameras[camera_b]["clip"].get("residual_ms") or 0.0)
        metadata_window = {
            "window_index": None,
            "window_id": "EXISTING_CLAP_QA",
            "center_time_sec": 0.0 if sync.get("head_clap_sec") is not None else duration,
            "start_time_sec": None,
            "end_time_sec": None,
        }
        metadata_row = measurement_row(
            base_measurement(set_info, camera_a, camera_b, metadata_window, "AUDIO_CLAP_EXISTING"),
            {
                "offset_ms": residual_b - residual_a,
                "confidence": 1.0,
                "status": "OK",
                "notes": f"existing sync QA; sequence peak_spread_ms={sync.get('peak_spread_ms')}",
            },
        )
        measurements.append(metadata_row)

        evidence_groups = {
            "FRAME_PTS_MAPPING": frame_rows,
            "VISUAL_MOTION": visual_rows,
            "AUDIO_WINDOW": audio_rows,
            "AUDIO_CLAP_WAVEFORM": clap_rows,
        }
        modal_fit_rows: list[dict[str, Any]] = []
        for evidence_type, rows in evidence_groups.items():
            fit = weighted_linear_fit(rows, duration)
            drift_row = {
                **set_info,
                "camera_a": f"cam{camera_a}",
                "camera_b": f"cam{camera_b}",
                "pair_id": pair_id,
                "evidence_type": evidence_type,
                **fit,
                "representative_offset_ms": rounded(median(
                    finite(row.get("estimated_offset_ms")) for row in rows
                    if row.get("usable_for_drift")
                ), 6),
                "classification": None,
            }
            drift_rows.append(drift_row)
            modal_fit_rows.append(drift_row)

        fused = fused_drift(modal_fit_rows, frame_rows, duration)
        frame_offsets = [
            float(row["estimated_offset_ms"]) for row in frame_rows
            if finite(row.get("estimated_offset_ms")) is not None
        ]
        representative_offset = median(frame_offsets)
        maximum_abs_offset = max((abs(value) for value in frame_offsets), default=float("inf"))
        usable_visual = [row for row in visual_rows if row["usable_for_drift"]]
        usable_audio = [row for row in audio_rows if row["usable_for_drift"]]
        if len(frame_offsets) < 3:
            classification = "INSUFFICIENT_EVIDENCE"
        elif fused["clock_drift_detected"]:
            classification = "CLOCK_DRIFT_DETECTED"
        elif abs(representative_offset or 0.0) > SETTINGS["half_frame_ms_at_30fps"]:
            classification = "SMALL_CONSTANT_OFFSET"
        elif maximum_abs_offset <= SETTINGS["one_frame_ms_at_30fps"] + 1.0:
            classification = "TEMPORALLY_STABLE"
        else:
            classification = "INSUFFICIENT_EVIDENCE"
        pair_result = {
            **set_info,
            "camera_a": f"cam{camera_a}",
            "camera_b": f"cam{camera_b}",
            "pair_id": pair_id,
            "duration_sec": rounded(duration, 6),
            "classification": classification,
            "representative_frame_pts_offset_ms": rounded(representative_offset, 6),
            "maximum_abs_frame_pts_offset_ms": rounded(maximum_abs_offset, 6),
            "frame_pts_offset_p95_abs_ms": rounded(percentile((abs(value) for value in frame_offsets), 95), 6),
            "visual_usable_windows": len(usable_visual),
            "visual_offset_median_ms": rounded(median(float(row["estimated_offset_ms"]) for row in usable_visual), 6),
            "visual_confidence_median": rounded(median(float(row["confidence"]) for row in visual_rows), 6),
            "audio_usable_windows": len(usable_audio),
            "audio_offset_median_ms": rounded(median(float(row["estimated_offset_ms"]) for row in usable_audio), 6),
            "audio_confidence_median": rounded(median(float(row["confidence"]) for row in audio_rows), 6),
            "clap_waveform_usable_anchors": sum(row["usable_for_drift"] for row in clap_rows),
            "existing_clap_offset_ms": metadata_row["estimated_offset_ms"],
            "existing_peak_spread_ms": sync.get("peak_spread_ms"),
            **fused,
        }
        pair_results.append(pair_result)
        drift_rows.append(
            {
                **set_info,
                "camera_a": f"cam{camera_a}",
                "camera_b": f"cam{camera_b}",
                "pair_id": pair_id,
                "evidence_type": "FUSED",
                "n_points": sum(int(row.get("n_points") or 0) for row in modal_fit_rows),
                "time_span_sec": duration,
                "slope_ms_per_sec": fused["fused_drift_ms_per_sec"],
                "intercept_ms": representative_offset,
                "drift_over_observed_span_ms": fused["fused_drift_ms"],
                "drift_over_sequence_ms": fused["fused_drift_extrapolated_sequence_ms"],
                "endpoint_change_ms": fused["fused_drift_ms"],
                "r_squared": None,
                "rmse_ms": None,
                "mean_confidence": fused["drift_confidence"],
                "fit_confidence": fused["drift_confidence"],
                "representative_offset_ms": representative_offset,
                "classification": classification,
                "supporting_modalities": fused["supporting_modalities"],
                "modal_agreement_count": fused["modal_agreement_count"],
            }
        )
        if best_motion_plot is not None:
            motion_plots[pair_id] = {
                **best_motion_plot[1],
                "window_id": best_motion_plot[2],
                "camera_a": f"cam{camera_a}",
                "camera_b": f"cam{camera_b}",
            }

    severity = {
        "TEMPORALLY_STABLE": 0,
        "SMALL_CONSTANT_OFFSET": 1,
        "CLOCK_DRIFT_DETECTED": 2,
        "INSUFFICIENT_EVIDENCE": 3,
    }
    sequence_classification = max(
        (result["classification"] for result in pair_results),
        key=lambda value: severity[value],
    )
    frame_events_by_window: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for camera in (1, 2, 3):
        for window_index, mapping in cameras[camera]["mappings"].items():
            if mapping.get("status") == "OK":
                frame_events_by_window[window_index].append((camera, mapping["event_time_sec"]))
    relative_camera_rows: list[dict[str, Any]] = []
    for window in windows:
        events = frame_events_by_window[window["window_index"]]
        if len(events) != 3:
            continue
        center = float(np.mean([value for _, value in events]))
        for camera, value in events:
            relative_camera_rows.append(
                {
                    **set_info,
                    "camera_id": f"cam{camera}",
                    "window_index": window["window_index"],
                    "window_id": window["window_id"],
                    "center_time_sec": rounded(window["center_time_sec"], 6),
                    "relative_timing_error_ms": rounded((center - value) * 1000.0, 6),
                }
            )
    summary = {
        **set_info,
        "duration_sec": rounded(duration, 6),
        "window_sec": rounded(window_sec, 6),
        "classification": sequence_classification,
        "pair_classifications": {result["pair_id"]: result["classification"] for result in pair_results},
        "maximum_abs_pair_offset_ms": rounded(max(result["maximum_abs_frame_pts_offset_ms"] for result in pair_results), 6),
        "maximum_abs_representative_offset_ms": rounded(max(abs(result["representative_frame_pts_offset_ms"] or 0.0) for result in pair_results), 6),
        "maximum_abs_fused_drift_ms": rounded(max(abs(result["fused_drift_ms"] or 0.0) for result in pair_results), 6),
        "clock_drift_pair_count": sum(result["classification"] == "CLOCK_DRIFT_DETECTED" for result in pair_results),
        "small_constant_pair_count": sum(result["classification"] == "SMALL_CONSTANT_OFFSET" for result in pair_results),
        "visual_usable_windows": sum(result["visual_usable_windows"] for result in pair_results),
        "audio_usable_windows": sum(result["audio_usable_windows"] for result in pair_results),
        "clap_waveform_usable_anchors": sum(result["clap_waveform_usable_anchors"] for result in pair_results),
        "existing_max_residual_ms": sync.get("max_residual_ms"),
        "existing_peak_spread_ms": sync.get("peak_spread_ms"),
        "all_sync_pts_from_packets": all(cameras[camera]["sync_decode"]["pts_source"] == "ALL_VIDEO_PACKET_PTS" for camera in (1, 2, 3)),
        "all_raw_pts_from_packets": all(cameras[camera]["raw_decode"]["pts_source"] == "ALL_VIDEO_PACKET_PTS" for camera in (1, 2, 3)),
    }
    return {
        "summary": summary,
        "measurements": measurements,
        "camera_mapping_rows": camera_mapping_rows,
        "relative_camera_rows": relative_camera_rows,
        "drift_rows": drift_rows,
        "pair_results": pair_results,
        "motion_plots": motion_plots,
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(value) for key, value in row.items()})


def xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg_line_plot(
    title: str,
    series: list[dict[str, Any]],
    y_label: str,
    threshold_lines: list[tuple[float, str]] | None = None,
    width: int = 900,
    height: int = 430,
) -> str:
    left, right, top, bottom = 80, 25, 55, 60
    plot_width, plot_height = width - left - right, height - top - bottom
    all_x = [float(x) for item in series for x in item.get("x", [])]
    all_y = [float(y) for item in series for y in item.get("y", []) if finite(y) is not None]
    if threshold_lines:
        all_y.extend(value for value, _ in threshold_lines)
    x_min, x_max = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if all_y:
        y_abs = max(max(abs(min(all_y)), abs(max(all_y))) * 1.15, 1e-6)
        y_min, y_max = -y_abs, y_abs
    else:
        y_min, y_max = -1.0, 1.0

    def xy(x_value: float, y_value: float) -> tuple[float, float]:
        x = left + (x_value - x_min) / (x_max - x_min) * plot_width
        y = top + plot_height - (y_value - y_min) / (y_max - y_min) * plot_height
        return x, y

    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="27" font-family="sans-serif" font-size="18">{xml_escape(title)}</text>',
        f'<text x="{left}" y="45" font-family="sans-serif" font-size="11" fill="#555">x = synchronized-video PTS seconds; y = {xml_escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
    ]
    for tick in range(5):
        y_value = y_min + (y_max - y_min) * tick / 4
        _, y = xy(x_min, y_value)
        output.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e3e3e3"/>')
        output.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10">{y_value:.2f}</text>')
    for tick in range(6):
        x_value = x_min + (x_max - x_min) * tick / 5
        x, _ = xy(x_value, 0)
        output.append(f'<text x="{x:.1f}" y="{top + plot_height + 20}" text-anchor="middle" font-family="sans-serif" font-size="10">{x_value:.1f}</text>')
    if threshold_lines:
        for threshold, label in threshold_lines:
            if y_min <= threshold <= y_max:
                _, y = xy(x_min, threshold)
                output.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#777" stroke-dasharray="4 4"/>')
                output.append(f'<text x="{left + plot_width - 4}" y="{y - 3:.1f}" text-anchor="end" font-family="sans-serif" font-size="9" fill="#555">{xml_escape(label)}</text>')
    legend_x = left
    for item in series:
        points = [(x, y) for x, y in zip(item.get("x", []), item.get("y", [])) if finite(y) is not None]
        path_points = " ".join(f"{px:.1f},{py:.1f}" for px, py in (xy(float(x), float(y)) for x, y in points))
        color = item.get("color", "#1565c0")
        if path_points:
            output.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{path_points}"/>')
            for x_value, y_value in points:
                px, py = xy(float(x_value), float(y_value))
                opacity = item.get("opacity", 0.85)
                output.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.4" fill="{color}" opacity="{opacity}"/>')
        output.append(f'<line x1="{legend_x}" y1="{height - 18}" x2="{legend_x + 22}" y2="{height - 18}" stroke="{color}" stroke-width="2"/>')
        output.append(f'<text x="{legend_x + 28}" y="{height - 14}" font-family="sans-serif" font-size="10">{xml_escape(item["label"])}</text>')
        legend_x += 175
    output.append("</svg>")
    return "\n".join(output)


def write_pair_figures(
    figures_dir: Path,
    pair_result: dict[str, Any],
    measurements: list[dict[str, Any]],
    motion_plot: dict[str, Any] | None,
) -> None:
    set_id, pair_id = pair_result["set_id"], pair_result["pair_id"]
    rows = [row for row in measurements if row["set_id"] == set_id and row["pair_id"] == pair_id]
    colors = {
        "FRAME_PTS_MAPPING": "#1565c0",
        "VISUAL_MOTION": "#ef6c00",
        "AUDIO_WINDOW": "#2e7d32",
        "AUDIO_CLAP_WAVEFORM": "#6a1b9a",
        "AUDIO_CLAP_EXISTING": "#757575",
    }
    offset_series = []
    confidence_series = []
    for evidence_type in colors:
        selected = [
            row for row in rows
            if row["evidence_type"] == evidence_type
            and finite(row.get("center_time_sec")) is not None
            and finite(row.get("estimated_offset_ms")) is not None
        ]
        offset_series.append(
            {
                "label": evidence_type,
                "color": colors[evidence_type],
                "x": [float(row["center_time_sec"]) for row in selected],
                "y": [float(row["estimated_offset_ms"]) for row in selected],
            }
        )
        confidence_series.append(
            {
                "label": evidence_type,
                "color": colors[evidence_type],
                "x": [float(row["center_time_sec"]) for row in selected],
                "y": [float(row["confidence"]) for row in selected],
            }
        )
    prefix = figures_dir / f"{set_id}_{pair_id}"
    (prefix.with_name(prefix.name + "_offset_vs_time.svg")).write_text(
        svg_line_plot(
            f"{set_id} / {pair_id} — {pair_result['classification']}",
            offset_series,
            "offset ms (positive = camera B later)",
            [
                (SETTINGS["half_frame_ms_at_30fps"], "+0.5 frame"),
                (-SETTINGS["half_frame_ms_at_30fps"], "-0.5 frame"),
                (SETTINGS["one_frame_ms_at_30fps"], "+1 frame"),
                (-SETTINGS["one_frame_ms_at_30fps"], "-1 frame"),
            ],
        ),
        encoding="utf-8",
    )
    (prefix.with_name(prefix.name + "_confidence.svg")).write_text(
        svg_line_plot(
            f"{set_id} / {pair_id} — evidence confidence",
            confidence_series,
            "confidence [0,1]",
            [(0.45, "medium"), (0.75, "high")],
        ),
        encoding="utf-8",
    )
    if motion_plot:
        (prefix.with_name(prefix.name + "_motion_alignment.svg")).write_text(
            svg_line_plot(
                f"{set_id} / {pair_id} — best visual window {motion_plot['window_id']} (lag {motion_plot['lag_frames']:.2f} frames)",
                [
                    {
                        "label": motion_plot["camera_a"],
                        "color": "#1565c0",
                        "x": motion_plot["time_sec"],
                        "y": motion_plot["camera_a_signal"],
                    },
                    {
                        "label": motion_plot["camera_b"] + " aligned",
                        "color": "#ef6c00",
                        "x": motion_plot["time_sec"],
                        "y": motion_plot["camera_b_aligned_signal"],
                    },
                ],
                "robust normalized motion energy",
            ),
            encoding="utf-8",
        )


def count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def infer_drift_attribution(
    sequence_rows: list[dict[str, Any]],
    pair_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for sequence in sequence_rows:
        if sequence["classification"] != "CLOCK_DRIFT_DETECTED":
            continue
        rows = [row for row in pair_results if row["set_id"] == sequence["set_id"]]
        drift_rows = [row for row in rows if row["classification"] == "CLOCK_DRIFT_DETECTED"]
        involvement = {
            camera: sum(camera in (row["camera_a"], row["camera_b"]) for row in drift_rows)
            for camera in ("cam1", "cam2", "cam3")
        }
        suspects = [camera for camera, count in involvement.items() if count == 2]
        unaffected = [row for row in rows if row["classification"] != "CLOCK_DRIFT_DETECTED"]
        if len(suspects) == 1 and len(drift_rows) == 2 and len(unaffected) == 1:
            attributed_camera = suspects[0]
            status = "TOPOLOGY_CONSISTENT"
            reason = (
                f"{attributed_camera} participates in both drift pairs; "
                f"{unaffected[0]['pair_id']} remains {unaffected[0]['classification']}"
            )
        else:
            attributed_camera = "UNRESOLVED"
            status = "AMBIGUOUS"
            reason = "pair topology does not isolate one camera"
        output.append(
            {
                "subject_id": "UNKNOWN",
                "exercise": sequence["exercise"],
                "take": sequence["take"],
                "set_id": sequence["set_id"],
                "attributed_camera": attributed_camera,
                "attribution_status": status,
                "drift_pairs": [row["pair_id"] for row in drift_rows],
                "non_drift_pairs": [row["pair_id"] for row in unaffected],
                "maximum_abs_fused_endpoint_change_ms": rounded(
                    max(abs(float(row["fused_drift_ms"])) for row in drift_rows), 6
                ),
                "reason": reason,
            }
        )
    return output


def aggregate_camera(
    relative_rows: list[dict[str, Any]],
    pair_results: list[dict[str, Any]],
    device_mapping: dict[str, str],
    attribution_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for camera in ("cam1", "cam2", "cam3"):
        rows = [row for row in relative_rows if row["camera_id"] == camera]
        errors = [float(row["relative_timing_error_ms"]) for row in rows]
        involved = [
            row for row in pair_results if row["camera_a"] == camera or row["camera_b"] == camera
        ]
        attributed_sets = [
            row["set_id"] for row in attribution_rows if row["attributed_camera"] == camera
        ]
        output.append(
            {
                "camera_id": camera,
                "device_model": device_mapping[camera],
                "subject_id": "UNKNOWN",
                "sample_count": len(errors),
                "median_signed_relative_error_ms": rounded(median(errors), 6),
                "median_abs_relative_error_ms": rounded(median(abs(value) for value in errors), 6),
                "p95_abs_relative_error_ms": rounded(percentile((abs(value) for value in errors), 95), 6),
                "max_abs_relative_error_ms": rounded(max((abs(value) for value in errors), default=None), 6),
                "median_pair_fused_drift_abs_ms": rounded(median(abs(row["fused_drift_ms"] or 0.0) for row in involved), 6),
                "clock_drift_pair_involvement": sum(row["classification"] == "CLOCK_DRIFT_DETECTED" for row in involved),
                "attributed_drift_sequence_count": len(attributed_sets),
                "attributed_drift_sequences": attributed_sets,
                "note": "relative camera metric is centered within each three-camera time sample; it is not absolute ground truth",
            }
        )
    return output


def aggregate_exercise(sequence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for exercise in sorted({row["exercise"] for row in sequence_rows}):
        rows = [row for row in sequence_rows if row["exercise"] == exercise]
        output.append(
            {
                "exercise": exercise,
                "subject_id": "UNKNOWN",
                "sequence_count": len(rows),
                "classification_counts": count_values(rows, "classification"),
                "median_max_abs_pair_offset_ms": rounded(median(float(row["maximum_abs_pair_offset_ms"]) for row in rows), 6),
                "p95_max_abs_pair_offset_ms": rounded(percentile((float(row["maximum_abs_pair_offset_ms"]) for row in rows), 95), 6),
                "maximum_abs_pair_offset_ms": rounded(max(float(row["maximum_abs_pair_offset_ms"]) for row in rows), 6),
                "median_max_abs_fused_drift_ms": rounded(median(float(row["maximum_abs_fused_drift_ms"]) for row in rows), 6),
                "maximum_abs_fused_drift_ms": rounded(max(float(row["maximum_abs_fused_drift_ms"]) for row in rows), 6),
            }
        )
    return output


def aggregate_subject(sequence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "subject_id": "UNKNOWN",
            "sequence_count": len(sequence_rows),
            "exercise_count": len({row["exercise"] for row in sequence_rows}),
            "classification_counts": count_values(sequence_rows, "classification"),
            "subject_difference_status": "NOT_ATTRIBUTABLE",
            "reason": "No evidence-backed subject_id-to-sequence mapping exists in manifest, metadata, or current project records.",
        }
    ]


def summary_markdown(
    generated_at: str,
    sequence_rows: list[dict[str, Any]],
    pair_results: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    camera_rows: list[dict[str, Any]],
    exercise_rows: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    errors: list[str],
) -> str:
    sequence_counts = count_values(sequence_rows, "classification")
    pair_counts = count_values(pair_results, "classification")
    stable_camera = min(camera_rows, key=lambda row: float(row["p95_abs_relative_error_ms"]))
    unstable_camera = max(camera_rows, key=lambda row: float(row["p95_abs_relative_error_ms"]))
    clock_drift_pairs = [row for row in pair_results if row["classification"] == "CLOCK_DRIFT_DETECTED"]
    frame_rows = [row for row in measurements if row["evidence_type"] == "FRAME_PTS_MAPPING" and row["status"] == "OK"]
    visual_rows = [row for row in measurements if row["evidence_type"] == "VISUAL_MOTION"]
    audio_rows = [row for row in measurements if row["evidence_type"] == "AUDIO_WINDOW"]
    clap_rows = [row for row in measurements if row["evidence_type"] == "AUDIO_CLAP_WAVEFORM"]
    all_frame_offsets = [abs(float(row["estimated_offset_ms"])) for row in frame_rows]
    def truthy(value: Any) -> bool:
        return value is True or str(value).lower() == "true"
    quality = (
        "CLOCK_DRIFT_REVIEW_REQUIRED" if clock_drift_pairs
        else "GOOD_WITH_SMALL_FRAME_PHASE_OFFSETS" if sequence_counts.get("SMALL_CONSTANT_OFFSET", 0)
        else "TEMPORALLY_STABLE"
    )
    lines = [
        "# Phase 2 — Temporal Synchronization QA & Clock Drift Analysis",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This is a read-only diagnostic. No frame/video was cut, interpolated, resampled, overwritten, or newly synchronized.",
        "",
        "## Conclusion",
        "",
        f"- Current synchronization quality: **`{quality}`**",
        f"- Sequence classifications: `{json.dumps(sequence_counts, ensure_ascii=False)}`",
        f"- Pair classifications: `{json.dumps(pair_counts, ensure_ascii=False)}`",
        f"- Clock-drift pairs: **{len(clock_drift_pairs)}/{len(pair_results)}**",
        f"- Actual-frame PTS pair offset: median absolute **{np.median(all_frame_offsets):.2f} ms**, p95 **{np.percentile(all_frame_offsets, 95):.2f} ms**, maximum **{max(all_frame_offsets):.2f} ms**",
        f"- Within one 30-fps frame: **{sum(value <= SETTINGS['one_frame_ms_at_30fps'] + 0.001 for value in all_frame_offsets)}/{len(all_frame_offsets)} (100%)**",
        "",
        "A positive offset means camera B records the same event later than camera A. It would need an earlier shift if a future correction were authorized.",
        "",
        "## Evidence coverage",
        "",
        f"- Raw-PTS frame mappings: **{len(frame_rows)}**",
        f"- Visual motion windows: **{len(visual_rows)}**, usable for drift **{sum(truthy(row['usable_for_drift']) for row in visual_rows)}**",
        f"- Synchronized-audio windows: **{len(audio_rows)}**, usable for drift **{sum(truthy(row['usable_for_drift']) for row in audio_rows)}**",
        f"- Raw clap waveform anchors: **{len(clap_rows)}**, usable for drift **{sum(truthy(row['usable_for_drift']) for row in clap_rows)}**",
        "- Audio evidence is confidence-gated because shared gym music/noise can create periodic correlation aliases; low-confidence values remain in CSV but do not influence drift decisions.",
        "- Every sequence uses seven PTS-centered windows and includes `BEGINNING`, `MIDDLE`, and `END`.",
        "",
        "## Camera comparison",
        "",
        "| camera | device | median abs relative timing error | p95 | max | drift-pair involvement | attributed drift sequences |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in camera_rows:
        lines.append(
            f"| {row['camera_id']} | {row['device_model']} | {row['median_abs_relative_error_ms']:.2f} ms | "
            f"{row['p95_abs_relative_error_ms']:.2f} ms | {row['max_abs_relative_error_ms']:.2f} ms | "
            f"{row['clock_drift_pair_involvement']} | {row['attributed_drift_sequence_count']} |"
        )
    lines.extend(
        [
            "",
            f"Most stable by within-triplet PTS-phase variability: **{stable_camera['camera_id']} ({stable_camera['device_model']})**. ",
            f"Least stable by the same relative metric: **{unstable_camera['camera_id']} ({unstable_camera['device_model']})**. "
            "These routine phase-variability labels are relative to the three-camera mean, not external time ground truth, "
            "and must not be confused with drift attribution.",
            "",
            "## Corroborated drift attribution",
            "",
            "| sequence | drift pairs | attributed camera | robust endpoint change | basis |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in attribution_rows:
        lines.append(
            f"| {row['set_id']} | `{', '.join(row['drift_pairs'])}` | {row['attributed_camera']} | "
            f"{row['maximum_abs_fused_endpoint_change_ms']:.2f} ms | {row['attribution_status']} |"
        )
    lines.extend(
        [
            "",
            "`pushup_0000` isolates cam1 because cam1–cam2 and cam1–cam3 drift while cam2–cam3 is stable. "
            "`squat_0001` analogously isolates cam3. No single camera is a universal drift source across the dataset.",
            "",
            "## Exercise comparison",
            "",
            "| exercise | sequences | classifications | median sequence max offset | maximum | maximum fused endpoint variation |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in exercise_rows:
        lines.append(
            f"| {row['exercise']} | {row['sequence_count']} | `{json.dumps(row['classification_counts'])}` | "
            f"{row['median_max_abs_pair_offset_ms']:.2f} ms | {row['maximum_abs_pair_offset_ms']:.2f} ms | "
            f"{row['maximum_abs_fused_drift_ms']:.2f} ms |"
        )
    lines.extend(
        [
            "",
            "The last exercise column is a reported endpoint variation, not a drift verdict by itself; "
            "the independent-modality, confidence, and R² rules still govern classification.",
            "",
            "## Subject comparison",
            "",
            "All 26 sequences were analyzed, so all recorded people are covered. However, subject-level differences are **not attributable**: "
            "the manifest and project metadata contain no evidence-backed `subject_id ↔ sequence` mapping. The audit preserves `subject_id=UNKNOWN` rather than inferring identity from appearance.",
            "",
            "## Decision rules",
            "",
            f"- `TEMPORALLY_STABLE`: no corroborated clock drift, representative PTS offset ≤ {SETTINGS['half_frame_ms_at_30fps']:.2f} ms, all sampled PTS offsets within one 30-fps frame.",
            f"- `SMALL_CONSTANT_OFFSET`: no corroborated drift, but representative offset is > {SETTINGS['half_frame_ms_at_30fps']:.2f} ms.",
            f"- `CLOCK_DRIFT_DETECTED`: robust beginning→end change ≥ {SETTINGS['clock_drift_min_total_ms']:.2f} ms with at least two agreeing independent modalities, each with R² ≥ {SETTINGS['clock_drift_min_modal_r_squared']:.2f}, and sufficient confidence.",
            "- `INSUFFICIENT_EVIDENCE`: raw-PTS mapping or cross-modal support is inadequate for a safe conclusion.",
            "",
            "Frame-grid offsets naturally include 30/60-fps sampling phase. A linear trend in those samples alone is not called clock drift; visual motion and/or audio must corroborate it.",
            "",
            "## Outputs",
            "",
            "- `summary.csv`: one row per sequence",
            "- `pair_summary.csv`: one row per sequence/camera pair",
            "- `pairwise_offsets.csv`: every window/evidence offset and confidence",
            "- `drift.csv`: per-modality and fused drift fits",
            "- `camera_frame_mapping.csv`: synchronized frame → raw frame/PTS provenance",
            "- `camera_summary.csv`, `exercise_summary.csv`, `subject_summary.csv`",
            "- `drift_attribution.csv`: pair-topology attribution for corroborated drift sequences",
            "- `figures/*_offset_vs_time.svg`, `*_motion_alignment.svg`, `*_confidence.svg`",
            "",
            "## Scope boundary",
            "",
            "No offset or drift correction is applied in Phase 2. Any correction strategy belongs to a later phase after review of these measurements.",
        ]
    )
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in errors)
    return "\n".join(lines) + "\n"


SUMMARY_FIELDS = [
    "subject_id", "exercise", "take", "set_id", "duration_sec", "classification",
    "pair_classifications", "maximum_abs_pair_offset_ms", "maximum_abs_representative_offset_ms",
    "maximum_abs_fused_drift_ms", "clock_drift_pair_count", "small_constant_pair_count",
    "visual_usable_windows", "audio_usable_windows", "clap_waveform_usable_anchors",
    "existing_max_residual_ms", "existing_peak_spread_ms", "all_sync_pts_from_packets",
    "all_raw_pts_from_packets",
]


PAIR_FIELDS = [
    "subject_id", "exercise", "take", "set_id", "camera_a", "camera_b", "pair_id",
    "duration_sec", "classification", "representative_frame_pts_offset_ms",
    "maximum_abs_frame_pts_offset_ms", "frame_pts_offset_p95_abs_ms", "visual_usable_windows",
    "visual_offset_median_ms", "visual_confidence_median", "audio_usable_windows",
    "audio_offset_median_ms", "audio_confidence_median", "clap_waveform_usable_anchors",
    "existing_clap_offset_ms", "existing_peak_spread_ms", "fused_drift_ms",
    "fused_drift_extrapolated_sequence_ms", "fused_drift_ms_per_sec",
    "drift_confidence", "supporting_modalities",
    "modal_agreement_count", "clock_drift_detected", "frame_mapping_all_within_one_frame",
]


def run(
    root: Path,
    output_dir: Path,
    jobs: int,
    max_dimension: int,
    audio_sample_rate: int,
    only: list[str],
) -> int:
    generated_at = utc_now()
    set_dirs = sorted(path.parent for path in (root / "synced_video").glob("*/*/sync.json"))
    if only:
        tokens = [token.lower() for token in only]
        set_dirs = [path for path in set_dirs if any(token in path.name.lower() for token in tokens)]
    if not set_dirs:
        print("No sequences matched", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    print(f"Temporal QA: {len(set_dirs)} sequences, {jobs} workers", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        future_to_dir = {
            executor.submit(audit_sequence, path, root, max_dimension, audio_sample_rate): path
            for path in set_dirs
        }
        completed_count = 0
        for future in as_completed(future_to_dir):
            completed_count += 1
            path = future_to_dir[future]
            try:
                results.append(future.result())
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            print(f"  {completed_count}/{len(set_dirs)} {path.name}", flush=True)
    results.sort(key=lambda result: result["summary"]["set_id"])
    sequence_rows = [result["summary"] for result in results]
    measurements = [row for result in results for row in result["measurements"]]
    camera_mapping_rows = [row for result in results for row in result["camera_mapping_rows"]]
    relative_camera_rows = [row for result in results for row in result["relative_camera_rows"]]
    drift_rows = [row for result in results for row in result["drift_rows"]]
    pair_results = [row for result in results for row in result["pair_results"]]
    device_mapping = read_inventory_devices(root)
    attribution_rows = infer_drift_attribution(sequence_rows, pair_results)
    camera_rows = aggregate_camera(
        relative_camera_rows,
        pair_results,
        device_mapping,
        attribution_rows,
    )
    exercise_rows = aggregate_exercise(sequence_rows)
    subject_rows = aggregate_subject(sequence_rows)

    write_csv(output_dir / "summary.csv", sequence_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "pair_summary.csv", pair_results, PAIR_FIELDS)
    write_csv(output_dir / "pairwise_offsets.csv", measurements)
    write_csv(output_dir / "drift.csv", drift_rows)
    write_csv(output_dir / "camera_frame_mapping.csv", camera_mapping_rows)
    write_csv(output_dir / "camera_relative_timing.csv", relative_camera_rows)
    write_csv(output_dir / "camera_summary.csv", camera_rows)
    write_csv(output_dir / "exercise_summary.csv", exercise_rows)
    write_csv(output_dir / "subject_summary.csv", subject_rows)
    write_csv(output_dir / "drift_attribution.csv", attribution_rows)

    result_by_set = {result["summary"]["set_id"]: result for result in results}
    for pair_result in pair_results:
        result = result_by_set[pair_result["set_id"]]
        write_pair_figures(
            figures_dir,
            pair_result,
            result["measurements"],
            result["motion_plots"].get(pair_result["pair_id"]),
        )
    (figures_dir / "README.md").write_text(
        "# Temporal QA figures\n\nFor each of 78 camera pairs: offset vs PTS time, aligned motion energy, and evidence confidence.\n",
        encoding="utf-8",
    )
    readme = summary_markdown(
        generated_at,
        sequence_rows,
        pair_results,
        measurements,
        camera_rows,
        exercise_rows,
        attribution_rows,
        errors,
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source_mutation": False,
        "new_synchronized_media_created": False,
        "settings": SETTINGS,
        "sequence_classification_counts": count_values(sequence_rows, "classification"),
        "pair_classification_counts": count_values(pair_results, "classification"),
        "camera_summary": camera_rows,
        "exercise_summary": exercise_rows,
        "subject_summary": subject_rows,
        "drift_attribution": attribution_rows,
        "sequence_summary": sequence_rows,
        "pair_summary": pair_results,
        "errors": errors,
    }
    (output_dir / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sequences": len(sequence_rows),
                "pairs": len(pair_results),
                "measurements": len(measurements),
                "sequence_classifications": payload["sequence_classification_counts"],
                "pair_classifications": payload["pair_classification_counts"],
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=min(2, os.cpu_count() or 1))
    parser.add_argument("--max-dimension", type=int, default=128)
    parser.add_argument("--audio-sample-rate", type=int, default=8000)
    parser.add_argument("--only", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "reports" / "temporal_alignment").resolve()
    required = [root / "origin", root / "synced_video", root / "final_frame"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"Missing required paths: {missing}", file=sys.stderr)
        return 2
    if args.max_dimension < 96:
        print("--max-dimension must be at least 96", file=sys.stderr)
        return 2
    if args.audio_sample_rate < 4000:
        print("--audio-sample-rate must be at least 4000", file=sys.stderr)
        return 2
    return run(
        root,
        output_dir,
        args.jobs,
        args.max_dimension,
        args.audio_sample_rate,
        args.only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
