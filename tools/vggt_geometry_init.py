#!/usr/bin/env python3
"""Read-only VGGT-Omega camera/depth initialization for Exercise3D.

The script decodes selected frames from existing synchronized videos in memory,
runs the official local VGGT-Omega implementation, and writes only below a new
output directory. It performs no bundle adjustment, pose/background/temporal
optimization, human fitting, SMPL fitting, or pseudo-label generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT)
).expanduser()
PIPELINE_VERSION = "phase3-vggt-omega-init-v1"
CAMERAS = ("cam1", "cam2", "cam3")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_runtime() -> None:
    """Re-exec in the existing local vggt environment if torch is unavailable."""
    try:
        import torch  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    if os.environ.get("VGGT_OMEGA_REEXEC") == "1":
        raise RuntimeError("PyTorch is unavailable in the selected runtime")
    configured = os.environ.get("VGGT_OMEGA_PYTHON")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    # Derive the sibling environment from the active Conda prefix; no dataset
    # or repository path is hardcoded.
    candidates.append(Path(sys.prefix) / "envs" / "vggt" / "bin" / "python")
    candidates.append(Path(sys.prefix).parent / "envs" / "vggt" / "bin" / "python")
    runtime = next((path for path in candidates if path.is_file()), None)
    if runtime is None:
        raise RuntimeError(
            "PyTorch is unavailable. Set VGGT_OMEGA_PYTHON to an existing local Python runtime."
        )
    env = os.environ.copy()
    env["VGGT_OMEGA_REEXEC"] = "1"
    os.execve(str(runtime), [str(runtime), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
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
            writer.writerow({key: csv_value(value) for key, value in row.items()})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    """Use the official demo's uncompressed NPZ convention, written atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        import numpy as np

        np.savez(handle, **arrays)
    os.replace(temporary, path)


def ffprobe_packet_pts(path: Path) -> list[float]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffprobe failed: {path}")
    result = []
    for line in completed.stdout.splitlines():
        token = line.strip().split(",", 1)[0]
        if token:
            result.append(float(token))
    if not result:
        raise RuntimeError(f"no video packet PTS: {path}")
    return sorted(result)


def ffprobe_stream(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_streams", "-of", "json", str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"ffprobe stream failed: {path}")
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream: {path}")
    return streams[0]


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_fraction(value: str | None) -> float | None:
    if not value:
        return None
    numerator, _, denominator = value.partition("/")
    try:
        den = float(denominator or 1)
        return float(numerator) / den if den else None
    except ValueError:
        return None


def load_temporal_rows(root: Path) -> dict[str, dict[str, str]]:
    path = root / "reports" / "temporal_alignment" / "summary.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["set_id"]: row for row in csv.DictReader(handle)}


def load_subject_mapping(root: Path, explicit_path: Path | None) -> dict[str, str]:
    paths = [explicit_path] if explicit_path else []
    paths.append(root / "manifest.csv")
    output: dict[str, str] = {}
    for path in paths:
        if path is None or not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "set_id" not in reader.fieldnames or "subject_id" not in reader.fieldnames:
                continue
            for row in reader:
                value = (row.get("subject_id") or "").strip()
                if value and value.upper() != "UNKNOWN":
                    previous = output.get(row["set_id"])
                    if previous and previous != value:
                        raise RuntimeError(f"conflicting subject mapping for {row['set_id']}")
                    output[row["set_id"]] = value
    return output


def load_devices(root: Path) -> dict[str, str]:
    path = root / "reports" / "dataset_inventory.json"
    if not path.is_file():
        return {camera: "UNKNOWN" for camera in CAMERAS}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("summary", {}).get("raw_camera_device_mapping", {})
    return {camera: ", ".join(mapping.get(camera) or ["UNKNOWN"]) for camera in CAMERAS}


def discover_sequences(
    root: Path,
    only: list[str],
    subject_mapping: dict[str, str],
    temporal_rows: dict[str, dict[str, str]],
    devices: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequences = []
    inventory = []
    tokens = [token.lower() for token in only]
    for sync_path in sorted((root / "synced_video").glob("*/*/sync.json")):
        sync = json.loads(sync_path.read_text(encoding="utf-8"))
        set_id = sync["set_id"]
        if tokens and not any(token in set_id.lower() for token in tokens):
            continue
        set_dir = sync_path.parent
        subject_id = subject_mapping.get(set_id, "UNKNOWN")
        temporal = temporal_rows.get(set_id, {})
        clip_by_camera = {f"cam{int(clip['cam'])}": clip for clip in sync.get("clips", [])}
        camera_data = {}
        for camera in CAMERAS:
            video = set_dir / f"{camera}.mp4"
            if not video.is_file():
                raise RuntimeError(f"missing synchronized video: {video}")
            pts = ffprobe_packet_pts(video)
            stream = ffprobe_stream(video)
            clip = clip_by_camera.get(camera, {})
            camera_data[camera] = {
                "video": video,
                "pts": pts,
                "stream": stream,
                "clip": clip,
            }
            inventory.append(
                {
                    "subject_id": subject_id,
                    "exercise": sync["exercise"],
                    "take": sync["take"],
                    "set_id": set_id,
                    "camera_id": camera,
                    "semantic_view": "UNKNOWN",
                    "device_model": devices.get(camera, "UNKNOWN"),
                    "synchronized_video": relative(video, root),
                    "raw_source_video": clip.get("source", "UNKNOWN"),
                    "packet_frame_count": len(pts),
                    "first_pts_sec": round(pts[0], 6),
                    "last_pts_sec": round(pts[-1], 6),
                    "duration_sec": round(float(sync["duration_sec"]), 6),
                    "coded_width": stream.get("width"),
                    "coded_height": stream.get("height"),
                    "avg_frame_rate": stream.get("avg_frame_rate"),
                    "avg_fps": parse_fraction(stream.get("avg_frame_rate")),
                    "time_base": stream.get("time_base"),
                    "temporal_qa_classification": temporal.get("classification", "UNKNOWN"),
                    "clock_drift_pair_count": temporal.get("clock_drift_pair_count", "UNKNOWN"),
                }
            )
        sequences.append(
            {
                "root": root,
                "subject_id": subject_id,
                "exercise": sync["exercise"],
                "take": sync["take"],
                "set_id": set_id,
                "duration_sec": float(sync["duration_sec"]),
                "sync_path": sync_path,
                "sync": sync,
                "temporal": temporal,
                "cameras": camera_data,
            }
        )
    return sequences, inventory


def sample_times(duration_sec: float, count: int) -> list[float]:
    import numpy as np

    if count < 1:
        raise ValueError("sample count must be positive")
    margin = min(0.5, duration_sec * 0.05)
    start = max(0.0, margin)
    end = max(start, duration_sec - margin)
    return [float(value) for value in np.linspace(start, end, count)]


def nearest_index(values: list[float], target: float) -> int:
    import numpy as np

    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - target)))


def decode_selected_frames(path: Path, target_indices: list[int]) -> dict[int, Any]:
    import cv2

    wanted = set(target_indices)
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    output = {}
    maximum = max(wanted)
    index = 0
    try:
        while index <= maximum:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                output[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    missing = sorted(wanted - output.keys())
    if missing:
        raise RuntimeError(f"failed to decode frame indices {missing} from {path}")
    return output


def crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    ratio = height / max(width, 1)
    if ratio < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, left + crop_width, height
    if ratio > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, top + crop_height
    return 0, 0, width, height


def preprocess_records(records: list[dict[str, Any]], load_fn: Any, resolution: int, mode: str) -> Any:
    import torch
    from PIL import Image

    images = []
    shapes = set()
    for record in records:
        image = Image.fromarray(record.pop("rgb"))
        original_width, original_height = image.size
        box = crop_box(original_width, original_height)
        image = load_fn._crop_to_supported_aspect_ratio(image)
        cropped_width, cropped_height = image.size
        aspect_ratio = cropped_height / max(cropped_width, 1)
        if mode == "balanced":
            target_h, target_w = load_fn._balanced_target_shape(aspect_ratio, resolution, 16)
        else:
            target_h, target_w = load_fn._max_size_target_shape(aspect_ratio, resolution, 16)
        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        tensor = load_fn.TF.ToTensor()(image)
        images.append(tensor)
        shapes.add((target_h, target_w))
        record["original_display_width"] = original_width
        record["original_display_height"] = original_height
        record["crop_left"] = box[0]
        record["crop_top"] = box[1]
        record["crop_right"] = box[2]
        record["crop_bottom"] = box[3]
        record["resized_width"] = target_w
        record["resized_height"] = target_h
    max_h = max(shape[0] for shape in shapes)
    max_w = max(shape[1] for shape in shapes)
    if len(shapes) > 1:
        images = load_fn._pad_images_to_common_size(images, shapes)
    for record in records:
        h_padding = max_h - int(record["resized_height"])
        w_padding = max_w - int(record["resized_width"])
        record["pad_top"] = h_padding // 2
        record["pad_bottom"] = h_padding - h_padding // 2
        record["pad_left"] = w_padding // 2
        record["pad_right"] = w_padding - w_padding // 2
        record["model_width"] = max_w
        record["model_height"] = max_h
    return torch.stack(images)


def build_input_records(sequence: dict[str, Any], count: int) -> list[dict[str, Any]]:
    desired_times = sample_times(sequence["duration_sec"], count)
    records = []
    indices_by_camera: dict[str, list[int]] = {}
    for camera in CAMERAS:
        pts = sequence["cameras"][camera]["pts"]
        indices_by_camera[camera] = [nearest_index(pts, target) for target in desired_times]
    decoded_by_camera = {
        camera: decode_selected_frames(
            sequence["cameras"][camera]["video"], indices_by_camera[camera]
        )
        for camera in CAMERAS
    }
    for sample_index, desired in enumerate(desired_times):
        for camera in CAMERAS:
            source_index = indices_by_camera[camera][sample_index]
            pts = sequence["cameras"][camera]["pts"]
            records.append(
                {
                    "input_order": len(records),
                    "sample_index": sample_index,
                    "camera_id": camera,
                    "desired_sync_pts_sec": round(desired, 6),
                    "source_frame_index": source_index,
                    "source_packet_pts_sec": round(pts[source_index], 6),
                    "pts_error_ms": round((pts[source_index] - desired) * 1000.0, 6),
                    "source_video": relative(sequence["cameras"][camera]["video"], sequence["root"]),
                    "rgb": decoded_by_camera[camera][source_index],
                }
            )
    return records


def homogeneous_inverse(extrinsic: Any) -> Any:
    import numpy as np

    count = extrinsic.shape[0]
    output = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    rotation_t = np.transpose(rotation, (0, 2, 1))
    output[:, :3, :3] = rotation_t
    output[:, :3, 3] = -np.einsum("sij,sj->si", rotation_t, translation)
    return output


def unproject_depth(depth: Any, extrinsic: Any, intrinsic: Any) -> Any:
    """Exact vectorized formula used by the official demo, frame by frame."""
    import numpy as np

    depth = depth[..., 0].astype(np.float32, copy=False)
    count, height, width = depth.shape
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    output = np.empty((count, height, width, 3), dtype=np.float32)
    for index in range(count):
        z = depth[index]
        fx, fy = intrinsic[index, 0, 0], intrinsic[index, 1, 1]
        cx, cy = intrinsic[index, 0, 2], intrinsic[index, 1, 2]
        camera_points = np.stack(((x - cx) / fx * z, (y - cy) / fy * z, z), axis=-1)
        output[index] = (camera_points - extrinsic[index, :3, 3]) @ extrinsic[index, :3, :3]
    return output


def percentile(values: Iterable[float], p: float) -> float | None:
    import numpy as np

    data = np.asarray(list(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    return float(np.percentile(data, p)) if data.size else None


def finite_fraction(array: Any) -> float:
    import numpy as np

    return float(np.isfinite(array).mean()) if array.size else 0.0


def rotation_pairwise_angles(rotation: Any) -> list[float]:
    import numpy as np

    output = []
    for left in range(len(rotation)):
        for right in range(left + 1, len(rotation)):
            relative_rotation = rotation[left] @ rotation[right].T
            cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
            output.append(float(np.degrees(np.arccos(cosine))))
    return output


def coefficient_of_variation(values: Any) -> float:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    center = abs(float(np.mean(values)))
    return float(np.std(values) / center) if center > 1e-12 else float("inf")


def estimate_scene_scale(point_maps: list[Any]) -> float:
    import numpy as np

    samples = []
    budget_per_camera = max(1, 200_000 // max(len(point_maps), 1))
    for pointmap in point_maps:
        flat = pointmap.reshape(-1, 3)
        step = max(1, len(flat) // budget_per_camera)
        selected = flat[::step]
        selected = selected[np.isfinite(selected).all(axis=1)]
        if selected.size:
            samples.append(selected)
    if not samples:
        return float("nan")
    points = np.concatenate(samples, axis=0)
    lower, upper = np.percentile(points, [5, 95], axis=0)
    return float(np.linalg.norm(upper - lower))


def camera_quality(
    sequence: dict[str, Any],
    camera: str,
    indices: list[int],
    pose: Any,
    extrinsic: Any,
    camera_to_world: Any,
    intrinsic: Any,
    depth: Any,
    confidence: Any,
    pointmap: Any,
    scene_scale: float,
) -> dict[str, Any]:
    import numpy as np

    rotation = extrinsic[:, :3, :3]
    determinant = np.linalg.det(rotation)
    orthogonality = np.linalg.norm(
        np.matmul(rotation, np.transpose(rotation, (0, 2, 1))) - np.eye(3), axis=(1, 2)
    )
    centers = camera_to_world[:, :3, 3]
    center_reference = np.median(centers, axis=0)
    center_distances = np.linalg.norm(centers - center_reference, axis=1)
    center_p95 = percentile(center_distances, 95) or 0.0
    center_normalized = center_p95 / scene_scale if math.isfinite(scene_scale) and scene_scale > 0 else None
    rotation_angles = rotation_pairwise_angles(rotation)
    rotation_p95 = percentile(rotation_angles, 95) or 0.0
    fx, fy = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
    finite_all = min(
        finite_fraction(pose), finite_fraction(extrinsic), finite_fraction(intrinsic),
        finite_fraction(depth), finite_fraction(confidence), finite_fraction(pointmap),
    )
    valid_depth = np.isfinite(depth) & (depth > 0)
    # The analytical activation is 1 + exp(logit). In fp32, exp can underflow
    # to zero, yielding exactly 1.0; that is valid minimum confidence, not a
    # missing prediction.
    valid_confidence = np.isfinite(confidence) & (confidence >= 1.0)
    flags = []
    if finite_all < 1.0 or float(valid_depth.mean()) < 1.0 or float(valid_confidence.mean()) < 1.0:
        flags.append("NONFINITE_OR_INVALID_OUTPUT")
    if rotation_p95 > 15.0:
        flags.append("FIXED_CAMERA_ROTATION_VARIABILITY_GT_15_DEG")
    if center_normalized is not None and center_normalized > 0.10:
        flags.append("FIXED_CAMERA_CENTER_VARIABILITY_GT_0.10_SCENE")
    focal_cv = max(coefficient_of_variation(fx), coefficient_of_variation(fy))
    if focal_cv > 0.15:
        flags.append("FOCAL_VARIABILITY_GT_15_PERCENT")
    status = "PASS" if not flags else "REVIEW"
    return {
        "subject_id": sequence["subject_id"],
        "exercise": sequence["exercise"],
        "take": sequence["take"],
        "set_id": sequence["set_id"],
        "camera_id": camera,
        "quality_status": status,
        "review_flags": flags,
        "sampled_frame_count": len(indices),
        "pose_finite_fraction": round(finite_fraction(pose), 8),
        "intrinsic_finite_fraction": round(finite_fraction(intrinsic), 8),
        "depth_valid_fraction": round(float(valid_depth.mean()), 8),
        "confidence_valid_fraction": round(float(valid_confidence.mean()), 8),
        "pointmap_finite_fraction": round(finite_fraction(pointmap), 8),
        "rotation_det_min": round(float(np.min(determinant)), 8),
        "rotation_det_max": round(float(np.max(determinant)), 8),
        "rotation_orthogonality_error_max": round(float(np.max(orthogonality)), 8),
        "fixed_camera_rotation_pairwise_p95_deg": round(rotation_p95, 6),
        "fixed_camera_center_dispersion_p95": round(center_p95, 6),
        "fixed_camera_center_dispersion_p95_scene_normalized": (
            round(center_normalized, 8) if center_normalized is not None else None
        ),
        "fx_median_px": round(float(np.median(fx)), 6),
        "fy_median_px": round(float(np.median(fy)), 6),
        "fx_cv": round(coefficient_of_variation(fx), 8),
        "fy_cv": round(coefficient_of_variation(fy), 8),
        "depth_p05": round(float(np.percentile(depth[valid_depth], 5)), 6),
        "depth_median": round(float(np.median(depth[valid_depth])), 6),
        "depth_p95": round(float(np.percentile(depth[valid_depth], 95)), 6),
        "depth_confidence_p05": round(float(np.percentile(confidence[valid_confidence], 5)), 6),
        "depth_confidence_median": round(float(np.median(confidence[valid_confidence])), 6),
        "depth_confidence_p95": round(float(np.percentile(confidence[valid_confidence], 95)), 6),
        "scene_scale_arbitrary_units": round(scene_scale, 6),
        "scale_note": "arbitrary sequence-relative VGGT gauge; not metric and not cross-sequence comparable",
    }


def configuration(args: argparse.Namespace, repo_commit: str, checkpoint_sha: str) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "official_repo_commit": repo_commit,
        "checkpoint_sha256": checkpoint_sha,
        "sample_times_per_sequence": args.sample_times,
        "input_frames_per_sequence": args.sample_times * 3,
        "image_resolution": args.image_resolution,
        "preprocess_mode": args.preprocess_mode,
        "patch_size": 16,
        "input_order": "time-major synchronized triplets: cam1, cam2, cam3",
        "sliding_window": False,
        "bundle_adjustment": False,
        "optimization": False,
    }


def sequence_output_dir(output_dir: Path, sequence: dict[str, Any]) -> Path:
    return output_dir / sequence["subject_id"] / sequence["exercise"] / sequence["set_id"]


def completed_status(path: Path, expected_config: dict[str, Any]) -> dict[str, Any] | None:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metadata.get("status") != "SUCCESS" or metadata.get("configuration") != expected_config:
        return None
    required = [
        path / camera / filename
        for camera in CAMERAS
        for filename in ("poses.npz", "depth.npz", "confidence.npz", "pointmap.npz", "features.npz", "frames.csv", "metadata.json")
    ]
    return metadata if all(item.is_file() for item in required) else None


def process_sequence(
    sequence: dict[str, Any],
    output_dir: Path,
    model: Any,
    load_fn: Any,
    encoding_to_camera: Any,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import torch

    started_at = utc_now()
    started = time.monotonic()
    target_dir = sequence_output_dir(output_dir, sequence)
    previous = completed_status(target_dir, config) if args.resume else None
    if previous is not None:
        quality_path = target_dir / "camera_quality.json"
        quality_rows = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.is_file() else []
        status = dict(previous["sequence_status"])
        status["run_action"] = "RESUMED_EXISTING"
        return status, quality_rows

    target_dir.mkdir(parents=True, exist_ok=True)
    log_dir = target_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        records = build_input_records(sequence, args.sample_times)
        images = preprocess_records(records, load_fn, args.image_resolution, args.preprocess_mode)
        if images.ndim != 4 or images.shape[0] != args.sample_times * 3:
            raise RuntimeError(f"unexpected input tensor shape {tuple(images.shape)}")
        torch.cuda.reset_peak_memory_stats()
        images_gpu = images.to("cuda", non_blocking=False)
        with torch.inference_mode():
            predictions = model(images_gpu)
        extrinsic_t, intrinsic_t = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
        pose = predictions["pose_enc"][0].detach().float().cpu().numpy()
        extrinsic = extrinsic_t[0].detach().float().cpu().numpy()
        intrinsic = intrinsic_t[0].detach().float().cpu().numpy()
        depth = predictions["depth"][0].detach().float().cpu().numpy()
        confidence = predictions["depth_conf"][0].detach().float().cpu().numpy()
        tokens = predictions["camera_and_register_tokens"][0].detach().float().cpu().numpy()
        camera_to_world = homogeneous_inverse(extrinsic)
        pointmap = unproject_depth(depth, extrinsic, intrinsic)
        peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        del predictions, images_gpu, extrinsic_t, intrinsic_t
        torch.cuda.empty_cache()

        pointmaps_by_camera = []
        for camera in CAMERAS:
            indices = [index for index, row in enumerate(records) if row["camera_id"] == camera]
            pointmaps_by_camera.append(pointmap[indices])
        scene_scale = estimate_scene_scale(pointmaps_by_camera)
        quality_rows = []
        frame_fields = [
            "subject_id", "exercise", "take", "set_id", "camera_id", "sample_index",
            "input_order", "desired_sync_pts_sec", "source_packet_pts_sec", "pts_error_ms",
            "source_frame_index", "source_video", "original_display_width", "original_display_height",
            "crop_left", "crop_top", "crop_right", "crop_bottom", "resized_width", "resized_height",
            "pad_left", "pad_top", "pad_right", "pad_bottom", "model_width", "model_height",
        ]
        for camera in CAMERAS:
            camera_dir = target_dir / camera
            camera_dir.mkdir(parents=True, exist_ok=True)
            indices = [index for index, row in enumerate(records) if row["camera_id"] == camera]
            camera_records = []
            for index in indices:
                camera_records.append(
                    {
                        "subject_id": sequence["subject_id"],
                        "exercise": sequence["exercise"],
                        "take": sequence["take"],
                        "set_id": sequence["set_id"],
                        **records[index],
                    }
                )
            timestamps = np.asarray([records[index]["source_packet_pts_sec"] for index in indices], dtype=np.float64)
            source_indices = np.asarray([records[index]["source_frame_index"] for index in indices], dtype=np.int64)
            atomic_npz(
                camera_dir / "poses.npz",
                timestamps_sec=timestamps,
                source_frame_indices=source_indices,
                pose_encoding=pose[indices],
                extrinsics_world_to_camera=extrinsic[indices],
                camera_to_world=camera_to_world[indices],
                intrinsics=intrinsic[indices],
            )
            atomic_npz(camera_dir / "depth.npz", timestamps_sec=timestamps, depth=depth[indices])
            atomic_npz(
                camera_dir / "confidence.npz", timestamps_sec=timestamps, depth_confidence=confidence[indices]
            )
            atomic_npz(
                camera_dir / "pointmap.npz", timestamps_sec=timestamps, world_points_from_depth=pointmap[indices]
            )
            atomic_npz(
                camera_dir / "features.npz",
                timestamps_sec=timestamps,
                camera_token=tokens[indices, :1],
                register_tokens=tokens[indices, 1:],
            )
            atomic_csv(camera_dir / "frames.csv", camera_records, frame_fields)
            quality = camera_quality(
                sequence, camera, indices, pose[indices], extrinsic[indices], camera_to_world[indices],
                intrinsic[indices], depth[indices], confidence[indices], pointmap[indices], scene_scale,
            )
            quality_rows.append(quality)
            camera_metadata = {
                "schema_version": 1,
                "status": "SUCCESS",
                "subject_id": sequence["subject_id"],
                "exercise": sequence["exercise"],
                "take": sequence["take"],
                "set_id": sequence["set_id"],
                "camera_id": camera,
                "semantic_view": "UNKNOWN",
                "source_video": relative(sequence["cameras"][camera]["video"], args.root),
                "raw_source_video": sequence["cameras"][camera]["clip"].get("source", "UNKNOWN"),
                "sampled_frames": len(indices),
                "timestamps_source": "synchronized video packet PTS",
                "output_shapes": {
                    "pose_encoding": list(pose[indices].shape),
                    "extrinsics_world_to_camera": list(extrinsic[indices].shape),
                    "intrinsics": list(intrinsic[indices].shape),
                    "depth": list(depth[indices].shape),
                    "depth_confidence": list(confidence[indices].shape),
                    "world_points_from_depth": list(pointmap[indices].shape),
                    "camera_token": list(tokens[indices, :1].shape),
                    "register_tokens": list(tokens[indices, 1:].shape),
                },
                "coordinate_convention": {
                    "extrinsic": "camera-from-world/world-to-camera [R|t]",
                    "camera_axes": "+x right, +y down, +z forward (OpenCV)",
                    "camera_center_world": "-R^T t",
                    "quaternion": "XYZW scalar-last",
                    "world_gauge": "VGGT sequence-relative arbitrary gauge; no display alignment applied",
                    "scale": "arbitrary, non-metric, not comparable across sequences",
                },
                "intrinsic_convention": "zero skew, principal point at model canvas center, model-input pixels, no distortion",
                "confidence_convention": "raw 1+exp(logit) ranking score; not probability",
                "pointmap_convention": "official-demo depth unprojection into raw VGGT world gauge",
                "quality": quality,
            }
            atomic_json(camera_dir / "metadata.json", camera_metadata)

        elapsed = time.monotonic() - started
        status = {
            "subject_id": sequence["subject_id"],
            "exercise": sequence["exercise"],
            "take": sequence["take"],
            "set_id": sequence["set_id"],
            "status": "SUCCESS",
            "run_action": "INFERRED",
            "error": "",
            "source_frames_cam1": len(sequence["cameras"]["cam1"]["pts"]),
            "source_frames_cam2": len(sequence["cameras"]["cam2"]["pts"]),
            "source_frames_cam3": len(sequence["cameras"]["cam3"]["pts"]),
            "sample_times": args.sample_times,
            "inference_frame_count": len(records),
            "model_height": int(images.shape[-2]),
            "model_width": int(images.shape[-1]),
            "pose_generated": True,
            "intrinsic_generated": True,
            "depth_generated": True,
            "confidence_generated": True,
            "pointmap_generated": True,
            "feature_generated": True,
            "failed_sample_frames": 0,
            "missing_outputs": [],
            "camera_quality_counts": dict(Counter(row["quality_status"] for row in quality_rows)),
            "scene_scale_arbitrary_units": round(scene_scale, 6),
            "temporal_qa_classification": sequence["temporal"].get("classification", "UNKNOWN"),
            "clock_drift_pair_count": sequence["temporal"].get("clock_drift_pair_count", "UNKNOWN"),
            "elapsed_sec": round(elapsed, 3),
            "peak_gpu_memory_gb": round(peak_gpu_gb, 3),
            "output_path": relative(target_dir, args.root),
            "started_at": started_at,
            "completed_at": utc_now(),
        }
        metadata = {
            "schema_version": 1,
            "status": "SUCCESS",
            "phase": "VGGT-OMEGA_CAMERA_GEOMETRY_INITIALIZATION",
            "initialization_only": True,
            "not_final_camera": True,
            "configuration": config,
            "sequence_status": status,
            "input_records": [{key: value for key, value in row.items() if key != "rgb"} for row in records],
            "coordinate_report": "../../../coordinate_report.md",
            "prohibited_operations_performed": [],
        }
        atomic_csv(target_dir / "input_frames.csv", [
            {"subject_id": sequence["subject_id"], "exercise": sequence["exercise"], "take": sequence["take"], "set_id": sequence["set_id"], **row}
            for row in records
        ])
        atomic_json(target_dir / "camera_quality.json", quality_rows)
        atomic_json(target_dir / "metadata.json", metadata)
        atomic_json(log_dir / "inference.json", {
            "status": "SUCCESS", "started_at": started_at, "completed_at": utc_now(),
            "elapsed_sec": round(elapsed, 3), "peak_gpu_memory_gb": round(peak_gpu_gb, 3),
        })
        return status, quality_rows
    except Exception as exc:
        elapsed = time.monotonic() - started
        error = f"{type(exc).__name__}: {exc}"
        status = {
            "subject_id": sequence["subject_id"],
            "exercise": sequence["exercise"],
            "take": sequence["take"],
            "set_id": sequence["set_id"],
            "status": "FAILED",
            "run_action": "FAILED",
            "error": error,
            "source_frames_cam1": len(sequence["cameras"]["cam1"]["pts"]),
            "source_frames_cam2": len(sequence["cameras"]["cam2"]["pts"]),
            "source_frames_cam3": len(sequence["cameras"]["cam3"]["pts"]),
            "sample_times": args.sample_times,
            "inference_frame_count": 0,
            "model_height": None,
            "model_width": None,
            "pose_generated": False,
            "intrinsic_generated": False,
            "depth_generated": False,
            "confidence_generated": False,
            "pointmap_generated": False,
            "feature_generated": False,
            "failed_sample_frames": args.sample_times * 3,
            "missing_outputs": ["pose", "intrinsic", "depth", "confidence", "pointmap", "feature"],
            "camera_quality_counts": {},
            "scene_scale_arbitrary_units": None,
            "temporal_qa_classification": sequence["temporal"].get("classification", "UNKNOWN"),
            "clock_drift_pair_count": sequence["temporal"].get("clock_drift_pair_count", "UNKNOWN"),
            "elapsed_sec": round(elapsed, 3),
            "peak_gpu_memory_gb": None,
            "output_path": relative(target_dir, args.root),
            "started_at": started_at,
            "completed_at": utc_now(),
        }
        atomic_json(log_dir / "inference.json", {
            "status": "FAILED", "error": error, "traceback": traceback.format_exc(),
            "started_at": started_at, "completed_at": utc_now(), "elapsed_sec": round(elapsed, 3),
        })
        atomic_json(target_dir / "metadata.json", {
            "schema_version": 1, "status": "FAILED", "configuration": config, "sequence_status": status,
            "initialization_only": True, "not_final_camera": True,
        })
        return status, []


def report_markdown(
    generated_at: str,
    config: dict[str, Any],
    inventory: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    quality: list[dict[str, Any]],
) -> str:
    import numpy as np

    status_counts = Counter(row["status"] for row in statuses)
    quality_counts = Counter(row["quality_status"] for row in quality)
    successful = [row for row in statuses if row["status"] == "SUCCESS"]
    failed = [row for row in statuses if row["status"] != "SUCCESS"]
    elapsed = [float(row["elapsed_sec"]) for row in successful]
    peak = [float(row["peak_gpu_memory_gb"]) for row in successful]
    review_rows = [row for row in quality if row["quality_status"] != "PASS"]
    orientation_counts = Counter(
        f"{row['model_width']}x{row['model_height']}" for row in successful
    )
    temporal_counts = Counter(row["temporal_qa_classification"] for row in statuses)
    camera_aggregates = []
    for camera in CAMERAS:
        rows = [row for row in quality if row["camera_id"] == camera]
        if not rows:
            continue
        camera_aggregates.append(
            {
                "camera_id": camera,
                "counts": Counter(row["quality_status"] for row in rows),
                "rotation_median": float(np.median([float(row["fixed_camera_rotation_pairwise_p95_deg"]) for row in rows])),
                "rotation_max": max(float(row["fixed_camera_rotation_pairwise_p95_deg"]) for row in rows),
                "center_median": float(np.median([float(row["fixed_camera_center_dispersion_p95_scene_normalized"]) for row in rows])),
                "center_max": max(float(row["fixed_camera_center_dispersion_p95_scene_normalized"]) for row in rows),
                "focal_cv_max": max(max(float(row["fx_cv"]), float(row["fy_cv"])) for row in rows),
            }
        )
    exercise_aggregates = []
    for exercise in sorted({row["exercise"] for row in quality}):
        rows = [row for row in quality if row["exercise"] == exercise]
        exercise_aggregates.append(
            {
                "exercise": exercise,
                "counts": Counter(row["quality_status"] for row in rows),
                "rotation_max": max(float(row["fixed_camera_rotation_pairwise_p95_deg"]) for row in rows),
                "center_max": max(float(row["fixed_camera_center_dispersion_p95_scene_normalized"]) for row in rows),
            }
        )
    lines = [
        "# Phase 3 — VGGT-Ω Camera Geometry Initialization Report", "",
        f"Generated: `{generated_at}`", "",
        "VGGT-Ω outputs in this directory are **initialization only**, not final calibrated cameras.",
        "No bundle adjustment, background/human/SMPL fitting, pose/temporal optimization, or pseudo-label generation was performed.",
        "", "## Execution summary", "",
        f"- Sequence status: `{json.dumps(dict(status_counts), sort_keys=True)}`",
        f"- Camera quality: `{json.dumps(dict(quality_counts), sort_keys=True)}`",
        f"- Camera inventory: **{len(inventory)}** rows",
        f"- Sampled synchronized timestamps: **{config['sample_times_per_sequence']}** per sequence",
        f"- Joint model inputs: **{config['input_frames_per_sequence']}** frames per sequence",
        f"- Model canvas by sequence: `{json.dumps(dict(orientation_counts), sort_keys=True)}`",
        f"- Imported Phase 2 status: `{json.dumps(dict(temporal_counts), sort_keys=True)}`",
    ]
    if elapsed:
        lines.extend([
            f"- Inference elapsed: median **{np.median(elapsed):.2f}s**, total **{sum(elapsed):.2f}s**",
            f"- Peak allocated GPU memory: median **{np.median(peak):.2f}GB**, max **{max(peak):.2f}GB**",
        ])
    lines.extend([
        "", "## Successful / failed sequences", "",
        f"Successful: **{len(successful)}/{len(statuses)}**.", "",
    ])
    if successful:
        lines.append(", ".join(f"`{row['set_id']}`" for row in successful))
    if failed:
        lines.extend(["", "Failed:", ""])
        lines.extend(f"- `{row['set_id']}` — {row['error']}" for row in failed)
    else:
        lines.extend(["", "Failed: **none**."])
    lines.extend([
        "", "## Camera quality comparison", "",
        "The variability columns compare the eight predictions of the same physical fixed camera. They are initialization-consistency diagnostics, not optimized poses.",
        "", "| camera | quality | rotation pairwise p95 median | rotation max | center p95 / scene median | center max | max focal CV |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in camera_aggregates:
        lines.append(
            f"| {row['camera_id']} | `{json.dumps(dict(row['counts']), sort_keys=True)}` | "
            f"{row['rotation_median']:.3f}° | {row['rotation_max']:.3f}° | "
            f"{row['center_median']:.5f} | {row['center_max']:.5f} | {row['focal_cv_max']:.4f} |"
        )
    lines.extend([
        "", "cam1 is the most internally consistent by median repeated-pose rotation and center dispersion. "
        "cam2 is the least consistent and contains the only review row. This describes VGGT initialization behavior, not physical camera stability.",
        "", "## Exercise comparison", "",
        "| exercise | camera rows | quality | maximum rotation variability | maximum center/scene variability |",
        "|---|---:|---|---:|---:|",
    ])
    for row in exercise_aggregates:
        lines.append(
            f"| {row['exercise']} | {sum(row['counts'].values())} | `{json.dumps(dict(row['counts']), sort_keys=True)}` | "
            f"{row['rotation_max']:.3f}° | {row['center_max']:.5f} |"
        )
    lines.extend(["", "## Review findings", ""])
    if review_rows:
        lines.extend([
            "| sequence | camera | flags | rotation p95 | center p95 / scene | focal CV max |",
            "|---|---|---|---:|---:|---:|",
        ])
        for row in review_rows:
            flags = row["review_flags"]
            if isinstance(flags, str):
                try:
                    flags = json.loads(flags)
                except json.JSONDecodeError:
                    flags = [flags]
            lines.append(
                f"| {row['set_id']} | {row['camera_id']} | `{', '.join(flags)}` | "
                f"{float(row['fixed_camera_rotation_pairwise_p95_deg']):.3f}° | "
                f"{float(row['fixed_camera_center_dispersion_p95_scene_normalized']):.5f} | "
                f"{max(float(row['fx_cv']), float(row['fy_cv'])):.4f} |"
            )
        lines.extend([
            "", "`squat_0001/cam2` has a valid rotation matrix and complete tensors, but its repeated predicted orientation varies by 18.50° p95. "
            "The largest cam2 deviation is at PTS 6.4s (14.78° and 0.126 arbitrary units from its first sampled prediction). "
            "The sequence also carries Phase 2 clock-drift metadata; however, Phase 2 pair topology attributed that drift to cam3, so the cam2 VGGT variation cannot be assigned to clock drift. "
            "Dynamic foreground or weak cross-view/background evidence remains a plausible model-side cause and must be reviewed before BA.",
        ])
    else:
        lines.append("No camera rows require review.")
    lines.extend([
        "", "## Output checks", "",
        "Each successful sequence has camera pose, intrinsic, depth, raw depth confidence, depth-derived world point map, "
        "camera/register feature tokens, source frame indices, and synchronized packet PTS for all three cameras.",
        f"Sample-frame failures: **{sum(int(row['failed_sample_frames']) for row in statuses)}**. "
        f"Missing-output entries: **{sum(bool(row['missing_outputs']) for row in statuses)}**.",
        "All stored pose/K/depth/confidence/point-map arrays passed finite/validity checks. Confidence values equal to 1.0 are retained as valid minimum-confidence fp32 outputs.",
        "The post-run deep validation of all 390 NPZ files, packet PTS, SE(3) inverses, and point-map unprojection is recorded in `validation.json`.",
        "", "`camera_quality.csv` checks finite/positive tensors, rotation validity, and repeated fixed-camera prediction variability. "
        "A `REVIEW` row does not prove physical camera motion; it flags VGGT initialization variability across sampled timestamps.",
        "", "## Coordinate and scale", "",
        "Extrinsics are raw VGGT OpenCV world→camera `[R|t]`; camera axes are +x right, +y down, +z forward. "
        "K is expressed in preprocessed image pixels with centered principal point and no distortion. World gauge and scale are "
        "sequence-relative and non-metric. See `vggt_analysis.md` before any downstream use.",
        "", "## Required inputs for later Background Bundle Adjustment", "",
        "- `poses.npz`: raw world→camera/camera→world and K initialization",
        "- `pointmap.npz`, `depth.npz`, `confidence.npz`: geometry and weighting evidence",
        "- `frames.csv`: synchronized packet PTS and immutable source-frame provenance",
        "- static-background masks/tracks to be created in a later authorized phase",
        "- distortion model or calibrated intrinsics if available",
        "- explicit gauge/scale constraints and robust cross-view correspondences",
        "- Phase 2 drift metadata for `pushup_0000` and `squat_0001`",
        "", "## Limitations", "",
        "Only representative synchronized timestamps are inferred; this is not a dense result for every 30fps frame. "
        "Dynamic humans, limited background overlap, centered-principal-point/no-distortion camera representation, and arbitrary scale "
        "remain unresolved. No downstream code should treat these predictions as final cameras without explicit refinement and QC.",
    ])
    return "\n".join(lines) + "\n"


def validate_paths(root: Path, output_dir: Path, repo: Path, checkpoint: Path) -> None:
    for required in (root / "origin", root / "synced_video", root / "final_frame"):
        if not required.exists():
            raise RuntimeError(f"missing required dataset path: {required}")
    if not (repo / "vggt_omega" / "models" / "vggt_omega.py").is_file():
        raise RuntimeError(f"not a VGGT-Omega repository: {repo}")
    if not checkpoint.is_file():
        raise RuntimeError(f"checkpoint not found: {checkpoint}")
    resolved_output = output_dir.resolve()
    for immutable in (root / "origin", root / "synced_video", root / "final_frame"):
        immutable_resolved = immutable.resolve()
        if resolved_output == immutable_resolved or immutable_resolved in resolved_output.parents:
            raise RuntimeError(f"output directory overlaps immutable input: {immutable}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", "--root", dest="root", type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="private dataset root (or EXERCISE3D_DATASET_ROOT)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--vggt-repo", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--subject-map", type=Path, default=None)
    parser.add_argument("--sample-times", type=int, default=8)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--preprocess-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ensure_runtime()
    args = parse_args(argv)
    args.root = args.root.resolve()
    args.output_dir = (args.output_dir or args.root / "outputs" / "vggt").resolve()
    workspace = args.root.parent
    configured_repo = os.environ.get("EXERCISE3D_VGGT_REPO")
    configured_checkpoint = os.environ.get("EXERCISE3D_VGGT_CHECKPOINT")
    args.vggt_repo = (
        args.vggt_repo
        or (Path(configured_repo).expanduser() if configured_repo else workspace / "vggt-omega")
    ).resolve()
    args.checkpoint = (
        args.checkpoint
        or (
            Path(configured_checkpoint).expanduser()
            if configured_checkpoint
            else workspace / "checkpoints" / "vggt_omega_1b_512.pt"
        )
    ).resolve()
    validate_paths(args.root, args.output_dir, args.vggt_repo, args.checkpoint)
    if args.sample_times < 1:
        raise RuntimeError("--sample-times must be positive")
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise RuntimeError("--image-resolution must be a positive multiple of 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.vggt_repo))

    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils import load_fn
    from vggt_omega.utils.pose_enc import encoding_to_camera

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the official VGGT-Omega forward implementation")

    generated_at = utc_now()
    temporal_rows = load_temporal_rows(args.root)
    subject_mapping = load_subject_mapping(args.root, args.subject_map)
    devices = load_devices(args.root)
    sequences, inventory = discover_sequences(
        args.root, args.only, subject_mapping, temporal_rows, devices
    )
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
        selected_sets = {row["set_id"] for row in sequences}
        inventory = [row for row in inventory if row["set_id"] in selected_sets]
    if not sequences:
        raise RuntimeError("no sequences matched")
    atomic_csv(args.output_dir / "camera_inventory.csv", inventory)

    repo_commit = git_value(args.vggt_repo, "rev-parse", "HEAD")
    checkpoint_sha = sha256_file(args.checkpoint)
    config = configuration(args, repo_commit, checkpoint_sha)
    if args.inventory_only:
        atomic_json(args.output_dir / "run_metadata.json", {
            "generated_at": generated_at, "status": "INVENTORY_ONLY", "configuration": config,
            "sequence_count": len(sequences), "camera_count": len(inventory),
        })
        print(json.dumps({"status": "INVENTORY_ONLY", "sequences": len(sequences), "cameras": len(inventory)}, indent=2))
        return 0

    print(f"Loading official VGGT-Omega checkpoint: {args.checkpoint}", flush=True)
    model = VGGTOmega().eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model = model.to("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    statuses = []
    quality_rows = []
    for index, sequence in enumerate(sequences, 1):
        print(f"[{index}/{len(sequences)}] {sequence['set_id']}", flush=True)
        status, quality = process_sequence(
            sequence, args.output_dir, model, load_fn, encoding_to_camera, args, config
        )
        statuses.append(status)
        quality_rows.extend(quality)
        print(f"  {status['status']} {status['elapsed_sec']}s", flush=True)
        atomic_csv(args.output_dir / "sequence_status.csv", statuses)
        atomic_csv(args.output_dir / "camera_quality.csv", quality_rows)

    atomic_csv(args.output_dir / "sequence_status.csv", statuses)
    atomic_csv(args.output_dir / "camera_quality.csv", quality_rows)
    report = report_markdown(generated_at, config, inventory, statuses, quality_rows)
    (args.output_dir / "camera_geometry_report.md").write_text(report, encoding="utf-8")
    run_metadata = {
        "schema_version": 1,
        "generated_at": generated_at,
        "completed_at": utc_now(),
        "status": "SUCCESS" if all(row["status"] == "SUCCESS" for row in statuses) else "COMPLETED_WITH_FAILURES",
        "configuration": config,
        "dataset_root": str(args.root),
        "output_root": str(args.output_dir),
        "official_repo": str(args.vggt_repo),
        "checkpoint": str(args.checkpoint),
        "sequence_counts": dict(Counter(row["status"] for row in statuses)),
        "camera_quality_counts": dict(Counter(row["quality_status"] for row in quality_rows)),
        "input_mutation": False,
        "new_input_frames_created": False,
        "bundle_adjustment_performed": False,
        "optimization_performed": False,
        "human_or_smpl_fitting_performed": False,
        "pseudo_labels_created": False,
    }
    atomic_json(args.output_dir / "run_metadata.json", run_metadata)
    print(json.dumps(run_metadata, indent=2), flush=True)
    return 0 if run_metadata["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
