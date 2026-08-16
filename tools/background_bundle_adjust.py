#!/usr/bin/env python3
"""Fixed-camera static-background bundle adjustment for Exercise3D VGGT outputs.

The tool reads immutable synchronized videos and Phase 2/3 QA products.  It
aggregates eight noisy VGGT poses into one pose per physical camera, extracts
temporally persistent SIFT background landmarks, constructs robust cross-view
tracks, and optimizes exactly three shared camera poses plus sparse 3D points.
It never writes to source videos, working frames, or VGGT initialization files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT)
).expanduser()
CAMERAS = ("cam1", "cam2", "cam3")
CAMERA_INDEX = {name: index for index, name in enumerate(CAMERAS)}
PIPELINE_VERSION = "phase4-fixed-camera-background-ba-v1"


def ensure_runtime() -> None:
    try:
        import cv2  # noqa: F401
        import scipy  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    if os.environ.get("EXERCISE3D_BACKGROUND_BA_REEXEC") == "1":
        raise RuntimeError("SciPy/OpenCV are unavailable in the selected runtime")
    configured = os.environ.get("EXERCISE3D_BACKGROUND_BA_PYTHON")
    candidates = [Path(configured)] if configured else []
    candidates.extend(
        [
            Path(sys.prefix) / "envs" / "c4g-fresh" / "bin" / "python",
            Path(sys.prefix).parent / "envs" / "c4g-fresh" / "bin" / "python",
        ]
    )
    runtime = next((candidate for candidate in candidates if candidate.is_file()), None)
    if runtime is None:
        raise RuntimeError(
            "SciPy/OpenCV runtime unavailable; set EXERCISE3D_BACKGROUND_BA_PYTHON to an existing runtime"
        )
    conda_root = runtime.parents[3]
    env = os.environ.copy()
    env["EXERCISE3D_BACKGROUND_BA_REEXEC"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    library_path = str(conda_root / "lib")
    if env.get("LD_LIBRARY_PATH"):
        library_path += os.pathsep + env["LD_LIBRARY_PATH"]
    env["LD_LIBRARY_PATH"] = library_path
    os.execve(str(runtime), [str(runtime), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def jsonable(value: Any) -> Any:
    import numpy as np

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
            writer.writerow(
                {
                    key: json.dumps(jsonable(value), ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else jsonable(value)
                    for key, value in row.items()
                }
            )
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def resolve_sequence(vggt_root: Path, sequence_id: str) -> Path:
    candidates = sorted(
        path.parent
        for path in vggt_root.glob(f"*/*/{sequence_id}/metadata.json")
        if path.parent.name == sequence_id
    )
    if not candidates:
        raise RuntimeError(f"VGGT sequence not found below {vggt_root}: {sequence_id}")
    if len(candidates) != 1:
        raise RuntimeError(f"ambiguous sequence id {sequence_id}: {candidates}")
    return candidates[0]


def project_so3(matrix: Any) -> Any:
    import numpy as np

    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def make_extrinsic(rotation: Any, center: Any) -> Any:
    import numpy as np

    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = -rotation @ center
    return output


def invert_extrinsic(extrinsic: Any) -> Any:
    import numpy as np

    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation.T
    output[:3, 3] = -rotation.T @ translation
    return output


def rotation_angle_deg(left: Any, right: Any) -> float:
    import numpy as np

    relative = project_so3(left) @ project_so3(right).T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def robust_scale(values: Any, floor: float) -> float:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    return max(mad, floor)


def weighted_median(values: Any, weights: Any) -> float:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, cumulative[-1] * 0.5)])


def aggregate_camera_pose(extrinsics: Any, intrinsics: Any, scene_scale: float) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    rotations = np.stack([project_so3(matrix[:3, :3]) for matrix in extrinsics])
    translations = extrinsics[:, :3, 3]
    centers = -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), translations)
    pairwise = np.asarray(
        [
            [rotation_angle_deg(rotations[i], rotations[j]) for j in range(len(rotations))]
            for i in range(len(rotations))
        ]
    )
    medoid = int(np.argmin(np.median(pairwise, axis=1)))
    mean_rotation = rotations[medoid]
    mean_center = np.median(centers, axis=0)
    weights = np.ones(len(rotations), dtype=np.float64)
    for _ in range(12):
        rot_residual = np.asarray(
            [rotation_angle_deg(rotation, mean_rotation) for rotation in rotations]
        )
        center_residual = np.linalg.norm(centers - mean_center, axis=1)
        rot_sigma = robust_scale(rot_residual, 0.35)
        center_sigma = robust_scale(center_residual, max(scene_scale * 0.001, 1e-5))
        normalized = np.sqrt((rot_residual / rot_sigma) ** 2 + (center_residual / center_sigma) ** 2)
        new_weights = 1.0 / (1.0 + (normalized / 2.5) ** 4)
        new_weights = np.clip(new_weights, 1e-3, 1.0)
        updated_rotation = Rotation.from_matrix(rotations).mean(weights=new_weights).as_matrix()
        updated_center = np.average(centers, axis=0, weights=new_weights)
        if rotation_angle_deg(updated_rotation, mean_rotation) < 1e-6 and np.linalg.norm(
            updated_center - mean_center
        ) < 1e-9:
            weights = new_weights
            mean_rotation, mean_center = updated_rotation, updated_center
            break
        weights = new_weights
        mean_rotation, mean_center = updated_rotation, updated_center
    mean_rotation = project_so3(mean_rotation)
    intrinsic = np.eye(3, dtype=np.float64)
    for row, column in ((0, 0), (1, 1), (0, 2), (1, 2)):
        intrinsic[row, column] = weighted_median(intrinsics[:, row, column], weights)
    intrinsic[2, 2] = 1.0
    rot_residual = np.asarray(
        [rotation_angle_deg(rotation, mean_rotation) for rotation in rotations]
    )
    center_residual = np.linalg.norm(centers - mean_center, axis=1)
    sample_rows = []
    for index, weight in enumerate(weights):
        status = "GOOD" if weight >= 0.50 else "DOWNWEIGHT" if weight >= 0.10 else "REJECT"
        sample_rows.append(
            {
                "sample_index": index,
                "weight": float(weight),
                "status": status,
                "rotation_residual_deg": float(rot_residual[index]),
                "center_residual": float(center_residual[index]),
            }
        )
    return {
        "rotation": mean_rotation,
        "center": mean_center,
        "extrinsic": make_extrinsic(mean_rotation, mean_center),
        "intrinsic": intrinsic,
        "sample_rows": sample_rows,
        "rotation_pairwise_p95_deg": float(np.percentile(pairwise[np.triu_indices(len(rotations), 1)], 95)),
        "center_dispersion_p95": float(np.percentile(center_residual, 95)),
    }


def load_camera(camera_dir: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(camera_dir / "poses.npz") as archive:
        pose = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(camera_dir / "confidence.npz") as archive:
        confidence = np.asarray(archive["depth_confidence"])
        confidence_timestamps = np.asarray(archive["timestamps_sec"])
    with np.load(camera_dir / "pointmap.npz") as archive:
        pointmap = np.asarray(archive["world_points_from_depth"])
        pointmap_timestamps = np.asarray(archive["timestamps_sec"])
    frames = read_csv(camera_dir / "frames.csv")
    count = len(frames)
    if pose["extrinsics_world_to_camera"].shape != (count, 3, 4):
        raise RuntimeError(f"invalid pose shape: {camera_dir}")
    if confidence.shape != pointmap.shape[:3] or pointmap.shape != (*confidence.shape, 3):
        raise RuntimeError(f"confidence/pointmap shape mismatch: {camera_dir}")
    timestamps = pose["timestamps_sec"]
    if not np.allclose(timestamps, confidence_timestamps) or not np.allclose(
        timestamps, pointmap_timestamps
    ):
        raise RuntimeError(f"timestamp arrays disagree: {camera_dir}")
    arrays = [
        pose["extrinsics_world_to_camera"], pose["intrinsics"], confidence, pointmap
    ]
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"non-finite VGGT input: {camera_dir}")
    for index, row in enumerate(frames):
        if int(row["source_frame_index"]) != int(pose["source_frame_indices"][index]):
            raise RuntimeError(f"source index mismatch: {camera_dir}, sample {index}")
        if abs(float(row["source_packet_pts_sec"]) - float(timestamps[index])) > 1e-6:
            raise RuntimeError(f"PTS mismatch: {camera_dir}, sample {index}")
    return {
        **pose,
        "confidence": confidence,
        "pointmap": pointmap,
        "frames": frames,
        "height": confidence.shape[1],
        "width": confidence.shape[2],
    }


def load_temporal_models(root: Path, sequence_id: str) -> dict[str, dict[str, Any]]:
    path = root / "reports" / "temporal_alignment" / "pair_summary.csv"
    models = {}
    for row in read_csv(path):
        if row["set_id"] != sequence_id:
            continue
        models[row["pair_id"]] = {
            "classification": row["classification"],
            "representative_offset_ms": float(row["representative_frame_pts_offset_ms"]),
            "drift_ms_per_sec": float(row["fused_drift_ms_per_sec"])
            if row.get("fused_drift_ms_per_sec")
            else 0.0,
            "duration_sec": float(row["duration_sec"]),
        }
    if len(models) != 3:
        raise RuntimeError(f"missing Phase 2 pair metadata for {sequence_id}")
    return models


def validate_eis(root: Path, sequence_id: str) -> None:
    rows = [
        row for row in read_csv(root / "reports" / "eis_audit" / "summary.csv")
        if row["set_id"] == sequence_id
    ]
    if len(rows) != 3 or any(row["recommendation"] != "FIXED_CAMERA_OK" for row in rows):
        raise RuntimeError(f"fixed-camera QA prerequisite not satisfied: {sequence_id}")


def decode_preprocessed_images(root: Path, data: dict[str, Any]) -> list[Any]:
    import cv2
    import numpy as np

    frames = data["frames"]
    source = root / frames[0]["source_video"]
    indices = {int(row["source_frame_index"]) for row in frames}
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {source}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    decoded = {}
    index = 0
    try:
        while index <= max(indices):
            ok, frame = capture.read()
            if not ok:
                break
            if index in indices:
                decoded[index] = frame
            index += 1
    finally:
        capture.release()
    if indices - decoded.keys():
        raise RuntimeError(f"failed to decode {sorted(indices - decoded.keys())}: {source}")
    output = []
    for row in frames:
        image = decoded[int(row["source_frame_index"])]
        image = image[
            int(row["crop_top"]):int(row["crop_bottom"]),
            int(row["crop_left"]):int(row["crop_right"]),
        ]
        image = cv2.resize(
            image,
            (int(row["resized_width"]), int(row["resized_height"])),
            interpolation=cv2.INTER_CUBIC,
        )
        image = cv2.copyMakeBorder(
            image,
            int(row["pad_top"]), int(row["pad_bottom"]),
            int(row["pad_left"]), int(row["pad_right"]),
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
        if image.shape[:2] != (data["height"], data["width"]):
            raise RuntimeError(f"preprocessed RGB shape mismatch: {source}")
        output.append(image)
    return output


def dynamic_foreground_mask(gray: Any, background: Any) -> Any:
    import cv2
    import numpy as np

    difference = cv2.GaussianBlur(cv2.absdiff(gray, background), (5, 5), 0)
    binary = (difference >= 18).astype(np.uint8) * 255
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    foreground = np.zeros_like(binary)
    minimum_area = max(64, round(gray.size * 0.001))
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            foreground[labels == label] = 255
    return cv2.dilate(
        foreground, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1
    )


def extract_static_features(
    camera: str,
    data: dict[str, Any],
    images: list[Any],
    args: argparse.Namespace,
    debug_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    import cv2
    import numpy as np

    gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    stack = np.stack(gray)
    background = np.median(stack, axis=0).astype(np.uint8)
    temporal_mad = np.median(np.abs(stack.astype(np.float32) - background), axis=0)
    mad_threshold = max(float(np.percentile(temporal_mad, args.static_mad_percentile)), 2.0)
    stable = temporal_mad <= mad_threshold
    height, width = background.shape
    edge = max(4, round(min(height, width) * args.border_fraction))
    stable[:edge] = False
    stable[-edge:] = False
    stable[:, :edge] = False
    stable[:, -edge:] = False
    detector = cv2.SIFT_create(
        nfeatures=args.max_features,
        contrastThreshold=args.sift_contrast,
        edgeThreshold=15,
    )
    feature_sets = []
    masks = []
    stats_rows = []
    for sample_index, image in enumerate(gray):
        foreground = dynamic_foreground_mask(image, background)
        confidence = data["confidence"][sample_index]
        confidence_threshold = float(np.percentile(confidence, args.confidence_percentile))
        mask = stable & (foreground == 0) & (confidence >= confidence_threshold)
        mask_u8 = mask.astype(np.uint8) * 255
        keypoints, descriptors = detector.detectAndCompute(image, mask_u8)
        if descriptors is None:
            keypoints, descriptors = [], np.empty((0, 128), dtype=np.float32)
        xy = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32).reshape(-1, 2)
        responses = np.asarray([keypoint.response for keypoint in keypoints], dtype=np.float32)
        point_confidence = np.empty(len(xy), dtype=np.float32)
        world_points = np.empty((len(xy), 3), dtype=np.float32)
        for feature_index, (x, y) in enumerate(xy):
            pixel_x = int(np.clip(round(float(x)), 0, width - 1))
            pixel_y = int(np.clip(round(float(y)), 0, height - 1))
            point_confidence[feature_index] = confidence[pixel_y, pixel_x]
            world_points[feature_index] = data["pointmap"][sample_index, pixel_y, pixel_x]
        feature_sets.append(
            {
                "xy": xy,
                "descriptors": descriptors.astype(np.float32, copy=False),
                "responses": responses,
                "confidence": point_confidence,
                "world_points": world_points,
            }
        )
        masks.append(mask_u8)
        stats_rows.append(
            {
                "camera_id": camera,
                "sample_index": sample_index,
                "timestamp_sec": float(data["timestamps_sec"][sample_index]),
                "static_mask_fraction": float(mask.mean()),
                "dynamic_foreground_fraction": float((foreground > 0).mean()),
                "confidence_percentile_removed": args.confidence_percentile,
                "confidence_threshold": confidence_threshold,
                "temporal_mad_threshold": mad_threshold,
                "sift_feature_count": len(xy),
            }
        )
        if debug_dir is not None:
            target = debug_dir / "masks" / f"{camera}_sample{sample_index}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), mask_u8):
                raise RuntimeError(f"failed to write mask: {target}")
    return feature_sets, masks, stats_rows


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def build_persistent_landmarks(
    camera: str, feature_sets: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    from scipy.spatial import cKDTree

    sample_ids = []
    feature_ids = []
    xy_chunks = []
    descriptor_chunks = []
    for sample_index, features in enumerate(feature_sets):
        count = len(features["xy"])
        sample_ids.extend([sample_index] * count)
        feature_ids.extend(range(count))
        xy_chunks.append(features["xy"])
        descriptor_chunks.append(features["descriptors"])
    if not xy_chunks or sum(len(chunk) for chunk in xy_chunks) == 0:
        return [], {"camera_id": camera, "raw_features": 0, "persistent_landmarks": 0}
    xy = np.concatenate(xy_chunks)
    descriptors = np.concatenate(descriptor_chunks)
    sample_ids = np.asarray(sample_ids, dtype=np.int16)
    feature_ids = np.asarray(feature_ids, dtype=np.int32)
    union = UnionFind(len(xy))
    for left, right in cKDTree(xy).query_pairs(args.temporal_cluster_radius):
        if sample_ids[left] == sample_ids[right]:
            continue
        if np.linalg.norm(descriptors[left] - descriptors[right]) <= args.temporal_descriptor_distance:
            union.union(left, right)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(xy)):
        groups[union.find(index)].append(index)
    landmarks = []
    conflict_groups = 0
    spread_rejected = 0
    for group in groups.values():
        median_xy = np.median(xy[group], axis=0)
        by_sample: dict[int, int] = {}
        for index in group:
            sample_index = int(sample_ids[index])
            previous = by_sample.get(sample_index)
            if previous is None or np.linalg.norm(xy[index] - median_xy) < np.linalg.norm(
                xy[previous] - median_xy
            ):
                by_sample[sample_index] = index
        if len(by_sample) != len(group):
            conflict_groups += 1
        chosen = list(by_sample.values())
        if len(chosen) < args.min_track_length:
            continue
        spread = np.percentile(np.linalg.norm(xy[chosen] - np.median(xy[chosen], axis=0), axis=1), 90)
        if spread > args.temporal_cluster_radius * 1.5:
            spread_rejected += 1
            continue
        observations = []
        for index in chosen:
            sample_index = int(sample_ids[index])
            feature_index = int(feature_ids[index])
            features = feature_sets[sample_index]
            observations.append(
                {
                    "camera_id": camera,
                    "camera_index": CAMERA_INDEX[camera],
                    "sample_index": sample_index,
                    "xy": features["xy"][feature_index].astype(np.float64),
                    "descriptor": features["descriptors"][feature_index],
                    "response": float(features["responses"][feature_index]),
                    "confidence": float(features["confidence"][feature_index]),
                    "world_point": features["world_points"][feature_index].astype(np.float64),
                }
            )
        landmarks.append(
            {
                "camera_id": camera,
                "xy": np.median(np.stack([observation["xy"] for observation in observations]), axis=0),
                "descriptor": np.median(
                    np.stack([observation["descriptor"] for observation in observations]), axis=0
                ).astype(np.float32),
                "observations": observations,
                "temporal_length": len(observations),
                "spread_p90_px": float(spread),
            }
        )
    return landmarks, {
        "camera_id": camera,
        "raw_features": len(xy),
        "raw_spatial_components": len(groups),
        "components_with_sample_conflicts": conflict_groups,
        "spread_rejected": spread_rejected,
        "persistent_landmarks": len(landmarks),
        "persistent_length_median": float(
            np.median([landmark["temporal_length"] for landmark in landmarks])
        ) if landmarks else 0.0,
        "persistent_length_p90": float(
            np.percentile([landmark["temporal_length"] for landmark in landmarks], 90)
        ) if landmarks else 0.0,
    }


def skew(vector: Any) -> Any:
    import numpy as np

    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def fundamental_from_cameras(camera_a: dict[str, Any], camera_b: dict[str, Any]) -> Any:
    import numpy as np

    extrinsic_a, extrinsic_b = camera_a["extrinsic"], camera_b["extrinsic"]
    rotation_a, translation_a = extrinsic_a[:3, :3], extrinsic_a[:3, 3]
    rotation_b, translation_b = extrinsic_b[:3, :3], extrinsic_b[:3, 3]
    relative_rotation = rotation_b @ rotation_a.T
    relative_translation = translation_b - relative_rotation @ translation_a
    essential = skew(relative_translation) @ relative_rotation
    fundamental = (
        np.linalg.inv(camera_b["intrinsic"]).T
        @ essential
        @ np.linalg.inv(camera_a["intrinsic"])
    )
    norm = np.linalg.norm(fundamental)
    return fundamental / norm if norm > 0 else fundamental


def sampson_error_px(fundamental: Any, points_a: Any, points_b: Any) -> Any:
    import numpy as np

    points_a = np.column_stack([points_a, np.ones(len(points_a))])
    points_b = np.column_stack([points_b, np.ones(len(points_b))])
    line_b = (fundamental @ points_a.T).T
    line_a = (fundamental.T @ points_b.T).T
    numerator = np.sum(points_b * line_b, axis=1) ** 2
    denominator = line_b[:, 0] ** 2 + line_b[:, 1] ** 2 + line_a[:, 0] ** 2 + line_a[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def corrected_temporal_pairings(
    camera_a: str,
    camera_b: str,
    data: dict[str, dict[str, Any]],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    import numpy as np

    rows = []
    timestamps_a = data[camera_a]["timestamps_sec"]
    timestamps_b = data[camera_b]["timestamps_sec"]
    midpoint = model["duration_sec"] * 0.5
    use_drift = model["classification"] == "CLOCK_DRIFT_DETECTED"
    for sample_index, timestamp in enumerate(timestamps_a):
        offset_ms = model["representative_offset_ms"]
        if use_drift:
            offset_ms += model["drift_ms_per_sec"] * (float(timestamp) - midpoint)
        corrected_target = float(timestamp) + offset_ms / 1000.0
        target_index = int(np.argmin(np.abs(timestamps_b - corrected_target)))
        rows.append(
            {
                "pair_id": f"{camera_a}-{camera_b}",
                "classification": model["classification"],
                "source_camera": camera_a,
                "source_sample_index": sample_index,
                "source_pts_sec": float(timestamp),
                "estimated_offset_ms_positive_b_later": float(offset_ms),
                "corrected_target_pts_sec": corrected_target,
                "target_camera": camera_b,
                "target_sample_index": target_index,
                "actual_target_pts_sec": float(timestamps_b[target_index]),
                "pairing_error_ms": float((timestamps_b[target_index] - corrected_target) * 1000.0),
                "interpolation_or_frame_generation": False,
            }
        )
    return rows


def temporal_descriptor_evidence(
    landmark_a: dict[str, Any],
    landmark_b: dict[str, Any],
    temporal_pairings: list[dict[str, Any]],
) -> tuple[int, float | None]:
    """Evaluate landmark descriptors at Phase-2-corrected sampled PTS pairs."""
    import numpy as np

    observations_a = {
        observation["sample_index"]: observation for observation in landmark_a["observations"]
    }
    observations_b = {
        observation["sample_index"]: observation for observation in landmark_b["observations"]
    }
    distances = []
    for pairing in temporal_pairings:
        observation_a = observations_a.get(pairing["source_sample_index"])
        observation_b = observations_b.get(pairing["target_sample_index"])
        if observation_a is None or observation_b is None:
            continue
        distances.append(
            float(np.linalg.norm(observation_a["descriptor"] - observation_b["descriptor"]))
        )
    return len(distances), float(np.median(distances)) if distances else None


def match_camera_landmarks(
    camera_a: str,
    camera_b: str,
    landmarks: dict[str, list[dict[str, Any]]],
    cameras_initial: dict[str, dict[str, Any]],
    gauge_transform: Any,
    scene_scale: float,
    temporal_pairings: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import cv2
    import numpy as np
    from scipy.spatial import cKDTree

    left, right = landmarks[camera_a], landmarks[camera_b]
    if len(left) < 8 or len(right) < 8:
        return [], {
            "pair_id": f"{camera_a}-{camera_b}", "status": "INSUFFICIENT_LANDMARKS",
            "left_landmarks": len(left), "right_landmarks": len(right),
        }
    descriptors_a = np.stack([landmark["descriptor"] for landmark in left]).astype(np.float32)
    descriptors_b = np.stack([landmark["descriptor"] for landmark in right]).astype(np.float32)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = []
    for pair in matcher.knnMatch(descriptors_a, descriptors_b, k=2):
        if len(pair) != 2:
            continue
        best, runner_up = pair
        if best.distance < args.ratio_test * runner_up.distance:
            candidates.append(best)
    unique_target: dict[int, Any] = {}
    for match in candidates:
        previous = unique_target.get(match.trainIdx)
        if previous is None or match.distance < previous.distance:
            unique_target[match.trainIdx] = match
    candidates = list(unique_target.values())
    if len(candidates) < 8:
        return [], {
            "pair_id": f"{camera_a}-{camera_b}", "status": "INSUFFICIENT_RATIO_MATCHES",
            "left_landmarks": len(left), "right_landmarks": len(right),
            "ratio_matches": len(candidates),
        }
    points_a = np.asarray([left[match.queryIdx]["xy"] for match in candidates], np.float32)
    points_b = np.asarray([right[match.trainIdx]["xy"] for match in candidates], np.float32)
    fundamental, inlier_mask = cv2.findFundamentalMat(
        points_a, points_b, cv2.USAC_MAGSAC,
        args.magsac_threshold, 0.999, args.magsac_iterations,
    )
    if fundamental is None or fundamental.shape != (3, 3) or inlier_mask is None:
        return [], {
            "pair_id": f"{camera_a}-{camera_b}", "status": "MAGSAC_FAILED",
            "ratio_matches": len(candidates),
        }
    inlier = inlier_mask.reshape(-1).astype(bool)
    initial_fundamental = fundamental_from_cameras(
        cameras_initial[camera_a], cameras_initial[camera_b]
    )
    initial_error = sampson_error_px(initial_fundamental, points_a, points_b)
    inlier_error = initial_error[inlier]
    center = float(np.median(inlier_error)) if len(inlier_error) else float("inf")
    spread = robust_scale(inlier_error, 1.0) if len(inlier_error) else float("inf")
    threshold = min(args.initial_epipolar_max, max(6.0, center + 3.0 * spread))
    keep = inlier & (initial_error <= threshold)
    fallback = False
    edges = []
    temporal_supported = 0
    temporal_rejected = 0
    for match_index in np.flatnonzero(keep):
        match = candidates[int(match_index)]
        aligned_support, aligned_distance = temporal_descriptor_evidence(
            left[match.queryIdx], right[match.trainIdx], temporal_pairings
        )
        if aligned_support:
            temporal_supported += 1
            if aligned_distance > args.temporal_crossview_descriptor_distance:
                temporal_rejected += 1
                continue
        edges.append(
            {
                "camera_a": camera_a,
                "landmark_a": int(match.queryIdx),
                "camera_b": camera_b,
                "landmark_b": int(match.trainIdx),
                "descriptor_distance": float(match.distance),
                "magsac_sampson_error_px": float(
                    sampson_error_px(
                        fundamental,
                        points_a[match_index:match_index + 1],
                        points_b[match_index:match_index + 1],
                    )[0]
                ),
                "initial_sampson_error_px": float(initial_error[match_index]),
                "temporal_aligned_support": aligned_support,
                "temporal_aligned_descriptor_distance": aligned_distance,
                "evidence": "SIFT_RATIO+USAC_MAGSAC+INIT_EPIPOLAR",
            }
        )

    # VGGT point maps are used only as a loose data-association guide.  The
    # observations remain real SIFT pixels and BA optimizes their reprojection.
    transformed_points = {}
    for camera in (camera_a, camera_b):
        transformed_points[camera] = np.stack(
            [
                np.median(
                    np.stack(
                        [
                            gauge_transform[:3, :3] @ observation["world_point"]
                            + gauge_transform[:3, 3]
                            for observation in landmark["observations"]
                        ]
                    ),
                    axis=0,
                )
                for landmark in landmarks[camera]
            ]
        )
    target_tree = cKDTree(transformed_points[camera_b])
    guided_candidates = []
    radius = args.guided_pointmap_radius * scene_scale
    for left_index, landmark in enumerate(left):
        target_indices = target_tree.query_ball_point(
            transformed_points[camera_a][left_index], radius
        )
        if not target_indices:
            continue
        distances = np.asarray(
            [
                np.linalg.norm(right[index]["descriptor"] - landmark["descriptor"])
                for index in target_indices
            ]
        )
        order = np.argsort(distances)
        if distances[order[0]] > args.guided_descriptor_distance:
            continue
        if len(order) > 1 and distances[order[0]] > args.guided_ratio_test * distances[order[1]]:
            continue
        right_index = int(target_indices[int(order[0])])
        aligned_support, aligned_distance = temporal_descriptor_evidence(
            landmark, right[right_index], temporal_pairings
        )
        if aligned_support:
            temporal_supported += 1
            if aligned_distance > args.temporal_crossview_descriptor_distance:
                temporal_rejected += 1
                continue
        guided_candidates.append(
            {
                "left": left_index,
                "right": right_index,
                "descriptor_distance": float(distances[order[0]]),
                "pointmap_distance": float(
                    np.linalg.norm(
                        transformed_points[camera_a][left_index]
                        - transformed_points[camera_b][right_index]
                    )
                ),
                "temporal_aligned_support": aligned_support,
                "temporal_aligned_descriptor_distance": aligned_distance,
            }
        )
    unique_guided = {}
    for candidate in guided_candidates:
        previous = unique_guided.get(candidate["right"])
        if previous is None or candidate["descriptor_distance"] < previous["descriptor_distance"]:
            unique_guided[candidate["right"]] = candidate
    guided_candidates = list(unique_guided.values())
    guided_magsac = np.zeros(len(guided_candidates), dtype=bool)
    if len(guided_candidates) >= 8:
        guided_a = np.asarray(
            [left[candidate["left"]]["xy"] for candidate in guided_candidates], np.float32
        )
        guided_b = np.asarray(
            [right[candidate["right"]]["xy"] for candidate in guided_candidates], np.float32
        )
        _, guided_mask = cv2.findFundamentalMat(
            guided_a, guided_b, cv2.USAC_MAGSAC,
            args.magsac_threshold, 0.999, args.magsac_iterations,
        )
        if guided_mask is not None:
            guided_magsac = guided_mask.reshape(-1).astype(bool)
        guided_initial_error = sampson_error_px(initial_fundamental, guided_a, guided_b)
        for index, candidate in enumerate(guided_candidates):
            if guided_initial_error[index] > args.guided_epipolar_max:
                continue
            edges.append(
                {
                    "camera_a": camera_a,
                    "landmark_a": candidate["left"],
                    "camera_b": camera_b,
                    "landmark_b": candidate["right"],
                    "descriptor_distance": candidate["descriptor_distance"],
                    "magsac_sampson_error_px": None,
                    "initial_sampson_error_px": float(guided_initial_error[index]),
                    "pointmap_distance": candidate["pointmap_distance"],
                    "guided_magsac_inlier": bool(guided_magsac[index]),
                    "temporal_aligned_support": candidate["temporal_aligned_support"],
                    "temporal_aligned_descriptor_distance": candidate[
                        "temporal_aligned_descriptor_distance"
                    ],
                    "evidence": "PERSISTENT_SIFT+POINTMAP_RADIUS+INIT_EPIPOLAR",
                }
            )
    deduplicated = {}
    for edge in edges:
        key = (edge["landmark_a"], edge["landmark_b"])
        previous = deduplicated.get(key)
        if previous is None or edge["initial_sampson_error_px"] < previous["initial_sampson_error_px"]:
            deduplicated[key] = edge
    edges = list(deduplicated.values())
    return edges, {
        "pair_id": f"{camera_a}-{camera_b}",
        "status": "OK" if len(edges) >= 8 else "INSUFFICIENT_INLIERS",
        "left_landmarks": len(left),
        "right_landmarks": len(right),
        "ratio_matches": len(candidates),
        "magsac_inliers": int(inlier.sum()),
        "initial_epipolar_threshold_px": threshold,
        "initial_epipolar_fallback": fallback,
        "descriptor_magsac_init_accepted": int(keep.sum()),
        "guided_candidates": len(guided_candidates),
        "guided_magsac_inliers": int(guided_magsac.sum()),
        "guided_epipolar_accepted": sum(
            edge["evidence"].startswith("PERSISTENT") for edge in edges
        ),
        "temporal_corrected_candidates_with_aligned_support": temporal_supported,
        "temporal_corrected_descriptor_rejected": temporal_rejected,
        "accepted_edges": len(edges),
    }


def build_cross_camera_tracks(
    landmarks: dict[str, list[dict[str, Any]]], edges: list[dict[str, Any]], max_tracks: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    offsets = {}
    total = 0
    for camera in CAMERAS:
        offsets[camera] = total
        total += len(landmarks[camera])
    union = UnionFind(total)
    edge_lookup: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        left = offsets[edge["camera_a"]] + edge["landmark_a"]
        right = offsets[edge["camera_b"]] + edge["landmark_b"]
        union.union(left, right)
        edge_lookup[tuple(sorted((left, right)))].append(edge)
    components: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for camera in CAMERAS:
        for landmark_index in range(len(landmarks[camera])):
            global_index = offsets[camera] + landmark_index
            components[union.find(global_index)].append((camera, landmark_index, global_index))
    tracks = []
    conflicts = 0
    singleton = 0
    for component in components.values():
        cameras = [item[0] for item in component]
        if len(set(cameras)) < 2:
            singleton += 1
            continue
        if len(set(cameras)) != len(cameras):
            conflicts += 1
            continue
        observations = []
        temporal_lengths = []
        for camera, landmark_index, _ in component:
            landmark = landmarks[camera][landmark_index]
            observations.extend(landmark["observations"])
            temporal_lengths.append(landmark["temporal_length"])
        component_edges = []
        global_ids = [item[2] for item in component]
        for left_index in range(len(global_ids)):
            for right_index in range(left_index + 1, len(global_ids)):
                component_edges.extend(
                    edge_lookup.get(tuple(sorted((global_ids[left_index], global_ids[right_index]))), [])
                )
        tracks.append(
            {
                "members": [(item[0], item[1]) for item in component],
                "observations": observations,
                "camera_count": len(set(cameras)),
                "edge_count": len(component_edges),
                "descriptor_distance_median": float(
                    __import__("numpy").median(
                        [edge["descriptor_distance"] for edge in component_edges]
                    )
                ) if component_edges else float("inf"),
                "score": sum(temporal_lengths) + 3 * len(set(cameras)),
            }
        )
    tracks.sort(key=lambda track: (-track["score"], track["descriptor_distance_median"]))
    truncated = max(0, len(tracks) - max_tracks)
    return tracks[:max_tracks], {
        "cross_edges": len(edges),
        "raw_multicamera_components": len(tracks) + conflicts,
        "conflicting_components_rejected": conflicts,
        "single_camera_components_ignored": singleton,
        "tracks_before_max_cap": len(tracks),
        "tracks_truncated_by_max_cap": truncated,
        "tracks_after_graph_filter": min(len(tracks), max_tracks),
    }


def triangulate_track(track: dict[str, Any], cameras: dict[str, dict[str, Any]]) -> Any | None:
    import numpy as np

    rows = []
    for camera in sorted({observation["camera_id"] for observation in track["observations"]}):
        observations = [
            observation for observation in track["observations"] if observation["camera_id"] == camera
        ]
        xy = np.median(np.stack([observation["xy"] for observation in observations]), axis=0)
        projection = cameras[camera]["intrinsic"] @ cameras[camera]["extrinsic"][:3, :]
        rows.append(xy[0] * projection[2] - projection[0])
        rows.append(xy[1] * projection[2] - projection[1])
    if len(rows) < 4:
        return None
    _, _, vt = np.linalg.svd(np.stack(rows))
    homogeneous = vt[-1]
    if abs(homogeneous[3]) < 1e-10:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.isfinite(point).all() else None


def project_points(camera: dict[str, Any], points: Any) -> tuple[Any, Any]:
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    extrinsic, intrinsic = camera["extrinsic"], camera["intrinsic"]
    camera_points = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    projected_h = camera_points @ intrinsic.T
    projected = projected_h[:, :2] / np.maximum(projected_h[:, 2:3], 1e-12)
    return projected, camera_points[:, 2]


def initial_track_metrics(
    track: dict[str, Any], point: Any, cameras: dict[str, dict[str, Any]], scene_scale: float,
    gauge_transform: Any,
) -> dict[str, Any]:
    import numpy as np

    residuals = []
    depths = []
    rays = []
    transformed_vggt_points = []
    for observation in track["observations"]:
        camera = cameras[observation["camera_id"]]
        projected, depth = project_points(camera, point[None])
        residuals.append(float(np.linalg.norm(projected[0] - observation["xy"])))
        depths.append(float(depth[0]))
        center = invert_extrinsic(camera["extrinsic"])[:3, 3]
        ray = point - center
        norm = np.linalg.norm(ray)
        if norm > 0:
            rays.append(ray / norm)
        world_point = observation["world_point"]
        transformed_vggt_points.append(
            gauge_transform[:3, :3] @ world_point + gauge_transform[:3, 3]
        )
    angles = []
    for left in range(len(rays)):
        for right in range(left + 1, len(rays)):
            cosine = np.clip(float(np.dot(rays[left], rays[right])), -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cosine))))
    pointmap_distance = np.linalg.norm(np.stack(transformed_vggt_points) - point, axis=1)
    return {
        "reprojection_median_px": float(np.median(residuals)),
        "reprojection_p90_px": float(np.percentile(residuals, 90)),
        "positive_depth_fraction": float(np.mean(np.asarray(depths) > 0)),
        "maximum_parallax_deg": max(angles) if angles else 0.0,
        "pointmap_distance_median_normalized": float(np.median(pointmap_distance) / scene_scale),
    }


def filter_and_triangulate_tracks(
    tracks: list[dict[str, Any]], cameras: dict[str, dict[str, Any]], scene_scale: float,
    gauge_transform: Any, args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    candidates = []
    failure_counts: dict[str, int] = defaultdict(int)
    for track in tracks:
        point = triangulate_track(track, cameras)
        if point is None:
            failure_counts["triangulation_failed"] += 1
            continue
        metrics = initial_track_metrics(track, point, cameras, scene_scale, gauge_transform)
        if metrics["positive_depth_fraction"] < 1.0:
            failure_counts["cheirality"] += 1
            continue
        if metrics["maximum_parallax_deg"] < args.min_parallax_deg:
            failure_counts["low_parallax"] += 1
            continue
        candidates.append({**track, "point_initial": point, "initial_metrics": metrics})
    if not candidates:
        return [], {"input_tracks": len(tracks), "candidates": 0, **failure_counts}
    reprojection = np.asarray(
        [track["initial_metrics"]["reprojection_median_px"] for track in candidates]
    )
    reprojection_p90 = np.asarray(
        [track["initial_metrics"]["reprojection_p90_px"] for track in candidates]
    )
    pointmap = np.asarray(
        [track["initial_metrics"]["pointmap_distance_median_normalized"] for track in candidates]
    )
    reprojection_limit = min(
        args.initial_reprojection_max,
        max(6.0, float(np.median(reprojection)) + 3.0 * robust_scale(reprojection, 1.0)),
    )
    p90_limit = min(
        args.initial_reprojection_max * 1.5,
        max(12.0, float(np.median(reprojection_p90)) + 3.0 * robust_scale(reprojection_p90, 2.0)),
    )
    pointmap_limit = min(
        args.pointmap_consistency_max,
        max(0.05, float(np.median(pointmap)) + 3.0 * robust_scale(pointmap, 0.01)),
    )
    kept = []
    duplicate_cells = set()
    for track in sorted(
        candidates,
        key=lambda item: (
            item["initial_metrics"]["reprojection_median_px"],
            -item["camera_count"],
            -len(item["observations"]),
        ),
    ):
        metrics = track["initial_metrics"]
        if metrics["reprojection_median_px"] > reprojection_limit:
            failure_counts["initial_reprojection_median"] += 1
            continue
        if metrics["reprojection_p90_px"] > p90_limit:
            failure_counts["initial_reprojection_p90"] += 1
            continue
        if metrics["pointmap_distance_median_normalized"] > pointmap_limit:
            failure_counts["pointmap_consistency"] += 1
            continue
        primary = min(track["observations"], key=lambda item: item["camera_index"])
        cell = (
            primary["camera_id"],
            round(float(primary["xy"][0]) / args.duplicate_radius_px),
            round(float(primary["xy"][1]) / args.duplicate_radius_px),
        )
        if cell in duplicate_cells:
            failure_counts["near_duplicate"] += 1
            continue
        duplicate_cells.add(cell)
        track["track_id"] = len(kept)
        kept.append(track)
    return kept, {
        "input_tracks": len(tracks),
        "triangulation_candidates": len(candidates),
        "initial_reprojection_median_limit_px": reprojection_limit,
        "initial_reprojection_p90_limit_px": p90_limit,
        "pointmap_consistency_limit_scene_fraction": pointmap_limit,
        "accepted_tracks": len(kept),
        "rejected_by_reason": dict(failure_counts),
    }


def tracks_to_arrays(tracks: list[dict[str, Any]], data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import numpy as np

    points = np.stack([track["point_initial"] for track in tracks]).astype(np.float64)
    observation_track = []
    observation_camera = []
    observation_sample = []
    observation_xy = []
    observation_timestamp = []
    observation_confidence = []
    observation_response = []
    for track_index, track in enumerate(tracks):
        for observation in track["observations"]:
            observation_track.append(track_index)
            observation_camera.append(observation["camera_index"])
            observation_sample.append(observation["sample_index"])
            observation_xy.append(observation["xy"])
            observation_timestamp.append(
                data[observation["camera_id"]]["timestamps_sec"][observation["sample_index"]]
            )
            observation_confidence.append(observation["confidence"])
            observation_response.append(observation["response"])
    return {
        "points_initial": points,
        "obs_track": np.asarray(observation_track, dtype=np.int32),
        "obs_camera": np.asarray(observation_camera, dtype=np.int8),
        "obs_sample": np.asarray(observation_sample, dtype=np.int8),
        "obs_xy": np.asarray(observation_xy, dtype=np.float64),
        "obs_timestamp": np.asarray(observation_timestamp, dtype=np.float64),
        "obs_confidence": np.asarray(observation_confidence, dtype=np.float32),
        "obs_response": np.asarray(observation_response, dtype=np.float32),
        "track_length": np.asarray([len(track["observations"]) for track in tracks], dtype=np.int16),
        "track_camera_count": np.asarray([track["camera_count"] for track in tracks], dtype=np.int8),
    }


def decode_ba_state(
    parameters: Any,
    base_cameras: list[dict[str, Any]],
    base_intrinsics: list[Any],
    scene_scale: float,
    point_count: int,
    intrinsics_mode: str,
    image_shapes: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    camera_parameters = parameters[:12].reshape(2, 6)
    point_start = 12
    points = parameters[point_start:point_start + point_count * 3].reshape(point_count, 3)
    intrinsics_start = point_start + point_count * 3
    cameras = []
    for camera_index in range(3):
        base = base_cameras[camera_index]
        if camera_index == 0:
            rotation = base["rotation"]
            center = base["center"]
        else:
            delta = camera_parameters[camera_index - 1]
            rotation = Rotation.from_rotvec(delta[:3]).as_matrix() @ base["rotation"]
            center = base["center"] + delta[3:] * scene_scale
        intrinsic = base_intrinsics[camera_index].copy()
        if intrinsics_mode == "limited":
            delta_k = parameters[intrinsics_start + camera_index * 4:intrinsics_start + (camera_index + 1) * 4]
            height, width = image_shapes[camera_index]
            intrinsic[0, 0] *= math.exp(float(delta_k[0]))
            intrinsic[1, 1] *= math.exp(float(delta_k[1]))
            intrinsic[0, 2] += float(delta_k[2]) * width
            intrinsic[1, 2] += float(delta_k[3]) * height
        cameras.append(
            {
                "rotation": rotation,
                "center": center,
                "extrinsic": make_extrinsic(rotation, center),
                "intrinsic": intrinsic,
            }
        )
    return cameras, points


def ba_residual_vector(
    parameters: Any,
    base_cameras: list[dict[str, Any]],
    base_intrinsics: list[Any],
    scene_scale: float,
    points_count: int,
    intrinsics_mode: str,
    image_shapes: list[tuple[int, int]],
    base_points: Any,
    observations: dict[str, Any],
    observation_weights: Any,
    pose_prior_weight: float,
    intrinsic_prior_weight: float,
    point_prior_weight: float,
) -> Any:
    import numpy as np

    cameras, points = decode_ba_state(
        parameters, base_cameras, base_intrinsics, scene_scale, points_count,
        intrinsics_mode, image_shapes,
    )
    residual = np.empty((len(observations["track"]), 2), dtype=np.float64)
    for camera_index in range(3):
        mask = observations["camera"] == camera_index
        if not mask.any():
            continue
        track_ids = observations["track"][mask]
        projected, _ = project_points(cameras[camera_index], points[track_ids])
        residual[mask] = projected - observations["xy"][mask]
    residual *= np.sqrt(observation_weights)[:, None]
    parts = [residual.reshape(-1)]
    camera_delta = parameters[:12].reshape(2, 6)
    if pose_prior_weight > 0:
        rotation_unit = math.radians(1.0)
        parts.append(camera_delta[:, :3].reshape(-1) / rotation_unit * pose_prior_weight)
        parts.append(camera_delta[:, 3:].reshape(-1) / 0.01 * pose_prior_weight)
    if intrinsics_mode == "limited" and intrinsic_prior_weight > 0:
        intrinsics_start = 12 + points_count * 3
        intrinsic_delta = parameters[intrinsics_start:intrinsics_start + 12].reshape(3, 4)
        parts.append(intrinsic_delta[:, :2].reshape(-1) / math.log(1.01) * intrinsic_prior_weight)
        parts.append(intrinsic_delta[:, 2:].reshape(-1) / 0.005 * intrinsic_prior_weight)
    if point_prior_weight > 0:
        # A very weak DLT-depth anchor removes the along-ray numerical null
        # direction without treating VGGT-derived points as fixed geometry.
        parts.append(
            ((points - base_points) / max(scene_scale * 0.05, 1e-9)).reshape(-1)
            * point_prior_weight
        )
    return np.concatenate(parts)


def ba_jacobian_sparsity(
    point_count: int,
    observations: dict[str, Any],
    intrinsics_mode: str,
    include_pose_prior: bool,
    include_intrinsic_prior: bool,
    include_point_prior: bool,
) -> Any:
    import numpy as np
    from scipy.sparse import lil_matrix

    observation_count = len(observations["track"])
    residual_count = observation_count * 2
    if include_pose_prior:
        residual_count += 12
    if intrinsics_mode == "limited" and include_intrinsic_prior:
        residual_count += 12
    if include_point_prior:
        residual_count += point_count * 3
    variable_count = 12 + point_count * 3 + (12 if intrinsics_mode == "limited" else 0)
    sparsity = lil_matrix((residual_count, variable_count), dtype=np.int8)
    for observation_index, (camera_index, point_index) in enumerate(
        zip(observations["camera"], observations["track"])
    ):
        rows = slice(observation_index * 2, observation_index * 2 + 2)
        if camera_index > 0:
            start = (int(camera_index) - 1) * 6
            sparsity[rows, start:start + 6] = 1
        point_start = 12 + int(point_index) * 3
        sparsity[rows, point_start:point_start + 3] = 1
        if intrinsics_mode == "limited":
            intrinsic_start = 12 + point_count * 3 + int(camera_index) * 4
            sparsity[rows, intrinsic_start:intrinsic_start + 4] = 1
    row = observation_count * 2
    if include_pose_prior:
        sparsity[row:row + 12, :12] = np.eye(12)
        row += 12
    if intrinsics_mode == "limited" and include_intrinsic_prior:
        start = 12 + point_count * 3
        sparsity[row:row + 12, start:start + 12] = np.eye(12)
        row += 12
    if include_point_prior:
        start = 12
        sparsity[row:row + point_count * 3, start:start + point_count * 3] = np.eye(
            point_count * 3
        )
    return sparsity.tocsr()


def normalize_scale_gauge(
    cameras: list[dict[str, Any]], points: Any, initial_baseline: float
) -> tuple[list[dict[str, Any]], Any, float]:
    import numpy as np

    baseline = float(np.linalg.norm(cameras[1]["center"] - cameras[0]["center"]))
    scale = initial_baseline / baseline if baseline > 1e-12 else 1.0
    origin = cameras[0]["center"].copy()
    output = []
    for camera in cameras:
        center = origin + (camera["center"] - origin) * scale
        output.append(
            {
                **camera,
                "center": center,
                "extrinsic": make_extrinsic(camera["rotation"], center),
            }
        )
    normalized_points = origin + (points - origin) * scale
    return output, normalized_points, scale


def run_bundle_adjustment(
    base_cameras: list[dict[str, Any]],
    base_intrinsics: list[Any],
    points: Any,
    observations: dict[str, Any],
    weights: Any,
    scene_scale: float,
    image_shapes: list[tuple[int, int]],
    args: argparse.Namespace,
    max_nfev: int | None = None,
    verbose: int = 0,
) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import least_squares

    point_count = len(points)
    parameter_count = 12 + point_count * 3 + (12 if args.intrinsics == "limited" else 0)
    initial = np.zeros(parameter_count, dtype=np.float64)
    initial[12:12 + point_count * 3] = points.reshape(-1)
    lower = np.full(parameter_count, -np.inf)
    upper = np.full(parameter_count, np.inf)
    rotation_limit = math.radians(args.max_camera_rotation_change)
    lower[:12] = np.tile([-rotation_limit] * 3 + [-args.max_camera_center_change] * 3, 2)
    upper[:12] = np.tile([rotation_limit] * 3 + [args.max_camera_center_change] * 3, 2)
    if args.intrinsics == "limited":
        start = 12 + point_count * 3
        focal_limit = math.log(1.0 + args.focal_refinement_fraction)
        lower[start:] = np.tile(
            [-focal_limit, -focal_limit, -args.principal_refinement_fraction, -args.principal_refinement_fraction],
            3,
        )
        upper[start:] = np.tile(
            [focal_limit, focal_limit, args.principal_refinement_fraction, args.principal_refinement_fraction],
            3,
        )
    residual_args = (
        base_cameras, base_intrinsics, scene_scale, point_count, args.intrinsics,
        image_shapes, points, observations, weights, args.pose_prior_weight,
        args.intrinsic_prior_weight, args.point_prior_weight,
    )
    sparsity = ba_jacobian_sparsity(
        point_count, observations, args.intrinsics,
        args.pose_prior_weight > 0, args.intrinsic_prior_weight > 0,
        args.point_prior_weight > 0,
    )
    started = time.monotonic()
    result = least_squares(
        ba_residual_vector,
        initial,
        args=residual_args,
        jac_sparsity=sparsity,
        method="trf",
        loss=args.robust_loss,
        f_scale=args.robust_scale,
        x_scale="jac",
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
        max_nfev=args.max_nfev if max_nfev is None else max_nfev,
        verbose=verbose,
    )
    cameras, optimized_points = decode_ba_state(
        result.x, base_cameras, base_intrinsics, scene_scale, point_count,
        args.intrinsics, image_shapes,
    )
    initial_baseline = float(np.linalg.norm(base_cameras[1]["center"] - base_cameras[0]["center"]))
    cameras, optimized_points, gauge_scale = normalize_scale_gauge(
        cameras, optimized_points, initial_baseline
    )
    return {
        "cameras": cameras,
        "points": optimized_points,
        "result": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "cost": float(result.cost),
            "optimality": float(result.optimality),
            "nfev": int(result.nfev),
            "njev": int(result.njev) if result.njev is not None else None,
            "elapsed_sec": time.monotonic() - started,
            "active_mask_count": int(np.count_nonzero(result.active_mask)),
            "scale_gauge_normalization": gauge_scale,
        },
    }


def observation_errors(cameras: list[dict[str, Any]], points: Any, observations: dict[str, Any]) -> Any:
    import numpy as np

    errors = np.empty(len(observations["track"]), dtype=np.float64)
    for camera_index in range(3):
        mask = observations["camera"] == camera_index
        if not mask.any():
            continue
        projected, _ = project_points(cameras[camera_index], points[observations["track"][mask]])
        errors[mask] = np.linalg.norm(projected - observations["xy"][mask], axis=1)
    return errors


def observation_residual_vectors(
    cameras: list[dict[str, Any]], points: Any, observations: dict[str, Any]
) -> Any:
    import numpy as np

    residuals = np.empty((len(observations["track"]), 2), dtype=np.float64)
    for camera_index in range(3):
        mask = observations["camera"] == camera_index
        if not mask.any():
            continue
        projected, _ = project_points(
            cameras[camera_index], points[observations["track"][mask]]
        )
        residuals[mask] = projected - observations["xy"][mask]
    return residuals


def radial_residual_diagnostic(
    cameras: list[dict[str, Any]],
    residual_vectors: Any,
    errors: Any,
    observations: dict[str, Any],
    accepted_mask: Any,
    image_shapes: list[tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    import numpy as np

    output = {}
    for camera_index, camera_id in enumerate(CAMERAS):
        mask = accepted_mask & (observations["camera"] == camera_index)
        xy = observations["xy"][mask]
        vectors = residual_vectors[mask]
        values = errors[mask]
        intrinsic = cameras[camera_index]["intrinsic"]
        center = intrinsic[:2, 2]
        displacement = xy - center
        radius_px = np.linalg.norm(displacement, axis=1)
        unit = displacement / np.maximum(radius_px[:, None], 1e-9)
        radial = np.sum(vectors * unit, axis=1)
        height, width = image_shapes[camera_index]
        radius_norm = radius_px / max(math.hypot(width, height), 1.0)
        predictor = radius_norm ** 3
        correlation = 0.0
        if len(radial) >= 3 and np.std(radial) > 1e-9 and np.std(predictor) > 1e-9:
            correlation = float(np.corrcoef(radial, predictor)[0, 1])
        split = float(np.median(radius_norm)) if len(radius_norm) else 0.0
        inner = np.abs(radial[radius_norm <= split])
        outer = np.abs(radial[radius_norm > split])
        inner_median = float(np.median(inner)) if len(inner) else 0.0
        outer_median = float(np.median(outer)) if len(outer) else 0.0
        review = (
            len(radial) >= 80
            and abs(correlation) >= 0.55
            and outer_median >= max(1.0, inner_median * 2.0)
        )
        output[camera_id] = {
            "support": len(radial),
            "pearson_radial_residual_vs_normalized_radius_cubed": correlation,
            "median_signed_radial_residual_px": float(np.median(radial)) if len(radial) else None,
            "inner_median_absolute_radial_residual_px": inner_median,
            "outer_median_absolute_radial_residual_px": outer_median,
            "outer_p90_reprojection_px": float(
                np.percentile(values[radius_norm > split], 90)
            ) if np.any(radius_norm > split) else None,
            "classification": "RADIAL_PATTERN_REVIEW" if review else "NO_STRONG_RADIAL_PATTERN",
        }
    return output


def robust_upper(values: Any, floor: float, multiplier: float = 3.0) -> float:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return max(floor, float(np.median(values)) + multiplier * robust_scale(values, floor * 0.25))


def classify_samples(errors: Any, observations: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    import numpy as np

    overall_median = float(np.median(errors))
    q25, q75 = np.percentile(errors, [25, 75])
    inlier_threshold = min(
        8.0, max(0.75, overall_median * 2.5, float(q75 + 1.5 * (q75 - q25)))
    )
    groups = []
    for camera_index in range(3):
        for sample_index in range(8):
            mask = (observations["camera"] == camera_index) & (
                observations["sample"] == sample_index
            )
            values = errors[mask]
            groups.append(
                {
                    "camera_id": CAMERAS[camera_index],
                    "camera_index": camera_index,
                    "sample_index": sample_index,
                    "support": len(values),
                    "median_px": float(np.median(values)) if len(values) else None,
                    "p90_px": float(np.percentile(values, 90)) if len(values) else None,
                    "inlier_ratio": float(np.mean(values <= inlier_threshold)) if len(values) else 0.0,
                }
            )
    supported = [group for group in groups if group["support"] > 0]
    support_values = np.asarray([group["support"] for group in supported], dtype=np.float64)
    median_values = np.asarray([group["median_px"] for group in supported], dtype=np.float64)
    p90_values = np.asarray([group["p90_px"] for group in supported], dtype=np.float64)
    inlier_values = np.asarray([group["inlier_ratio"] for group in supported], dtype=np.float64)
    support_low = max(4, float(np.median(support_values)) * 0.30) if len(support_values) else 4
    median_review = robust_upper(median_values, max(1.0, overall_median * 1.5)) if len(median_values) else 1
    median_reject = robust_upper(median_values, max(2.0, overall_median * 2.5), 4.5) if len(median_values) else 2
    p90_review = robust_upper(p90_values, max(2.0, inlier_threshold)) if len(p90_values) else 2
    p90_reject = robust_upper(p90_values, max(4.0, inlier_threshold * 1.5), 4.5) if len(p90_values) else 4
    inlier_low = max(0.20, float(np.median(inlier_values)) - 3.0 * robust_scale(inlier_values, 0.05)) if len(inlier_values) else 0.2
    for group in groups:
        if group["support"] == 0:
            group["gate"] = "REJECT"
            group["reasons"] = ["NO_TRACK_SUPPORT"]
            group["weight"] = 0.0
            continue
        review_reasons = []
        severe = []
        if group["support"] < support_low:
            review_reasons.append("LOW_RELATIVE_SUPPORT")
        if group["median_px"] > median_review:
            review_reasons.append("MEDIAN_RESIDUAL_OUTLIER")
        if group["p90_px"] > p90_review:
            review_reasons.append("P90_RESIDUAL_OUTLIER")
        if group["inlier_ratio"] < inlier_low:
            review_reasons.append("LOW_INLIER_RATIO")
        if group["median_px"] > median_reject:
            severe.append("SEVERE_MEDIAN_RESIDUAL")
        if group["p90_px"] > p90_reject:
            severe.append("SEVERE_P90_RESIDUAL")
        if group["inlier_ratio"] < max(0.10, inlier_low * 0.5):
            severe.append("SEVERE_LOW_INLIER_RATIO")
        if len(severe) >= 2:
            group["gate"] = "REJECT"
            group["reasons"] = severe + review_reasons
            group["weight"] = 0.0
        elif review_reasons or severe:
            group["gate"] = "DOWNWEIGHT"
            group["reasons"] = severe + review_reasons
            group["weight"] = 0.25
        else:
            group["gate"] = "GOOD"
            group["reasons"] = []
            group["weight"] = 1.0
    return groups, inlier_threshold


def classify_tracks(errors: Any, observations: dict[str, Any], inlier_threshold: float) -> list[dict[str, Any]]:
    import numpy as np

    track_rows = []
    for track_index in range(int(observations["track"].max()) + 1):
        values = errors[observations["track"] == track_index]
        track_rows.append(
            {
                "track_id": track_index,
                "support": len(values),
                "median_px": float(np.median(values)),
                "p90_px": float(np.percentile(values, 90)),
                "inlier_ratio": float(np.mean(values <= inlier_threshold)),
            }
        )
    medians = [row["median_px"] for row in track_rows]
    p90s = [row["p90_px"] for row in track_rows]
    median_limit = min(
        12.0, robust_upper(medians, max(2.0, float(np.median(medians)) * 2.5), 4.0)
    )
    p90_limit = min(
        16.0, robust_upper(p90s, max(4.0, float(np.median(p90s)) * 2.5), 4.0)
    )
    for row in track_rows:
        reasons = []
        if row["median_px"] > median_limit:
            reasons.append("RESIDUAL_OUTLIER")
        if row["inlier_ratio"] < 0.20:
            reasons.append("LOW_INLIER_RATIO")
        if row["p90_px"] > p90_limit and row["inlier_ratio"] < 0.75:
            reasons.append("UNSTABLE_TAIL_RESIDUAL")
        row["status"] = "REJECT" if reasons else "KEEP"
        row["reasons"] = reasons
    return track_rows


def summarize_errors(values: Any, inlier_threshold: float) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "p95_px": float(np.percentile(values, 95)),
        "inlier_threshold_px": inlier_threshold,
        "inlier_ratio": float(np.mean(values <= inlier_threshold)),
        "observation_count": len(values),
    }


def camera_json(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        "intrinsic": camera["intrinsic"],
        "extrinsic_world_to_camera": camera["extrinsic"][:3],
        "camera_to_world": invert_extrinsic(camera["extrinsic"]),
        "camera_center_world": camera["center"],
        "rotation_convention": "OpenCV world-to-camera; Xc=R*Xw+t",
    }


def rebuild_aggregate_csv(output_root: Path) -> None:
    sequence_rows = []
    camera_rows = []
    for path in sorted(output_root.glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        visual_path = path.parent / "visual_qa.json"
        visual = (
            json.loads(visual_path.read_text(encoding="utf-8"))
            if visual_path.is_file() else None
        )
        numeric_status = metrics["acceptance"]["status"]
        visual_status = visual["status"] if visual is not None else "PENDING"
        status_rank = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
        final_status = numeric_status
        if visual_status in status_rank and status_rank[visual_status] > status_rank[numeric_status]:
            final_status = visual_status
        final_reasons = list(metrics["acceptance"]["reasons"])
        if visual is not None:
            final_reasons.extend(
                reason for reason in visual.get("reasons", []) if reason not in final_reasons
            )
        sequence_rows.append(
            {
                "sequence": metrics["sequence"],
                "exercise": metrics["exercise"],
                "status": final_status,
                "numeric_status": numeric_status,
                "visual_status": visual_status,
                "status_reasons": final_reasons,
                "intrinsics_mode": metrics["configuration"]["intrinsics"],
                "robust_loss": metrics["configuration"]["robust_loss"],
                "points_initial": metrics["tracks"]["ba_track_count_initial"],
                "points_final": metrics["tracks"]["ba_track_count_final"],
                "observations_extracted": metrics["tracks"].get(
                    "observation_count_extracted",
                    sum(row["support"] for row in metrics["sample_gating"]["rows"]),
                ),
                "observations_stage2": metrics["tracks"].get(
                    "observation_count_stage2",
                    sum(row["support"] for row in metrics["tracks"]["post_ba_track_gating"]),
                ),
                "observations_initial": metrics["reprojection_pre"]["observation_count"],
                "observations_final": metrics["reprojection_post"]["observation_count"],
                "median_pre_px": metrics["reprojection_pre"]["median_px"],
                "median_post_px": metrics["reprojection_post"]["median_px"],
                "p90_pre_px": metrics["reprojection_pre"]["p90_px"],
                "p90_post_px": metrics["reprojection_post"]["p90_px"],
                "inlier_ratio_pre": metrics["reprojection_pre"]["inlier_ratio"],
                "inlier_ratio_post": metrics["reprojection_post"]["inlier_ratio"],
                "rejected_tracks": (
                    metrics["tracks"]["stage1_rejected_tracks"]
                    + metrics["tracks"].get("post_ba_rejected_tracks", 0)
                ),
                "rejected_samples": metrics["sample_gating"]["rejected_samples"],
                "downweighted_samples": metrics["sample_gating"]["downweighted_samples"],
                "converged": metrics["optimization"]["stage2"]["success"],
                "final_cost": metrics["optimization"]["stage2"]["cost"],
                "output_path": str(path.parent),
            }
        )
        for camera, row in metrics["camera_comparison"].items():
            camera_rows.append(
                {
                    "sequence": metrics["sequence"],
                    "exercise": metrics["exercise"],
                    "sequence_status": final_status,
                    "camera_id": camera,
                    **row,
                }
            )
    atomic_csv(output_root / "sequence_summary.csv", sequence_rows)
    atomic_csv(output_root / "camera_summary.csv", camera_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--dataset-root", "--root", dest="root", type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="private dataset root (or EXERCISE3D_DATASET_ROOT)",
    )
    parser.add_argument("--vggt-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--intrinsics", choices=("fixed", "limited"), default="fixed")
    parser.add_argument("--robust-loss", choices=("huber", "cauchy"), default="huber")
    parser.add_argument("--robust-scale", type=float, default=5.0)
    parser.add_argument("--max-tracks", type=int, default=800)
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--sift-contrast", type=float, default=0.005)
    parser.add_argument("--confidence-percentile", type=float, default=20.0)
    parser.add_argument("--static-mad-percentile", type=float, default=80.0)
    parser.add_argument("--border-fraction", type=float, default=0.025)
    parser.add_argument("--temporal-cluster-radius", type=float, default=3.0)
    parser.add_argument("--temporal-descriptor-distance", type=float, default=220.0)
    parser.add_argument("--ratio-test", type=float, default=0.95)
    parser.add_argument("--magsac-threshold", type=float, default=2.5)
    parser.add_argument("--magsac-iterations", type=int, default=100000)
    parser.add_argument("--initial-epipolar-max", type=float, default=30.0)
    parser.add_argument("--guided-pointmap-radius", type=float, default=0.02,
                        help="point-map association radius as a fraction of VGGT scene scale")
    parser.add_argument("--guided-epipolar-max", type=float, default=20.0)
    parser.add_argument("--guided-descriptor-distance", type=float, default=340.0)
    parser.add_argument("--guided-ratio-test", type=float, default=0.98)
    parser.add_argument(
        "--temporal-crossview-descriptor-distance", type=float, default=510.0,
        help="maximum SIFT L2 distance at a Phase-2-corrected sampled PTS pair",
    )
    parser.add_argument("--initial-reprojection-max", type=float, default=30.0)
    parser.add_argument("--pointmap-consistency-max", type=float, default=0.25)
    parser.add_argument("--min-parallax-deg", type=float, default=0.5)
    parser.add_argument("--duplicate-radius-px", type=float, default=2.0)
    parser.add_argument("--pose-prior-weight", type=float, default=1.0)
    parser.add_argument("--intrinsic-prior-weight", type=float, default=0.20)
    parser.add_argument("--point-prior-weight", type=float, default=0.05)
    parser.add_argument("--max-camera-rotation-change", type=float, default=20.0)
    parser.add_argument("--max-camera-center-change", type=float, default=0.25)
    parser.add_argument("--focal-refinement-fraction", type=float, default=0.05)
    parser.add_argument("--principal-refinement-fraction", type=float, default=0.02)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument(
        "--stage2-max-nfev", type=int, default=None,
        help="recovery-only Stage 2 budget; Stage 1 keeps --max-nfev",
    )
    parser.add_argument(
        "--optimizer-verbose", type=int, choices=(0, 1, 2), default=0,
        help="SciPy least_squares diagnostic verbosity; objective is unchanged",
    )
    parser.add_argument("--save-debug-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ensure_runtime()
    import cv2
    import numpy as np
    import scipy

    args = parse_args(argv)
    args.root = args.root.resolve()
    vggt_root = (args.vggt_root or args.root / "outputs" / "vggt").resolve()
    output_root = (args.output_root or args.root / "outputs" / "background_ba").resolve()
    immutable_roots = [
        (args.root / name).resolve()
        for name in ("origin", "synced_video", "final_frame")
    ]
    if any(output_root == item or item in output_root.parents for item in immutable_roots):
        raise RuntimeError(f"output root overlaps immutable dataset input: {output_root}")
    if not 0 <= args.confidence_percentile < 100 or not 0 < args.static_mad_percentile <= 100:
        raise RuntimeError("invalid percentile configuration")
    if args.max_nfev < 1 or (args.stage2_max_nfev is not None and args.stage2_max_nfev < 1):
        raise RuntimeError("optimization budgets must be positive")
    sequence_dir = resolve_sequence(vggt_root, args.sequence)
    sequence_metadata = json.loads((sequence_dir / "metadata.json").read_text(encoding="utf-8"))
    validate_eis(args.root, args.sequence)
    temporal_models = load_temporal_models(args.root, args.sequence)
    data = {camera: load_camera(sequence_dir / camera) for camera in CAMERAS}
    scene_scale = float(sequence_metadata["sequence_status"]["scene_scale_arbitrary_units"])
    if args.dry_run:
        payload = {
            "sequence": args.sequence,
            "sequence_path": sequence_dir,
            "output_path": output_root / args.sequence,
            "sample_count": {camera: len(data[camera]["timestamps_sec"]) for camera in CAMERAS},
            "scene_scale": scene_scale,
            "temporal_models": temporal_models,
            "fixed_camera_prerequisite": "3/3 FIXED_CAMERA_OK",
            "writes_performed": False,
        }
        print(json.dumps(jsonable(payload), indent=2))
        return 0

    started = time.monotonic()
    sequence_output = output_root / args.sequence
    debug_dir = sequence_output / "debug" if args.save_debug_images else None
    aggregated_old = {}
    for camera in CAMERAS:
        extrinsics = np.tile(np.eye(4), (len(data[camera]["timestamps_sec"]), 1, 1))
        extrinsics[:, :3, :4] = data[camera]["extrinsics_world_to_camera"]
        aggregated_old[camera] = aggregate_camera_pose(
            extrinsics, data[camera]["intrinsics"], scene_scale
        )
    gauge_transform = aggregated_old["cam1"]["extrinsic"]
    gauge_inverse = np.linalg.inv(gauge_transform)
    cameras_initial = {}
    for camera in CAMERAS:
        extrinsic = aggregated_old[camera]["extrinsic"] @ gauge_inverse
        extrinsic[:3, :3] = project_so3(extrinsic[:3, :3])
        center = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
        cameras_initial[camera] = {
            "rotation": extrinsic[:3, :3],
            "center": center,
            "extrinsic": make_extrinsic(extrinsic[:3, :3], center),
            "intrinsic": aggregated_old[camera]["intrinsic"],
            "height": data[camera]["height"],
            "width": data[camera]["width"],
        }
    cameras_initial["cam1"]["rotation"] = np.eye(3)
    cameras_initial["cam1"]["center"] = np.zeros(3)
    cameras_initial["cam1"]["extrinsic"] = np.eye(4)

    images = {camera: decode_preprocessed_images(args.root, data[camera]) for camera in CAMERAS}
    feature_sets = {}
    static_rows = []
    mask_arrays = {}
    landmarks = {}
    persistence_rows = []
    for camera in CAMERAS:
        feature_sets[camera], masks, rows = extract_static_features(
            camera, data[camera], images[camera], args, debug_dir
        )
        mask_arrays[camera] = np.stack(masks)
        static_rows.extend(rows)
        landmarks[camera], persistence = build_persistent_landmarks(
            camera, feature_sets[camera], args
        )
        persistence_rows.append(persistence)
    edges = []
    match_rows = []
    temporal_pairing_rows = []
    for camera_a, camera_b in (("cam1", "cam2"), ("cam1", "cam3"), ("cam2", "cam3")):
        pairings = corrected_temporal_pairings(
            camera_a, camera_b, data, temporal_models[f"{camera_a}-{camera_b}"]
        )
        pair_edges, match_stats = match_camera_landmarks(
            camera_a, camera_b, landmarks, cameras_initial, gauge_transform, scene_scale,
            pairings, args,
        )
        edges.extend(pair_edges)
        match_rows.append(match_stats)
        temporal_pairing_rows.extend(pairings)
    raw_tracks, graph_stats = build_cross_camera_tracks(landmarks, edges, args.max_tracks)
    tracks, triangulation_stats = filter_and_triangulate_tracks(
        raw_tracks, cameras_initial, scene_scale, gauge_transform, args
    )
    if len(tracks) < 8:
        raise RuntimeError(
            f"insufficient static cross-camera tracks after filtering: {len(tracks)} ({args.sequence})"
        )
    arrays = tracks_to_arrays(tracks, data)
    observations = {
        "track": arrays["obs_track"],
        "camera": arrays["obs_camera"],
        "sample": arrays["obs_sample"],
        "xy": arrays["obs_xy"],
    }
    initial_camera_list = [cameras_initial[camera] for camera in CAMERAS]
    initial_intrinsics = [cameras_initial[camera]["intrinsic"] for camera in CAMERAS]
    image_shapes = [(data[camera]["height"], data[camera]["width"]) for camera in CAMERAS]
    initial_errors = observation_errors(initial_camera_list, arrays["points_initial"], observations)
    initial_residual_vectors = observation_residual_vectors(
        initial_camera_list, arrays["points_initial"], observations
    )
    if args.optimizer_verbose:
        print("=== OPTIMIZER_TRACE stage1 ===", flush=True)
    stage1 = run_bundle_adjustment(
        initial_camera_list, initial_intrinsics, arrays["points_initial"], observations,
        np.ones(len(initial_errors)), scene_scale, image_shapes, args,
        max_nfev=args.max_nfev, verbose=args.optimizer_verbose,
    )
    stage1_errors = observation_errors(stage1["cameras"], stage1["points"], observations)
    stage1_residual_vectors = observation_residual_vectors(
        stage1["cameras"], stage1["points"], observations
    )
    sample_rows, inlier_threshold = classify_samples(stage1_errors, observations)
    track_rows = classify_tracks(stage1_errors, observations, inlier_threshold)
    sample_gate = {
        (CAMERA_INDEX[row["camera_id"]], row["sample_index"]): row for row in sample_rows
    }
    track_keep = np.asarray([row["status"] == "KEEP" for row in track_rows])
    observation_keep = track_keep[observations["track"]] & np.asarray(
        [
            sample_gate[(int(camera), int(sample))]["gate"] != "REJECT"
            for camera, sample in zip(observations["camera"], observations["sample"])
        ]
    )
    kept_old_tracks = np.flatnonzero(track_keep)
    old_to_new = np.full(len(track_keep), -1, dtype=np.int32)
    old_to_new[kept_old_tracks] = np.arange(len(kept_old_tracks))
    active_observation_mask = observation_keep & (old_to_new[observations["track"]] >= 0)
    stage2_observations = {
        "track": old_to_new[observations["track"][active_observation_mask]],
        "camera": observations["camera"][active_observation_mask],
        "sample": observations["sample"][active_observation_mask],
        "xy": observations["xy"][active_observation_mask],
    }
    stage2_weights = np.asarray(
        [
            sample_gate[(int(camera), int(sample))]["weight"]
            for camera, sample in zip(stage2_observations["camera"], stage2_observations["sample"])
        ],
        dtype=np.float64,
    )
    if len(kept_old_tracks) < 8 or len(stage2_observations["track"]) < 24:
        raise RuntimeError("robust gates removed too much support for stage 2")
    if args.optimizer_verbose:
        print("=== OPTIMIZER_TRACE stage2 ===", flush=True)
    stage2 = run_bundle_adjustment(
        stage1["cameras"], [camera["intrinsic"] for camera in stage1["cameras"]],
        stage1["points"][kept_old_tracks], stage2_observations, stage2_weights,
        scene_scale, image_shapes, args,
        max_nfev=args.stage2_max_nfev or args.max_nfev,
        verbose=args.optimizer_verbose,
    )
    points_refined_all = stage1["points"].copy()
    points_refined_all[kept_old_tracks] = stage2["points"]
    final_cameras = stage2["cameras"]
    final_errors_all = observation_errors(final_cameras, points_refined_all, observations)
    final_residual_vectors = observation_residual_vectors(
        final_cameras, points_refined_all, observations
    )
    stage2_errors = observation_errors(
        final_cameras, stage2["points"], stage2_observations
    )
    post_track_rows = classify_tracks(stage2_errors, stage2_observations, inlier_threshold)
    post_track_keep = np.asarray([row["status"] == "KEEP" for row in post_track_rows])
    final_track_keep = np.zeros(len(track_keep), dtype=bool)
    final_track_keep[kept_old_tracks[post_track_keep]] = True
    final_observation_mask = active_observation_mask & final_track_keep[observations["track"]]
    active_final_errors = final_errors_all[final_observation_mask]
    active_initial_errors = initial_errors[final_observation_mask]
    final_kept_old_tracks = np.flatnonzero(final_track_keep)

    camera_comparison = {}
    for camera_index, camera in enumerate(CAMERAS):
        initial = initial_camera_list[camera_index]
        refined = final_cameras[camera_index]
        camera_comparison[camera] = {
            "vggt_rotation_pairwise_p95_deg": aggregated_old[camera]["rotation_pairwise_p95_deg"],
            "vggt_center_dispersion_p95": aggregated_old[camera]["center_dispersion_p95"],
            "refined_rotation_change_from_robust_init_deg": rotation_angle_deg(
                refined["rotation"], initial["rotation"]
            ),
            "refined_center_change_from_robust_init": float(
                np.linalg.norm(refined["center"] - initial["center"])
            ),
            "refined_center_change_scene_fraction": float(
                np.linalg.norm(refined["center"] - initial["center"]) / scene_scale
            ),
            "fx_change_fraction": float(
                refined["intrinsic"][0, 0] / initial["intrinsic"][0, 0] - 1.0
            ),
            "fy_change_fraction": float(
                refined["intrinsic"][1, 1] / initial["intrinsic"][1, 1] - 1.0
            ),
            "principal_point_change_px": float(
                np.linalg.norm(refined["intrinsic"][:2, 2] - initial["intrinsic"][:2, 2])
            ),
        }
    pre_summary = summarize_errors(active_initial_errors, inlier_threshold)
    post_summary = summarize_errors(active_final_errors, inlier_threshold)
    radial_diagnostic = radial_residual_diagnostic(
        final_cameras, final_residual_vectors, final_errors_all, observations,
        final_observation_mask, image_shapes,
    )
    camera_graph_connected = all(
        any(camera in [member[0] for member in track["members"]] for track in tracks)
        for camera in CAMERAS
    )
    max_rotation_change = max(
        row["refined_rotation_change_from_robust_init_deg"] for row in camera_comparison.values()
    )
    max_center_change = max(
        row["refined_center_change_scene_fraction"] for row in camera_comparison.values()
    )
    acceptance_reasons = []
    fatal = []
    review = []
    if not stage2["result"]["success"] or not np.isfinite(active_final_errors).all():
        fatal.append("OPTIMIZATION_NOT_CONVERGED_OR_NONFINITE")
    if not camera_graph_connected:
        fatal.append("THREE_CAMERA_GRAPH_DISCONNECTED")
    if len(final_kept_old_tracks) < 15 or len(active_final_errors) < 60:
        fatal.append("INSUFFICIENT_FINAL_SUPPORT")
    if post_summary["median_px"] > pre_summary["median_px"] * 1.05:
        review.append("MEDIAN_REPROJECTION_NOT_IMPROVED")
    if (
        post_summary["p90_px"] > pre_summary["p90_px"] * 1.10
        or post_summary["p95_px"] > pre_summary["p95_px"] * 1.20
    ):
        review.append("TAIL_REPROJECTION_WORSENED")
    if post_summary["inlier_ratio"] < 0.50:
        review.append("LOW_FINAL_INLIER_RATIO")
    if len(final_kept_old_tracks) < 40 or len(active_final_errors) < 200:
        review.append("LIMITED_TRACK_SUPPORT")
    if max_rotation_change > 10.0 or max_center_change > 0.15:
        review.append("LARGE_CAMERA_CHANGE")
    if any(
        row["classification"] == "RADIAL_PATTERN_REVIEW"
        for row in radial_diagnostic.values()
    ):
        review.append("RADIAL_DISTORTION_PATTERN")
    if fatal:
        acceptance_status = "FAIL"
        acceptance_reasons = fatal + review
    elif review:
        acceptance_status = "REVIEW"
        acceptance_reasons = review
    else:
        acceptance_status = "PASS"

    cameras_initial_payload = {
        "schema_version": 1,
        "sequence": args.sequence,
        "coordinate_convention": "OpenCV world-to-camera; cam1 robust physical pose fixed to identity",
        "scale": "arbitrary; cam1-cam2 initial baseline preserved as gauge",
        "source_vggt_world_to_ba_world": gauge_transform,
        "ba_world_to_source_vggt_world": gauge_inverse,
        "aggregation": "IRLS robust SO(3) mean + robust camera-center mean; no element-wise matrix average",
        "cameras": {
            camera: {
                **camera_json(cameras_initial[camera]),
                "vggt_pose_samples": [
                    {
                        **row,
                        "timestamp_sec": float(data[camera]["timestamps_sec"][row["sample_index"]]),
                    }
                    for row in aggregated_old[camera]["sample_rows"]
                ],
                "rotation_pairwise_p95_deg": aggregated_old[camera]["rotation_pairwise_p95_deg"],
                "center_dispersion_p95": aggregated_old[camera]["center_dispersion_p95"],
            }
            for camera in CAMERAS
        },
    }
    cameras_refined_payload = {
        "schema_version": 1,
        "sequence": args.sequence,
        "shared_pose_constraint": "exactly one optimized pose per physical camera; cam1 fixed",
        "coordinate_convention": "OpenCV world-to-camera",
        "initialization_only": False,
        "not_metric": True,
        "cameras": {
            camera: {
                **camera_json(final_cameras[index]),
                "comparison": camera_comparison[camera],
            }
            for index, camera in enumerate(CAMERAS)
        },
    }
    residual_rows = []
    for observation_index in range(len(observations["track"])):
        camera_index = int(observations["camera"][observation_index])
        sample_index = int(observations["sample"][observation_index])
        track_index = int(observations["track"][observation_index])
        gate = sample_gate[(camera_index, sample_index)]
        residual_rows.append(
            {
                "sequence": args.sequence,
                "track_id": track_index,
                "camera_id": CAMERAS[camera_index],
                "sample_index": sample_index,
                "timestamp_sec": float(arrays["obs_timestamp"][observation_index]),
                "x_px": float(observations["xy"][observation_index, 0]),
                "y_px": float(observations["xy"][observation_index, 1]),
                "vggt_confidence": float(arrays["obs_confidence"][observation_index]),
                "pre_error_px": float(initial_errors[observation_index]),
                "pre_residual_x_px": float(initial_residual_vectors[observation_index, 0]),
                "pre_residual_y_px": float(initial_residual_vectors[observation_index, 1]),
                "stage1_error_px": float(stage1_errors[observation_index]),
                "stage1_residual_x_px": float(stage1_residual_vectors[observation_index, 0]),
                "stage1_residual_y_px": float(stage1_residual_vectors[observation_index, 1]),
                "post_error_px": float(final_errors_all[observation_index]),
                "post_residual_x_px": float(final_residual_vectors[observation_index, 0]),
                "post_residual_y_px": float(final_residual_vectors[observation_index, 1]),
                "sample_gate": gate["gate"],
                "sample_weight": gate["weight"],
                "track_status": track_rows[track_index]["status"],
                "post_ba_track_status": (
                    post_track_rows[old_to_new[track_index]]["status"]
                    if old_to_new[track_index] >= 0 else "NOT_OPTIMIZED"
                ),
                "active_in_stage2": bool(active_observation_mask[observation_index]),
                "accepted_final": bool(final_observation_mask[observation_index]),
            }
        )
    metrics = {
        "schema_version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "created_at": utc_now(),
        "sequence": args.sequence,
        "subject_id": sequence_metadata["sequence_status"]["subject_id"],
        "exercise": sequence_metadata["sequence_status"]["exercise"],
        "initialization_source": str(sequence_dir),
        "scene_scale_arbitrary_units": scene_scale,
        "configuration": vars(args),
        "runtime": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "opencv": cv2.__version__,
            "elapsed_sec": time.monotonic() - started,
        },
        "physical_camera_model": {
            "camera_pose_variable_count": 3,
            "cam1_pose_fixed": True,
            "timestamp_pose_variables": 0,
            "global_scale_metric": False,
        },
        "static_background": {
            "method": "temporal median/MAD + broad motion components + VGGT confidence percentile + border exclusion + persistent SIFT",
            "sample_stats": static_rows,
            "persistence": persistence_rows,
        },
        "matching": {
            "method": "persistent SIFT + Phase-2-corrected PTS descriptor validation + ratio/MAGSAC and pointmap-guided association + VGGT-init epipolar gate",
            "pair_stats": match_rows,
            "temporal_pairings": temporal_pairing_rows,
        },
        "tracks": {
            "graph": graph_stats,
            "triangulation": triangulation_stats,
            "ba_track_count_initial": len(tracks),
            "ba_track_count_final": len(final_kept_old_tracks),
            "observation_count_extracted": len(initial_errors),
            "observation_count_stage2": len(stage2_errors),
            "observation_count_final": len(active_final_errors),
            "observation_track_length_median": float(np.median(arrays["track_length"])),
            "observation_track_length_p90": float(np.percentile(arrays["track_length"], 90)),
            "three_camera_track_count": int(np.sum(arrays["track_camera_count"] == 3)),
            "stage1_rejected_tracks": int(np.sum(~track_keep)),
            "post_ba_rejected_tracks": int(np.sum(~post_track_keep)),
            "track_gating": track_rows,
            "post_ba_track_gating": post_track_rows,
        },
        "sample_gating": {
            "rows": sample_rows,
            "rejected_samples": sum(row["gate"] == "REJECT" for row in sample_rows),
            "downweighted_samples": sum(row["gate"] == "DOWNWEIGHT" for row in sample_rows),
            "good_samples": sum(row["gate"] == "GOOD" for row in sample_rows),
        },
        "objective": {
            "reprojection": "pixel reprojection residual on static tracks",
            "robust_loss": args.robust_loss,
            "robust_scale": args.robust_scale,
            "weak_vggt_pose_prior": args.pose_prior_weight,
            "weak_dlt_point_prior": args.point_prior_weight,
            "intrinsics": args.intrinsics,
            "stage1": "all filtered tracks/observations",
            "stage2": "track outliers removed; sample GOOD=1, DOWNWEIGHT=0.25, REJECT=0",
            "optimization_budget": {
                "stage1_max_nfev": args.max_nfev,
                "stage2_max_nfev": args.stage2_max_nfev or args.max_nfev,
                "only_budget_differs_in_recovery": args.stage2_max_nfev is not None,
            },
        },
        "reprojection_pre": pre_summary,
        "reprojection_post": post_summary,
        "optimization": {"stage1": stage1["result"], "stage2": stage2["result"]},
        "camera_comparison": camera_comparison,
        "radial_distortion_diagnostic": radial_diagnostic,
        "acceptance": {
            "status": acceptance_status,
            "reasons": acceptance_reasons,
            "visual_review_pending": True,
            "single_absolute_threshold_used": False,
        },
        "forbidden_operations": {
            "source_media_modified": False,
            "frame_interpolation_or_generation": False,
            "human_pose_or_smpl": False,
            "pseudo_labels": False,
        },
    }

    atomic_json(sequence_output / "cameras_initial.json", cameras_initial_payload)
    atomic_json(sequence_output / "cameras_refined.json", cameras_refined_payload)
    atomic_npz(
        sequence_output / "tracks.npz",
        obs_track_id=arrays["obs_track"],
        obs_camera_index=arrays["obs_camera"],
        obs_sample_index=arrays["obs_sample"],
        obs_xy=arrays["obs_xy"],
        obs_timestamp_sec=arrays["obs_timestamp"],
        obs_vggt_confidence=arrays["obs_confidence"],
        obs_sift_response=arrays["obs_response"],
        track_length=arrays["track_length"],
        track_camera_count=arrays["track_camera_count"],
        active_stage2=active_observation_mask,
        accepted_final=final_observation_mask,
    )
    atomic_npz(
        sequence_output / "points3d.npz",
        points_initial=arrays["points_initial"],
        points_stage1=stage1["points"],
        points_refined=points_refined_all,
        accepted_track_mask=final_track_keep,
        coordinate_note=np.asarray("cam1-reference arbitrary-scale world", dtype="U64"),
    )
    atomic_csv(sequence_output / "residuals.csv", residual_rows)
    atomic_csv(sequence_output / "sample_gating.csv", sample_rows)
    atomic_json(sequence_output / "metrics.json", metrics)
    atomic_csv(sequence_output / "debug" / "static_feature_stats.csv", static_rows)
    atomic_csv(sequence_output / "debug" / "persistent_landmarks.csv", persistence_rows)
    atomic_csv(sequence_output / "debug" / "pair_matches.csv", match_rows)
    atomic_csv(sequence_output / "debug" / "temporal_pairings.csv", temporal_pairing_rows)
    atomic_npz(
        sequence_output / "debug" / "static_masks.npz",
        **{camera: mask_arrays[camera] for camera in CAMERAS},
    )
    rebuild_aggregate_csv(output_root)
    print(json.dumps(jsonable({
        "sequence": args.sequence,
        "status": acceptance_status,
        "tracks": {"initial": len(tracks), "final": len(final_kept_old_tracks)},
        "observations": {"initial": len(initial_errors), "final": len(active_final_errors)},
        "median_reprojection_px": {"pre": pre_summary["median_px"], "post": post_summary["median_px"]},
        "p90_reprojection_px": {"pre": pre_summary["p90_px"], "post": post_summary["p90_px"]},
        "sample_gates": {
            "good": metrics["sample_gating"]["good_samples"],
            "downweight": metrics["sample_gating"]["downweighted_samples"],
            "reject": metrics["sample_gating"]["rejected_samples"],
        },
        "output": sequence_output,
    }), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
