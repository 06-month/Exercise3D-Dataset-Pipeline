#!/usr/bin/env python3
"""Build a read-only provenance and timing inventory for an Exercise3D dataset.

The script never mutates source data.  It probes the authoritative MOV files,
the synchronized per-camera MP4 derivatives, the extracted JPEG trees, and the
manifest.  Reports are written below ``reports/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT)
).expanduser()
RAW_NAME_RE = re.compile(
    r"^(?P<exercise>[a-z]+)_(?P<view>0[123])_(?P<take>\d{4})\.(?:mov|mp4)$",
    re.IGNORECASE,
)
SET_NAME_RE = re.compile(r"^(?P<exercise>[a-z]+)_(?P<take>\d{4})$")
SYNC_CAM_RE = re.compile(r"^cam(?P<cam>[123])\.mp4$", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def relpath(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def parse_fraction(value: Any) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    try:
        text = str(value)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_f = float(denominator)
            return float(numerator) / denominator_f if denominator_f else None
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def rounded(value: float | None, digits: int = 9) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_json(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "unknown ffprobe error"
        raise RuntimeError(f"{' '.join(command[:4])}: {message}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON: {exc}") from exc


def ffprobe_metadata(path: Path) -> dict[str, Any]:
    return run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )


def ffprobe_video_packet_pts(path: Path) -> list[float]:
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
        raise RuntimeError(completed.stderr.strip() or "ffprobe packet scan failed")
    timestamps: list[float] = []
    for line in completed.stdout.splitlines():
        token = line.strip().split(",", 1)[0]
        value = parse_float(token)
        if value is not None and math.isfinite(value):
            timestamps.append(value)
    return timestamps


def first_stream(probe: dict[str, Any], codec_type: str) -> dict[str, Any]:
    return next(
        (stream for stream in probe.get("streams", []) if stream.get("codec_type") == codec_type),
        {},
    )


def rotation_degrees(video_stream: dict[str, Any]) -> int:
    tags = video_stream.get("tags") or {}
    rotation = parse_float(tags.get("rotate"))
    if rotation is None:
        for side_data in video_stream.get("side_data_list") or []:
            rotation = parse_float(side_data.get("rotation"))
            if rotation is not None:
                break
    return int(round(rotation or 0)) % 360


def merge_tags(probe: dict[str, Any], video_stream: dict[str, Any]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for source in (probe.get("format", {}).get("tags") or {}, video_stream.get("tags") or {}):
        for key, value in source.items():
            merged[str(key).lower()] = str(value)
    return merged


def first_tag(tags: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = tags.get(name.lower())
        if value:
            return value
    return None


def pts_statistics(timestamps_in_packet_order: list[float]) -> dict[str, Any]:
    if not timestamps_in_packet_order:
        return {
            "packet_count": 0,
            "first_pts_sec": None,
            "last_pts_sec": None,
            "actual_fps_pts": None,
            "pts_delta_median_sec": None,
            "pts_delta_p05_sec": None,
            "pts_delta_p95_sec": None,
            "pts_delta_min_sec": None,
            "pts_delta_max_sec": None,
            "unique_delta_count_rounded_us": 0,
            "duplicate_pts_count": 0,
            "cadence_class": "INSUFFICIENT_EVIDENCE",
        }

    # Presentation timestamps can be out of order in packet order because of B
    # frames.  Cadence is therefore measured in presentation order.
    ordered = sorted(timestamps_in_packet_order)
    deltas = [current - previous for previous, current in zip(ordered, ordered[1:])]
    positive = [delta for delta in deltas if delta > 1e-9]
    duplicate_count = len(deltas) - len(positive)
    first_pts = ordered[0]
    last_pts = ordered[-1]
    span = last_pts - first_pts
    actual_fps = (len(ordered) - 1) / span if len(ordered) > 1 and span > 0 else None
    median_delta = statistics.median(positive) if positive else None
    p05 = percentile(positive, 5)
    p95 = percentile(positive, 95)
    delta_range = (p95 - p05) if p95 is not None and p05 is not None else None
    if len(positive) < 2 or median_delta is None:
        cadence_class = "INSUFFICIENT_EVIDENCE"
    elif duplicate_count:
        cadence_class = "VFR_OR_IRREGULAR"
    elif delta_range is not None and delta_range <= max(2e-6, median_delta * 0.005):
        cadence_class = "CFR_LIKE"
    else:
        cadence_class = "VFR_OR_IRREGULAR"

    return {
        "packet_count": len(ordered),
        "first_pts_sec": rounded(first_pts, 6),
        "last_pts_sec": rounded(last_pts, 6),
        "actual_fps_pts": rounded(actual_fps, 6),
        "pts_delta_median_sec": rounded(median_delta, 9),
        "pts_delta_p05_sec": rounded(p05, 9),
        "pts_delta_p95_sec": rounded(p95, 9),
        "pts_delta_min_sec": rounded(min(positive), 9) if positive else None,
        "pts_delta_max_sec": rounded(max(positive), 9) if positive else None,
        "unique_delta_count_rounded_us": len({round(delta, 6) for delta in positive}),
        "duplicate_pts_count": duplicate_count,
        "cadence_class": cadence_class,
    }


def probe_video(spec: dict[str, Any], root: Path, with_hash: bool) -> dict[str, Any]:
    path = Path(spec["absolute_path"])
    probe = ffprobe_metadata(path)
    video = first_stream(probe, "video")
    audio = first_stream(probe, "audio")
    if not video:
        raise RuntimeError(f"video stream missing: {path}")
    packet_stats = pts_statistics(ffprobe_video_packet_pts(path))
    format_info = probe.get("format") or {}
    tags = merge_tags(probe, video)
    rotation = rotation_degrees(video)
    coded_width = parse_int(video.get("width"))
    coded_height = parse_int(video.get("height"))
    if rotation in (90, 270):
        display_width, display_height = coded_height, coded_width
    else:
        display_width, display_height = coded_width, coded_height
    stat = path.stat()

    result = {
        key: value for key, value in spec.items() if key != "absolute_path"
    }
    result.update(
        {
            "path": relpath(path, root),
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "sha256": sha256_file(path) if with_hash else None,
            "format_name": format_info.get("format_name"),
            "format_duration_sec": rounded(parse_float(format_info.get("duration")), 6),
            "format_start_time_sec": rounded(parse_float(format_info.get("start_time")), 6),
            "format_bit_rate": parse_int(format_info.get("bit_rate")),
            "video_codec": video.get("codec_name"),
            "video_profile": video.get("profile"),
            "video_pix_fmt": video.get("pix_fmt"),
            "coded_width": coded_width,
            "coded_height": coded_height,
            "display_width": display_width,
            "display_height": display_height,
            "rotation_degrees": rotation,
            "sample_aspect_ratio": video.get("sample_aspect_ratio"),
            "r_frame_rate": video.get("r_frame_rate"),
            "r_frame_rate_fps": rounded(parse_fraction(video.get("r_frame_rate")), 6),
            "avg_frame_rate": video.get("avg_frame_rate"),
            "avg_frame_rate_fps": rounded(parse_fraction(video.get("avg_frame_rate")), 6),
            "video_time_base": video.get("time_base"),
            "video_start_time_sec": rounded(parse_float(video.get("start_time")), 6),
            "video_duration_sec": rounded(parse_float(video.get("duration")), 6),
            "container_nb_frames": parse_int(video.get("nb_frames")),
            "audio_present": bool(audio),
            "audio_codec": audio.get("codec_name"),
            "audio_sample_rate_hz": parse_int(audio.get("sample_rate")),
            "audio_channels": parse_int(audio.get("channels")),
            "audio_channel_layout": audio.get("channel_layout"),
            "audio_time_base": audio.get("time_base"),
            "audio_start_time_sec": rounded(parse_float(audio.get("start_time")), 6),
            "audio_duration_sec": rounded(parse_float(audio.get("duration")), 6),
            "creation_time": first_tag(tags, ["creation_time", "com.apple.quicktime.creationdate"]),
            "device_make": first_tag(
                tags,
                ["com.apple.quicktime.make", "make", "manufacturer", "android manufacturer"],
            ),
            "device_model": first_tag(
                tags,
                ["com.apple.quicktime.model", "model", "android model"],
            ),
            "software": first_tag(tags, ["software", "encoder"]),
            # Exact GPS is not needed for provenance and is privacy-sensitive.
            # Record only whether location metadata exists.
            "location_metadata_present": bool(
                first_tag(
                    tags,
                    ["com.apple.quicktime.location.iso6709", "location", "location-eng"],
                )
            ),
            **packet_stats,
        }
    )
    return result


def discover_raw(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    specs: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted((root / "origin").rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mov", ".mp4"}:
            continue
        match = RAW_NAME_RE.match(path.name)
        if not match:
            errors.append(f"unrecognized raw video name: {relpath(path, root)}")
            continue
        exercise = match.group("exercise").lower()
        view = int(match.group("view"))
        take = match.group("take")
        specs.append(
            {
                "absolute_path": str(path),
                "asset_kind": "raw_video",
                "exercise": exercise,
                "take": take,
                "set_id": f"{exercise}_{take}",
                "camera_id": f"cam{view}",
                "source_view_id": f"{view:02d}",
                "subject_id": "UNKNOWN",
                "semantic_view": "UNKNOWN",
                "physical_parent": path.parent.name,
                "source_path": None,
            }
        )
    return specs, errors


def load_sync_sources(root: Path) -> tuple[dict[tuple[str, int], str], list[str]]:
    sources: dict[tuple[str, int], str] = {}
    errors: list[str] = []
    for sync_path in sorted((root / "synced_video").glob("*/*/sync.json")):
        try:
            payload = json.loads(sync_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {relpath(sync_path, root)}: {exc}")
            continue
        set_id = str(payload.get("set_id") or sync_path.parent.name)
        for clip in payload.get("clips") or []:
            cam = parse_int(clip.get("cam"))
            source = clip.get("source")
            if cam in (1, 2, 3) and source:
                sources[(set_id, cam)] = str(source)
    return sources, errors


def discover_synced(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    source_map, errors = load_sync_sources(root)
    specs: list[dict[str, Any]] = []
    for path in sorted((root / "synced_video").glob("*/*/cam?.mp4")):
        set_match = SET_NAME_RE.match(path.parent.name)
        cam_match = SYNC_CAM_RE.match(path.name)
        if not set_match or not cam_match:
            errors.append(f"unrecognized synchronized path: {relpath(path, root)}")
            continue
        exercise = set_match.group("exercise").lower()
        take = set_match.group("take")
        cam = int(cam_match.group("cam"))
        set_id = f"{exercise}_{take}"
        source = source_map.get((set_id, cam))
        specs.append(
            {
                "absolute_path": str(path),
                "asset_kind": "synchronized_video",
                "exercise": exercise,
                "take": take,
                "set_id": set_id,
                "camera_id": f"cam{cam}",
                "source_view_id": f"{cam:02d}",
                "subject_id": "UNKNOWN",
                "semantic_view": "UNKNOWN",
                "physical_parent": path.parent.parent.name,
                "source_path": source,
            }
        )
        if not source:
            errors.append(f"missing sync.json source relation: {set_id}/cam{cam}")
    return specs, errors


def jpeg_size(path: Path) -> tuple[int, int]:
    # Small stdlib JPEG SOF parser, used only for representative files in each
    # frame directory so this audit does not require Pillow.
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise ValueError("not a JPEG")
        while True:
            prefix = handle.read(1)
            if not prefix:
                break
            if prefix != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if not marker or marker in {b"\xd8", b"\xd9"}:
                continue
            if marker == b"\xda":
                break
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = int.from_bytes(length_bytes, "big")
            if marker[0] in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                payload = handle.read(segment_length - 2)
                if len(payload) < 5:
                    break
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                return width, height
            handle.seek(max(segment_length - 2, 0), os.SEEK_CUR)
    raise ValueError(f"JPEG dimensions not found: {path}")


def inventory_frame_directories(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for camera_dir in sorted((root / "final_frame").glob("*/*/cam[123]")):
        set_match = SET_NAME_RE.match(camera_dir.parent.name)
        cam_match = re.match(r"cam([123])$", camera_dir.name)
        if not set_match or not cam_match:
            errors.append(f"unrecognized frame directory: {relpath(camera_dir, root)}")
            continue
        files = sorted(path for path in camera_dir.iterdir() if path.suffix.lower() == ".jpg")
        indices: list[int] = []
        for path in files:
            try:
                indices.append(int(path.stem))
            except ValueError:
                errors.append(f"non-numeric frame name: {relpath(path, root)}")
        contiguous = indices == list(range(len(indices)))
        dimensions: set[tuple[int, int]] = set()
        sample_positions = sorted({0, len(files) // 2, len(files) - 1}) if files else []
        for index in sample_positions:
            try:
                dimensions.add(jpeg_size(files[index]))
            except (OSError, ValueError) as exc:
                errors.append(f"cannot inspect {relpath(files[index], root)}: {exc}")
        exercise = set_match.group("exercise").lower()
        take = set_match.group("take")
        rows.append(
            {
                "asset_kind": "frame_directory",
                "path": relpath(camera_dir, root),
                "exercise": exercise,
                "take": take,
                "set_id": f"{exercise}_{take}",
                "camera_id": f"cam{cam_match.group(1)}",
                "subject_id": "UNKNOWN",
                "semantic_view": "UNKNOWN",
                "frame_count": len(files),
                "first_frame_index": indices[0] if indices else None,
                "last_frame_index": indices[-1] if indices else None,
                "indices_contiguous_zero_based": contiguous,
                "sampled_dimensions": [f"{width}x{height}" for width, height in sorted(dimensions)],
            }
        )
        if not contiguous:
            errors.append(f"non-contiguous frame indices: {relpath(camera_dir, root)}")
    return rows, errors


def inventory_manifest(root: Path) -> tuple[dict[str, Any], list[str]]:
    manifest_path = root / "manifest.csv"
    errors: list[str] = []
    rows = 0
    sample_ids: set[str] = set()
    sets: set[str] = set()
    exercises: set[str] = set()
    missing_paths = 0
    duplicate_sample_ids = 0
    split_counts: dict[str, int] = {}
    qc_counts: dict[str, int] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"sample_id", "set_id", "exercise", "cam1", "cam2", "cam3"}
        absent = expected - set(reader.fieldnames or [])
        if absent:
            errors.append(f"manifest columns missing: {sorted(absent)}")
        for row in reader:
            rows += 1
            sample_id = row.get("sample_id", "")
            if sample_id in sample_ids:
                duplicate_sample_ids += 1
            sample_ids.add(sample_id)
            sets.add(row.get("set_id", ""))
            exercises.add(row.get("exercise", ""))
            split = row.get("split", "")
            qc = row.get("qc", "")
            split_counts[split] = split_counts.get(split, 0) + 1
            qc_counts[qc] = qc_counts.get(qc, 0) + 1
            for camera in ("cam1", "cam2", "cam3"):
                image_path = root / row.get(camera, "")
                if not image_path.is_file():
                    missing_paths += 1
    if duplicate_sample_ids:
        errors.append(f"duplicate manifest sample_id values: {duplicate_sample_ids}")
    if missing_paths:
        errors.append(f"missing manifest image paths: {missing_paths}")
    return (
        {
            "path": relpath(manifest_path, root),
            "row_count": rows,
            "unique_sample_ids": len(sample_ids),
            "set_count": len(sets),
            "exercise_count": len(exercises),
            "image_references": rows * 3,
            "missing_image_paths": missing_paths,
            "duplicate_sample_ids": duplicate_sample_ids,
            "split_counts": dict(sorted(split_counts.items())),
            "qc_counts": dict(sorted(qc_counts.items())),
        },
        errors,
    )


def validate_inventory(
    root: Path,
    videos: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    raw = [row for row in videos if row["asset_kind"] == "raw_video"]
    synced = [row for row in videos if row["asset_kind"] == "synchronized_video"]
    for name, rows, expected in (("raw video", raw, 78), ("synchronized video", synced, 78)):
        if len(rows) != expected:
            errors.append(f"expected {expected} {name} assets, found {len(rows)}")
        identities = [(row["set_id"], row["camera_id"]) for row in rows]
        if len(identities) != len(set(identities)):
            errors.append(f"duplicate logical identity among {name} assets")
    if len(frames) != 78:
        errors.append(f"expected 78 frame directories, found {len(frames)}")
    if sum(row["frame_count"] for row in frames) != manifest["image_references"]:
        errors.append(
            "frame directory total does not equal manifest image references: "
            f"{sum(row['frame_count'] for row in frames)} != {manifest['image_references']}"
        )

    raw_paths = {row["path"] for row in raw}
    for row in synced:
        source = row.get("source_path")
        if source not in raw_paths:
            errors.append(f"invalid source relation for {row['path']}: {source}")
        elif not (root / source).is_file():
            errors.append(f"source path absent for {row['path']}: {source}")

    raw_by_set: dict[str, list[dict[str, Any]]] = {}
    synced_by_set: dict[str, list[dict[str, Any]]] = {}
    frames_by_set: dict[str, list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_set.setdefault(row["set_id"], []).append(row)
    for row in synced:
        synced_by_set.setdefault(row["set_id"], []).append(row)
    for row in frames:
        frames_by_set.setdefault(row["set_id"], []).append(row)
    all_sets = set(raw_by_set) | set(synced_by_set) | set(frames_by_set)
    for set_id in sorted(all_sets):
        if len(raw_by_set.get(set_id, [])) != 3:
            errors.append(f"{set_id}: raw view count is not 3")
        if len(synced_by_set.get(set_id, [])) != 3:
            errors.append(f"{set_id}: synchronized view count is not 3")
        set_frames = frames_by_set.get(set_id, [])
        if len(set_frames) != 3:
            errors.append(f"{set_id}: frame camera directory count is not 3")
        elif len({row["frame_count"] for row in set_frames}) != 1:
            errors.append(f"{set_id}: per-camera frame counts differ")
    return errors


VIDEO_CSV_FIELDS = [
    "asset_kind",
    "path",
    "exercise",
    "take",
    "set_id",
    "camera_id",
    "source_view_id",
    "subject_id",
    "semantic_view",
    "physical_parent",
    "source_path",
    "size_bytes",
    "sha256",
    "format_name",
    "format_duration_sec",
    "format_start_time_sec",
    "video_codec",
    "video_profile",
    "video_pix_fmt",
    "coded_width",
    "coded_height",
    "display_width",
    "display_height",
    "rotation_degrees",
    "r_frame_rate",
    "r_frame_rate_fps",
    "avg_frame_rate",
    "avg_frame_rate_fps",
    "actual_fps_pts",
    "video_time_base",
    "video_start_time_sec",
    "video_duration_sec",
    "container_nb_frames",
    "packet_count",
    "first_pts_sec",
    "last_pts_sec",
    "pts_delta_median_sec",
    "pts_delta_p05_sec",
    "pts_delta_p95_sec",
    "pts_delta_min_sec",
    "pts_delta_max_sec",
    "unique_delta_count_rounded_us",
    "duplicate_pts_count",
    "cadence_class",
    "audio_present",
    "audio_codec",
    "audio_sample_rate_hz",
    "audio_channels",
    "audio_time_base",
    "audio_start_time_sec",
    "audio_duration_sec",
    "creation_time",
    "device_make",
    "device_model",
    "software",
    "location_metadata_present",
    "mtime_utc",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(
    root: Path,
    videos: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    raw = [row for row in videos if row["asset_kind"] == "raw_video"]
    synced = [row for row in videos if row["asset_kind"] == "synchronized_video"]

    def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key))
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    known_device_models = sorted({row["device_model"] for row in raw if row.get("device_model")})
    known_device_makes = sorted({row["device_make"] for row in raw if row.get("device_make")})
    camera_device_mapping: dict[str, list[str]] = {}
    for camera_id in sorted({row["camera_id"] for row in raw}):
        camera_device_mapping[camera_id] = sorted(
            {row["device_model"] for row in raw if row["camera_id"] == camera_id and row.get("device_model")}
        )
    return {
        "raw_video_count": len(raw),
        "synchronized_camera_video_count": len(synced),
        "synchronized_preview_video_count_excluded": len(list(root.glob("synced_video/*/*/preview.mp4"))),
        "sequence_count": len({row["set_id"] for row in raw}),
        "exercise_count": len({row["exercise"] for row in raw}),
        "frame_directory_count": len(frames),
        "jpeg_count": sum(row["frame_count"] for row in frames),
        "manifest_triplet_count": manifest["row_count"],
        "raw_by_source_view": count_by(raw, "source_view_id"),
        "raw_by_cadence_class": count_by(raw, "cadence_class"),
        "synced_by_cadence_class": count_by(synced, "cadence_class"),
        "raw_nominal_fps_counts": count_by(raw, "r_frame_rate_fps"),
        "synced_nominal_fps_counts": count_by(synced, "r_frame_rate_fps"),
        "raw_rotation_counts": count_by(raw, "rotation_degrees"),
        "synced_rotation_counts": count_by(synced, "rotation_degrees"),
        "raw_audio_present_count": sum(bool(row["audio_present"]) for row in raw),
        "synced_audio_present_count": sum(bool(row["audio_present"]) for row in synced),
        "known_raw_device_makes": known_device_makes,
        "known_raw_device_models": known_device_models,
        "raw_camera_device_mapping": camera_device_mapping,
        "raw_videos_without_subject_id": len(raw),
        "raw_videos_without_semantic_view": len(raw),
        "raw_videos_with_location_metadata": sum(
            bool(row["location_metadata_present"]) for row in raw
        ),
    }


def write_markdown(
    path: Path,
    generated_at: str,
    summary: dict[str, Any],
    errors: list[str],
    videos: list[dict[str, Any]],
) -> None:
    raw = [row for row in videos if row["asset_kind"] == "raw_video"]
    synced = [row for row in videos if row["asset_kind"] == "synchronized_video"]
    lines = [
        "# Exercise3D source-data inventory",
        "",
        f"Generated: `{generated_at}`",
        "",
        "This is a read-only audit. `origin/`, `_replaced_originals/`, `synced_video/`, and "
        "`final_frame/` were not modified.",
        "",
        "## Counts",
        "",
        f"- Raw camera videos: **{summary['raw_video_count']}**",
        f"- Synchronized camera videos: **{summary['synchronized_camera_video_count']}**",
        f"- Sequences: **{summary['sequence_count']}**",
        f"- Frame directories / JPEGs: **{summary['frame_directory_count']} / {summary['jpeg_count']}**",
        f"- Manifest triplets: **{summary['manifest_triplet_count']}**",
        "",
        "## Timing and metadata",
        "",
        f"- Raw nominal FPS: `{json.dumps(summary['raw_nominal_fps_counts'], ensure_ascii=False)}`",
        f"- Raw PTS cadence: `{json.dumps(summary['raw_by_cadence_class'], ensure_ascii=False)}`",
        f"- Synchronized PTS cadence: `{json.dumps(summary['synced_by_cadence_class'], ensure_ascii=False)}`",
        f"- Raw audio present: **{summary['raw_audio_present_count']}/{len(raw)}**",
        f"- Synchronized audio present: **{summary['synced_audio_present_count']}/{len(synced)}**",
        f"- Device makes: `{summary['known_raw_device_makes'] or ['UNKNOWN']}`",
        f"- Device models: `{summary['known_raw_device_models'] or ['UNKNOWN']}`",
        f"- Camera/device mapping: `{json.dumps(summary['raw_camera_device_mapping'], ensure_ascii=False)}`",
        f"- Location metadata present: **{summary['raw_videos_with_location_metadata']}/{len(raw)}** "
        "(exact coordinates deliberately omitted from reports)",
        "- Subject IDs: **UNKNOWN** (no evidence-backed mapping in current metadata)",
        "- Front/Left/Right mapping: **UNKNOWN** (camera IDs remain cam1/cam2/cam3)",
        "",
        "`actual_fps_pts` and cadence statistics use presentation timestamps from every video packet. "
        "Packet order is sorted by PTS before cadence measurement because B-frame packet order is not presentation order.",
        "",
        "## Validation",
        "",
    ]
    if errors:
        lines.append(f"**FAIL — {len(errors)} issue(s)**")
        lines.append("")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("**PASS — no structural or provenance errors found.**")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `dataset_inventory.csv`: one row per raw or synchronized camera video",
            "- `dataset_inventory.json`: full video, frame-directory, manifest, and validation records",
            "- SHA-256 is included unless the script was run with `--no-hash`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_inventory(root: Path, output_dir: Path, jobs: int, with_hash: bool) -> int:
    generated_at = utc_now()
    raw_specs, errors = discover_raw(root)
    synced_specs, sync_errors = discover_synced(root)
    errors.extend(sync_errors)
    specs = raw_specs + synced_specs
    print(f"Probing {len(specs)} videos with {jobs} workers...", flush=True)
    videos: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        future_to_spec = {
            executor.submit(probe_video, spec, root, with_hash): spec for spec in specs
        }
        completed_count = 0
        for future in as_completed(future_to_spec):
            spec = future_to_spec[future]
            completed_count += 1
            try:
                videos.append(future.result())
            except Exception as exc:  # preserve all other audit results
                path = Path(spec["absolute_path"])
                errors.append(f"probe failed for {relpath(path, root)}: {exc}")
            if completed_count % 10 == 0 or completed_count == len(specs):
                print(f"  {completed_count}/{len(specs)}", flush=True)
    videos.sort(key=lambda row: (row["asset_kind"], row["set_id"], row["camera_id"]))

    frames, frame_errors = inventory_frame_directories(root)
    errors.extend(frame_errors)
    manifest, manifest_errors = inventory_manifest(root)
    errors.extend(manifest_errors)
    errors.extend(validate_inventory(root, videos, frames, manifest))
    errors = sorted(set(errors))
    summary = summarize(root, videos, frames, manifest)

    payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "dataset_root": ".",
        "read_only_source_policy": [
            "origin/",
            "_replaced_originals/",
            "synced_video/",
            "final_frame/",
        ],
        "identity_notes": {
            "subject_id": "UNKNOWN: no evidence-backed subject-to-sequence mapping is present",
            "semantic_view": "UNKNOWN: cam1/cam2/cam3 are source view IDs, not inferred Front/Left/Right labels",
            "raw_deadlift_physical_folder": "origin/deaflift; logical identity comes from the filename prefix deadlift",
            "replaced_barbellrow_files": "_replaced_originals/ is provenance-only and excluded from active source counts",
            "synchronized_preview_videos": "preview.mp4 files are visualization composites, not camera assets, and are excluded",
        },
        "summary": summary,
        "manifest": manifest,
        "videos": videos,
        "frame_directories": frames,
        "validation": {"status": "PASS" if not errors else "FAIL", "errors": errors},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "dataset_inventory.csv", videos, VIDEO_CSV_FIELDS)
    (output_dir / "dataset_inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(
        output_dir / "dataset_inventory_summary.md",
        generated_at,
        summary,
        errors,
        videos,
    )
    print(json.dumps({"summary": summary, "validation": payload["validation"]}, indent=2))
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
        help="report directory (default: DATASET_ROOT/reports; set explicitly for read-only sources)",
    )
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--no-hash", action="store_true", help="skip SHA-256 calculation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "reports").resolve()
    required = [root / "origin", root / "synced_video", root / "final_frame", root / "manifest.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"Required dataset assets missing: {missing}", file=sys.stderr)
        return 2
    return build_inventory(root, output_dir, args.jobs, not args.no_hash)


if __name__ == "__main__":
    raise SystemExit(main())
