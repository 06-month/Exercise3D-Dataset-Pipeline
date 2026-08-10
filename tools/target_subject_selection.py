#!/usr/bin/env python3
"""Select one temporally persistent subject from cached DETR person candidates.

The selector is deliberately pose-model independent: it consumes the complete,
ragged DETR output produced by ``sapiens2_pose_pipeline.py`` and never uses a
Sapiens2 result to decide identity.  All candidate boxes remain in private
metadata while only a bidirectionally agreed target is eligible for pose
inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


CAMERAS = ("cam1", "cam2", "cam3")
PILOT_SEQUENCES = (
    "barbellrow_0000",
    "squat_0001",
    "pushup_0001",
    "benchpress_0003",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSOCIATION_WEIGHTS = {
    "bbox_iou": 0.50,
    "normalized_center": 0.25,
    "log_area_continuity": 0.15,
    "log_aspect_continuity": 0.10,
}
TRACK_QUALITY_WEIGHTS = {
    "sequence_coverage": 0.30,
    "track_span": 0.10,
    "initial_window_coverage": 0.16,
    "terminal_window_coverage": 0.04,
    "association_continuity": 0.16,
    "relative_bbox_area": 0.12,
    "detector_score": 0.07,
    "sequence_location_prior": 0.05,
}


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detections-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, default=list(PILOT_SEQUENCES))
    parser.add_argument("--cameras", type=parse_list, default=list(CAMERAS))
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--association-threshold", type=float, default=0.30)
    parser.add_argument("--ambiguity-confidence", type=float, default=0.45)
    parser.add_argument("--global-margin-scale", type=float, default=0.20)
    parser.add_argument("--save-overlays", type=int, default=12)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="skip cameras whose consolidated all-detections output is not ready",
    )
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_dir(dataset_root: Path, sequence: str, camera: str) -> Path:
    exercise = sequence.rsplit("_", 1)[0]
    return dataset_root / "final_frame" / exercise / sequence / camera


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    result = sorted(item for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)
    if not result:
        raise RuntimeError(f"no image frames: {path}")
    return result


def box_iou(left: np.ndarray, right: np.ndarray) -> float:
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    extent = np.maximum(bottom_right - top_left, 0.0)
    intersection = float(extent[0] * extent[1])
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def intersection_over_min_area(left: np.ndarray, right: np.ndarray) -> float:
    top_left = np.maximum(left[:2], right[:2])
    bottom_right = np.minimum(left[2:], right[2:])
    extent = np.maximum(bottom_right - top_left, 0.0)
    intersection = float(extent[0] * extent[1])
    left_area = float(np.prod(np.maximum(left[2:] - left[:2], 0.0)))
    right_area = float(np.prod(np.maximum(right[2:] - right[:2], 0.0)))
    denominator = min(left_area, right_area)
    return intersection / denominator if denominator > 0 else 0.0


def box_geometry(box: np.ndarray) -> tuple[np.ndarray, float, float]:
    size = np.maximum(box[2:] - box[:2], 1e-6)
    return (box[:2] + box[2:]) * 0.5, float(size[0] * size[1]), float(size[0] / size[1])


def association_affinity(previous: np.ndarray, current: np.ndarray, gap: int) -> float:
    previous_center, previous_area, previous_aspect = box_geometry(previous)
    current_center, current_area, current_aspect = box_geometry(current)
    center_distance = float(np.linalg.norm(current_center - previous_center))
    log_area_change = abs(math.log(max(current_area, 1e-9) / max(previous_area, 1e-9)))
    log_aspect_change = abs(
        math.log(max(current_aspect, 1e-9) / max(previous_aspect, 1e-9))
    )
    overlap = box_iou(previous, current)
    permitted_distance = 0.16 + 0.06 * max(gap - 1, 0)
    if overlap < 0.005 and center_distance > permitted_distance:
        return 0.0
    if log_area_change > 1.35 + 0.15 * max(gap - 1, 0):
        return 0.0
    center_similarity = math.exp(-((center_distance / (0.10 + 0.04 * gap)) ** 2))
    area_similarity = math.exp(-(log_area_change / 0.50))
    aspect_similarity = math.exp(-(log_aspect_change / 0.55))
    gap_penalty = math.exp(-0.12 * max(gap - 1, 0))
    return float(
        gap_penalty
        * (
            ASSOCIATION_WEIGHTS["bbox_iou"] * overlap
            + ASSOCIATION_WEIGHTS["normalized_center"] * center_similarity
            + ASSOCIATION_WEIGHTS["log_area_continuity"] * area_similarity
            + ASSOCIATION_WEIGHTS["log_aspect_continuity"] * aspect_similarity
        )
    )


def minimum_cost_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Pure NumPy rectangular Hungarian assignment (minimum cost)."""

    if cost.size == 0:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    matrix = cost.T if transposed else cost
    rows, columns = matrix.shape
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    p = np.zeros(columns + 1, dtype=np.int64)
    way = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        min_value = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=np.bool_)
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                value = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                if value < min_value[column]:
                    min_value[column] = value
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = [(int(p[column] - 1), column - 1) for column in range(1, columns + 1) if p[column]]
    if transposed:
        return [(column, row) for row, column in pairs]
    return pairs


@dataclass
class Observation:
    frame: int
    candidate: int
    box: np.ndarray
    score: float
    affinity: float = 1.0


@dataclass
class Track:
    identifier: int
    observations: dict[int, Observation] = field(default_factory=dict)

    @property
    def first_frame(self) -> int:
        return min(self.observations)

    @property
    def last_frame(self) -> int:
        return max(self.observations)

    def temporal_last(self, reverse: bool) -> Observation:
        frame = min(self.observations) if reverse else max(self.observations)
        return self.observations[frame]


def track_candidates(
    boxes: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    max_gap: int,
    threshold: float,
    reverse: bool,
) -> tuple[list[Track], list[np.ndarray]]:
    frame_count = len(boxes)
    candidate_track = [np.full(len(item), -1, dtype=np.int32) for item in boxes]
    tracks: list[Track] = []
    active: list[Track] = []
    order = range(frame_count - 1, -1, -1) if reverse else range(frame_count)
    for frame in order:
        candidates = boxes[frame]
        active = [
            track
            for track in active
            if abs(frame - track.temporal_last(reverse).frame) <= max_gap
        ]
        affinities = np.zeros((len(active), len(candidates)), dtype=np.float64)
        for track_index, track in enumerate(active):
            previous = track.temporal_last(reverse)
            gap = abs(frame - previous.frame)
            for candidate_index, candidate in enumerate(candidates):
                affinities[track_index, candidate_index] = association_affinity(
                    previous.box, candidate, gap
                )
        matched_candidates: set[int] = set()
        if affinities.size:
            for track_index, candidate_index in minimum_cost_assignment(-affinities):
                affinity = float(affinities[track_index, candidate_index])
                if affinity < threshold:
                    continue
                track = active[track_index]
                observation = Observation(
                    frame=frame,
                    candidate=candidate_index,
                    box=candidates[candidate_index],
                    score=float(scores[frame][candidate_index]),
                    affinity=affinity,
                )
                track.observations[frame] = observation
                candidate_track[frame][candidate_index] = track.identifier
                matched_candidates.add(candidate_index)
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in matched_candidates:
                continue
            track = Track(identifier=len(tracks))
            track.observations[frame] = Observation(
                frame=frame,
                candidate=candidate_index,
                box=candidate,
                score=float(scores[frame][candidate_index]),
            )
            tracks.append(track)
            active.append(track)
            candidate_track[frame][candidate_index] = track.identifier
    return tracks, candidate_track


def track_quality(
    track: Track,
    boxes: Sequence[np.ndarray],
    frame_count: int,
) -> dict[str, float]:
    observations = [track.observations[frame] for frame in sorted(track.observations)]
    observed = len(observations)
    initialization_frames = max(12, min(90, int(round(frame_count * 0.10))))
    terminal_start = max(0, frame_count - initialization_frames)
    relative_area = []
    center_prior = []
    for observation in observations:
        _, area, _ = box_geometry(observation.box)
        candidate_areas = [box_geometry(candidate)[1] for candidate in boxes[observation.frame]]
        relative_area.append(area / max(max(candidate_areas, default=area), 1e-9))
        center, _, _ = box_geometry(observation.box)
        center_prior.append(math.exp(-float(np.linalg.norm(center - 0.5)) / 0.45))
    affinities = [item.affinity for item in observations if item.affinity < 1.0]
    coverage = observed / max(frame_count, 1)
    span = (track.last_frame - track.first_frame + 1) / max(frame_count, 1)
    initial = sum(item.frame < initialization_frames for item in observations) / initialization_frames
    terminal = sum(item.frame >= terminal_start for item in observations) / initialization_frames
    continuity = float(np.median(affinities)) if affinities else 0.0
    detector_score = float(np.median([item.score for item in observations]))
    area_rank = float(np.median(relative_area))
    location = float(np.median(center_prior))
    composite = (
        TRACK_QUALITY_WEIGHTS["sequence_coverage"] * coverage
        + TRACK_QUALITY_WEIGHTS["track_span"] * span
        + TRACK_QUALITY_WEIGHTS["initial_window_coverage"] * min(initial, 1.0)
        + TRACK_QUALITY_WEIGHTS["terminal_window_coverage"] * min(terminal, 1.0)
        + TRACK_QUALITY_WEIGHTS["association_continuity"] * continuity
        + TRACK_QUALITY_WEIGHTS["relative_bbox_area"] * area_rank
        + TRACK_QUALITY_WEIGHTS["detector_score"] * detector_score
        + TRACK_QUALITY_WEIGHTS["sequence_location_prior"] * location
    )
    return {
        "composite": float(composite),
        "coverage": float(coverage),
        "span": float(span),
        "initial_coverage": float(min(initial, 1.0)),
        "terminal_coverage": float(min(terminal, 1.0)),
        "continuity": continuity,
        "relative_area": area_rank,
        "detector_score": detector_score,
        "location_prior": location,
    }


def rank_tracks(
    tracks: Sequence[Track], boxes: Sequence[np.ndarray], frame_count: int
) -> list[tuple[Track, dict[str, float]]]:
    result = [(track, track_quality(track, boxes, frame_count)) for track in tracks]
    return sorted(result, key=lambda item: item[1]["composite"], reverse=True)


def motion_correlation(primary: Track, secondary: Track) -> float:
    shared = sorted(set(primary.observations) & set(secondary.observations))
    if len(shared) < 20:
        return 0.0
    primary_features = []
    secondary_features = []
    for frame in shared:
        first_center, first_area, first_aspect = box_geometry(primary.observations[frame].box)
        second_center, second_area, second_aspect = box_geometry(secondary.observations[frame].box)
        primary_features.append([*first_center, math.log(first_area), math.log(first_aspect)])
        secondary_features.append([*second_center, math.log(second_area), math.log(second_aspect)])
    first_delta = np.diff(np.asarray(primary_features, dtype=np.float64), axis=0)
    second_delta = np.diff(np.asarray(secondary_features, dtype=np.float64), axis=0)
    correlations = []
    for column in range(first_delta.shape[1]):
        if np.std(first_delta[:, column]) < 1e-6 or np.std(second_delta[:, column]) < 1e-6:
            continue
        correlations.append(abs(float(np.corrcoef(first_delta[:, column], second_delta[:, column])[0, 1])))
    return float(np.mean(correlations)) if correlations else 0.0


def load_candidates(
    path: Path, frame_count: int, width: int, height: int
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "person_count",
            "detector_fallback",
            "instance_offsets",
            "all_bboxes_xyxy",
            "all_bbox_scores",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise RuntimeError(f"{path}: missing {', '.join(missing)}")
        raw = {key: payload[key].copy() for key in required}
    offsets = raw["instance_offsets"].astype(np.int64)
    if len(offsets) != frame_count + 1 or int(offsets[-1]) != len(raw["all_bboxes_xyxy"]):
        raise RuntimeError(f"{path}: invalid ragged candidate offsets")
    normalization = np.asarray([width, height, width, height], dtype=np.float32)
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    fallback = raw["detector_fallback"].astype(np.bool_)
    for frame in range(frame_count):
        start, end = int(offsets[frame]), int(offsets[frame + 1])
        if fallback[frame]:
            boxes.append(np.empty((0, 4), dtype=np.float32))
            scores.append(np.empty(0, dtype=np.float32))
        else:
            boxes.append(raw["all_bboxes_xyxy"][start:end].astype(np.float32) / normalization)
            scores.append(raw["all_bbox_scores"][start:end].astype(np.float32))
    return boxes, scores, fallback, raw


def load_timestamps(path: Path, frames: Sequence[Path]) -> np.ndarray:
    if not path.exists():
        return np.full(len(frames), np.nan, dtype=np.float64)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(frames):
        raise RuntimeError(f"{path}: timestamp row count does not match source")
    timestamps = np.full(len(frames), np.nan, dtype=np.float64)
    for index, (row, frame) in enumerate(zip(rows, frames)):
        if row.get("frame_name") != frame.name:
            raise RuntimeError(f"{path}: frame name mismatch at row {index}")
        value = row.get("timestamp_pts_seconds", "")
        if value:
            timestamps[index] = float(value)
    return timestamps


def select_camera(
    dataset_root: Path,
    detections_root: Path,
    output_root: Path,
    sequence: str,
    camera: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frames = list_images(frame_dir(dataset_root, sequence, camera))
    first_image = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"failed to decode {frames[0]}")
    height, width = first_image.shape[:2]
    detection_path = detections_root / sequence / camera / "bboxes.npz"
    boxes, scores, fallback, raw = load_candidates(detection_path, len(frames), width, height)
    timestamps = load_timestamps(detection_path.parent / "frames.csv", frames)
    selection_started = time.perf_counter()
    forward_tracks, forward_map = track_candidates(
        boxes, scores, args.max_gap, args.association_threshold, reverse=False
    )
    backward_tracks, backward_map = track_candidates(
        boxes, scores, args.max_gap, args.association_threshold, reverse=True
    )
    forward_ranked = rank_tracks(forward_tracks, boxes, len(frames))
    backward_ranked = rank_tracks(backward_tracks, boxes, len(frames))
    empty_quality = {
        "composite": 0.0,
        "coverage": 0.0,
        "span": 0.0,
        "initial_coverage": 0.0,
        "terminal_coverage": 0.0,
        "continuity": 0.0,
        "relative_area": 0.0,
        "detector_score": 0.0,
        "location_prior": 0.0,
    }
    if forward_ranked:
        forward_primary, forward_quality = forward_ranked[0]
    else:
        forward_primary, forward_quality = Track(-1), empty_quality.copy()
    if backward_ranked:
        backward_primary, backward_quality = backward_ranked[0]
    else:
        backward_primary, backward_quality = Track(-1), empty_quality.copy()
    forward_second = forward_ranked[1][1]["composite"] if len(forward_ranked) > 1 else 0.0
    backward_second = backward_ranked[1][1]["composite"] if len(backward_ranked) > 1 else 0.0
    forward_margin = forward_quality["composite"] - forward_second
    backward_margin = backward_quality["composite"] - backward_second
    global_margin = min(forward_margin, backward_margin)
    margin_confidence = min(max(global_margin / args.global_margin_scale, 0.0), 1.0)
    forward_competitor = forward_ranked[1][0] if len(forward_ranked) > 1 else None
    backward_competitor = backward_ranked[1][0] if len(backward_ranked) > 1 else None
    persistent_close_competitor = (
        global_margin < 0.04
        and (
            (len(forward_ranked) > 1 and forward_ranked[1][1]["coverage"] >= 0.45)
            or (
                len(backward_ranked) > 1
                and backward_ranked[1][1]["coverage"] >= 0.45
            )
        )
    )

    forward_choice = np.full(len(frames), -1, dtype=np.int32)
    backward_choice = np.full(len(frames), -1, dtype=np.int32)
    for frame, observation in forward_primary.observations.items():
        forward_choice[frame] = observation.candidate
    for frame, observation in backward_primary.observations.items():
        backward_choice[frame] = observation.candidate

    counts = np.asarray([len(item) for item in boxes], dtype=np.int16)
    selected = np.full(len(frames), -1, dtype=np.int32)
    confidence = np.zeros(len(frames), dtype=np.float32)
    ambiguous = np.zeros(len(frames), dtype=np.bool_)
    no_target = counts == 0
    status = np.full(len(frames), "NO_TARGET", dtype="<U20")
    global_track_ambiguity = np.zeros(len(frames), dtype=np.bool_)
    local_affinity = np.zeros(len(frames), dtype=np.float32)
    for frame, observation in forward_primary.observations.items():
        local_affinity[frame] = max(local_affinity[frame], observation.affinity)
    for frame, observation in backward_primary.observations.items():
        local_affinity[frame] = max(local_affinity[frame], observation.affinity)
    for frame in range(len(frames)):
        if no_target[frame]:
            continue
        agreement = forward_choice[frame] >= 0 and forward_choice[frame] == backward_choice[frame]
        candidate_score = float(scores[frame][forward_choice[frame]]) if forward_choice[frame] >= 0 else 0.0
        confidence[frame] = float(
            0.45 * margin_confidence
            + 0.30 * local_affinity[frame]
            + 0.15 * candidate_score
            + 0.10 * min(forward_quality["coverage"], backward_quality["coverage"])
        )
        competitor_present = (
            forward_competitor is not None
            and forward_competitor.identifier in forward_map[frame]
        ) or (
            backward_competitor is not None
            and backward_competitor.identifier in backward_map[frame]
        )
        if persistent_close_competitor and competitor_present:
            global_track_ambiguity[frame] = True
            ambiguous[frame] = True
            status[frame] = "TARGET_AMBIGUOUS"
            confidence[frame] = min(confidence[frame], 0.44)
            continue
        if not agreement or confidence[frame] < args.ambiguity_confidence:
            ambiguous[frame] = True
            status[frame] = "TARGET_AMBIGUOUS"
            continue
        selected[frame] = forward_choice[frame]
        status[frame] = "TARGET"

    # Candidate crossings can be locally underdetermined even when the same
    # global track wins in both directions.  Compare the chosen association
    # with the best competing candidate from both temporal directions and
    # abstain when the margin is small.
    association_ambiguity = np.zeros(len(frames), dtype=np.bool_)
    association_margin = np.ones(len(frames), dtype=np.float32)
    for frame in range(len(frames)):
        target_index = int(selected[frame])
        if target_index < 0 or len(boxes[frame]) < 2:
            continue
        directional_margins = []
        for neighbor in (frame - 1, frame + 1):
            if neighbor < 0 or neighbor >= len(frames) or selected[neighbor] < 0:
                continue
            affinities = np.asarray(
                [
                    association_affinity(
                        boxes[neighbor][selected[neighbor]], candidate, 1
                    )
                    for candidate in boxes[frame]
                ],
                dtype=np.float32,
            )
            competitors = np.delete(affinities, target_index)
            if len(competitors) and float(np.max(competitors)) >= args.association_threshold:
                directional_margins.append(
                    float(affinities[target_index] - np.max(competitors))
                )
        if directional_margins:
            association_margin[frame] = min(directional_margins)
            if association_margin[frame] < 0.06:
                association_ambiguity[frame] = True
                ambiguous[frame] = True
                selected[frame] = -1
                status[frame] = "TARGET_AMBIGUOUS"
                confidence[frame] = min(confidence[frame], 0.44)

    identity_switch_risk = np.zeros(len(frames), dtype=np.bool_)
    previous_frame = -1
    for frame in range(len(frames)):
        if selected[frame] < 0:
            continue
        if previous_frame == frame - 1:
            affinity = association_affinity(
                boxes[previous_frame][selected[previous_frame]],
                boxes[frame][selected[frame]],
                1,
            )
            if affinity < max(args.association_threshold + 0.05, 0.38):
                identity_switch_risk[frame] = True
                ambiguous[frame] = True
                selected[frame] = -1
                status[frame] = "TARGET_AMBIGUOUS"
                confidence[frame] = min(confidence[frame], 0.44)
        previous_frame = frame if selected[frame] >= 0 else -1

    occlusion_risk = np.zeros(len(frames), dtype=np.bool_)
    duplicate_count = np.zeros(len(frames), dtype=np.int16)
    background_count = counts.astype(np.int32) - (selected >= 0).astype(np.int32)
    background_offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    background_boxes: list[np.ndarray] = []
    normalization = np.asarray([width, height, width, height], dtype=np.float32)
    for frame in range(len(frames)):
        for candidate, box in enumerate(boxes[frame]):
            if candidate == selected[frame]:
                continue
            background_boxes.append(box * normalization)
            if selected[frame] < 0:
                continue
            target = boxes[frame][selected[frame]]
            overlap = box_iou(target, box)
            containment = intersection_over_min_area(target, box)
            target_center, target_area, _ = box_geometry(target)
            other_center, _, _ = box_geometry(box)
            normalized_distance = float(
                np.linalg.norm(target_center - other_center) / max(math.sqrt(target_area), 1e-6)
            )
            if overlap >= 0.05 or containment >= 0.15 or normalized_distance <= 0.65:
                occlusion_risk[frame] = True
            _, other_area, _ = box_geometry(box)
            area_ratio = min(target_area, other_area) / max(target_area, other_area, 1e-9)
            if overlap >= 0.65 or (containment >= 0.80 and area_ratio >= 0.50):
                duplicate_count[frame] += 1
        background_offsets[frame + 1] = len(background_boxes)

    possible_reflection_tracks: set[int] = set()
    for track, quality in forward_ranked[1:]:
        shared = sorted(set(forward_primary.observations) & set(track.observations))
        if not shared or quality["coverage"] < 0.45:
            continue
        median_overlap = float(
            np.median(
                [
                    box_iou(
                        forward_primary.observations[frame].box,
                        track.observations[frame].box,
                    )
                    for frame in shared
                ]
            )
        )
        # A long-lived, spatially separate, much smaller copy is useful mirror
        # evidence even when the exercising subject has too little translation
        # for a reliable motion-correlation estimate.  This flag is diagnostic;
        # it never participates in target selection.
        correlation = motion_correlation(forward_primary, track)
        if median_overlap < 0.10 and (
            correlation >= 0.35 or quality["relative_area"] <= 0.45
        ):
            possible_reflection_tracks.add(track.identifier)
    possible_reflection_count = np.zeros(len(frames), dtype=np.int16)
    for frame in range(len(frames)):
        for candidate in range(len(boxes[frame])):
            if candidate != selected[frame] and int(forward_map[frame][candidate]) in possible_reflection_tracks:
                possible_reflection_count[frame] += 1
    selection_elapsed = time.perf_counter() - selection_started

    output_dir = output_root / sequence / camera
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    candidate_offsets[1:] = np.cumsum(counts, dtype=np.int64)
    all_boxes = (
        np.concatenate(boxes, axis=0).astype(np.float32) * normalization
        if int(counts.sum())
        else np.empty((0, 4), dtype=np.float32)
    )
    all_scores = (
        np.concatenate(scores, axis=0).astype(np.float32)
        if int(counts.sum())
        else np.empty(0, dtype=np.float32)
    )
    atomic_savez(
        output_dir / "target_selection.npz",
        frame_index=np.arange(len(frames), dtype=np.int32),
        frame_name=np.asarray([path.name for path in frames]),
        timestamp_pts_seconds=timestamps,
        num_person_candidates=counts,
        target_candidate_index=selected,
        forward_candidate_index=forward_choice,
        backward_candidate_index=backward_choice,
        background_person_count=background_count.astype(np.int16),
        target_selection_confidence=confidence,
        target_ambiguous=ambiguous,
        no_target=no_target,
        detector_fallback=fallback,
        target_status=status,
        occlusion_risk=occlusion_risk,
        association_ambiguity=association_ambiguity,
        association_margin=association_margin,
        identity_switch_risk=identity_switch_risk,
        global_track_ambiguity=global_track_ambiguity,
        detector_duplicate_count=duplicate_count,
        possible_reflection_count=possible_reflection_count,
        candidate_offsets=candidate_offsets,
        all_person_detections_xyxy=all_boxes,
        all_person_detection_scores=all_scores,
        background_instance_offsets=background_offsets,
        background_bboxes_xyxy=(
            np.stack(background_boxes).astype(np.float32)
            if background_boxes
            else np.empty((0, 4), dtype=np.float32)
        ),
    )
    frame_rows = []
    for frame, path in enumerate(frames):
        frame_rows.append(
            {
                "frame_index": frame,
                "frame_name": path.name,
                "timestamp_pts_seconds": (
                    f"{timestamps[frame]:.9f}" if np.isfinite(timestamps[frame]) else ""
                ),
                "num_person_candidates": int(counts[frame]),
                "target_candidate_index": int(selected[frame]),
                "background_person_count": int(background_count[frame]),
                "target_selection_confidence": f"{confidence[frame]:.6f}",
                "target_ambiguous": bool(ambiguous[frame]),
                "no_target": bool(no_target[frame]),
                "detector_fallback": bool(fallback[frame]),
                "target_status": str(status[frame]),
                "occlusion_risk": bool(occlusion_risk[frame]),
                "association_ambiguity": bool(association_ambiguity[frame]),
                "identity_switch_risk": bool(identity_switch_risk[frame]),
                "global_track_ambiguity": bool(global_track_ambiguity[frame]),
                "detector_duplicate_count": int(duplicate_count[frame]),
                "possible_reflection_count": int(possible_reflection_count[frame]),
            }
        )
    atomic_write_csv(output_dir / "frames.csv", list(frame_rows[0]), frame_rows)

    top_tracks = []
    for track, quality in forward_ranked[:8]:
        top_tracks.append({"track_id": track.identifier, "observations": len(track.observations), **quality})
    geometry_path = Path("outputs/background_ba") / sequence / "cameras_refined.json"
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": sequence,
        "camera": camera,
        "frame_count": len(frames),
        "source_detection_file": str(detection_path),
        "privacy": "private coordinate payload; do not commit",
        "selection_policy": "multi-frame track persistence plus bidirectional association",
        "association_weights": ASSOCIATION_WEIGHTS,
        "track_quality_weights": TRACK_QUALITY_WEIGHTS,
        "appearance_embedding_used": False,
        "full_triangulation_used": False,
        "refined_camera_geometry_interface": str(geometry_path),
        "refined_camera_geometry_available": (PROJECT_ROOT / geometry_path).exists(),
        "timestamp_coverage": float(np.isfinite(timestamps).mean()),
        "forward_primary_track": forward_primary.identifier,
        "backward_primary_track": backward_primary.identifier,
        "forward_global_margin": forward_margin,
        "backward_global_margin": backward_margin,
        "persistent_close_competitor": persistent_close_competitor,
        "forward_primary_quality": forward_quality,
        "backward_primary_quality": backward_quality,
        "forward_top_tracks": top_tracks,
        "possible_reflection_track_ids": sorted(possible_reflection_tracks),
    }
    atomic_write_text(output_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    save_overlays(output_dir, frames, boxes, scores, selected, confidence, ambiguous, no_target, occlusion_risk, possible_reflection_count, args.save_overlays)

    accepted = selected >= 0
    center_jumps = []
    log_area_jumps = []
    for frame in range(1, len(frames)):
        if selected[frame - 1] < 0 or selected[frame] < 0:
            continue
        previous_center, previous_area, _ = box_geometry(
            boxes[frame - 1][selected[frame - 1]]
        )
        current_center, current_area, _ = box_geometry(boxes[frame][selected[frame]])
        center_jumps.append(
            float(
                np.linalg.norm(current_center - previous_center)
                / max(math.sqrt(current_area), 1e-6)
            )
        )
        log_area_jumps.append(
            abs(math.log(max(current_area, 1e-9) / max(previous_area, 1e-9)))
        )
    visibility_transitions = int(np.count_nonzero(np.diff(accepted.astype(np.int8))))
    target_gap_segments = int(
        np.count_nonzero(accepted & ~np.concatenate(([False], accepted[:-1])))
        - int(accepted[0])
    )
    summary = {
        "sequence": sequence,
        "camera": camera,
        "frame_count": len(frames),
        "total_person_candidates": int(counts.sum()),
        "mean_detected_persons_per_frame": float(counts.mean()),
        "all_detections_sapiens_crops": int(raw["person_count"].sum()),
        "target_only_sapiens_crops": int(accepted.sum()),
        "crop_reduction_fraction": float(1.0 - accepted.sum() / max(raw["person_count"].sum(), 1)),
        "target_frame_count": int(accepted.sum()),
        "target_ambiguous_count": int(ambiguous.sum()),
        "no_target_count": int(no_target.sum()),
        "background_person_count": int(background_count.sum()),
        "multi_person_frame_count": int((counts > 1).sum()),
        "occlusion_risk_count": int(occlusion_risk.sum()),
        "detector_duplicate_candidate_count": int(duplicate_count.sum()),
        "possible_reflection_candidate_count": int(possible_reflection_count.sum()),
        "forward_backward_disagreement_count": int(
            ((forward_choice != backward_choice) & ~no_target).sum()
        ),
        "selection_confidence_median": float(np.median(confidence[accepted])) if accepted.any() else 0.0,
        "selection_confidence_p10": float(np.percentile(confidence[accepted], 10)) if accepted.any() else 0.0,
        "target_bbox_center_jump_normalized_p95": (
            float(np.percentile(center_jumps, 95)) if center_jumps else 0.0
        ),
        "target_bbox_log_area_jump_p95": (
            float(np.percentile(log_area_jumps, 95)) if log_area_jumps else 0.0
        ),
        "target_visibility_transition_count": visibility_transitions,
        "target_gap_segment_count": target_gap_segments,
        "selection_elapsed_seconds": selection_elapsed,
        "selection_frames_per_second": len(frames) / max(selection_elapsed, 1e-12),
        "identity_switch_count": int(identity_switch_risk.sum()),
        "status": (
            "PASS"
            if accepted.any()
            and not ambiguous.any()
            and not no_target.any()
            and not identity_switch_risk.any()
            else "REVIEW"
        ),
    }
    atomic_write_text(output_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    return summary


def choose_overlay_indices(
    counts: np.ndarray,
    selected: np.ndarray,
    boxes: Sequence[np.ndarray],
    ambiguous: np.ndarray,
    occlusion: np.ndarray,
    reflection: np.ndarray,
    limit: int,
) -> list[int]:
    frame_count = len(counts)
    categories: list[int] = [0, frame_count // 2, frame_count - 1, int(np.argmax(counts))]
    candidate_index_changes = np.flatnonzero(np.diff(selected) != 0) + 1
    if len(candidate_index_changes):
        categories.extend(
            [
                int(candidate_index_changes[0]),
                int(candidate_index_changes[len(candidate_index_changes) // 2]),
                int(candidate_index_changes[-1]),
            ]
        )
    for mask in (ambiguous, occlusion, reflection > 0):
        indices = np.flatnonzero(mask)
        if len(indices):
            categories.extend([int(indices[0]), int(indices[len(indices) // 2]), int(indices[-1])])
    inversions = []
    for frame, target_index in enumerate(selected):
        if target_index < 0 or len(boxes[frame]) < 2:
            continue
        target_area = box_geometry(boxes[frame][target_index])[1]
        if any(box_geometry(box)[1] > target_area for index, box in enumerate(boxes[frame]) if index != target_index):
            inversions.append(frame)
    if inversions:
        categories.extend([inversions[0], inversions[len(inversions) // 2], inversions[-1]])
    for index in np.linspace(0, frame_count - 1, min(limit, frame_count), dtype=np.int64):
        categories.append(int(index))
    unique = list(dict.fromkeys(categories))
    return unique[:limit]


def save_overlays(
    output_dir: Path,
    frames: Sequence[Path],
    boxes: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    selected: np.ndarray,
    confidence: np.ndarray,
    ambiguous: np.ndarray,
    no_target: np.ndarray,
    occlusion: np.ndarray,
    reflection: np.ndarray,
    limit: int,
) -> None:
    if limit < 1:
        return
    counts = np.asarray([len(item) for item in boxes])
    indices = choose_overlay_indices(counts, selected, boxes, ambiguous, occlusion, reflection, limit)
    overlay_dir = output_dir / "debug" / "target_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for frame in indices:
        image = cv2.imread(str(frames[frame]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        scale = np.asarray([width, height, width, height], dtype=np.float32)
        for candidate, normalized_box in enumerate(boxes[frame]):
            x1, y1, x2, y2 = np.rint(normalized_box * scale).astype(int)
            is_target = candidate == selected[frame]
            color = (40, 210, 40) if is_target else (40, 40, 220)
            thickness = 4 if is_target else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                image,
                f"{'TARGET' if is_target else 'BG'} {candidate} {scores[frame][candidate]:.3f}",
                (max(x1, 0), max(y1 - 7, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        if no_target[frame]:
            state = "NO_TARGET"
        elif ambiguous[frame]:
            state = "TARGET_AMBIGUOUS"
        else:
            state = "TARGET"
        label = (
            f"frame={frame} state={state} conf={confidence[frame]:.3f} "
            f"persons={counts[frame]} occ={int(occlusion[frame])} refl={int(reflection[frame])}"
        )
        cv2.rectangle(image, (0, 0), (min(width - 1, 1180), 38), (0, 0, 0), -1)
        cv2.putText(image, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(overlay_dir / f"frame_{frame:06d}.jpg"), image)
        reasons = []
        if frame == 0:
            reasons.append("sequence_start")
        if frame == len(frames) // 2:
            reasons.append("sequence_middle")
        if frame == len(frames) - 1:
            reasons.append("sequence_end")
        if counts[frame] == counts.max():
            reasons.append("max_person_candidates")
        if ambiguous[frame]:
            reasons.append("target_ambiguous")
        if occlusion[frame]:
            reasons.append("occlusion_risk")
        if reflection[frame] > 0:
            reasons.append("possible_reflection")
        if frame > 0 and selected[frame] != selected[frame - 1]:
            reasons.append("candidate_index_order_change")
        target_index = int(selected[frame])
        if target_index >= 0:
            target_area = box_geometry(boxes[frame][target_index])[1]
            if any(
                box_geometry(box)[1] > target_area
                for candidate, box in enumerate(boxes[frame])
                if candidate != target_index
            ):
                reasons.append("background_bbox_larger_than_target")
        if not reasons:
            reasons.append("temporal_sample")
        manifest_rows.append(
            {
                "frame_index": frame,
                "frame_name": frames[frame].name,
                "overlay_file": f"frame_{frame:06d}.jpg",
                "reasons": ";".join(reasons),
                "target_candidate_index": target_index,
                "target_selection_confidence": f"{confidence[frame]:.6f}",
                "num_person_candidates": int(counts[frame]),
            }
        )
    if manifest_rows:
        atomic_write_csv(
            overlay_dir / "manifest.csv", list(manifest_rows[0]), manifest_rows
        )


def cross_view_summary(output_root: Path, sequence: str, cameras: Sequence[str]) -> dict[str, Any]:
    payloads = []
    for camera in cameras:
        path = output_root / sequence / camera / "target_selection.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            payloads.append(
                {
                    "camera": camera,
                    "target": payload["target_candidate_index"].copy(),
                    "ambiguous": payload["target_ambiguous"].copy(),
                    "no_target": payload["no_target"].copy(),
                    "timestamp": payload["timestamp_pts_seconds"].copy(),
                    "summary": read_json(output_root / sequence / camera / "summary.json"),
                }
            )
    if not payloads:
        return {}
    frame_count = min(len(item["target"]) for item in payloads)
    visibility = np.stack([item["target"][:frame_count] >= 0 for item in payloads], axis=1)
    visible_views = visibility.sum(axis=1)
    timestamp_stack = np.stack(
        [item["timestamp"][:frame_count] for item in payloads], axis=1
    )
    timestamp_spread = np.nanmax(timestamp_stack, axis=1) - np.nanmin(
        timestamp_stack, axis=1
    )
    patterns = {str(count): int((visible_views == count).sum()) for count in range(len(payloads) + 1)}
    summary = {
        "sequence": sequence,
        "camera_count": len(payloads),
        "aligned_frame_count": frame_count,
        "visible_view_count_histogram": patterns,
        "all_views_target_visible_count": int((visible_views == len(payloads)).sum()),
        "partial_visibility_count": int(((visible_views > 0) & (visible_views < len(payloads))).sum()),
        "all_views_unavailable_count": int((visible_views == 0).sum()),
        "any_view_ambiguous_count": int(
            np.stack([item["ambiguous"][:frame_count] for item in payloads], axis=1).any(axis=1).sum()
        ),
        "identity_switch_count": int(
            sum(item["summary"]["identity_switch_count"] for item in payloads)
        ),
        "per_camera_identity_switch_count": {
            item["camera"]: int(item["summary"]["identity_switch_count"])
            for item in payloads
        },
        "per_camera_target_visibility_transition_count": {
            item["camera"]: int(item["summary"]["target_visibility_transition_count"])
            for item in payloads
        },
        "geometry_interface": str(Path("outputs/background_ba") / sequence / "cameras_refined.json"),
        "alignment_interface": "PTS-derived timestamps plus refined camera geometry",
        "timestamp_spread_seconds_p95": float(np.nanpercentile(timestamp_spread, 95)),
        "timestamp_spread_seconds_max": float(np.nanmax(timestamp_spread)),
        "triangulation_performed": False,
        "status": (
            "PASS"
            if (visible_views == len(payloads)).all()
            and all(item["summary"]["identity_switch_count"] == 0 for item in payloads)
            else "REVIEW"
        ),
    }
    path = output_root / sequence / "cross_view_summary.json"
    atomic_write_text(path, json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if args.max_gap < 1 or not 0.0 <= args.association_threshold <= 1.0:
        raise RuntimeError("invalid tracking thresholds")
    dataset_root = args.dataset_root.expanduser().resolve()
    detections_root = args.detections_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runtime_dir = args.runtime_dir.expanduser().resolve()
    summaries: list[dict[str, Any]] = []
    skipped: list[str] = []
    for sequence in args.sequences:
        for camera in args.cameras:
            detection_path = detections_root / sequence / camera / "bboxes.npz"
            if not detection_path.exists() and args.allow_incomplete:
                skipped.append(f"{sequence}/{camera}")
                continue
            summary = select_camera(
                dataset_root, detections_root, output_root, sequence, camera, args
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    cross_view = [
        summary
        for sequence in args.sequences
        if (summary := cross_view_summary(output_root, sequence, args.cameras))
    ]
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if summaries:
        atomic_write_csv(
            runtime_dir / "target_selection_pilot.csv",
            sorted({key for row in summaries for key in row}),
            summaries,
        )
    result = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "camera_results": len(summaries),
        "skipped_incomplete": skipped,
        "frame_count": int(sum(row["frame_count"] for row in summaries)),
        "total_person_candidates": int(sum(row["total_person_candidates"] for row in summaries)),
        "all_detections_sapiens_crops": int(sum(row["all_detections_sapiens_crops"] for row in summaries)),
        "target_only_sapiens_crops": int(sum(row["target_only_sapiens_crops"] for row in summaries)),
        "target_ambiguous_count": int(sum(row["target_ambiguous_count"] for row in summaries)),
        "no_target_count": int(sum(row["no_target_count"] for row in summaries)),
        "occlusion_risk_count": int(sum(row["occlusion_risk_count"] for row in summaries)),
        "identity_switch_count": int(sum(row["identity_switch_count"] for row in summaries)),
        "cross_view": cross_view,
    }
    result["crop_reduction_fraction"] = float(
        1.0
        - result["target_only_sapiens_crops"]
        / max(result["all_detections_sapiens_crops"], 1)
    )
    atomic_write_text(runtime_dir / "target_selection_summary.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
