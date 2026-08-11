#!/usr/bin/env python3
"""Benchmark and run Sapiens2-5B on accepted primary-target crops only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sapiens2_pose_pipeline import (
    CAMERAS,
    PILOT_SEQUENCES,
    GPUMonitor,
    PrefetchLoader,
    Sapiens2BatchEngine,
    StageTimer,
    add_common_model_args,
    atomic_savez,
    atomic_write_csv,
    atomic_write_text,
    chunks,
    decode_image,
    evenly_spaced,
    ffprobe_pts,
    list_images,
    parse_int_list,
    parse_str_list,
    resolve_frame_dir,
    resolve_video,
    summarize_samples,
    utc_now,
)


def add_target_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser("benchmark", help="target-only batch scaling")
    add_common_model_args(benchmark)
    add_target_sources(benchmark)
    benchmark.add_argument("--sequence", default="barbellrow_0000")
    benchmark.add_argument("--camera", choices=CAMERAS, default="cam1")
    benchmark.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8, 12, 16])
    benchmark.add_argument("--sample-count", type=int, default=16)
    benchmark.add_argument("--warmup-count", type=int, default=1)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument(
        "--all-detections-root",
        type=Path,
        help="optional completed baseline for selected-candidate pose equivalence",
    )
    benchmark.add_argument("--equivalence-xy-atol", type=float, default=0.5)
    benchmark.add_argument("--equivalence-score-atol", type=float, default=0.005)

    infer = commands.add_parser("infer", help="resumable target-only pilot inference")
    add_common_model_args(infer)
    add_target_sources(infer)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument("--runtime-dir", type=Path, required=True)
    infer.add_argument("--sequences", type=parse_str_list, default=list(PILOT_SEQUENCES))
    infer.add_argument("--cameras", type=parse_str_list, default=list(CAMERAS))
    infer.add_argument("--batch-size", type=int, required=True)
    infer.add_argument("--chunk-size", type=int, default=256)
    infer.add_argument("--loader-workers", type=int, default=8)
    infer.add_argument("--prefetch-batches", type=int, default=4)
    infer.add_argument("--retry-failures", type=int, default=1)
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--save-overlays", type=int, default=6)

    materialize = commands.add_parser(
        "materialize-baseline",
        help="losslessly select accepted targets from a completed all-detections baseline",
    )
    add_target_sources(materialize)
    materialize.add_argument("--all-detections-root", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    materialize.add_argument("--runtime-dir", type=Path, required=True)
    materialize.add_argument(
        "--sequences", type=parse_str_list, default=list(PILOT_SEQUENCES)
    )
    materialize.add_argument("--cameras", type=parse_str_list, default=list(CAMERAS))
    materialize.add_argument("--chunk-size", type=int, default=256)
    materialize.set_defaults(batch_size=0)

    verify = commands.add_parser("verify", help="verify target-only output schema")
    add_target_sources(verify)
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument(
        "--all-detections-root",
        type=Path,
        help="optional ALL_DETECTIONS_BASELINE output for selected-pose equivalence",
    )
    verify.add_argument("--equivalence-xy-atol", type=float, default=0.5)
    verify.add_argument("--equivalence-score-atol", type=float, default=0.005)
    verify.add_argument("--report", type=Path)
    verify.add_argument("--sequences", type=parse_str_list, default=list(PILOT_SEQUENCES))
    verify.add_argument("--cameras", type=parse_str_list, default=list(CAMERAS))
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TargetSelection:
    def __init__(self, path: Path, frame_paths: Sequence[Path]) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "frame_index",
                "frame_name",
                "num_person_candidates",
                "target_candidate_index",
                "target_selection_confidence",
                "target_ambiguous",
                "no_target",
                "target_status",
                "occlusion_risk",
                "candidate_offsets",
                "all_person_detections_xyxy",
                "all_person_detection_scores",
            }
            missing = sorted(required - set(payload.files))
            if missing:
                raise RuntimeError(f"{path}: missing {', '.join(missing)}")
            for key in required:
                setattr(self, key, payload[key].copy())
        expected_index = np.arange(len(frame_paths), dtype=np.int32)
        expected_name = np.asarray([item.name for item in frame_paths])
        if not np.array_equal(self.frame_index, expected_index):
            raise RuntimeError(f"{path}: frame indices do not match source")
        if not np.array_equal(self.frame_name, expected_name):
            raise RuntimeError(f"{path}: frame names do not match source")
        if len(self.candidate_offsets) != len(frame_paths) + 1:
            raise RuntimeError(f"{path}: invalid candidate offsets")
        if int(self.candidate_offsets[-1]) != len(self.all_person_detections_xyxy):
            raise RuntimeError(f"{path}: candidate payload length mismatch")
        accepted = self.target_candidate_index >= 0
        if np.any(accepted & (self.target_ambiguous | self.no_target)):
            raise RuntimeError(f"{path}: ambiguous/no-target frame was force-selected")
        if np.any(self.target_candidate_index[accepted] >= self.num_person_candidates[accepted]):
            raise RuntimeError(f"{path}: selected candidate index is out of bounds")

    def detection(self, frame: int) -> tuple[np.ndarray, np.ndarray, bool]:
        candidate = int(self.target_candidate_index[frame])
        if candidate < 0:
            raise ValueError(f"frame {frame} is not target-eligible")
        absolute = int(self.candidate_offsets[frame]) + candidate
        return (
            self.all_person_detections_xyxy[absolute : absolute + 1].astype(np.float32),
            self.all_person_detection_scores[absolute : absolute + 1].astype(np.float32),
            False,
        )


def selected_sample(
    dataset_root: Path,
    selection_root: Path,
    sequence: str,
    camera: str,
    count: int,
) -> tuple[list[Path], list[int], TargetSelection]:
    paths = list_images(resolve_frame_dir(dataset_root, sequence, camera))
    selection = TargetSelection(
        selection_root / sequence / camera / "target_selection.npz", paths
    )
    eligible = np.flatnonzero(selection.target_candidate_index >= 0)
    if not len(eligible):
        raise RuntimeError(f"no accepted target frames in {sequence}/{camera}")
    indices = evenly_spaced([Path(str(index)) for index in eligible], count)
    frame_indices = [int(item.name) for item in indices]
    return [paths[index] for index in frame_indices], frame_indices, selection


def run_target_batch(
    engine: Sapiens2BatchEngine,
    paths: Sequence[Path],
    frame_indices: Sequence[int],
    selection: TargetSelection,
    batch_size: int,
    monitor: GPUMonitor,
    serialize_dir: Path | None = None,
) -> tuple[dict[str, np.ndarray], StageTimer, float]:
    timer = StageTimer()
    points: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    started = time.perf_counter()
    for batch_index, start in enumerate(range(0, len(paths), batch_size)):
        path_batch = paths[start : start + batch_size]
        index_batch = frame_indices[start : start + batch_size]
        monitor.set_phase("image_load", len(path_batch))
        with timer.measure("image_load"):
            images = [decode_image(path) for path in path_batch]
        monitor.set_phase("detector", len(path_batch))
        engine.detect(images, timer)
        detections = [selection.detection(index) for index in index_batch]
        predictions = engine.process_detections(
            path_batch,
            images,
            detections,
            batch_size,
            timer=timer,
            monitor=monitor,
        )
        batch_points = np.stack([item.keypoints_xy[0] for item in predictions])
        batch_scores = np.stack([item.confidence[0] for item in predictions])
        points.append(batch_points)
        scores.append(batch_scores)
        if serialize_dir is not None:
            monitor.set_phase("serialization", len(path_batch))
            with timer.measure("serialization"):
                atomic_savez(
                    serialize_dir / f"batch_{batch_index:04d}.npz",
                    keypoints_xy=batch_points,
                    confidence=batch_scores,
                )
    engine.torch.cuda.synchronize(engine.device)
    return {
        "keypoints_xy": np.concatenate(points),
        "confidence": np.concatenate(scores),
    }, timer, time.perf_counter() - started


def all_detections_reference(
    root: Path,
    sequence: str,
    camera: str,
    frame_indices: Sequence[int],
    selection: TargetSelection,
) -> dict[str, np.ndarray]:
    output_dir = root / sequence / camera
    with np.load(output_dir / "poses_2d.npz", allow_pickle=False) as pose, np.load(
        output_dir / "bboxes.npz", allow_pickle=False
    ) as bbox:
        if not np.array_equal(pose["instance_offsets"], selection.candidate_offsets):
            raise RuntimeError("baseline and target selector candidate offsets differ")
        absolute = np.asarray(
            [
                int(selection.candidate_offsets[frame])
                + int(selection.target_candidate_index[frame])
                for frame in frame_indices
            ],
            dtype=np.int64,
        )
        selected_boxes = np.stack(
            [selection.detection(frame)[0][0] for frame in frame_indices]
        )
        baseline_boxes = bbox["all_bboxes_xyxy"][absolute]
        if not np.allclose(selected_boxes, baseline_boxes, rtol=0.0, atol=1e-5):
            raise RuntimeError("baseline and target selector boxes differ")
        return {
            "keypoints_xy": pose["all_keypoints_xy"][absolute].copy(),
            "confidence": pose["all_confidence"][absolute].copy(),
        }


def equivalence_metrics(
    arrays: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, float]:
    delta = np.linalg.norm(
        arrays["keypoints_xy"] - reference["keypoints_xy"], axis=-1
    )
    confident = np.minimum(arrays["confidence"], reference["confidence"]) >= 0.3
    return {
        "max_xy": float(np.max(delta)),
        "p95_xy": float(np.percentile(delta, 95)),
        "max_confident_xy": float(np.max(delta[confident])) if confident.any() else 0.0,
        "max_confidence": float(
            np.max(np.abs(arrays["confidence"] - reference["confidence"]))
        ),
    }


def benchmark_command(args: argparse.Namespace) -> int:
    if args.sample_count < 1 or args.warmup_count < 0 or 1 not in args.batch_sizes:
        raise RuntimeError("valid samples/warmup and batch 1 reference are required")
    args.batch_sizes = sorted(set(args.batch_sizes))
    dataset_root = args.dataset_root.expanduser().resolve()
    selection_root = args.selection_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, frame_indices, selection = selected_sample(
        dataset_root, selection_root, args.sequence, args.camera, args.sample_count
    )
    baseline_reference = (
        all_detections_reference(
            args.all_detections_root.expanduser().resolve(),
            args.sequence,
            args.camera,
            frame_indices,
            selection,
        )
        if args.all_detections_root is not None
        else None
    )
    selection_summary = json.loads(
        (selection_root / args.sequence / args.camera / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    selector_seconds_per_frame = float(
        selection_summary["selection_elapsed_seconds"]
    ) / max(int(selection_summary["frame_count"]), 1)
    engine = Sapiens2BatchEngine(args)
    monitor_path = output_dir / "target_only_gpu_utilization.csv"
    monitor = GPUMonitor(monitor_path, args.gpu_sample_interval, engine.device_index)
    monitor.start()
    records: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    reference: dict[str, np.ndarray] | None = None
    try:
        for batch_size in args.batch_sizes:
            engine.torch.cuda.empty_cache()
            engine.reset_peak_memory()
            try:
                warmup_count = min(batch_size, len(paths))
                for _ in range(args.warmup_count):
                    warmup_paths = paths[:warmup_count]
                    warmup_indices = frame_indices[:warmup_count]
                    warmup_images = [decode_image(path) for path in warmup_paths]
                    monitor.set_phase("detector", len(warmup_paths))
                    engine.detect(warmup_images)
                    engine.process_detections(
                        warmup_paths,
                        warmup_images,
                        [selection.detection(index) for index in warmup_indices],
                        batch_size,
                        monitor=monitor,
                    )
                with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
                    arrays, timer, elapsed = run_target_batch(
                        engine,
                        paths,
                        frame_indices,
                        selection,
                        batch_size,
                        monitor,
                        Path(temporary),
                    )
                if reference is None:
                    reference = arrays
                    batch_metrics = {
                        "max_xy": 0.0,
                        "p95_xy": 0.0,
                        "max_confident_xy": 0.0,
                        "max_confidence": 0.0,
                    }
                else:
                    batch_metrics = equivalence_metrics(arrays, reference)
                baseline_metrics = (
                    equivalence_metrics(arrays, baseline_reference)
                    if baseline_reference is not None
                    else {
                        "max_xy": 0.0,
                        "p95_xy": 0.0,
                        "max_confident_xy": 0.0,
                        "max_confidence": 0.0,
                    }
                )
                batch_equivalent = (
                    batch_metrics["max_confident_xy"] <= args.equivalence_xy_atol
                    and batch_metrics["max_confidence"]
                    <= args.equivalence_score_atol
                )
                baseline_equivalent = (
                    baseline_reference is None
                    or (
                        baseline_metrics["max_confident_xy"]
                        <= args.equivalence_xy_atol
                        and baseline_metrics["max_confidence"]
                        <= args.equivalence_score_atol
                    )
                )
                equivalent = batch_equivalent and baseline_equivalent
                allocated, reserved = engine.memory()
                record = {
                    "batch_size": batch_size,
                    "status": "PASS" if equivalent else "NON_EQUIVALENT",
                    "images": len(paths),
                    "person_crops": len(paths),
                    "elapsed_seconds": elapsed,
                    "images_per_second": len(paths) / elapsed,
                    "person_crops_per_second": len(paths) / elapsed,
                    "effective_seconds_per_image": elapsed / len(paths),
                    "image_load_seconds": timer.total("image_load"),
                    "detector_preprocess_seconds": timer.total("detector_preprocess"),
                    "detector_forward_seconds": timer.total("detector_forward"),
                    "detector_postprocess_seconds": timer.total("detector_postprocess"),
                    "target_selection_seconds_estimate": selector_seconds_per_frame
                    * len(paths),
                    "target_selection_timing_included": False,
                    "crop_preprocess_seconds": timer.total("crop_preprocess"),
                    "host_to_device_preprocess_seconds": timer.total("host_to_device_preprocess"),
                    "pose_forward_seconds": timer.total("pose_forward"),
                    "flip_forward_seconds": timer.total("flip_forward"),
                    "heatmap_transfer_seconds": timer.total("heatmap_transfer"),
                    "postprocess_seconds": timer.total("postprocess"),
                    "serialization_seconds": timer.total("serialization"),
                    "detector_images_per_second": len(paths)
                    / max(timer.total("detector_forward"), 1e-12),
                    "pose_person_crops_per_second": len(paths)
                    / max(
                        timer.total("pose_forward") + timer.total("flip_forward"),
                        1e-12,
                    ),
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "max_xy_delta_vs_batch1_px": batch_metrics["max_xy"],
                    "p95_xy_delta_vs_batch1_px": batch_metrics["p95_xy"],
                    "max_confident_xy_delta_vs_batch1_px": batch_metrics[
                        "max_confident_xy"
                    ],
                    "max_confidence_delta_vs_batch1": batch_metrics[
                        "max_confidence"
                    ],
                    "max_confident_xy_delta_vs_all_detections_baseline_px": baseline_metrics[
                        "max_confident_xy"
                    ],
                    "max_confidence_delta_vs_all_detections_baseline": baseline_metrics[
                        "max_confidence"
                    ],
                    "batch_numerically_equivalent": batch_equivalent,
                    "all_detections_baseline_equivalent": baseline_equivalent,
                    "numerically_equivalent": equivalent,
                    "error": "",
                }
                for stage, values in timer.values.items():
                    profile_rows.append(
                        {
                            "batch_size": batch_size,
                            "stage": stage,
                            "measurement_count": len(values),
                            "total_seconds": float(sum(values)),
                            "mean_seconds": float(np.mean(values)),
                            "median_seconds": float(np.median(values)),
                            "p90_seconds": float(np.percentile(values, 90)),
                            "seconds_per_image": float(sum(values) / len(paths)),
                        }
                    )
            except (engine.torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if not (
                    isinstance(exc, engine.torch.cuda.OutOfMemoryError)
                    or "out of memory" in str(exc).lower()
                ):
                    raise
                engine.torch.cuda.empty_cache()
                allocated, reserved = engine.memory()
                record = {
                    "batch_size": batch_size,
                    "status": "OOM",
                    "images": len(paths),
                    "person_crops": "",
                    "peak_allocated_bytes": allocated,
                    "peak_reserved_bytes": reserved,
                    "numerically_equivalent": False,
                    "error": str(exc).splitlines()[0][:500],
                }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        monitor.stop()
    gpu_rows = read_csv(monitor_path)
    for record in records:
        if record["status"] != "PASS":
            continue
        for phase in ("detector", "pose"):
            for key, value in summarize_samples(
                gpu_rows, phase, int(record["batch_size"])
            ).items():
                record[f"{phase}_{key}"] = value
    atomic_write_csv(
        output_dir / "target_only_batch_scaling.csv",
        sorted({key for row in records for key in row}),
        records,
    )
    atomic_write_csv(
        output_dir / "target_only_stage_profile.csv",
        (
            "batch_size",
            "stage",
            "measurement_count",
            "total_seconds",
            "mean_seconds",
            "median_seconds",
            "p90_seconds",
            "seconds_per_image",
        ),
        profile_rows,
    )
    passing = [row for row in records if row["status"] == "PASS"]
    if not passing:
        raise RuntimeError("no target-only batch passed")
    best = max(passing, key=lambda row: float(row["images_per_second"]))
    plateau = [
        row
        for row in passing
        if float(row["images_per_second"])
        >= 0.99 * float(best["images_per_second"])
    ]
    recommended = min(plateau, key=lambda row: int(row["batch_size"]))
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workload": "TARGET_ONLY",
        "sequence": args.sequence,
        "camera": args.camera,
        "sample_count": len(paths),
        "frame_indices": frame_indices,
        "candidate_source": "cached official facebook/detr-resnet-101-dc5 target selection; official DETR rerun for timing",
        "target_selector": "bidirectional sequence-level temporal tracker",
        "detector_timing_included": True,
        "target_selection_timing_included": False,
        "target_selection_seconds_per_frame": selector_seconds_per_frame,
        "all_detections_baseline_equivalence_checked": baseline_reference is not None,
        "model": "facebook/sapiens2-pose-5b",
        "pose_input_hw": [1024, 768],
        "precision": "float32",
        "flip_test": True,
        "best_batch_size": int(best["batch_size"]),
        "best_images_per_second": float(best["images_per_second"]),
        "best_person_crops_per_second": float(best["person_crops_per_second"]),
        "plateau_threshold_fraction_of_fastest": 0.99,
        "recommended_batch_size": int(recommended["batch_size"]),
        "recommended_images_per_second": float(recommended["images_per_second"]),
        "all_stable_batches_equivalent": all(bool(row["numerically_equivalent"]) for row in passing),
        "records": records,
    }
    atomic_write_text(
        output_dir / "target_only_benchmark_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def target_chunk_path(output_dir: Path, start: int, end: int) -> Path:
    return output_dir / "chunks" / f"chunk_{start:06d}_{end:06d}.npz"


def save_target_chunk(
    path: Path,
    frame_paths: Sequence[Path],
    frame_start: int,
    selection: TargetSelection,
    predictions: dict[int, Any],
) -> None:
    count = len(frame_paths)
    xy = np.full((count, 308, 2), np.nan, dtype=np.float32)
    confidence = np.full((count, 308), np.nan, dtype=np.float32)
    bbox = np.full((count, 4), np.nan, dtype=np.float32)
    bbox_score = np.full(count, np.nan, dtype=np.float32)
    for local, prediction in predictions.items():
        xy[local] = prediction.keypoints_xy[0]
        confidence[local] = prediction.confidence[0]
        bbox[local] = prediction.bboxes[0]
        bbox_score[local] = prediction.bbox_scores[0]
    indices = np.arange(frame_start, frame_start + count, dtype=np.int32)
    atomic_savez(
        path,
        keypoints_xy=xy,
        confidence=confidence,
        valid_mask=np.isfinite(xy).all(axis=-1) & np.isfinite(confidence),
        bbox_xyxy=bbox,
        bbox_score=bbox_score,
        target_present=np.asarray([local in predictions for local in range(count)], dtype=np.bool_),
        frame_index=indices,
        frame_name=np.asarray([item.name for item in frame_paths]),
        num_person_candidates=selection.num_person_candidates[indices],
        target_candidate_index=selection.target_candidate_index[indices],
        target_selection_confidence=selection.target_selection_confidence[indices],
        target_ambiguous=selection.target_ambiguous[indices],
        no_target=selection.no_target[indices],
        target_status=selection.target_status[indices],
        occlusion_risk=selection.occlusion_risk[indices],
    )


def materialize_baseline_chunk(
    path: Path,
    frame_paths: Sequence[Path],
    frame_start: int,
    selection: TargetSelection,
    all_keypoints_xy: np.ndarray,
    all_confidence: np.ndarray,
    all_bboxes_xyxy: np.ndarray,
    all_bbox_scores: np.ndarray,
) -> None:
    count = len(frame_paths)
    indices = np.arange(frame_start, frame_start + count, dtype=np.int32)
    xy = np.full((count, 308, 2), np.nan, dtype=np.float32)
    confidence = np.full((count, 308), np.nan, dtype=np.float32)
    bbox = np.full((count, 4), np.nan, dtype=np.float32)
    bbox_score = np.full(count, np.nan, dtype=np.float32)
    present = selection.target_candidate_index[indices] >= 0
    for local in np.flatnonzero(present):
        frame = int(indices[local])
        absolute = (
            int(selection.candidate_offsets[frame])
            + int(selection.target_candidate_index[frame])
        )
        xy[local] = all_keypoints_xy[absolute]
        confidence[local] = all_confidence[absolute]
        bbox[local] = all_bboxes_xyxy[absolute]
        bbox_score[local] = all_bbox_scores[absolute]
    atomic_savez(
        path,
        keypoints_xy=xy,
        confidence=confidence,
        valid_mask=np.isfinite(xy).all(axis=-1) & np.isfinite(confidence),
        bbox_xyxy=bbox,
        bbox_score=bbox_score,
        target_present=present,
        frame_index=indices,
        frame_name=np.asarray([item.name for item in frame_paths]),
        num_person_candidates=selection.num_person_candidates[indices],
        target_candidate_index=selection.target_candidate_index[indices],
        target_selection_confidence=selection.target_selection_confidence[indices],
        target_ambiguous=selection.target_ambiguous[indices],
        no_target=selection.no_target[indices],
        target_status=selection.target_status[indices],
        occlusion_risk=selection.occlusion_risk[indices],
    )


TARGET_ARRAY_KEYS = (
    "keypoints_xy",
    "confidence",
    "valid_mask",
    "bbox_xyxy",
    "bbox_score",
    "target_present",
    "frame_index",
    "frame_name",
    "num_person_candidates",
    "target_candidate_index",
    "target_selection_confidence",
    "target_ambiguous",
    "no_target",
    "target_status",
    "occlusion_risk",
)


def target_chunk_matches_selection(
    payload: Any,
    frame_paths: Sequence[Path],
    frame_start: int,
    selection: TargetSelection,
) -> bool:
    """Return whether a resumable chunk is tied to the current selector output.

    Frame names alone are insufficient: a rerun may preserve the same source frames
    while changing abstention or target-candidate decisions.  Reusing such a chunk
    would silently mix identities.  Compare the selector fields and the selected
    detection payload before accepting a resume hit.
    """
    count = len(frame_paths)
    indices = np.arange(frame_start, frame_start + count, dtype=np.int32)
    required = {
        "frame_index",
        "frame_name",
        "target_present",
        "num_person_candidates",
        "target_candidate_index",
        "target_selection_confidence",
        "target_ambiguous",
        "no_target",
        "target_status",
        "occlusion_risk",
        "bbox_xyxy",
        "bbox_score",
    }
    if not required.issubset(payload.files):
        return False
    exact = {
        "frame_index": indices,
        "frame_name": np.asarray([item.name for item in frame_paths]),
        "target_present": selection.target_candidate_index[indices] >= 0,
        "num_person_candidates": selection.num_person_candidates[indices],
        "target_candidate_index": selection.target_candidate_index[indices],
        "target_ambiguous": selection.target_ambiguous[indices],
        "no_target": selection.no_target[indices],
        "target_status": selection.target_status[indices],
        "occlusion_risk": selection.occlusion_risk[indices],
    }
    if any(not np.array_equal(payload[key], value) for key, value in exact.items()):
        return False
    if not np.allclose(
        payload["target_selection_confidence"],
        selection.target_selection_confidence[indices],
        rtol=0.0,
        atol=1e-7,
        equal_nan=True,
    ):
        return False
    present = exact["target_present"]
    expected_bbox = np.full((count, 4), np.nan, dtype=np.float32)
    expected_score = np.full(count, np.nan, dtype=np.float32)
    for local in np.flatnonzero(present):
        detection_bbox, detection_score, _ = selection.detection(int(indices[local]))
        expected_bbox[local] = detection_bbox[0]
        expected_score[local] = detection_score[0]
    return bool(
        np.allclose(
            payload["bbox_xyxy"], expected_bbox, rtol=0.0, atol=1e-4, equal_nan=True
        )
        and np.allclose(
            payload["bbox_score"], expected_score, rtol=0.0, atol=1e-6, equal_nan=True
        )
    )


def consolidate_target_camera(
    output_dir: Path,
    frame_paths: Sequence[Path],
    selection_path: Path,
    sequence: str,
    camera: str,
    engine: Sapiens2BatchEngine,
    args: argparse.Namespace,
) -> dict[str, Any]:
    parts: dict[str, list[np.ndarray]] = {key: [] for key in TARGET_ARRAY_KEYS}
    for start in range(0, len(frame_paths), args.chunk_size):
        end = min(start + args.chunk_size, len(frame_paths))
        path = target_chunk_path(output_dir, start, end)
        if not path.exists():
            raise RuntimeError(f"missing target-only chunk: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(payload["frame_index"], np.arange(start, end, dtype=np.int32)):
                raise RuntimeError(f"invalid target-only chunk indices: {path}")
            if not np.array_equal(payload["frame_name"], np.asarray([item.name for item in frame_paths[start:end]])):
                raise RuntimeError(f"invalid target-only chunk names: {path}")
            for key in TARGET_ARRAY_KEYS:
                parts[key].append(payload[key])
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    pts = ffprobe_pts(resolve_video(args.dataset_root.expanduser().resolve(), sequence, camera))
    timestamps = np.full(len(frame_paths), np.nan, dtype=np.float64)
    timestamps[: min(len(pts), len(frame_paths))] = pts[: min(len(pts), len(frame_paths))]
    atomic_savez(
        output_dir / "poses_2d.npz",
        keypoints_xy=arrays["keypoints_xy"],
        confidence=arrays["confidence"],
        valid_mask=arrays["valid_mask"],
        target_present=arrays["target_present"],
        frame_index=arrays["frame_index"],
        timestamp_pts_seconds=timestamps,
    )
    atomic_savez(
        output_dir / "bboxes.npz",
        bbox_xyxy=arrays["bbox_xyxy"],
        bbox_score=arrays["bbox_score"],
        num_person_candidates=arrays["num_person_candidates"],
        target_candidate_index=arrays["target_candidate_index"],
        target_selection_confidence=arrays["target_selection_confidence"],
        target_ambiguous=arrays["target_ambiguous"],
        no_target=arrays["no_target"],
        target_status=arrays["target_status"],
        occlusion_risk=arrays["occlusion_risk"],
    )
    rows = [
        {
            "frame_index": frame,
            "frame_name": frame_paths[frame].name,
            "timestamp_pts_seconds": f"{timestamps[frame]:.9f}" if np.isfinite(timestamps[frame]) else "",
            "target_status": str(arrays["target_status"][frame]),
            "target_selection_confidence": f"{arrays['target_selection_confidence'][frame]:.6f}",
            "target_ambiguous": bool(arrays["target_ambiguous"][frame]),
            "no_target": bool(arrays["no_target"][frame]),
            "occlusion_risk": bool(arrays["occlusion_risk"][frame]),
        }
        for frame in range(len(frame_paths))
    ]
    atomic_write_csv(output_dir / "frames.csv", list(rows[0]), rows)
    present = arrays["target_present"]
    qa = {
        "sequence": sequence,
        "camera": camera,
        "frame_count": len(frame_paths),
        "target_pose_count": int(present.sum()),
        "target_ambiguous_count": int(arrays["target_ambiguous"].sum()),
        "no_target_count": int(arrays["no_target"].sum()),
        "occlusion_risk_count": int(arrays["occlusion_risk"].sum()),
        "finite_selected_pose_fraction": float(np.isfinite(arrays["keypoints_xy"][present]).all(axis=-1).mean()) if present.any() else 0.0,
        "forced_ambiguous_pose_count": int((present & arrays["target_ambiguous"]).sum()),
        "forced_no_target_pose_count": int((present & arrays["no_target"]).sum()),
        "timestamp_coverage": float(np.isfinite(timestamps).mean()),
    }
    qa["status"] = "PASS" if (
        qa["target_pose_count"] > 0
        and qa["finite_selected_pose_fraction"] == 1.0
        and qa["forced_ambiguous_pose_count"] == 0
        and qa["forced_no_target_pose_count"] == 0
        and qa["timestamp_coverage"] == 1.0
    ) else "FAIL"
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": sequence,
        "camera": camera,
        "workload": "TARGET_ONLY",
        "target_selection_file": str(selection_path),
        "all_person_detection_payload_retained_in": str(selection_path),
        "model": "facebook/sapiens2-pose-5b",
        "detector": "facebook/detr-resnet-101-dc5 (cached official candidates)",
        "keypoint_count": 308,
        "keypoint_names": engine.keypoint_id2name,
        "pose_input_hw": [1024, 768],
        "precision": "float32",
        "flip_test": True,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "ambiguous_policy": "do not run Sapiens2; preserve NaN pose and status",
        "no_target_policy": "do not run Sapiens2; preserve NaN pose and status",
        "qa": qa,
    }
    atomic_write_text(output_dir / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return qa


def save_pose_overlays(
    output_dir: Path,
    frame_paths: Sequence[Path],
    count: int,
    engine: Sapiens2BatchEngine,
) -> None:
    if count < 1:
        return
    import cv2

    tools_dir = engine.pose_root / "tools" / "vis"
    os.sys.path.insert(0, str(tools_dir))
    from pose_render_utils import visualize_keypoints

    with np.load(output_dir / "poses_2d.npz", allow_pickle=False) as payload:
        xy = payload["keypoints_xy"]
        confidence = payload["confidence"]
        present = payload["target_present"]
    eligible = np.flatnonzero(present)
    if not len(eligible):
        return
    selected = np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int)
    overlay_dir = output_dir / "debug" / "target_pose_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for position in selected:
        frame = int(eligible[position])
        image = decode_image(frame_paths[frame])
        rendered = visualize_keypoints(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            keypoints=[xy[frame]],
            keypoints_visible=[np.ones(308, dtype=np.bool_)],
            keypoint_scores=[confidence[frame]],
            radius=3,
            thickness=2,
            kpt_thr=0.3,
            skeleton=engine.model.pose_metainfo["skeleton_links"],
            kpt_color=engine.model.pose_metainfo["keypoint_colors"],
            link_color=engine.model.pose_metainfo["skeleton_link_colors"],
        )
        cv2.imwrite(
            str(overlay_dir / f"frame_{frame:06d}.jpg"),
            cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR),
        )


def infer_camera(
    engine: Sapiens2BatchEngine,
    args: argparse.Namespace,
    sequence: str,
    camera: str,
    monitor: GPUMonitor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_root = args.dataset_root.expanduser().resolve()
    selection_root = args.selection_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    frame_paths = list_images(resolve_frame_dir(dataset_root, sequence, camera))
    selection_path = selection_root / sequence / camera / "target_selection.npz"
    selection = TargetSelection(selection_path, frame_paths)
    output_dir = output_root / sequence / camera
    (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
    prior_metadata_path = output_dir / "metadata.json"
    prior_metadata = (
        json.loads(prior_metadata_path.read_text(encoding="utf-8"))
        if prior_metadata_path.exists()
        else {}
    )
    timer = StageTimer()
    skipped = retries = processed_crops = 0
    started = time.perf_counter()
    for chunk_start in range(0, len(frame_paths), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(frame_paths))
        chunk_path = target_chunk_path(output_dir, chunk_start, chunk_end)
        if chunk_path.exists() and not args.overwrite:
            with np.load(chunk_path, allow_pickle=False) as payload:
                if target_chunk_matches_selection(
                    payload,
                    frame_paths[chunk_start:chunk_end],
                    chunk_start,
                    selection,
                ):
                    skipped += int(payload["target_present"].sum())
                    continue
        eligible = [
            frame
            for frame in range(chunk_start, chunk_end)
            if selection.target_candidate_index[frame] >= 0
        ]
        selected_paths = [frame_paths[frame] for frame in eligible]
        frame_lookup = {path: frame for path, frame in zip(selected_paths, eligible)}
        predictions: dict[int, Any] = {}
        for path_batch, images, wait_seconds in PrefetchLoader(
            selected_paths,
            args.batch_size,
            args.loader_workers,
            args.prefetch_batches,
        ):
            timer.values.setdefault("image_load_wait", []).append(wait_seconds)
            indices = [frame_lookup[path] for path in path_batch]
            attempts = 0
            while True:
                try:
                    result = engine.process_detections(
                        path_batch,
                        images,
                        [selection.detection(frame) for frame in indices],
                        args.batch_size,
                        timer=timer,
                        monitor=monitor,
                    )
                    for frame, prediction in zip(indices, result):
                        predictions[frame - chunk_start] = prediction
                    break
                except Exception:
                    attempts += 1
                    retries += 1
                    if attempts > args.retry_failures:
                        raise
                    engine.torch.cuda.empty_cache()
        monitor.set_phase("serialization", len(predictions))
        with timer.measure("serialization"):
            save_target_chunk(
                chunk_path,
                frame_paths[chunk_start:chunk_end],
                chunk_start,
                selection,
                predictions,
            )
        processed_crops += len(predictions)
        print(
            f"[{sequence}/{camera}] target crops {processed_crops + skipped}/"
            f"{int((selection.target_candidate_index >= 0).sum())}; frames {chunk_end}/{len(frame_paths)}",
            flush=True,
        )
    qa = consolidate_target_camera(
        output_dir, frame_paths, selection_path, sequence, camera, engine, args
    )
    metadata = json.loads(prior_metadata_path.read_text(encoding="utf-8"))
    metadata["pose_inference_performed_in_this_stage"] = processed_crops > 0
    metadata["resume_skipped_target_crops"] = skipped
    if (
        processed_crops == 0
        and skipped == qa["target_pose_count"]
        and prior_metadata.get("materialization_source")
    ):
        for key in (
            "materialization_source",
            "materialization_equivalence",
            "baseline_repository_output",
        ):
            metadata[key] = prior_metadata[key]
        metadata["resume_provenance"] = (
            "all target chunks selection-bound and reused from lossless baseline gather"
        )
    elif skipped:
        metadata["resume_provenance"] = (
            "selection-bound existing chunks plus newly inferred target chunks"
        )
    atomic_write_text(
        prior_metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    save_pose_overlays(output_dir, frame_paths, args.save_overlays, engine)
    elapsed = time.perf_counter() - started
    metrics = {
        "sequence": sequence,
        "camera": camera,
        "frame_count": len(frame_paths),
        "target_pose_count": qa["target_pose_count"],
        "processed_target_crops": processed_crops,
        "elapsed_seconds": elapsed,
        "frames_per_second": len(frame_paths) / elapsed,
        "person_crops_per_second": processed_crops / max(elapsed, 1e-12),
        "resume_skipped_target_crops": skipped,
        "retry_count": retries,
        **{f"{key}_seconds": sum(value) for key, value in timer.values.items()},
        "status": qa["status"],
    }
    return qa, metrics


def infer_command(args: argparse.Namespace) -> int:
    if args.batch_size < 1 or args.chunk_size < args.batch_size:
        raise RuntimeError("batch size must be positive and not exceed chunk size")
    runtime_dir = args.runtime_dir.expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    engine = Sapiens2BatchEngine(args)
    engine.reset_peak_memory()
    monitor = GPUMonitor(
        runtime_dir / "target_only_pilot_gpu_utilization.csv",
        args.gpu_sample_interval,
        engine.device_index,
    )
    monitor.start()
    qa_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    try:
        for sequence in args.sequences:
            for camera in args.cameras:
                qa, metrics = infer_camera(engine, args, sequence, camera, monitor)
                qa_rows.append(qa)
                metrics_rows.append(metrics)
    finally:
        monitor.stop()
    allocated, reserved = engine.memory()
    atomic_write_csv(runtime_dir / "target_only_pilot_qa.csv", sorted({key for row in qa_rows for key in row}), qa_rows)
    atomic_write_csv(runtime_dir / "target_only_pilot_benchmark.csv", sorted({key for row in metrics_rows for key in row}), metrics_rows)
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workload": "TARGET_ONLY",
        "frame_count": int(sum(row["frame_count"] for row in metrics_rows)),
        "target_pose_count": int(sum(row["target_pose_count"] for row in metrics_rows)),
        "processed_target_crops": int(
            sum(row["processed_target_crops"] for row in metrics_rows)
        ),
        "resume_skipped_target_crops": int(
            sum(row["resume_skipped_target_crops"] for row in metrics_rows)
        ),
        "target_ambiguous_count": int(sum(row["target_ambiguous_count"] for row in qa_rows)),
        "no_target_count": int(sum(row["no_target_count"] for row in qa_rows)),
        "total_elapsed_seconds": float(sum(row["elapsed_seconds"] for row in metrics_rows)),
        "aggregate_frames_per_second": float(sum(row["frame_count"] for row in metrics_rows) / max(sum(row["elapsed_seconds"] for row in metrics_rows), 1e-12)),
        "aggregate_target_output_crops_per_second_including_resume": float(
            sum(row["target_pose_count"] for row in metrics_rows)
            / max(sum(row["elapsed_seconds"] for row in metrics_rows), 1e-12)
        ),
        "aggregate_new_person_crops_per_second": float(
            sum(row["processed_target_crops"] for row in metrics_rows)
            / max(sum(row["elapsed_seconds"] for row in metrics_rows), 1e-12)
        ),
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "pass_cameras": int(sum(row["status"] == "PASS" for row in qa_rows)),
        "fail_cameras": int(sum(row["status"] == "FAIL" for row in qa_rows)),
    }
    atomic_write_text(runtime_dir / "target_only_pilot_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["fail_cameras"] == 0 else 2


def materialize_baseline_command(args: argparse.Namespace) -> int:
    if args.chunk_size < 1:
        raise RuntimeError("chunk size must be positive")
    dataset_root = args.dataset_root.expanduser().resolve()
    selection_root = args.selection_root.expanduser().resolve()
    baseline_root = args.all_detections_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runtime_dir = args.runtime_dir.expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for sequence in args.sequences:
        for camera in args.cameras:
            started = time.perf_counter()
            frame_paths = list_images(resolve_frame_dir(dataset_root, sequence, camera))
            selection_path = (
                selection_root / sequence / camera / "target_selection.npz"
            )
            selection = TargetSelection(selection_path, frame_paths)
            baseline_dir = baseline_root / sequence / camera
            baseline_metadata = json.loads(
                (baseline_dir / "metadata.json").read_text(encoding="utf-8")
            )
            output_dir = output_root / sequence / camera
            (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
            with np.load(
                baseline_dir / "poses_2d.npz", allow_pickle=False
            ) as pose, np.load(
                baseline_dir / "bboxes.npz", allow_pickle=False
            ) as bbox:
                if not np.array_equal(
                    pose["instance_offsets"], selection.candidate_offsets
                ):
                    raise RuntimeError(
                        f"baseline candidate offsets differ: {sequence}/{camera}"
                    )
                if not np.array_equal(
                    bbox["instance_offsets"], selection.candidate_offsets
                ):
                    raise RuntimeError(
                        f"baseline bbox offsets differ: {sequence}/{camera}"
                    )
                if not np.allclose(
                    bbox["all_bboxes_xyxy"],
                    selection.all_person_detections_xyxy,
                    rtol=0.0,
                    atol=1e-5,
                ):
                    raise RuntimeError(
                        f"baseline candidate boxes differ: {sequence}/{camera}"
                    )
                if not np.allclose(
                    bbox["all_bbox_scores"],
                    selection.all_person_detection_scores,
                    rtol=0.0,
                    atol=1e-6,
                ):
                    raise RuntimeError(
                        f"baseline candidate scores differ: {sequence}/{camera}"
                    )
                for chunk_start in range(0, len(frame_paths), args.chunk_size):
                    chunk_end = min(chunk_start + args.chunk_size, len(frame_paths))
                    materialize_baseline_chunk(
                        target_chunk_path(output_dir, chunk_start, chunk_end),
                        frame_paths[chunk_start:chunk_end],
                        chunk_start,
                        selection,
                        pose["all_keypoints_xy"],
                        pose["all_confidence"],
                        bbox["all_bboxes_xyxy"],
                        bbox["all_bbox_scores"],
                    )

            class MetadataOnlyEngine:
                keypoint_id2name = baseline_metadata["keypoint_names"]

            qa = consolidate_target_camera(
                output_dir,
                frame_paths,
                selection_path,
                sequence,
                camera,
                MetadataOnlyEngine(),
                args,
            )
            metadata_path = output_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "pose_inference_performed_in_this_stage": False,
                    "materialization_source": "ALL_DETECTIONS_BASELINE",
                    "materialization_equivalence": "lossless accepted-candidate gather",
                    "baseline_repository_output": str(baseline_dir),
                    "batch_size": None,
                }
            )
            atomic_write_text(
                metadata_path,
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            )
            row = {
                **qa,
                "elapsed_seconds": time.perf_counter() - started,
                "materialization_source": "ALL_DETECTIONS_BASELINE",
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    atomic_write_csv(
        runtime_dir / "baseline_materialization_qa.csv",
        sorted({key for row in rows for key in row}),
        rows,
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workload": "TARGET_ONLY",
        "materialization_source": "ALL_DETECTIONS_BASELINE",
        "camera_count": len(rows),
        "frame_count": int(sum(row["frame_count"] for row in rows)),
        "target_pose_count": int(sum(row["target_pose_count"] for row in rows)),
        "pass_cameras": int(sum(row["status"] == "PASS" for row in rows)),
        "fail_cameras": int(sum(row["status"] == "FAIL" for row in rows)),
        "elapsed_seconds": time.perf_counter() - started_all,
        "new_pose_inference_performed": False,
    }
    atomic_write_text(
        runtime_dir / "baseline_materialization_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["fail_cameras"] == 0 else 2


def verify_command(args: argparse.Namespace) -> int:
    rows = []
    for sequence in args.sequences:
        for camera in args.cameras:
            frames = list_images(resolve_frame_dir(args.dataset_root.expanduser().resolve(), sequence, camera))
            selection = TargetSelection(
                args.selection_root.expanduser().resolve() / sequence / camera / "target_selection.npz",
                frames,
            )
            output_dir = args.output_root.expanduser().resolve() / sequence / camera
            errors = []
            max_confident_xy_delta = 0.0
            max_confidence_delta = 0.0
            try:
                with np.load(output_dir / "poses_2d.npz", allow_pickle=False) as pose, np.load(
                    output_dir / "bboxes.npz", allow_pickle=False
                ) as bbox:
                    present = pose["target_present"]
                    if pose["keypoints_xy"].shape != (len(frames), 308, 2):
                        errors.append("pose_shape")
                    if pose["confidence"].shape != (len(frames), 308):
                        errors.append("confidence_shape")
                    if not np.array_equal(present, selection.target_candidate_index >= 0):
                        errors.append("target_presence_mismatch")
                    if np.any(present & (bbox["target_ambiguous"] | bbox["no_target"])):
                        errors.append("forced_invalid_target")
                    if present.any() and not np.isfinite(pose["keypoints_xy"][present]).all():
                        errors.append("nonfinite_target_pose")
                    if (~present).any() and not np.isnan(pose["keypoints_xy"][~present]).all():
                        errors.append("invalid_frames_not_nan")
                    if args.all_detections_root is not None and present.any():
                        baseline_dir = (
                            args.all_detections_root.expanduser().resolve()
                            / sequence
                            / camera
                        )
                        with np.load(
                            baseline_dir / "poses_2d.npz", allow_pickle=False
                        ) as baseline_pose:
                            offsets = baseline_pose["instance_offsets"]
                            accepted_frames = np.flatnonzero(present)
                            absolute = np.asarray(
                                [
                                    int(offsets[frame])
                                    + int(selection.target_candidate_index[frame])
                                    for frame in accepted_frames
                                ],
                                dtype=np.int64,
                            )
                            baseline_xy = baseline_pose["all_keypoints_xy"][absolute]
                            baseline_confidence = baseline_pose["all_confidence"][absolute]
                            target_xy = pose["keypoints_xy"][accepted_frames]
                            target_confidence = pose["confidence"][accepted_frames]
                            delta = np.linalg.norm(target_xy - baseline_xy, axis=-1)
                            confident = np.minimum(
                                target_confidence, baseline_confidence
                            ) >= 0.3
                            max_confident_xy_delta = (
                                float(np.max(delta[confident])) if confident.any() else 0.0
                            )
                            max_confidence_delta = float(
                                np.max(np.abs(target_confidence - baseline_confidence))
                            )
                            if max_confident_xy_delta > args.equivalence_xy_atol:
                                errors.append("baseline_xy_non_equivalent")
                            if max_confidence_delta > args.equivalence_score_atol:
                                errors.append("baseline_confidence_non_equivalent")
            except (OSError, KeyError, ValueError) as exc:
                errors.append(str(exc))
            row = {
                "sequence": sequence,
                "camera": camera,
                "frame_count": len(frames),
                "target_pose_count": int((selection.target_candidate_index >= 0).sum()),
                "max_confident_xy_delta_vs_all_detections_px": max_confident_xy_delta,
                "max_confidence_delta_vs_all_detections": max_confidence_delta,
                "status": "PASS" if not errors else "FAIL",
                "errors": ";".join(errors),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    if args.report is not None and rows:
        atomic_write_csv(
            args.report.expanduser().resolve(),
            list(rows[0]),
            rows,
        )
    return 0 if all(row["status"] == "PASS" for row in rows) else 2


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "benchmark":
        return benchmark_command(args)
    if args.command == "infer":
        return infer_command(args)
    if args.command == "materialize-baseline":
        return materialize_baseline_command(args)
    if args.command == "verify":
        return verify_command(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
