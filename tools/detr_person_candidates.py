#!/usr/bin/env python3
"""Resumable official DETR person-candidate extraction without pose inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sapiens2_pose_pipeline import (
    CAMERAS,
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_SAPIENS2_ROOT,
    GPUMonitor,
    PrefetchLoader,
    StageTimer,
    atomic_savez,
    atomic_write_csv,
    atomic_write_text,
    ffprobe_pts,
    list_images,
    parse_str_list,
    resolve_frame_dir,
    resolve_video,
    utc_now,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument(
        "--sequences",
        type=parse_str_list,
        required=True,
        help="explicit sequence allowlist; no implicit full-dataset mode",
    )
    parser.add_argument("--cameras", type=parse_str_list, default=list(CAMERAS))
    parser.add_argument("--sapiens2-root", type=Path, default=DEFAULT_SAPIENS2_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument("--prefetch-batches", type=int, default=4)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


class OfficialDetrPersonDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from sapiens.pose.evaluators import nms
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self.torch = torch
        self.device = torch.device(args.device)
        self.device_index = self.device.index if self.device.index is not None else 0
        torch.cuda.set_device(self.device)
        self.bbox_thr = float(args.bbox_thr)
        self.nms_thr = float(args.nms_thr)
        self.nms = nms
        detector_path = (
            args.checkpoint_root.expanduser().resolve()
            / "sapiens2"
            / "detector"
            / "detr-resnet-101-dc5"
        )
        if not detector_path.exists():
            raise FileNotFoundError(detector_path)
        started = time.perf_counter()
        self.processor = DetrImageProcessor.from_pretrained(detector_path)
        self.model = DetrForObjectDetection.from_pretrained(detector_path).eval().to(self.device)
        torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - started

    def detect(
        self, images: Sequence[np.ndarray], timer: StageTimer, monitor: GPUMonitor
    ) -> list[tuple[np.ndarray, np.ndarray, bool]]:
        import cv2
        from PIL import Image

        monitor.set_phase("detector", len(images))
        with timer.measure("detector_preprocess"):
            rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            inputs = self.processor(
                images=[Image.fromarray(image) for image in rgb], return_tensors="pt"
            ).to(self.device)
        with timer.measure("detector_forward"):
            with self.torch.inference_mode():
                output = self.model(**inputs)
            self.torch.cuda.synchronize(self.device)
        with timer.measure("detector_postprocess"):
            sizes = self.torch.tensor(
                [image.shape[:2] for image in rgb], device=self.device
            )
            results = self.processor.post_process_object_detection(
                output, target_sizes=sizes, threshold=self.bbox_thr
            )
            detections = []
            for result in results:
                mask = result["labels"] == 1
                boxes = result["boxes"][mask].detach().cpu().numpy()
                scores = result["scores"][mask].detach().cpu().numpy()
                if len(boxes):
                    scored = np.concatenate([boxes, scores[:, None]], axis=1)
                    keep = np.asarray(self.nms(scored, self.nms_thr), dtype=np.int64)
                    boxes = boxes[keep].astype(np.float32, copy=False)
                    scores = scores[keep].astype(np.float32, copy=False)
                    fallback = False
                else:
                    # Unlike the upstream visualization fallback, an empty DETR
                    # result stays empty so the identity gate can emit NO_TARGET.
                    boxes = np.empty((0, 4), dtype=np.float32)
                    scores = np.empty(0, dtype=np.float32)
                    fallback = True
                detections.append((boxes, scores, fallback))
        return detections

    def reset_peak_memory(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.device_index)

    def memory(self) -> tuple[int, int]:
        return (
            int(self.torch.cuda.max_memory_allocated(self.device_index)),
            int(self.torch.cuda.max_memory_reserved(self.device_index)),
        )


def chunk_path(output_dir: Path, start: int, end: int) -> Path:
    return output_dir / "chunks" / f"chunk_{start:06d}_{end:06d}.npz"


def save_chunk(
    path: Path,
    frame_paths: Sequence[Path],
    frame_start: int,
    detections: Sequence[tuple[np.ndarray, np.ndarray, bool]],
) -> None:
    counts = np.asarray([len(item[0]) for item in detections], dtype=np.int16)
    offsets = np.zeros(len(detections) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    all_boxes = (
        np.concatenate([item[0] for item in detections], axis=0)
        if int(counts.sum())
        else np.empty((0, 4), dtype=np.float32)
    )
    all_scores = (
        np.concatenate([item[1] for item in detections], axis=0)
        if int(counts.sum())
        else np.empty(0, dtype=np.float32)
    )
    atomic_savez(
        path,
        frame_index=np.arange(
            frame_start, frame_start + len(frame_paths), dtype=np.int32
        ),
        frame_name=np.asarray([item.name for item in frame_paths]),
        person_count=counts,
        detector_fallback=np.asarray([item[2] for item in detections], dtype=np.bool_),
        instance_offsets=offsets,
        all_bboxes_xyxy=all_boxes.astype(np.float32, copy=False),
        all_bbox_scores=all_scores.astype(np.float32, copy=False),
    )


def consolidate(
    output_dir: Path,
    frames: Sequence[Path],
    timestamps: Sequence[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = []
    fallbacks = []
    boxes = []
    scores = []
    for start in range(0, len(frames), args.chunk_size):
        end = min(start + args.chunk_size, len(frames))
        path = chunk_path(output_dir, start, end)
        if not path.exists():
            raise RuntimeError(f"missing candidate chunk: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(payload["frame_index"], np.arange(start, end, dtype=np.int32)):
                raise RuntimeError(f"invalid frame indices: {path}")
            if not np.array_equal(payload["frame_name"], np.asarray([item.name for item in frames[start:end]])):
                raise RuntimeError(f"invalid frame names: {path}")
            counts.append(payload["person_count"])
            fallbacks.append(payload["detector_fallback"])
            boxes.append(payload["all_bboxes_xyxy"])
            scores.append(payload["all_bbox_scores"])
    person_count = np.concatenate(counts)
    fallback = np.concatenate(fallbacks)
    all_boxes = np.concatenate(boxes) if boxes else np.empty((0, 4), dtype=np.float32)
    all_scores = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(person_count, dtype=np.int64)
    atomic_savez(
        output_dir / "bboxes.npz",
        person_count=person_count,
        detector_fallback=fallback,
        instance_offsets=offsets,
        all_bboxes_xyxy=all_boxes,
        all_bbox_scores=all_scores,
    )
    pts = np.full(len(frames), np.nan, dtype=np.float64)
    pts[: min(len(frames), len(timestamps))] = timestamps[: min(len(frames), len(timestamps))]
    rows = [
        {
            "frame_index": index,
            "frame_name": frame.name,
            "timestamp_pts_seconds": f"{pts[index]:.9f}" if np.isfinite(pts[index]) else "",
            "pts_source": "synchronized_video_best_effort_timestamp",
        }
        for index, frame in enumerate(frames)
    ]
    atomic_write_csv(
        output_dir / "frames.csv",
        ("frame_index", "frame_name", "timestamp_pts_seconds", "pts_source"),
        rows,
    )
    qa = {
        "frame_count": len(frames),
        "total_person_candidates": int(person_count.sum()),
        "mean_person_candidates_per_frame": float(person_count.mean()),
        "multi_person_frame_count": int((person_count > 1).sum()),
        "no_person_frame_count": int((person_count == 0).sum()),
        "detector_fallback_count": int(fallback.sum()),
        "timestamp_coverage": float(np.isfinite(pts).mean()),
        "status": "PASS" if np.isfinite(pts).all() else "FAIL",
    }
    return qa


def infer_camera(
    detector: OfficialDetrPersonDetector,
    args: argparse.Namespace,
    sequence: str,
    camera: str,
    monitor: GPUMonitor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_root.expanduser().resolve() / sequence / camera
    (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
    frames = list_images(resolve_frame_dir(dataset_root, sequence, camera))
    timer = StageTimer()
    processed = skipped = 0
    started = time.perf_counter()
    for start in range(0, len(frames), args.chunk_size):
        end = min(start + args.chunk_size, len(frames))
        path = chunk_path(output_dir, start, end)
        if path.exists() and not args.overwrite:
            with np.load(path, allow_pickle=False) as payload:
                if np.array_equal(payload["frame_index"], np.arange(start, end, dtype=np.int32)) and np.array_equal(
                    payload["frame_name"], np.asarray([item.name for item in frames[start:end]])
                ):
                    skipped += end - start
                    continue
        detections = []
        selected = frames[start:end]
        for path_batch, images, wait_seconds in PrefetchLoader(
            selected, args.batch_size, args.loader_workers, args.prefetch_batches
        ):
            timer.values.setdefault("image_load_wait", []).append(wait_seconds)
            detections.extend(detector.detect(images, timer, monitor))
        monitor.set_phase("serialization", len(selected))
        with timer.measure("serialization"):
            save_chunk(path, selected, start, detections)
        processed += len(selected)
        print(f"[{sequence}/{camera}] {processed + skipped}/{len(frames)} frames", flush=True)
    qa = consolidate(
        output_dir,
        frames,
        ffprobe_pts(resolve_video(dataset_root, sequence, camera)),
        args,
    )
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": sequence,
        "camera": camera,
        "stage": "OFFICIAL_DETR_PERSON_CANDIDATES",
        "detector": "facebook/detr-resnet-101-dc5",
        "bbox_threshold": args.bbox_thr,
        "nms_threshold": args.nms_thr,
        "empty_detection_policy": "NO_TARGET candidate set; no full-frame pose fallback",
        "pose_inference_performed": False,
        "qa": qa,
    }
    atomic_write_text(output_dir / "metadata.json", json.dumps(metadata, indent=2) + "\n")
    elapsed = time.perf_counter() - started
    metrics = {
        "sequence": sequence,
        "camera": camera,
        "frame_count": len(frames),
        "elapsed_seconds": elapsed,
        "frames_per_second": (len(frames) - skipped) / max(elapsed, 1e-12),
        "resume_skipped_frames": skipped,
        **{f"{key}_seconds": sum(value) for key, value in timer.values.items()},
        "status": qa["status"],
    }
    return qa, metrics


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size < 1 or args.chunk_size < args.batch_size:
        raise RuntimeError("batch size must be positive and not exceed chunk size")
    if any(camera not in CAMERAS for camera in args.cameras):
        raise RuntimeError(f"unknown camera in {args.cameras}")
    # Validate the Sapiens package/root because its official NMS implementation
    # is intentionally retained even though the 5B pose model is not loaded.
    sapiens_root = args.sapiens2_root.expanduser().resolve()
    if not sapiens_root.exists():
        raise FileNotFoundError(sapiens_root)
    sys.path.insert(0, str(sapiens_root))
    runtime = args.runtime_dir.expanduser().resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    detector = OfficialDetrPersonDetector(args)
    detector.reset_peak_memory()
    monitor = GPUMonitor(
        runtime / "detr_candidate_gpu_utilization.csv",
        args.gpu_sample_interval,
        detector.device_index,
    )
    monitor.start()
    qa_rows = []
    metric_rows = []
    try:
        for sequence in args.sequences:
            for camera in args.cameras:
                qa, metrics = infer_camera(detector, args, sequence, camera, monitor)
                qa_rows.append({"sequence": sequence, "camera": camera, **qa})
                metric_rows.append(metrics)
    finally:
        monitor.stop()
    allocated, reserved = detector.memory()
    atomic_write_csv(
        runtime / "detr_candidate_qa.csv",
        sorted({key for row in qa_rows for key in row}),
        qa_rows,
    )
    atomic_write_csv(
        runtime / "detr_candidate_benchmark.csv",
        sorted({key for row in metric_rows for key in row}),
        metric_rows,
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence_allowlist": args.sequences,
        "cameras": args.cameras,
        "frame_count": int(sum(row["frame_count"] for row in metric_rows)),
        "total_person_candidates": int(sum(row["total_person_candidates"] for row in qa_rows)),
        "pose_inference_performed": False,
        "detector_load_seconds": detector.load_seconds,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "pass_cameras": int(sum(row["status"] == "PASS" for row in qa_rows)),
        "fail_cameras": int(sum(row["status"] == "FAIL" for row in qa_rows)),
    }
    atomic_write_text(
        runtime / "detr_candidate_summary.json", json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["fail_cameras"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
