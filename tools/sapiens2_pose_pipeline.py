#!/usr/bin/env python3
"""Batch/profile/resume Sapiens2-5B 308-keypoint inference without quality changes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = Path(
    os.environ.get("EXERCISE3D_CHECKPOINT_ROOT", PROJECT_ROOT.parent / "checkpoints")
).expanduser()
DEFAULT_SAPIENS2_ROOT = Path(
    os.environ.get("SAPIENS2_ROOT", PROJECT_ROOT.parent / "sapiens2")
).expanduser()
MODEL_CONFIG = (
    "configs/keypoints308/shutterstock_goliath_3po/"
    "sapiens2_5b_keypoints308_shutterstock_goliath_3po-1024x768.py"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CAMERAS = ("cam1", "cam2", "cam3")
PILOT_SEQUENCES = (
    "barbellrow_0000",
    "squat_0001",
    "pushup_0001",
    "benchpress_0003",
)


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated non-empty list")
    return values


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sapiens2-root", type=Path, default=DEFAULT_SAPIENS2_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--gpu-sample-interval", type=float, default=0.2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="batch scaling benchmark")
    add_common_model_args(benchmark)
    benchmark.add_argument("--dataset-root", type=Path, required=True)
    benchmark.add_argument("--sequence", default="barbellrow_0000")
    benchmark.add_argument("--camera", choices=CAMERAS, default="cam1")
    benchmark.add_argument("--batch-sizes", type=parse_int_list, default=[1, 2, 4, 8])
    benchmark.add_argument("--sample-count", type=int, default=16)
    benchmark.add_argument("--warmup-count", type=int, default=1)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--equivalence-xy-atol", type=float, default=0.5)
    benchmark.add_argument("--equivalence-score-atol", type=float, default=0.005)

    infer = subparsers.add_parser("infer", help="resumable sequence/camera inference")
    add_common_model_args(infer)
    infer.add_argument("--dataset-root", type=Path, required=True)
    infer.add_argument("--output-root", type=Path, required=True)
    infer.add_argument(
        "--sequences", type=parse_str_list, default=list(PILOT_SEQUENCES)
    )
    infer.add_argument("--cameras", type=parse_str_list, default=list(CAMERAS))
    infer.add_argument("--batch-size", type=int, required=True)
    infer.add_argument("--chunk-size", type=int, default=256)
    infer.add_argument("--loader-workers", type=int, default=8)
    infer.add_argument("--prefetch-batches", type=int, default=4)
    infer.add_argument("--retry-failures", type=int, default=1)
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--save-overlays", type=int, default=3)
    infer.add_argument("--runtime-dir", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify completed pilot outputs")
    verify.add_argument("--dataset-root", type=Path, required=True)
    verify.add_argument("--output-root", type=Path, required=True)
    verify.add_argument(
        "--sequences", type=parse_str_list, default=list(PILOT_SEQUENCES)
    )
    verify.add_argument("--cameras", type=parse_str_list, default=list(CAMERAS))

    reclassify = subparsers.add_parser(
        "reclassify-benchmark",
        help="recompute equivalence/status from an existing raw benchmark",
    )
    reclassify.add_argument("--output-dir", type=Path, required=True)
    reclassify.add_argument("--equivalence-xy-atol", type=float, default=0.5)
    reclassify.add_argument("--equivalence-score-atol", type=float, default=0.005)
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_savez(path: Path, *, compressed: bool = True, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    if compressed:
        np.savez_compressed(temporary, **arrays)
    else:
        np.savez(temporary, **arrays)
    os.replace(temporary, path)


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    images = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise RuntimeError(f"no image frames: {directory}")
    return images


def resolve_frame_dir(dataset_root: Path, sequence: str, camera: str) -> Path:
    exercise = sequence.rsplit("_", 1)[0]
    return dataset_root / "final_frame" / exercise / sequence / camera


def resolve_video(dataset_root: Path, sequence: str, camera: str) -> Path:
    exercise = sequence.rsplit("_", 1)[0]
    return dataset_root / "synced_video" / exercise / sequence / f"{camera}.mp4"


def evenly_spaced(items: Sequence[Path], count: int) -> list[Path]:
    count = min(count, len(items))
    if count < 1:
        return []
    indices = np.linspace(0, len(items) - 1, count, dtype=np.int64)
    return [items[int(index)] for index in indices]


def decode_image(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"image decode failed: {path}")
    return image


def chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@dataclass
class StageTimer:
    values: dict[str, list[float]] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        yield
        self.values.setdefault(name, []).append(time.perf_counter() - started)

    def total(self, name: str) -> float:
        return float(sum(self.values.get(name, [])))


class GPUMonitor:
    FIELDS = (
        "timestamp_utc",
        "elapsed_seconds",
        "phase",
        "batch_size",
        "gpu_utilization_pct",
        "memory_used_mib",
        "power_draw_w",
    )

    def __init__(self, output: Path, interval: float, gpu_index: int = 0) -> None:
        self.output = output
        self.interval = max(0.1, interval)
        self.gpu_index = gpu_index
        self.phase = "idle"
        self.batch_size = 0
        self._started = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        self._thread: threading.Thread | None = None
        self._handle: Any | None = None
        self._writer: csv.DictWriter | None = None

    def set_phase(self, phase: str, batch_size: int) -> None:
        with self._lock:
            self.phase = phase
            self.batch_size = batch_size

    def start(self) -> None:
        self._started = time.perf_counter()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.output.open(
            "w", newline="", encoding="utf-8", buffering=1
        )
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 4))
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()

    def _run(self) -> None:
        query = (
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            f"--id={self.gpu_index}",
        )
        while not self._stop.is_set():
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", *query], text=True, stderr=subprocess.DEVNULL
                ).strip()
                values = [item.strip() for item in raw.split(",")]
                with self._lock:
                    phase = self.phase
                    batch_size = self.batch_size
                row = {
                    "timestamp_utc": utc_now(),
                    "elapsed_seconds": f"{time.perf_counter() - self._started:.6f}",
                    "phase": phase,
                    "batch_size": batch_size,
                    "gpu_utilization_pct": values[0],
                    "memory_used_mib": values[1],
                    "power_draw_w": values[2],
                }
                self._rows.append(row)
                if self._writer is not None:
                    self._writer.writerow(row)
            except (OSError, subprocess.CalledProcessError, IndexError):
                pass
            self._stop.wait(self.interval)


@dataclass
class FramePrediction:
    path: Path
    keypoints_xy: np.ndarray
    confidence: np.ndarray
    bboxes: np.ndarray
    bbox_scores: np.ndarray
    detector_fallback: bool


class Sapiens2BatchEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(args.device)
        self.device_index = self.device.index if self.device.index is not None else 0
        torch.cuda.set_device(self.device)
        self.bbox_thr = float(args.bbox_thr)
        self.nms_thr = float(args.nms_thr)
        self.sapiens2_root = args.sapiens2_root.expanduser().resolve()
        self.checkpoint_root = args.checkpoint_root.expanduser().resolve()
        self.pose_root = self.sapiens2_root / "sapiens" / "pose"
        self.config = self.pose_root / MODEL_CONFIG
        self.checkpoint = (
            self.checkpoint_root / "sapiens2" / "pose" / "sapiens2_5b_pose.safetensors"
        )
        self.detector_path = (
            self.checkpoint_root / "sapiens2" / "detector" / "detr-resnet-101-dc5"
        )
        for path in (self.pose_root, self.config, self.checkpoint, self.detector_path):
            if not path.exists():
                raise FileNotFoundError(path)

        tools_dir = self.pose_root / "tools" / "vis"
        sys.path.insert(0, str(tools_dir))
        from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
        from sapiens.pose.evaluators import nms
        from sapiens.pose.models import init_model
        from transformers import DetrForObjectDetection, DetrImageProcessor

        self.nms = nms
        started = time.perf_counter()
        self.detector_processor = DetrImageProcessor.from_pretrained(self.detector_path)
        self.detector = (
            DetrForObjectDetection.from_pretrained(self.detector_path)
            .eval()
            .to(self.device)
        )
        self.detector_load_seconds = time.perf_counter() - started

        previous_cwd = Path.cwd()
        os.chdir(self.pose_root)
        try:
            started = time.perf_counter()
            self.model = init_model(
                str(self.config), str(self.checkpoint), device=str(self.device)
            )
            self.model.eval()
            self.model.pose_metainfo = parse_pose_metainfo(
                dict(from_file="configs/_base_/keypoints308.py")
            )
            codec_cfg = dict(self.model.cfg.codec)
            codec_type = codec_cfg.pop("type")
            if codec_type != "UDPHeatmap":
                raise RuntimeError(f"unexpected codec: {codec_type}")
            self.model.codec = UDPHeatmap(**codec_cfg)
            torch.cuda.synchronize(self.device)
            self.pose_load_seconds = time.perf_counter() - started
        finally:
            os.chdir(previous_cwd)

        self.flip_indices = list(self.model.pose_metainfo["flip_indices"])
        self.keypoint_id2name = [
            self.model.pose_metainfo["keypoint_id2name"][index] for index in range(308)
        ]

    def reset_peak_memory(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.device_index)

    def memory(self) -> tuple[int, int]:
        return (
            int(self.torch.cuda.max_memory_allocated(self.device_index)),
            int(self.torch.cuda.max_memory_reserved(self.device_index)),
        )

    def detect(
        self, images: Sequence[np.ndarray], timer: StageTimer | None = None
    ) -> list[tuple[np.ndarray, np.ndarray, bool]]:
        import cv2
        from PIL import Image

        timer = timer or StageTimer()
        with timer.measure("detector_preprocess"):
            rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
            pil = [Image.fromarray(image) for image in rgb]
            inputs = self.detector_processor(images=pil, return_tensors="pt").to(
                self.device
            )
        with timer.measure("detector_forward"):
            with self.torch.inference_mode():
                outputs = self.detector(**inputs)
            self.torch.cuda.synchronize(self.device)
        with timer.measure("detector_postprocess"):
            target_sizes = self.torch.tensor(
                [image.shape[:2] for image in rgb], device=self.device
            )
            results = self.detector_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.bbox_thr
            )
            detections: list[tuple[np.ndarray, np.ndarray, bool]] = []
            for image, result in zip(rgb, results):
                person_mask = result["labels"] == 1
                boxes = result["boxes"][person_mask].detach().cpu().numpy()
                scores = result["scores"][person_mask].detach().cpu().numpy()
                if len(boxes):
                    scored = np.concatenate([boxes, scores[:, None]], axis=1)
                    keep = np.asarray(self.nms(scored, self.nms_thr), dtype=np.int64)
                    boxes = boxes[keep].astype(np.float32, copy=False)
                    scores = scores[keep].astype(np.float32, copy=False)
                    fallback = False
                else:
                    height, width = image.shape[:2]
                    boxes = np.array(
                        [[0, 0, width - 1, height - 1]], dtype=np.float32
                    )
                    scores = np.array([0.0], dtype=np.float32)
                    fallback = True
                detections.append((boxes, scores, fallback))
        return detections

    def _prepare_crops(
        self,
        images: Sequence[np.ndarray],
        detections: Sequence[tuple[np.ndarray, np.ndarray, bool]],
        timer: StageTimer,
    ) -> tuple[list[Any], list[tuple[int, int]]]:
        packed: list[Any] = []
        owners: list[tuple[int, int]] = []
        previous_cwd = Path.cwd()
        os.chdir(self.pose_root)
        try:
            with timer.measure("crop_preprocess"):
                for image_index, (image, detection) in enumerate(zip(images, detections)):
                    boxes, _, _ = detection
                    for person_index, bbox in enumerate(boxes):
                        data_info = {
                            "img": image,
                            "bbox": bbox[None],
                            "bbox_score": np.ones(1, dtype=np.float32),
                        }
                        packed.append(self.model.pipeline(data_info))
                        owners.append((image_index, person_index))
        finally:
            os.chdir(previous_cwd)
        return packed, owners

    def _pose_subbatch(
        self, packed: Sequence[Any], timer: StageTimer
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        with timer.measure("host_to_device_preprocess"):
            inputs_cpu = self.torch.stack([item["inputs"] for item in packed], dim=0)
            try:
                inputs_cpu = inputs_cpu.pin_memory()
            except RuntimeError:
                pass
            processed = self.model.data_preprocessor(
                {
                    "inputs": inputs_cpu,
                    "data_samples": [item["data_samples"] for item in packed],
                }
            )
            inputs = processed["inputs"]
            self.torch.cuda.synchronize(self.device)

        with self.torch.inference_mode():
            with timer.measure("pose_forward"):
                pred = self.model(inputs)
                self.torch.cuda.synchronize(self.device)
            with timer.measure("flip_forward"):
                pred_flipped = self.model(inputs.flip(-1))
                pred_flipped = pred_flipped.flip(-1)[:, self.flip_indices]
                pred = (pred + pred_flipped) / 2.0
                self.torch.cuda.synchronize(self.device)
            with timer.measure("heatmap_transfer"):
                pred_numpy = pred.detach().cpu().numpy()

        keypoints: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        with timer.measure("postprocess"):
            for index, item in enumerate(packed):
                points, confidence = self.model.codec.decode(pred_numpy[index])
                meta = item["data_samples"]["meta"]
                points = (
                    points / meta["input_size"] * meta["bbox_scale"]
                    + meta["bbox_center"]
                    - 0.5 * meta["bbox_scale"]
                )
                keypoints.append(points[0].astype(np.float32, copy=False))
                scores.append(confidence[0].astype(np.float32, copy=False))
        return keypoints, scores

    def process(
        self,
        paths: Sequence[Path],
        images: Sequence[np.ndarray],
        pose_batch_size: int,
        timer: StageTimer | None = None,
        monitor: GPUMonitor | None = None,
    ) -> list[FramePrediction]:
        timer = timer or StageTimer()
        if monitor:
            monitor.set_phase("detector", len(images))
        detections = self.detect(images, timer)
        return self.process_detections(
            paths,
            images,
            detections,
            pose_batch_size,
            timer=timer,
            monitor=monitor,
        )

    def process_detections(
        self,
        paths: Sequence[Path],
        images: Sequence[np.ndarray],
        detections: Sequence[tuple[np.ndarray, np.ndarray, bool]],
        pose_batch_size: int,
        timer: StageTimer | None = None,
        monitor: GPUMonitor | None = None,
    ) -> list[FramePrediction]:
        """Run pose on caller-selected boxes without repeating DETR.

        This is used only after the sequence-level target selector has retained
        all official DETR candidates and accepted one primary identity.  Each
        detection entry must contain at least one box; ambiguous/no-target
        frames are deliberately omitted by the target-only caller.
        """

        if len(paths) != len(images) or len(paths) != len(detections):
            raise ValueError("paths, images, and detections must have equal length")
        if any(len(detection[0]) < 1 for detection in detections):
            raise ValueError("process_detections cannot force an empty target")
        timer = timer or StageTimer()
        packed, owners = self._prepare_crops(images, detections, timer)
        all_keypoints: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        for packed_batch in chunks(packed, pose_batch_size):
            if monitor:
                monitor.set_phase("pose", len(packed_batch))
            points, scores = self._pose_subbatch(packed_batch, timer)
            all_keypoints.extend(points)
            all_scores.extend(scores)

        per_frame_points: list[list[np.ndarray]] = [[] for _ in paths]
        per_frame_scores: list[list[np.ndarray]] = [[] for _ in paths]
        for owner, points, scores in zip(owners, all_keypoints, all_scores):
            image_index, _ = owner
            per_frame_points[image_index].append(points)
            per_frame_scores[image_index].append(scores)

        predictions = []
        for index, path in enumerate(paths):
            boxes, bbox_scores, fallback = detections[index]
            predictions.append(
                FramePrediction(
                    path=path,
                    keypoints_xy=np.stack(per_frame_points[index]),
                    confidence=np.stack(per_frame_scores[index]),
                    bboxes=boxes,
                    bbox_scores=bbox_scores,
                    detector_fallback=fallback,
                )
            )
        return predictions


def flatten_primary(predictions: Sequence[FramePrediction]) -> dict[str, np.ndarray]:
    return {
        "keypoints_xy": np.stack([item.keypoints_xy[0] for item in predictions]),
        "confidence": np.stack([item.confidence[0] for item in predictions]),
        "bbox_xyxy": np.stack([item.bboxes[0] for item in predictions]),
        "bbox_score": np.asarray([item.bbox_scores[0] for item in predictions]),
        "person_count": np.asarray([len(item.bboxes) for item in predictions], dtype=np.int16),
        "detector_fallback": np.asarray(
            [item.detector_fallback for item in predictions], dtype=np.bool_
        ),
    }


def summarize_samples(rows: Sequence[dict[str, str]], phase: str, batch_size: int) -> dict:
    selected = [
        row
        for row in rows
        if row.get("phase") == phase and int(row.get("batch_size", 0)) == batch_size
    ]
    result: dict[str, Any] = {"sample_count": len(selected)}
    for source, target in (
        ("gpu_utilization_pct", "gpu_utilization"),
        ("memory_used_mib", "memory_used_mib"),
        ("power_draw_w", "power_draw_w"),
    ):
        values = [float(row[source]) for row in selected if row.get(source)]
        result[f"{target}_mean"] = float(np.mean(values)) if values else None
        result[f"{target}_p90"] = float(np.percentile(values, 90)) if values else None
        result[f"{target}_p95"] = float(np.percentile(values, 95)) if values else None
        result[f"{target}_max"] = float(np.max(values)) if values else None
    return result


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_batch(
    engine: Sapiens2BatchEngine,
    paths: Sequence[Path],
    batch_size: int,
    monitor: GPUMonitor,
    serialize_dir: Path | None = None,
    official_single_image_pose_grouping: bool = False,
) -> tuple[list[FramePrediction], StageTimer, float]:
    timer = StageTimer()
    results: list[FramePrediction] = []
    started = time.perf_counter()
    for batch_index, path_batch in enumerate(chunks(paths, batch_size)):
        monitor.set_phase("image_load", len(path_batch))
        with timer.measure("image_load"):
            images = [decode_image(path) for path in path_batch]
        batch_predictions = engine.process(
            path_batch,
            images,
            1024 if official_single_image_pose_grouping else batch_size,
            timer=timer,
            monitor=monitor,
        )
        results.extend(batch_predictions)
        if serialize_dir is not None:
            monitor.set_phase("serialization", len(path_batch))
            with timer.measure("serialization"):
                atomic_savez(
                    serialize_dir / f"batch_{batch_index:04d}.npz",
                    **flatten_primary(batch_predictions),
                )
    engine.torch.cuda.synchronize(engine.device)
    return results, timer, time.perf_counter() - started


def benchmark_command(args: argparse.Namespace) -> int:
    if args.sample_count < 1 or args.warmup_count < 0:
        raise RuntimeError("sample and warmup counts must be valid")
    if 1 not in args.batch_sizes:
        raise RuntimeError("--batch-sizes must contain batch 1 as equivalence reference")
    args.batch_sizes = sorted(set(args.batch_sizes))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    paths = evenly_spaced(
        list_images(resolve_frame_dir(dataset_root, args.sequence, args.camera)),
        args.sample_count,
    )
    engine = Sapiens2BatchEngine(args)
    monitor_path = output_dir / "gpu_utilization.csv"
    monitor = GPUMonitor(monitor_path, args.gpu_sample_interval, engine.device_index)
    monitor.start()
    records: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    reference: dict[str, np.ndarray] | None = None
    try:
        for batch_size in args.batch_sizes:
            status = "PASS"
            error = ""
            engine.torch.cuda.empty_cache()
            engine.reset_peak_memory()
            try:
                warmup_paths = paths[: min(batch_size, len(paths))]
                for _ in range(args.warmup_count):
                    warmup_images = [decode_image(path) for path in warmup_paths]
                    engine.process(
                        warmup_paths,
                        warmup_images,
                        batch_size,
                        monitor=monitor,
                    )
                with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
                    predictions, timer, elapsed = run_batch(
                        engine,
                        paths,
                        batch_size,
                        monitor,
                        Path(temporary),
                        official_single_image_pose_grouping=batch_size == 1,
                    )
                arrays = flatten_primary(predictions)
                if reference is None:
                    reference = arrays
                    max_xy_delta = 0.0
                    max_score_delta = 0.0
                    bbox_delta = 0.0
                    equivalent = True
                else:
                    xy_delta = np.linalg.norm(
                        arrays["keypoints_xy"] - reference["keypoints_xy"], axis=-1
                    )
                    confident = np.minimum(
                        arrays["confidence"], reference["confidence"]
                    ) >= 0.3
                    max_xy_delta = float(np.max(xy_delta))
                    p95_xy_delta = float(np.percentile(xy_delta, 95))
                    max_confident_xy_delta = (
                        float(np.max(xy_delta[confident])) if confident.any() else 0.0
                    )
                    max_score_delta = float(
                        np.max(np.abs(arrays["confidence"] - reference["confidence"]))
                    )
                    bbox_delta = float(
                        np.max(np.abs(arrays["bbox_xyxy"] - reference["bbox_xyxy"]))
                    )
                    equivalent = (
                        max_confident_xy_delta <= args.equivalence_xy_atol
                        and max_score_delta <= args.equivalence_score_atol
                        and bbox_delta <= 0.05
                        and np.array_equal(
                            arrays["person_count"], reference["person_count"]
                        )
                    )
                    if not equivalent:
                        status = "NON_EQUIVALENT"
                peak_allocated, peak_reserved = engine.memory()
                record = {
                    "batch_size": batch_size,
                    "status": status,
                    "images": len(paths),
                    "persons": int(sum(len(item.bboxes) for item in predictions)),
                    "elapsed_seconds": elapsed,
                    "images_per_second": len(paths) / elapsed,
                    "effective_seconds_per_image": elapsed / len(paths),
                    "image_load_seconds": timer.total("image_load"),
                    "detector_preprocess_seconds": timer.total("detector_preprocess"),
                    "detector_forward_seconds": timer.total("detector_forward"),
                    "detector_postprocess_seconds": timer.total("detector_postprocess"),
                    "crop_preprocess_seconds": timer.total("crop_preprocess"),
                    "host_to_device_preprocess_seconds": timer.total(
                        "host_to_device_preprocess"
                    ),
                    "pose_forward_seconds": timer.total("pose_forward"),
                    "flip_forward_seconds": timer.total("flip_forward"),
                    "heatmap_transfer_seconds": timer.total("heatmap_transfer"),
                    "postprocess_seconds": timer.total("postprocess"),
                    "serialization_seconds": timer.total("serialization"),
                    "detector_images_per_second": len(paths)
                    / max(timer.total("detector_forward"), 1e-12),
                    "pose_persons_per_second": int(
                        sum(len(item.bboxes) for item in predictions)
                    )
                    / max(
                        timer.total("pose_forward") + timer.total("flip_forward"),
                        1e-12,
                    ),
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "max_xy_delta_vs_batch1_px": max_xy_delta,
                    "p95_xy_delta_vs_batch1_px": (
                        p95_xy_delta if reference is not None and batch_size != 1 else 0.0
                    ),
                    "max_confident_xy_delta_vs_batch1_px": (
                        max_confident_xy_delta
                        if reference is not None and batch_size != 1
                        else 0.0
                    ),
                    "max_confidence_delta_vs_batch1": max_score_delta,
                    "max_bbox_delta_vs_batch1_px": bbox_delta,
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
                is_oom = isinstance(
                    exc, engine.torch.cuda.OutOfMemoryError
                ) or "out of memory" in str(exc).lower()
                if not is_oom:
                    raise
                engine.torch.cuda.empty_cache()
                peak_allocated, peak_reserved = engine.memory()
                status = "OOM"
                error = str(exc).splitlines()[0][:500]
                record = {
                    "batch_size": batch_size,
                    "status": status,
                    "images": len(paths),
                    "persons": "",
                    "elapsed_seconds": "",
                    "images_per_second": "",
                    "effective_seconds_per_image": "",
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "max_xy_delta_vs_batch1_px": "",
                    "p95_xy_delta_vs_batch1_px": "",
                    "max_confident_xy_delta_vs_batch1_px": "",
                    "max_confidence_delta_vs_batch1": "",
                    "max_bbox_delta_vs_batch1_px": "",
                    "numerically_equivalent": False,
                    "error": error,
                }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    finally:
        monitor.stop()

    gpu_rows = read_csv_rows(monitor_path)
    for record in records:
        if record["status"] != "PASS":
            continue
        batch_size = int(record["batch_size"])
        for phase in ("detector", "pose"):
            summary = summarize_samples(gpu_rows, phase, batch_size)
            for key, value in summary.items():
                record[f"{phase}_{key}"] = value

    fieldnames = sorted({key for record in records for key in record})
    atomic_write_csv(output_dir / "batch_scaling.csv", fieldnames, records)
    atomic_write_csv(
        output_dir / "sapiens2_benchmark.csv",
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
    passing = [record for record in records if record["status"] == "PASS"]
    best = max(passing, key=lambda record: float(record["images_per_second"]))
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": args.sequence,
        "camera": args.camera,
        "sample_count": len(paths),
        "frame_selection": "evenly spaced",
        "model": "facebook/sapiens2-pose-5b",
        "detector": "facebook/detr-resnet-101-dc5",
        "pose_input_hw": [1024, 768],
        "precision": "float32",
        "flip_test": True,
        "detector_load_seconds": engine.detector_load_seconds,
        "pose_model_load_seconds": engine.pose_load_seconds,
        "best_batch_size": int(best["batch_size"]),
        "best_images_per_second": float(best["images_per_second"]),
        "best_seconds_per_image": float(best["effective_seconds_per_image"]),
        "equivalence_xy_atol_px": args.equivalence_xy_atol,
        "equivalence_score_atol": args.equivalence_score_atol,
        "all_stable_batches_equivalent": all(
            bool(record["numerically_equivalent"]) for record in passing
        ),
        "records": records,
    }
    atomic_write_text(
        output_dir / "benchmark_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


class PrefetchLoader:
    def __init__(
        self,
        paths: Sequence[Path],
        batch_size: int,
        workers: int,
        prefetch_batches: int,
    ) -> None:
        self.paths = paths
        self.batch_size = batch_size
        self.workers = workers
        self.prefetch_batches = prefetch_batches

    def __iter__(self) -> Iterator[tuple[Sequence[Path], list[np.ndarray], float]]:
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            pending: deque[tuple[Path, Any]] = deque()
            path_iterator = iter(self.paths)
            capacity = max(
                self.batch_size,
                self.batch_size * max(1, self.prefetch_batches),
            )

            def refill() -> None:
                while len(pending) < capacity:
                    try:
                        path = next(path_iterator)
                    except StopIteration:
                        return
                    pending.append((path, executor.submit(decode_image, path)))

            refill()
            while pending:
                current = [pending.popleft() for _ in range(min(self.batch_size, len(pending)))]
                started = time.perf_counter()
                path_batch = [item[0] for item in current]
                images = [item[1].result() for item in current]
                refill()
                yield path_batch, images, time.perf_counter() - started


def ffprobe_pts(video: Path) -> list[float]:
    if not video.exists():
        raise FileNotFoundError(video)
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        text=True,
    )
    payload = json.loads(output)
    return [
        float(frame["best_effort_timestamp_time"])
        for frame in payload.get("frames", [])
        if "best_effort_timestamp_time" in frame
    ]


def save_chunk(
    path: Path, predictions: Sequence[FramePrediction], frame_start: int
) -> None:
    primary = flatten_primary(predictions)
    all_points = np.concatenate([item.keypoints_xy for item in predictions], axis=0)
    all_confidence = np.concatenate([item.confidence for item in predictions], axis=0)
    all_bboxes = np.concatenate([item.bboxes for item in predictions], axis=0)
    all_bbox_scores = np.concatenate([item.bbox_scores for item in predictions], axis=0)
    offsets = np.zeros(len(predictions) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([len(item.bboxes) for item in predictions])
    atomic_savez(
        path,
        **primary,
        valid_mask=np.isfinite(primary["keypoints_xy"]).all(axis=-1)
        & np.isfinite(primary["confidence"]),
        frame_index=np.arange(
            frame_start, frame_start + len(predictions), dtype=np.int32
        ),
        frame_name=np.asarray([item.path.name for item in predictions]),
        instance_offsets=offsets,
        all_keypoints_xy=all_points,
        all_confidence=all_confidence,
        all_bboxes_xyxy=all_bboxes,
        all_bbox_scores=all_bbox_scores,
    )


def consolidate_camera(
    output_dir: Path,
    frame_paths: Sequence[Path],
    pts: Sequence[float],
    sequence: str,
    camera: str,
    engine: Sapiens2BatchEngine,
    args: argparse.Namespace,
) -> dict[str, Any]:
    chunk_specs = [
        (
            chunk_start,
            min(chunk_start + args.chunk_size, len(frame_paths)),
            output_dir
            / "chunks"
            / (
                f"chunk_{chunk_start:06d}_"
                f"{min(chunk_start + args.chunk_size, len(frame_paths)):06d}.npz"
            ),
        )
        for chunk_start in range(0, len(frame_paths), args.chunk_size)
    ]
    missing_chunks = [path.name for _, _, path in chunk_specs if not path.exists()]
    if missing_chunks:
        raise RuntimeError(
            f"missing chunks for {sequence}/{camera}: {', '.join(missing_chunks)}"
        )
    primary_keys = (
        "keypoints_xy",
        "confidence",
        "valid_mask",
        "frame_index",
        "frame_name",
    )
    bbox_keys = ("bbox_xyxy", "bbox_score", "person_count", "detector_fallback")
    primary_parts: dict[str, list[np.ndarray]] = {key: [] for key in primary_keys}
    bbox_parts: dict[str, list[np.ndarray]] = {key: [] for key in bbox_keys}
    all_keypoints_parts: list[np.ndarray] = []
    all_confidence_parts: list[np.ndarray] = []
    all_bboxes_parts: list[np.ndarray] = []
    all_bbox_score_parts: list[np.ndarray] = []
    person_counts: list[np.ndarray] = []
    for chunk_start, chunk_end, chunk_path in chunk_specs:
        with np.load(chunk_path, allow_pickle=False) as chunk:
            expected_indices = np.arange(chunk_start, chunk_end, dtype=np.int32)
            expected_names = np.asarray(
                [path.name for path in frame_paths[chunk_start:chunk_end]]
            )
            if not np.array_equal(chunk["frame_index"], expected_indices):
                raise RuntimeError(
                    f"invalid frame indices in {sequence}/{camera}/{chunk_path.name}"
                )
            if not np.array_equal(chunk["frame_name"], expected_names):
                raise RuntimeError(
                    f"invalid frame names in {sequence}/{camera}/{chunk_path.name}"
                )
            for key in primary_keys:
                primary_parts[key].append(chunk[key])
            for key in bbox_keys:
                bbox_parts[key].append(chunk[key])
            all_keypoints_parts.append(chunk["all_keypoints_xy"])
            all_confidence_parts.append(chunk["all_confidence"])
            all_bboxes_parts.append(chunk["all_bboxes_xyxy"])
            all_bbox_score_parts.append(chunk["all_bbox_scores"])
            person_counts.append(chunk["person_count"])
    primary = {key: np.concatenate(value) for key, value in primary_parts.items()}
    bbox = {key: np.concatenate(value) for key, value in bbox_parts.items()}
    all_keypoints = np.concatenate(all_keypoints_parts)
    all_confidence = np.concatenate(all_confidence_parts)
    all_bboxes = np.concatenate(all_bboxes_parts)
    all_bbox_scores = np.concatenate(all_bbox_score_parts)
    instance_offsets = np.zeros(len(frame_paths) + 1, dtype=np.int64)
    instance_offsets[1:] = np.cumsum(np.concatenate(person_counts), dtype=np.int64)
    expected_indices = np.arange(len(frame_paths), dtype=np.int32)
    expected_names = np.asarray([path.name for path in frame_paths])
    if not np.array_equal(primary["frame_index"], expected_indices):
        raise RuntimeError(
            f"incomplete or unordered chunks for {sequence}/{camera}: "
            f"{len(primary['frame_index'])}/{len(frame_paths)} frames"
        )
    if not np.array_equal(primary["frame_name"], expected_names):
        raise RuntimeError(
            f"chunk frame names do not match source for {sequence}/{camera}"
        )
    timestamps = np.full(len(frame_paths), np.nan, dtype=np.float64)
    timestamp_count = min(len(pts), len(frame_paths))
    timestamps[:timestamp_count] = np.asarray(pts[:timestamp_count], dtype=np.float64)
    atomic_savez(
        output_dir / "poses_2d.npz",
        keypoints_xy=primary["keypoints_xy"],
        confidence=primary["confidence"],
        valid_mask=primary["valid_mask"],
        frame_index=primary["frame_index"],
        timestamp_pts_seconds=timestamps,
        instance_offsets=instance_offsets,
        all_keypoints_xy=all_keypoints,
        all_confidence=all_confidence,
        all_valid_mask=np.isfinite(all_keypoints).all(axis=-1)
        & np.isfinite(all_confidence),
    )
    atomic_savez(
        output_dir / "bboxes.npz",
        **bbox,
        instance_offsets=instance_offsets,
        all_bboxes_xyxy=all_bboxes,
        all_bbox_scores=all_bbox_scores,
    )
    rows = [
        {
            "frame_index": int(index),
            "frame_name": frame_path.name,
            "timestamp_pts_seconds": (
                f"{timestamps[index]:.9f}" if np.isfinite(timestamps[index]) else ""
            ),
            "pts_source": "synchronized_video_best_effort_timestamp",
        }
        for index, frame_path in enumerate(frame_paths)
    ]
    atomic_write_csv(
        output_dir / "frames.csv",
        ["frame_index", "frame_name", "timestamp_pts_seconds", "pts_source"],
        rows,
    )

    xy = primary["keypoints_xy"].astype(np.float64)
    confidence = primary["confidence"].astype(np.float64)
    centers = (bbox["bbox_xyxy"][:, :2] + bbox["bbox_xyxy"][:, 2:]) / 2
    bbox_diag = np.linalg.norm(
        bbox["bbox_xyxy"][:, 2:] - bbox["bbox_xyxy"][:, :2], axis=1
    )
    bbox_jump = np.linalg.norm(np.diff(centers, axis=0), axis=1) / np.maximum(
        bbox_diag[1:], 1.0
    )
    joint_jump = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / np.maximum(
        bbox_diag[1:, None], 1.0
    )
    median_confidence = np.median(confidence, axis=1)
    qa = {
        "sequence": sequence,
        "camera": camera,
        "frame_count": len(frame_paths),
        "detector_success_count": int((~bbox["detector_fallback"]).sum()),
        "detector_success_rate": float((~bbox["detector_fallback"]).mean()),
        "detector_fallback_count": int(bbox["detector_fallback"].sum()),
        "multiple_person_frame_count": int((bbox["person_count"] > 1).sum()),
        "finite_keypoint_fraction": float(np.isfinite(xy).all(axis=-1).mean()),
        "finite_confidence_fraction": float(np.isfinite(confidence).mean()),
        "confidence_median": float(np.median(confidence)),
        "confidence_p10": float(np.percentile(confidence, 10)),
        "confidence_p90": float(np.percentile(confidence, 90)),
        "low_frame_confidence_p10": float(np.percentile(median_confidence, 10)),
        "bbox_jump_normalized_p95": float(np.percentile(bbox_jump, 95)),
        "keypoint_jump_normalized_p95": float(np.percentile(joint_jump, 95)),
        "timestamp_coverage": float(np.isfinite(timestamps).mean()),
        "coverage": float(len(primary["frame_index"]) / len(frame_paths)),
        "status": "PASS",
    }
    if (
        qa["coverage"] < 1.0
        or qa["finite_keypoint_fraction"] < 1.0
        or qa["finite_confidence_fraction"] < 1.0
        or qa["timestamp_coverage"] < 1.0
    ):
        qa["status"] = "FAIL"
    elif qa["detector_fallback_count"] > 0 or qa["multiple_person_frame_count"] > 0:
        qa["status"] = "REVIEW"
    metadata = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequence": sequence,
        "camera": camera,
        "source_frame_count": len(frame_paths),
        "source_provenance": {
            "frame_directory": "external dataset root/final_frame/<exercise>/<sequence>/<camera>",
            "timestamp_video": (
                "external dataset root/synced_video/"
                "<exercise>/<sequence>/<camera>.mp4"
            ),
            "timestamp_kind": "best_effort_timestamp_time (PTS-derived)",
        },
        "model": "facebook/sapiens2-pose-5b",
        "detector": "facebook/detr-resnet-101-dc5",
        "keypoint_count": 308,
        "keypoint_names": engine.keypoint_id2name,
        "coordinate_system": "original image pixel coordinates (x, y)",
        "pose_input_hw": [1024, 768],
        "precision": "float32",
        "flip_test": True,
        "batch_size": args.batch_size,
        "chunk_size": args.chunk_size,
        "bbox_threshold": args.bbox_thr,
        "nms_threshold": args.nms_thr,
        "official_full_frame_fallback_retained": True,
        "multi_person_storage": (
            "primary arrays plus lossless ragged all-instance arrays with instance_offsets"
        ),
        "qa": qa,
    }
    atomic_write_text(
        output_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    return qa


def save_overlays(
    output_dir: Path,
    frame_paths: Sequence[Path],
    count: int,
    engine: Sapiens2BatchEngine,
) -> None:
    if count < 1:
        return
    import cv2

    tools_dir = engine.pose_root / "tools" / "vis"
    sys.path.insert(0, str(tools_dir))
    from pose_render_utils import visualize_keypoints

    with np.load(output_dir / "poses_2d.npz", allow_pickle=False) as poses:
        xy = poses["keypoints_xy"]
        confidence = poses["confidence"]
    indices = np.linspace(0, len(frame_paths) - 1, min(count, len(frame_paths)), dtype=int)
    overlay_dir = output_dir / "debug" / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        image = decode_image(frame_paths[int(index)])
        rendered = visualize_keypoints(
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            keypoints=[xy[int(index)]],
            keypoints_visible=[np.ones(308, dtype=np.bool_)],
            keypoint_scores=[confidence[int(index)]],
            radius=3,
            thickness=2,
            kpt_thr=0.3,
            skeleton=engine.model.pose_metainfo["skeleton_links"],
            kpt_color=engine.model.pose_metainfo["keypoint_colors"],
            link_color=engine.model.pose_metainfo["skeleton_link_colors"],
        )
        cv2.imwrite(
            str(overlay_dir / f"frame_{int(index):06d}.jpg"),
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
    output_root = args.output_root.expanduser().resolve()
    frame_paths = list_images(resolve_frame_dir(dataset_root, sequence, camera))
    output_dir = output_root / sequence / camera
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    timer = StageTimer()
    processed = 0
    skipped_frames = 0
    retries = 0
    started = time.perf_counter()
    for chunk_start in range(0, len(frame_paths), args.chunk_size):
        chunk_end = min(chunk_start + args.chunk_size, len(frame_paths))
        chunk_path = chunks_dir / f"chunk_{chunk_start:06d}_{chunk_end:06d}.npz"
        if chunk_path.exists() and not args.overwrite:
            with np.load(chunk_path, allow_pickle=False) as payload:
                expected_indices = np.arange(chunk_start, chunk_end, dtype=np.int32)
                expected_names = np.asarray(
                    [path.name for path in frame_paths[chunk_start:chunk_end]]
                )
                if np.array_equal(
                    payload["frame_index"], expected_indices
                ) and np.array_equal(payload["frame_name"], expected_names):
                    processed += chunk_end - chunk_start
                    skipped_frames += chunk_end - chunk_start
                    continue
        selected = frame_paths[chunk_start:chunk_end]
        predictions: list[FramePrediction] = []
        for path_batch, images, wait_seconds in PrefetchLoader(
            selected,
            args.batch_size,
            args.loader_workers,
            args.prefetch_batches,
        ):
            timer.values.setdefault("image_load_wait", []).append(wait_seconds)
            attempts = 0
            while True:
                try:
                    predictions.extend(
                        engine.process(
                            path_batch,
                            images,
                            args.batch_size,
                            timer=timer,
                            monitor=monitor,
                        )
                    )
                    break
                except Exception:
                    attempts += 1
                    retries += 1
                    if attempts > args.retry_failures:
                        raise
                    engine.torch.cuda.empty_cache()
        monitor.set_phase("serialization", len(predictions))
        with timer.measure("serialization"):
            save_chunk(chunk_path, predictions, chunk_start)
        processed += len(predictions)
        print(
            f"[{sequence}/{camera}] {processed}/{len(frame_paths)} frames",
            flush=True,
        )
    pts = ffprobe_pts(resolve_video(dataset_root, sequence, camera))
    qa = consolidate_camera(
        output_dir, frame_paths, pts, sequence, camera, engine, args
    )
    save_overlays(output_dir, frame_paths, args.save_overlays, engine)
    elapsed = time.perf_counter() - started
    metrics = {
        "sequence": sequence,
        "camera": camera,
        "frame_count": len(frame_paths),
        "elapsed_seconds": elapsed,
        "images_per_second": len(frame_paths) / elapsed,
        "effective_seconds_per_image": elapsed / len(frame_paths),
        "resume_skipped_frames": skipped_frames,
        "retry_count": retries,
        **{f"{key}_seconds": sum(values) for key, values in timer.values.items()},
        "status": qa["status"],
    }
    return qa, metrics


def infer_command(args: argparse.Namespace) -> int:
    if args.batch_size < 1 or args.chunk_size < args.batch_size:
        raise RuntimeError("batch size must be positive and not exceed chunk size")
    if any(camera not in CAMERAS for camera in args.cameras):
        raise RuntimeError(f"unknown camera in {args.cameras}")
    runtime_dir = args.runtime_dir.expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    engine = Sapiens2BatchEngine(args)
    engine.reset_peak_memory()
    monitor_path = runtime_dir / "pilot_gpu_utilization.csv"
    monitor = GPUMonitor(monitor_path, args.gpu_sample_interval, engine.device_index)
    monitor.start()
    qa_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    try:
        for sequence in args.sequences:
            for camera in args.cameras:
                monitor.set_phase("pilot", args.batch_size)
                qa, metrics = infer_camera(engine, args, sequence, camera, monitor)
                qa_rows.append(qa)
                metrics_rows.append(metrics)
    finally:
        monitor.stop()
    peak_allocated, peak_reserved = engine.memory()
    fieldnames = sorted({key for row in qa_rows for key in row})
    atomic_write_csv(runtime_dir / "pilot_qa.csv", fieldnames, qa_rows)
    fieldnames = sorted({key for row in metrics_rows for key in row})
    atomic_write_csv(
        runtime_dir / "sapiens2_pilot_benchmark.csv", fieldnames, metrics_rows
    )
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sequences": args.sequences,
        "cameras": args.cameras,
        "frame_count": int(sum(row["frame_count"] for row in metrics_rows)),
        "detector_success_count": int(
            sum(row["detector_success_count"] for row in qa_rows)
        ),
        "detector_success_rate": float(
            sum(row["detector_success_count"] for row in qa_rows)
            / max(sum(row["frame_count"] for row in qa_rows), 1)
        ),
        "finite_keypoint_fraction": float(
            np.average(
                [row["finite_keypoint_fraction"] for row in qa_rows],
                weights=[row["frame_count"] for row in qa_rows],
            )
        ),
        "total_elapsed_seconds": float(
            sum(row["elapsed_seconds"] for row in metrics_rows)
        ),
        "aggregate_images_per_second": float(
            sum(row["frame_count"] for row in metrics_rows)
            / max(sum(row["elapsed_seconds"] for row in metrics_rows), 1e-12)
        ),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "pass_cameras": sum(row["status"] == "PASS" for row in qa_rows),
        "review_cameras": sum(row["status"] == "REVIEW" for row in qa_rows),
        "fail_cameras": sum(row["status"] == "FAIL" for row in qa_rows),
    }
    atomic_write_text(
        runtime_dir / "pilot_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["fail_cameras"] == 0 else 2


def verify_command(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    rows = []
    for sequence in args.sequences:
        for camera in args.cameras:
            source_count = len(
                list_images(resolve_frame_dir(dataset_root, sequence, camera))
            )
            output_dir = output_root / sequence / camera
            required = (
                output_dir / "poses_2d.npz",
                output_dir / "bboxes.npz",
                output_dir / "frames.csv",
                output_dir / "metadata.json",
            )
            missing = [path.name for path in required if not path.exists()]
            output_count = 0
            finite = False
            schema_valid = False
            errors: list[str] = []
            if not missing:
                try:
                    pose_offsets = np.empty(0, dtype=np.int64)
                    all_instance_count = -1
                    with np.load(required[0], allow_pickle=False) as poses:
                        pose_keys = {
                            "keypoints_xy",
                            "confidence",
                            "valid_mask",
                            "frame_index",
                            "timestamp_pts_seconds",
                            "instance_offsets",
                            "all_keypoints_xy",
                            "all_confidence",
                            "all_valid_mask",
                        }
                        absent = sorted(pose_keys - set(poses.files))
                        if absent:
                            errors.append(f"pose keys missing: {','.join(absent)}")
                        else:
                            output_count = len(poses["frame_index"])
                            pose_offsets = poses["instance_offsets"].copy()
                            all_instance_count = len(poses["all_keypoints_xy"])
                            pose_shape_valid = (
                                poses["keypoints_xy"].shape
                                == (source_count, 308, 2)
                                and poses["confidence"].shape == (source_count, 308)
                                and poses["valid_mask"].shape == (source_count, 308)
                                and poses["timestamp_pts_seconds"].shape
                                == (source_count,)
                                and poses["all_keypoints_xy"].shape
                                == (all_instance_count, 308, 2)
                                and poses["all_confidence"].shape
                                == (all_instance_count, 308)
                                and poses["all_valid_mask"].shape
                                == (all_instance_count, 308)
                            )
                            index_valid = np.array_equal(
                                poses["frame_index"],
                                np.arange(source_count, dtype=np.int32),
                            )
                            offsets_valid = (
                                pose_offsets.shape == (source_count + 1,)
                                and int(pose_offsets[0]) == 0
                                and int(pose_offsets[-1]) == all_instance_count
                                and bool(np.all(np.diff(pose_offsets) >= 1))
                            )
                            finite = bool(
                                np.isfinite(poses["keypoints_xy"]).all()
                                and np.isfinite(poses["confidence"]).all()
                                and np.isfinite(
                                    poses["timestamp_pts_seconds"]
                                ).all()
                                and np.isfinite(poses["all_keypoints_xy"]).all()
                                and np.isfinite(poses["all_confidence"]).all()
                            )
                            if not pose_shape_valid:
                                errors.append("pose array shape mismatch")
                            if not index_valid:
                                errors.append("frame index mismatch")
                            if not offsets_valid:
                                errors.append("pose instance offsets invalid")

                    with np.load(required[1], allow_pickle=False) as bboxes:
                        bbox_keys = {
                            "bbox_xyxy",
                            "bbox_score",
                            "person_count",
                            "detector_fallback",
                            "instance_offsets",
                            "all_bboxes_xyxy",
                            "all_bbox_scores",
                        }
                        absent = sorted(bbox_keys - set(bboxes.files))
                        if absent:
                            errors.append(f"bbox keys missing: {','.join(absent)}")
                        else:
                            bbox_instance_count = len(bboxes["all_bboxes_xyxy"])
                            bbox_shape_valid = (
                                bboxes["bbox_xyxy"].shape == (source_count, 4)
                                and bboxes["bbox_score"].shape == (source_count,)
                                and bboxes["person_count"].shape == (source_count,)
                                and bboxes["detector_fallback"].shape
                                == (source_count,)
                                and bboxes["all_bboxes_xyxy"].shape
                                == (bbox_instance_count, 4)
                                and bboxes["all_bbox_scores"].shape
                                == (bbox_instance_count,)
                            )
                            bbox_offsets_valid = (
                                np.array_equal(
                                    bboxes["instance_offsets"], pose_offsets
                                )
                                and bbox_instance_count == all_instance_count
                                and np.array_equal(
                                    bboxes["person_count"], np.diff(pose_offsets)
                                )
                            )
                            bbox_finite = bool(
                                np.isfinite(bboxes["bbox_xyxy"]).all()
                                and np.isfinite(bboxes["bbox_score"]).all()
                                and np.isfinite(bboxes["all_bboxes_xyxy"]).all()
                                and np.isfinite(bboxes["all_bbox_scores"]).all()
                            )
                            if not bbox_shape_valid:
                                errors.append("bbox array shape mismatch")
                            if not bbox_offsets_valid:
                                errors.append("bbox instance offsets invalid")
                            if not bbox_finite:
                                finite = False
                                errors.append("non-finite bbox output")

                    with required[2].open(newline="", encoding="utf-8") as handle:
                        frame_rows = list(csv.DictReader(handle))
                    if len(frame_rows) != source_count:
                        errors.append("frames.csv row count mismatch")

                    metadata = json.loads(required[3].read_text(encoding="utf-8"))
                    metadata_valid = (
                        metadata.get("source_frame_count") == source_count
                        and metadata.get("keypoint_count") == 308
                        and metadata.get("model") == "facebook/sapiens2-pose-5b"
                        and metadata.get("flip_test") is True
                    )
                    if not metadata_valid:
                        errors.append("metadata mismatch")
                    schema_valid = not errors
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"validation error: {type(exc).__name__}: {exc}")
            status = (
                "PASS"
                if not missing
                and output_count == source_count
                and finite
                and schema_valid
                else "FAIL"
            )
            rows.append(
                {
                    "sequence": sequence,
                    "camera": camera,
                    "source_count": source_count,
                    "output_count": output_count,
                    "finite": finite,
                    "schema_valid": schema_valid,
                    "missing": missing,
                    "errors": errors,
                    "status": status,
                }
            )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "PASS" for row in rows) else 2


def reclassify_benchmark_command(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    summary_path = output_dir / "benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gpu_rows = read_csv_rows(output_dir / "gpu_utilization.csv")
    for record in summary["records"]:
        if record["status"] == "OOM":
            continue
        equivalent = (
            float(record["max_confident_xy_delta_vs_batch1_px"])
            <= args.equivalence_xy_atol
            and float(record["max_confidence_delta_vs_batch1"])
            <= args.equivalence_score_atol
            and float(record["max_bbox_delta_vs_batch1_px"]) <= 0.05
        )
        record["numerically_equivalent"] = equivalent
        record["status"] = "PASS" if equivalent else "NON_EQUIVALENT"
        if equivalent:
            batch_size = int(record["batch_size"])
            for phase in ("detector", "pose"):
                sample_summary = summarize_samples(gpu_rows, phase, batch_size)
                for key, value in sample_summary.items():
                    record[f"{phase}_{key}"] = value
    passing = [record for record in summary["records"] if record["status"] == "PASS"]
    if not passing:
        raise RuntimeError("no equivalent batch configuration")
    best = max(passing, key=lambda record: float(record["images_per_second"]))
    summary["equivalence_xy_atol_px"] = args.equivalence_xy_atol
    summary["equivalence_score_atol"] = args.equivalence_score_atol
    summary["all_stable_batches_equivalent"] = all(
        record["status"] in {"PASS", "OOM"} for record in summary["records"]
    )
    summary["best_batch_size"] = int(best["batch_size"])
    summary["best_images_per_second"] = float(best["images_per_second"])
    summary["best_seconds_per_image"] = float(best["effective_seconds_per_image"])
    summary["equivalence_note"] = (
        "Confidence>=0.3 joints use subpixel coordinate tolerance; low-confidence "
        "argmax locations are reported but excluded from the semantic gate."
    )
    fieldnames = sorted({key for record in summary["records"] for key in record})
    atomic_write_csv(output_dir / "batch_scaling.csv", fieldnames, summary["records"])
    atomic_write_text(
        summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "benchmark":
        return benchmark_command(args)
    if args.command == "infer":
        return infer_command(args)
    if args.command == "verify":
        return verify_command(args)
    if args.command == "reclassify-benchmark":
        return reclassify_benchmark_command(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
