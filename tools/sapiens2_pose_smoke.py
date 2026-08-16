#!/usr/bin/env python3
"""Run one-image Sapiens2 5B pose smoke/benchmark with the official pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
BODY_FLIP_NAMES = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
    ("left_big_toe", "right_big_toe"),
    ("left_small_toe", "right_small_toe"),
    ("left_heel", "right_heel"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--sapiens2-root", type=Path, default=DEFAULT_SAPIENS2_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bbox-thr", type=float, default=0.3)
    parser.add_argument("--nms-thr", type=float, default=0.3)
    parser.add_argument("--kpt-thr", type=float, default=0.3)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-visualization", type=Path, default=None)
    return parser.parse_args()


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise RuntimeError("--warmup must be nonnegative and --repeats must be positive")
    image_path = args.image.resolve()
    sapiens2_root = args.sapiens2_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    pose_root = sapiens2_root / "sapiens" / "pose"
    checkpoint = checkpoint_root / "sapiens2" / "pose" / "sapiens2_5b_pose.safetensors"
    detector = checkpoint_root / "sapiens2" / "detector" / "detr-resnet-101-dc5"
    config = pose_root / MODEL_CONFIG
    for path in (image_path, checkpoint, detector, config):
        if not path.exists():
            raise FileNotFoundError(path)

    import cv2
    import torch

    tools_dir = pose_root / "tools" / "vis"
    sys.path.insert(0, str(tools_dir))
    from pose_render_utils import visualize_keypoints  # type: ignore
    from vis_pose import _get_detector, process_one_image  # type: ignore
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
    from sapiens.pose.models import init_model

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"could not decode image: {image_path}")
    image_height, image_width = image.shape[:2]
    run_args = SimpleNamespace(
        device=args.device,
        det_checkpoint=str(detector),
        bbox_thr=args.bbox_thr,
        nms_thr=args.nms_thr,
    )

    previous_cwd = Path.cwd()
    os.chdir(pose_root)
    try:
        torch_device = torch.device(args.device)
        device_index = torch_device.index if torch_device.index is not None else 0
        torch.cuda.set_device(torch_device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
        load_start = time.perf_counter()
        _get_detector(args.device, str(detector))
        detector_loaded_at = time.perf_counter()
        model = init_model(str(config), str(checkpoint), device=args.device)
        model.pose_metainfo = parse_pose_metainfo(dict(from_file="configs/_base_/keypoints308.py"))
        codec_type = model.cfg.codec.pop("type")
        if codec_type != "UDPHeatmap":
            raise RuntimeError(f"unexpected codec: {codec_type}")
        model.codec = UDPHeatmap(**model.cfg.codec)
        torch.cuda.synchronize()
        model_loaded_at = time.perf_counter()

        keypoints: list[np.ndarray] = []
        keypoint_scores: list[np.ndarray] = []
        bboxes = np.empty((0, 4), dtype=np.float32)
        for _ in range(args.warmup):
            keypoints, keypoint_scores, bboxes = process_one_image(run_args, image, model)
            torch.cuda.synchronize()

        latencies = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            keypoints, keypoint_scores, bboxes = process_one_image(run_args, image, model)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

        peak_allocated = torch.cuda.max_memory_allocated(device_index)
        peak_reserved = torch.cuda.max_memory_reserved(device_index)
        allocated_after = torch.cuda.memory_allocated(device_index)
        model_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
        flip_indices = list(model.pose_metainfo["flip_indices"])
        keypoint_name2id = dict(model.pose_metainfo["keypoint_name2id"])
    finally:
        os.chdir(previous_cwd)

    keypoint_counts = [int(np.asarray(item).shape[0]) for item in keypoints]
    coordinates_finite = all(np.isfinite(item).all() for item in keypoints)
    scores_finite = all(np.isfinite(item).all() for item in keypoint_scores)
    score_counts = [int(np.asarray(item).reshape(-1).shape[0]) for item in keypoint_scores]
    flip_mapping_involutive = all(
        0 <= paired < len(flip_indices) and flip_indices[paired] == index
        for index, paired in enumerate(flip_indices)
    )
    body_flip_pairs_valid = all(
        flip_indices[keypoint_name2id[left_name]] == keypoint_name2id[right_name]
        and flip_indices[keypoint_name2id[right_name]] == keypoint_name2id[left_name]
        for left_name, right_name in BODY_FLIP_NAMES
    )
    if keypoints:
        xy = np.concatenate([np.asarray(item, dtype=np.float64) for item in keypoints], axis=0)
        scores = np.concatenate(
            [np.asarray(item, dtype=np.float64).reshape(-1) for item in keypoint_scores]
        )
        confident = scores >= args.kpt_thr
        in_frame = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < image_width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < image_height)
        )
        confident_in_frame_fraction = (
            float(in_frame[confident].mean()) if confident.any() else None
        )
        coordinate_range = {
            "x_min": float(xy[:, 0].min()),
            "x_max": float(xy[:, 0].max()),
            "y_min": float(xy[:, 1].min()),
            "y_max": float(xy[:, 1].max()),
        }
    else:
        confident_in_frame_fraction = None
        coordinate_range = None

    status = "PASS" if all(
        (
            len(bboxes) > 0,
            bool(keypoint_counts),
            all(count == 308 for count in keypoint_counts),
            all(count == 308 for count in score_counts),
            coordinates_finite,
            scores_finite,
            flip_mapping_involutive,
            body_flip_pairs_valid,
            confident_in_frame_fraction is not None,
            confident_in_frame_fraction >= 0.95,
        )
    ) else "FAIL"
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "official_repository": "https://github.com/facebookresearch/sapiens2",
        "official_repository_commit": git_revision(sapiens2_root),
        "model_id": "facebook/sapiens2-pose-5b",
        "checkpoint_filename": checkpoint.name,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "detector_id": "facebook/detr-resnet-101-dc5",
        "detector_directory": detector.name,
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch_device),
            "compute_capability": list(torch.cuda.get_device_capability(torch_device)),
            "model_parameter_dtypes": model_dtypes,
        },
        "input": {
            "image_name": image_path.name,
            "image_width": image_width,
            "image_height": image_height,
            "pose_input_height": 1024,
            "pose_input_width": 768,
        },
        "load": {
            "detector_seconds": detector_loaded_at - load_start,
            "pose_model_seconds": model_loaded_at - detector_loaded_at,
            "total_seconds": model_loaded_at - load_start,
        },
        "inference": {
            "warmup_count": args.warmup,
            "repeat_count": args.repeats,
            "flip_test": True,
            "latency_seconds": latencies,
            "latency_median_seconds": float(np.median(latencies)),
            "latency_min_seconds": float(np.min(latencies)),
            "latency_max_seconds": float(np.max(latencies)),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocated_after_inference_bytes": allocated_after,
        },
        "output": {
            "person_count": int(len(bboxes)),
            "bboxes_xyxy": np.asarray(bboxes, dtype=float).tolist(),
            "keypoint_counts": keypoint_counts,
            "score_counts": score_counts,
            "coordinates_finite": coordinates_finite,
            "scores_finite": scores_finite,
            "coordinate_system": "original image pixel coordinates (x, y)",
            "coordinate_range": coordinate_range,
            "confident_in_frame_fraction": confident_in_frame_fraction,
            "body_left_right_flip_pairs_valid": body_flip_pairs_valid,
            "all_flip_indices_involutive": flip_mapping_involutive,
        },
    }

    if args.save_visualization is not None and keypoints:
        output_path = args.save_visualization.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualization = visualize_keypoints(
            image=image_rgb,
            keypoints=keypoints,
            keypoints_visible=np.ones_like(keypoint_scores) > 0,
            keypoint_scores=keypoint_scores,
            radius=4,
            thickness=2,
            kpt_thr=args.kpt_thr,
            skeleton=model.pose_metainfo["skeleton_links"],
            kpt_color=model.pose_metainfo["keypoint_colors"],
            link_color=model.pose_metainfo["skeleton_link_colors"],
        )
        cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        result["visualization_written"] = True

    if args.output_json is not None:
        output_json = args.output_json.resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
