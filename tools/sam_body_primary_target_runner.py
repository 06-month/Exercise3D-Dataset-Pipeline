#!/usr/bin/env python3
"""Run official SAM body implementations for exactly one preselected subject.

This is a project adapter, not an upstream CLI extension.  Mode A calls the
official SAM 3D Body estimator with one precomputed bbox per frame.  Modes B/C
instantiate the official SAM-Body4D classes, seed SAM 3 with exactly one bbox,
and bypass the upstream initial-frame all-human detector.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


MODE_A = "A"
MODE_B = "B"
MODE_C = "C"
BODY_PRIOR_KEYS = (
    "bbox",
    "focal_length",
    "pred_keypoints_3d",
    "pred_keypoints_2d",
    "pred_cam_t",
    "pred_pose_raw",
    "global_rot",
    "body_pose_params",
    "hand_pose_params",
    "scale_params",
    "shape_params",
    "expr_params",
    "pred_joint_coords",
    "pred_global_rots",
    "mhr_model_params",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(MODE_A, MODE_B, MODE_C), required=True)
    parser.add_argument("--input-frames", type=Path, required=True)
    parser.add_argument("--target-input", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--sam-3d-body-root", type=Path)
    parser.add_argument("--sam-body4d-root", type=Path)
    parser.add_argument("--body4d-config", type=Path)
    return parser.parse_args()


def sync_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def timed_call(torch_module: Any, function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    sync_cuda(torch_module)
    started = time.perf_counter()
    result = function(*args, **kwargs)
    sync_cuda(torch_module)
    return result, time.perf_counter() - started


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_target_input(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "frame_names",
            "source_frame_indices",
            "target_bboxes_xyxy",
            "target_valid",
            "target_selection_confidence",
            "target_ambiguous",
            "no_target",
            "occlusion_risk",
            "timestamp_pts_seconds",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"target input is missing fields: {sorted(missing)}")
        result = {key: archive[key].copy() for key in required}
        if "source_frame_names" in archive.files:
            result["source_frame_names"] = archive["source_frame_names"].copy()
    frame_count = len(result["frame_names"])
    if result["target_bboxes_xyxy"].shape != (frame_count, 4):
        raise RuntimeError("target input must contain exactly one bbox slot per frame")
    for key in required - {"target_bboxes_xyxy"}:
        if len(result[key]) != frame_count:
            raise RuntimeError(f"target input length mismatch: {key}")
    valid_boxes = result["target_bboxes_xyxy"][result["target_valid"]]
    if len(valid_boxes) and (
        not np.isfinite(valid_boxes).all()
        or (valid_boxes[:, 2:] <= valid_boxes[:, :2]).any()
    ):
        raise RuntimeError("target input contains invalid accepted bbox")
    return result


def numeric_payload(person: dict[str, Any]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for key, value in person.items():
        if value is None or isinstance(value, dict):
            continue
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            continue
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite SAM 3D Body output: {key}")
        payload[key] = array
    return payload


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def save_body_prior_numeric(
    outputs: list[dict[str, Any]] | None,
    image_path: str,
    object_ids: list[Any] | None,
    output_root: Path,
) -> tuple[int, set[str]]:
    """Persist compact MHR parameters before upstream converts them to PLY."""
    if outputs is None:
        return 0, set()
    if len(outputs) > 1:
        raise RuntimeError("primary-target invariant violated while saving MHR prior")
    saved = 0
    keys: set[str] = set()
    stem = Path(image_path).stem
    for index, person in enumerate(outputs):
        object_id = int(object_ids[index]) if object_ids else index + 1
        numeric = numeric_payload(person)
        payload = {key: numeric[key] for key in BODY_PRIOR_KEYS if key in numeric}
        if not payload:
            raise RuntimeError(f"no numeric MHR prior fields at frame {stem}")
        payload["object_id"] = np.asarray(object_id, dtype=np.int32)
        atomic_savez(output_root / str(object_id) / f"{stem}.npz", **payload)
        saved += 1
        keys.update(key for key in payload if key != "object_id")
    return saved, keys


def run_mode_a(args: argparse.Namespace, target: dict[str, np.ndarray]) -> dict[str, Any]:
    if args.sam_3d_body_root is None:
        raise RuntimeError("--sam-3d-body-root is required for mode A")
    upstream = args.sam_3d_body_root.expanduser().resolve()
    sys.path.insert(0, str(upstream))

    import torch
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    from tools.build_fov_estimator import FOVEstimator

    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoint = checkpoint_root / "sam-3d-body-dinov3" / "model.ckpt"
    mhr = checkpoint_root / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt"
    fov = checkpoint_root / "moge-2-vitl-normal" / "model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init_started = time.perf_counter()
    model, model_config = load_sam_3d_body(
        str(checkpoint), device=device, mhr_path=str(mhr)
    )
    fov_estimator = FOVEstimator(name="moge2", device=device, path=str(fov))
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_config,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )
    sync_cuda(torch)
    initialization_seconds = time.perf_counter() - init_started

    paths = [args.input_frames / str(name) for name in target["frame_names"]]
    output_dir = args.output_dir / "mode_a_numeric"
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_seconds = 0.0
    serialization_seconds = 0.0
    processed = 0
    skipped = 0
    output_keys: set[str] = set()
    for offset, (image_path, bbox, valid) in enumerate(
        zip(paths, target["target_bboxes_xyxy"], target["target_valid"])
    ):
        if not bool(valid):
            skipped += 1
            continue
        outputs, elapsed = timed_call(
            torch,
            estimator.process_one_image,
            str(image_path),
            bboxes=np.asarray([bbox], dtype=np.float32),
        )
        inference_seconds += elapsed
        if len(outputs) != 1:
            raise RuntimeError(
                f"primary-target invariant violated at clip frame {offset}: {len(outputs)} persons"
            )
        payload = numeric_payload(outputs[0])
        output_keys.update(payload)
        started = time.perf_counter()
        np.savez_compressed(output_dir / f"{offset:08d}.npz", **payload)
        serialization_seconds += time.perf_counter() - started
        processed += 1

    return {
        "mode": MODE_A,
        "completion_enabled": False,
        "target_seed_count": 1,
        "persons_processed": processed,
        "frames_processed": processed,
        "frames_skipped_ambiguous": skipped,
        "model_initialization_seconds": initialization_seconds,
        "mask_generation_seconds": 0.0,
        "base_body_inference_seconds": inference_seconds,
        "refinement_model_seconds": 0.0,
        "serialization_seconds": serialization_seconds,
        "output_keys": sorted(output_keys),
        "output_bytes": directory_size(args.output_dir),
    }


class TimedCallable:
    def __init__(self, wrapped: Any, torch_module: Any) -> None:
        self.wrapped = wrapped
        self.torch = torch_module
        self.calls = 0
        self.seconds = 0.0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result, elapsed = timed_call(self.torch, self.wrapped, *args, **kwargs)
        self.calls += 1
        self.seconds += elapsed
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


class TimedDepthModel:
    def __init__(self, wrapped: Any, torch_module: Any) -> None:
        self.wrapped = wrapped
        self.torch = torch_module
        self.calls = 0
        self.seconds = 0.0

    def infer_image(self, *args: Any, **kwargs: Any) -> Any:
        result, elapsed = timed_call(self.torch, self.wrapped.infer_image, *args, **kwargs)
        self.calls += 1
        self.seconds += elapsed
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.wrapped, name)


def load_offline_module(upstream: Path) -> Any:
    sys.path.insert(0, str(upstream))
    source = upstream / "scripts" / "offline_app.py"
    specification = importlib.util.spec_from_file_location(
        "exercise3d_sam_body4d_offline", source
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot import official offline runner: {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run_mode_body4d(args: argparse.Namespace, target: dict[str, np.ndarray]) -> dict[str, Any]:
    if args.sam_body4d_root is None or args.body4d_config is None:
        raise RuntimeError("--sam-body4d-root and --body4d-config are required for modes B/C")
    if not bool(target["target_valid"][0]):
        raise RuntimeError("Body4D clip frame 0 is not an accepted primary target seed")

    upstream = args.sam_body4d_root.expanduser().resolve()
    module = load_offline_module(upstream)
    torch = module.torch

    # The official OfflineApp builds ViTDet for its all-human initialization.
    # This adapter supplies the accepted Phase 6 bbox directly, so the detector
    # is intentionally omitted while the official estimator/FOV API is kept.
    def build_target_only_estimator(config: Any) -> Any:
        model, model_config = module.load_sam_3d_body(
            config.sam_3d_body["ckpt_path"],
            device=module.device,
            mhr_path=config.sam_3d_body["mhr_path"],
        )
        from models.sam_3d_body.tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name="moge2", device=module.device, path=config.sam_3d_body["fov_path"]
        )
        return module.SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_config,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=fov_estimator,
        )

    module.build_sam3_3d_body_config = build_target_only_estimator
    init_started = time.perf_counter()
    app = module.OfflineApp(config_path=str(args.body4d_config))
    sync_cuda(torch)
    initialization_seconds = time.perf_counter() - init_started

    mask_timer = TimedCallable(app.pipeline_mask, torch) if app.pipeline_mask is not None else None
    rgb_timer = TimedCallable(app.pipeline_rgb, torch) if app.pipeline_rgb is not None else None
    depth_timer = TimedDepthModel(app.depth_model, torch) if app.depth_model is not None else None
    if mask_timer is not None:
        app.pipeline_mask = mask_timer
    if rgb_timer is not None:
        app.pipeline_rgb = rgb_timer
    if depth_timer is not None:
        app.depth_model = depth_timer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    app.OUTPUT_DIR = str(args.output_dir)
    prior_state: dict[str, Any] = {"files": 0, "keys": set()}
    original_mesh_saver = getattr(module, "save_mesh_results", None)
    if original_mesh_saver is not None:
        def save_mesh_and_prior(*call_args: Any, **call_kwargs: Any) -> Any:
            outputs = call_kwargs.get("outputs", call_args[0] if call_args else None)
            image_path = call_kwargs.get(
                "image_path", call_args[4] if len(call_args) > 4 else ""
            )
            object_ids = call_kwargs.get(
                "id_current", call_args[5] if len(call_args) > 5 else None
            )
            saved, keys = save_body_prior_numeric(
                outputs,
                str(image_path),
                object_ids,
                args.output_dir / "mhr_numeric",
            )
            prior_state["files"] += saved
            prior_state["keys"].update(keys)
            return original_mesh_saver(*call_args, **call_kwargs)

        module.save_mesh_results = save_mesh_and_prior
    image_paths = [str(args.input_frames / str(name)) for name in target["frame_names"]]
    from PIL import Image

    width, height = Image.open(image_paths[0]).size
    xmin, ymin, xmax, ymax = target["target_bboxes_xyxy"][0]
    relative_box = np.asarray(
        [[xmin / width, ymin / height, xmax / width, ymax / height]],
        dtype=np.float32,
    )
    state = app.predictor.init_state(video_path=image_paths)
    app.predictor.clear_all_points_in_video(state)
    app.RUNTIME["inference_state"] = state
    app.RUNTIME["out_obj_ids"] = []
    _, app.RUNTIME["out_obj_ids"], _, _ = app.predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=0,
        obj_id=1,
        box=relative_box,
    )
    if len(app.RUNTIME["out_obj_ids"]) != 1:
        raise RuntimeError("SAM 3 primary-target seed did not produce exactly one object")

    _, mask_seconds = timed_call(
        torch,
        app.on_mask_generation,
        start_frame_idx=0,
        max_frame_num_to_track=len(image_paths),
    )
    with torch.autocast("cuda", enabled=False):
        _, body_seconds = timed_call(torch, app.on_4d_generation)

    refinement_seconds = sum(
        timer.seconds for timer in (mask_timer, rgb_timer, depth_timer) if timer is not None
    )
    refinement_calls = sum(
        timer.calls for timer in (mask_timer, rgb_timer, depth_timer) if timer is not None
    )
    mesh_count = len(list(args.output_dir.rglob("*.ply")))
    if mesh_count > len(image_paths):
        raise RuntimeError("primary-target invariant violated: more than one mesh per frame")
    return {
        "mode": args.mode,
        "completion_enabled": args.mode == MODE_C,
        "target_seed_count": 1,
        "persons_processed": 1,
        "frames_processed": len(image_paths),
        "frames_skipped_ambiguous": 0,
        "model_initialization_seconds": initialization_seconds,
        "mask_generation_seconds": mask_seconds,
        "body_stage_seconds": body_seconds,
        "base_body_and_serialization_residual_seconds": max(
            0.0, body_seconds - refinement_seconds
        ),
        "refinement_model_seconds": refinement_seconds,
        "refinement_model_calls": refinement_calls,
        "amodal_segmentation_calls": mask_timer.calls if mask_timer else 0,
        "content_completion_calls": rgb_timer.calls if rgb_timer else 0,
        "depth_inference_calls": depth_timer.calls if depth_timer else 0,
        "mesh_file_count": mesh_count,
        "numeric_prior_file_count": int(prior_state["files"]),
        "numeric_prior_keys": sorted(prior_state["keys"]),
        "output_bytes": directory_size(args.output_dir),
    }


def main() -> int:
    args = parse_args()
    target = load_target_input(args.target_input.expanduser().resolve())
    frame_names = [str(item) for item in target["frame_names"]]
    actual_names = [item.name for item in sorted(args.input_frames.glob("*.jpg"))]
    if frame_names != actual_names:
        raise RuntimeError("target input frame names do not match clip images")
    if not len(frame_names):
        raise RuntimeError("empty target input")
    atomic_savez(
        args.output_dir / "target_provenance.npz",
        **{key: value for key, value in target.items()},
    )
    started = time.perf_counter()
    profile = run_mode_a(args, target) if args.mode == MODE_A else run_mode_body4d(args, target)
    profile["elapsed_runner_seconds"] = time.perf_counter() - started
    profile["input_frames"] = len(frame_names)
    profile["target_input_schema"] = "one_bbox_slot_per_frame"
    args.profile_json.parent.mkdir(parents=True, exist_ok=True)
    args.profile_json.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
