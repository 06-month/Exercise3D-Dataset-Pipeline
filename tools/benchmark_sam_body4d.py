#!/usr/bin/env python3
"""Preflight and benchmark primary-target SAM 3D Body / SAM-Body4D modes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


MODE_A = "A"
MODE_B = "B"
MODE_C = "C"

COMMON_FILES = (
    "sam-3d-body-dinov3/model.ckpt",
    "sam-3d-body-dinov3/model_config.yaml",
    "sam-3d-body-dinov3/assets/mhr_model.pt",
    "moge-2-vitl-normal/model.pt",
)
MODE_FILES = {
    MODE_A: COMMON_FILES,
    MODE_B: COMMON_FILES + ("sam3/sam3.pt",),
    MODE_C: COMMON_FILES
    + (
        "sam3/sam3.pt",
        "depth_anything_v2_vitl.pth",
    ),
}
MODE_DIRS = {
    MODE_A: (),
    MODE_B: (),
    MODE_C: (
        "diffusion-vas-amodal-segmentation",
        "diffusion-vas-content-completion",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    os.replace(temporary, path)


def required_checkpoint_components(mode: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if mode not in MODE_FILES:
        raise ValueError(f"unknown SAM runtime mode: {mode}")
    return MODE_FILES[mode], MODE_DIRS[mode]


def missing_checkpoint_components(root: Path, mode: str) -> list[str]:
    required_files, required_dirs = required_checkpoint_components(mode)
    missing = [item for item in required_files if not (root / item).is_file()]
    missing.extend(
        item
        for item in required_dirs
        if not (root / item).is_dir() or not any((root / item).iterdir())
    )
    return missing


def prepare_body4d_config_text(
    source: str, checkpoint_root: Path, completion_enabled: bool
) -> str:
    checkpoint_placeholder = 'ckpt_root: "path to global checkpoint root"'
    if checkpoint_placeholder not in source:
        raise RuntimeError("unexpected SAM-Body4D checkpoint config template")
    prepared = source.replace(
        checkpoint_placeholder, f'ckpt_root: "{checkpoint_root}"', 1
    )
    completion_marker = "enable: true"
    if completion_marker not in prepared:
        raise RuntimeError("unexpected SAM-Body4D completion config template")
    return prepared.replace(
        completion_marker,
        f"enable: {'true' if completion_enabled else 'false'}",
        1,
    )


def prepare_target_input(
    input_frames: Path,
    selection_path: Path,
    source_start_index: int,
    frame_count: int,
    clip_dir: Path,
    target_path: Path,
) -> dict[str, Any]:
    source_images = sorted(input_frames.glob("*.jpg"))
    if source_start_index < 0 or frame_count < 1:
        raise RuntimeError("source start and frame count must be positive")
    source_end = min(source_start_index + frame_count, len(source_images))
    selected_images = source_images[source_start_index:source_end]
    if not selected_images:
        raise RuntimeError("no JPEG frames in requested source range")

    with np.load(selection_path, allow_pickle=False) as archive:
        required = {
            "frame_name",
            "candidate_offsets",
            "all_person_detections_xyxy",
            "target_candidate_index",
            "target_selection_confidence",
            "target_ambiguous",
            "no_target",
            "occlusion_risk",
        }
        missing = required - set(archive.files)
        if missing:
            raise RuntimeError(f"selection archive is missing: {sorted(missing)}")
        frame_names = archive["frame_name"].astype(str)
        offsets = archive["candidate_offsets"].astype(np.int64)
        all_boxes = archive["all_person_detections_xyxy"].astype(np.float32)
        target_indices = archive["target_candidate_index"].astype(np.int32)
        confidence = archive["target_selection_confidence"].astype(np.float32)
        ambiguous = archive["target_ambiguous"].astype(np.bool_)
        no_target = archive["no_target"].astype(np.bool_)
        occlusion = archive["occlusion_risk"].astype(np.bool_)

    if len(frame_names) != len(source_images):
        raise RuntimeError("selection and source frame counts differ")
    if list(frame_names) != [path.name for path in source_images]:
        raise RuntimeError("selection frame names do not match sorted source images")
    if len(offsets) != len(source_images) + 1:
        raise RuntimeError("invalid selection candidate offsets")

    indices = np.arange(source_start_index, source_end, dtype=np.int32)
    boxes = np.full((len(indices), 4), np.nan, dtype=np.float32)
    valid = np.zeros(len(indices), dtype=np.bool_)
    for local, source_index in enumerate(indices):
        candidate = int(target_indices[source_index])
        accepted = candidate >= 0 and not ambiguous[source_index] and not no_target[source_index]
        if not accepted:
            continue
        absolute = int(offsets[source_index]) + candidate
        if absolute < int(offsets[source_index]) or absolute >= int(offsets[source_index + 1]):
            raise RuntimeError("target candidate index is outside its frame slice")
        boxes[local] = all_boxes[absolute]
        valid[local] = True

    if not bool(valid[0]):
        raise RuntimeError("requested clip does not start with an accepted primary target")
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_names: list[str] = []
    for local, source in enumerate(selected_images):
        clip_name = f"{local:08d}.jpg"
        (clip_dir / clip_name).symlink_to(source)
        clip_names.append(clip_name)
    np.savez_compressed(
        target_path,
        frame_names=np.asarray(clip_names),
        source_frame_names=np.asarray([path.name for path in selected_images]),
        source_frame_indices=indices,
        target_bboxes_xyxy=boxes,
        target_valid=valid,
        target_selection_confidence=confidence[indices],
        occlusion_risk=occlusion[indices],
    )
    return {
        "frames_available": len(source_images),
        "frames_requested": len(indices),
        "target_valid_frame_count": int(valid.sum()),
        "target_ambiguous_or_missing_count": int((~valid).sum()),
        "target_seed_source_frame_index": int(indices[0]),
        "target_seed_count": 1,
        "occlusion_risk_frame_count": int(occlusion[indices].sum()),
        "target_selection_confidence_median": float(np.median(confidence[indices][valid])),
    }


class Monitor:
    FIELDS = (
        "timestamp_utc",
        "elapsed_seconds",
        "gpu_utilization_pct",
        "memory_used_mib",
        "power_draw_w",
    )

    def __init__(self, path: Path, interval: float) -> None:
        self.path = path
        self.interval = max(0.1, interval)
        self.rows: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.started = 0.0
        self.handle: Any | None = None
        self.writer: csv.DictWriter | None = None

    def start(self) -> None:
        self.started = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", newline="", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.handle is not None:
            self.handle.flush()
            self.handle.close()

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                utilization, memory, power = [value.strip() for value in raw.split(",")]
                row = {
                    "timestamp_utc": utc_now(),
                    "elapsed_seconds": time.perf_counter() - self.started,
                    "gpu_utilization_pct": utilization,
                    "memory_used_mib": memory,
                    "power_draw_w": power,
                }
                self.rows.append(row)
                if self.writer is not None:
                    self.writer.writerow(row)
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
            self.stop_event.wait(self.interval)


def summarize_values(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float | str]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not len(values):
        return {"mean": "", "p95": "", "max": ""}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def repository_revision(path: Path | None) -> str:
    if path is None or not (path / ".git").exists():
        return ""
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(MODE_A, MODE_B, MODE_C), required=True)
    parser.add_argument("--sam-body4d-root", type=Path)
    parser.add_argument("--sam-3d-body-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--input-frames", type=Path, required=True)
    parser.add_argument("--target-selection", type=Path, required=True)
    parser.add_argument("--source-start-index", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=1800)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = args.mode
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    input_frames = args.input_frames.expanduser().resolve()
    selection_path = args.target_selection.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sam_body4d_root = (
        args.sam_body4d_root.expanduser().resolve()
        if args.sam_body4d_root is not None
        else None
    )
    sam_3d_body_root = (
        args.sam_3d_body_root.expanduser().resolve()
        if args.sam_3d_body_root is not None
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    required_repository = sam_3d_body_root if mode == MODE_A else sam_body4d_root
    base: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "mode": mode,
        "implementation_repository": (
            "https://github.com/facebookresearch/sam-3d-body"
            if mode == MODE_A
            else "https://github.com/gaomingqi/sam-body4d"
        ),
        "repository_revision": repository_revision(required_repository),
        "completion_enabled": mode == MODE_C,
        "candidate": input_frames.parent.name,
        "camera": input_frames.name,
        "source_start_index": args.source_start_index,
        "frames_available": len(list(input_frames.glob("*.jpg"))),
        "frames_requested": args.frame_count,
        "frames_processed": 0,
        "persons_targeted": 1,
        "target_seed_count": 0,
        "target_valid_frame_count": 0,
        "target_ambiguous_or_missing_count": 0,
        "occlusion_risk_frame_count": 0,
        "elapsed_wall_seconds": "",
        "frames_per_second": "",
        "gpu_utilization_mean_pct": "",
        "gpu_utilization_p95_pct": "",
        "peak_nvidia_vram_mib": "",
        "power_mean_w": "",
        "power_max_w": "",
        "model_initialization_seconds": "",
        "mask_generation_seconds": "",
        "base_body_inference_seconds": "",
        "body_stage_seconds": "",
        "refinement_model_seconds": "",
        "serialization_seconds": "",
        "output_bytes": "",
        "missing_checkpoint_components": "",
        "status": "",
        "reason": "",
    }

    if required_repository is None or not required_repository.is_dir():
        base["status"] = "BLOCKED_REPOSITORY"
        base["reason"] = "required official implementation repository is unavailable"
        atomic_write_csv(output_dir / "sam_body_benchmark.csv", base)
        return 3
    if not selection_path.is_file():
        base["status"] = "BLOCKED_TARGET_SELECTION"
        base["reason"] = "target selection archive is unavailable"
        atomic_write_csv(output_dir / "sam_body_benchmark.csv", base)
        return 3

    missing = missing_checkpoint_components(checkpoint_root, mode)
    base["missing_checkpoint_components"] = ";".join(missing)
    try:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            clip_dir = temporary_root / "clip"
            target_input = temporary_root / "primary_target_input.npz"
            target_summary = prepare_target_input(
                input_frames,
                selection_path,
                args.source_start_index,
                args.frame_count,
                clip_dir,
                target_input,
            )
            base.update(target_summary)
            if missing:
                base["status"] = "BLOCKED_CHECKPOINT"
                base["reason"] = "required official pretrained payload is unavailable locally"
                atomic_write_csv(output_dir / "sam_body_benchmark.csv", base)
                print(json.dumps(base, ensure_ascii=False, indent=2))
                return 3
            if not args.run:
                base["status"] = "READY_NOT_RUN"
                base["reason"] = "pass --run after explicit approval to execute the pilot"
                atomic_write_csv(output_dir / "sam_body_benchmark.csv", base)
                print(json.dumps(base, ensure_ascii=False, indent=2))
                return 0

            execution_root = required_repository
            body4d_config: Path | None = None
            if mode in (MODE_B, MODE_C):
                execution_root = temporary_root / "sam-body4d"
                shutil.copytree(
                    required_repository,
                    execution_root,
                    ignore=shutil.ignore_patterns(".git", "outputs", "__pycache__"),
                )
                body4d_config = execution_root / "configs" / "body4d.yaml"
                body4d_config.write_text(
                    prepare_body4d_config_text(
                        body4d_config.read_text(encoding="utf-8"),
                        checkpoint_root,
                        completion_enabled=mode == MODE_C,
                    ),
                    encoding="utf-8",
                )

            run_output = output_dir / f"mode_{mode.lower()}_private_output"
            profile_json = output_dir / f"mode_{mode.lower()}_profile.json"
            log_path = output_dir / f"mode_{mode.lower()}.log"
            gpu_path = output_dir / f"mode_{mode.lower()}_gpu.csv"
            command = [
                str(args.python),
                str(Path(__file__).with_name("sam_body_primary_target_runner.py")),
                "--mode",
                mode,
                "--input-frames",
                str(clip_dir),
                "--target-input",
                str(target_input),
                "--checkpoint-root",
                str(checkpoint_root),
                "--output-dir",
                str(run_output),
                "--profile-json",
                str(profile_json),
            ]
            if mode == MODE_A:
                command.extend(["--sam-3d-body-root", str(execution_root)])
            else:
                command.extend(
                    [
                        "--sam-body4d-root",
                        str(execution_root),
                        "--body4d-config",
                        str(body4d_config),
                    ]
                )

            monitor = Monitor(gpu_path, args.sample_interval)
            started = time.perf_counter()
            monitor.start()
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=execution_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
            finally:
                monitor.stop()
            elapsed = time.perf_counter() - started
            gpu = summarize_values(monitor.rows, "gpu_utilization_pct")
            memory = summarize_values(monitor.rows, "memory_used_mib")
            power = summarize_values(monitor.rows, "power_draw_w")
            profile = (
                json.loads(profile_json.read_text(encoding="utf-8"))
                if profile_json.is_file()
                else {}
            )
            processed = int(profile.get("frames_processed", 0))
            base.update(
                {
                    "frames_processed": processed,
                    "elapsed_wall_seconds": elapsed,
                    "frames_per_second": processed / elapsed if processed else "",
                    "gpu_utilization_mean_pct": gpu["mean"],
                    "gpu_utilization_p95_pct": gpu["p95"],
                    "peak_nvidia_vram_mib": memory["max"],
                    "power_mean_w": power["mean"],
                    "power_max_w": power["max"],
                    "model_initialization_seconds": profile.get(
                        "model_initialization_seconds", ""
                    ),
                    "mask_generation_seconds": profile.get(
                        "mask_generation_seconds", ""
                    ),
                    "base_body_inference_seconds": profile.get(
                        "base_body_inference_seconds",
                        profile.get("base_body_and_serialization_residual_seconds", ""),
                    ),
                    "body_stage_seconds": profile.get("body_stage_seconds", ""),
                    "refinement_model_seconds": profile.get(
                        "refinement_model_seconds", ""
                    ),
                    "serialization_seconds": profile.get(
                        "serialization_seconds", ""
                    ),
                    "output_bytes": directory_size(run_output),
                    "status": "PASS" if process.returncode == 0 else "FAIL",
                    "reason": (
                        "" if process.returncode == 0 else f"runner exit code {process.returncode}"
                    ),
                }
            )
    except (OSError, RuntimeError, ValueError) as exc:
        base["status"] = "BLOCKED_TARGET_INPUT"
        base["reason"] = str(exc)
        atomic_write_csv(output_dir / "sam_body_benchmark.csv", base)
        print(json.dumps(base, ensure_ascii=False, indent=2))
        return 3

    atomic_write_csv(output_dir / "sam_body_benchmark.csv", base)
    print(json.dumps(base, ensure_ascii=False, indent=2))
    return 0 if base["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
