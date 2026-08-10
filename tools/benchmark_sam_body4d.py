#!/usr/bin/env python3
"""Preflight and benchmark the upstream SAM-Body4D offline pipeline."""

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
from typing import Any


REQUIRED_FILES = (
    "sam3/sam3.pt",
    "sam-3d-body-dinov3/model.ckpt",
    "sam-3d-body-dinov3/assets/mhr_model.pt",
    "moge-2-vitl-normal/model.pt",
    "depth_anything_v2_vitl.pth",
    "vitdet/model_final_f05665.pkl",
)
REQUIRED_DIRS = (
    "diffusion-vas-amodal-segmentation",
    "diffusion-vas-content-completion",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    os.replace(temporary, path)


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
        self.handle = self.path.open(
            "w", newline="", encoding="utf-8", buffering=1
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-body4d-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--input-frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("python"))
    parser.add_argument("--frame-count", type=int, default=1800)
    parser.add_argument("--refiner", choices=("on", "off"), default="on")
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.sam_body4d_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    input_frames = args.input_frames.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    images = sorted(input_frames.glob("*.jpg"))[: args.frame_count]
    missing_files = [item for item in REQUIRED_FILES if not (checkpoint_root / item).is_file()]
    missing_dirs = [
        item
        for item in REQUIRED_DIRS
        if not (checkpoint_root / item).is_dir()
        or not any((checkpoint_root / item).iterdir())
    ]
    missing = missing_files + missing_dirs
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    base = {
        "created_at_utc": utc_now(),
        "implementation_repository": "https://github.com/gaomingqi/sam-body4d",
        "repository_revision": revision,
        "configuration": f"refiner_{args.refiner}",
        "candidate": input_frames.parent.name,
        "camera": input_frames.name,
        "frames_requested": args.frame_count,
        "frames_available": len(list(input_frames.glob('*.jpg'))),
        "frames_processed": 0,
        "runtime_seconds": "",
        "seconds_per_frame": "",
        "peak_vram_mib": "",
        "gpu_utilization_mean_pct": "",
        "power_mean_w": "",
        "refiner_invocation_count": "",
        "temporary_disk_bytes": "",
        "output_bytes": "",
        "missing_checkpoint_components": ";".join(missing),
        "status": "",
        "reason": "",
    }
    if not images:
        base["status"] = "BLOCKED_INPUT"
        base["reason"] = "no JPEG input frames"
        write_csv(output_dir / "sam_body4d_benchmark.csv", base)
        return 3
    if missing:
        base["status"] = "BLOCKED_CHECKPOINT"
        base["reason"] = "required pretrained payload is unavailable locally"
        write_csv(output_dir / "sam_body4d_benchmark.csv", base)
        print(json.dumps(base, ensure_ascii=False, indent=2))
        return 3
    if not args.run:
        base["status"] = "READY_NOT_RUN"
        base["reason"] = "pass --run to execute the official offline pipeline"
        write_csv(output_dir / "sam_body4d_benchmark.csv", base)
        print(json.dumps(base, ensure_ascii=False, indent=2))
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_csv = output_dir / f"gpu_refiner_{args.refiner}.csv"
    monitor = Monitor(gpu_csv, args.sample_interval)
    log_path = output_dir / f"sam_body4d_refiner_{args.refiner}.log"
    before_disk = directory_size(output_dir)
    return_code = -1
    elapsed = 0.0
    with tempfile.TemporaryDirectory() as temporary:
        temporary_root = Path(temporary)
        execution_repo = temporary_root / "sam-body4d"
        shutil.copytree(
            repository,
            execution_repo,
            ignore=shutil.ignore_patterns(".git", "outputs", "__pycache__"),
        )
        config_path = execution_repo / "configs" / "body4d.yaml"
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        original_config = config_path.read_text(encoding="utf-8")
        checkpoint_placeholder = 'ckpt_root: "path to global checkpoint root"'
        if checkpoint_placeholder not in original_config:
            raise RuntimeError("unexpected SAM-Body4D checkpoint config template")
        prepared = original_config.replace(
            checkpoint_placeholder, f'ckpt_root: "{checkpoint_root}"'
        )
        if "enable: true" not in prepared:
            raise RuntimeError("unexpected SAM-Body4D completion config template")
        detector_placeholder = 'detector_path: ""'
        if detector_placeholder not in prepared:
            raise RuntimeError("unexpected SAM-Body4D detector config template")
        prepared = prepared.replace(
            detector_placeholder,
            f'detector_path: "{checkpoint_root / "vitdet"}"',
        )
        prepared = prepared.replace(
            "enable: true",
            f"enable: {'true' if args.refiner == 'on' else 'false'}",
            1,
        )
        config_path.write_text(prepared, encoding="utf-8")

        clip_dir = temporary_root / "clip"
        clip_dir.mkdir()
        for index, source in enumerate(images):
            (clip_dir / f"{index:08d}.jpg").symlink_to(source)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_output = output_dir / f"refiner_{args.refiner}_output_{run_id}_{os.getpid()}"
        started = time.perf_counter()
        try:
            monitor.start()
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    [
                        str(args.python),
                        str(execution_repo / "scripts" / "offline_app.py"),
                        "--input_video",
                        str(clip_dir),
                        "--output_dir",
                        str(run_output),
                    ],
                    cwd=execution_repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            return_code = process.returncode
        finally:
            monitor.stop()
            elapsed = time.perf_counter() - started
    gpu_values = [float(row["gpu_utilization_pct"]) for row in monitor.rows]
    memory_values = [float(row["memory_used_mib"]) for row in monitor.rows]
    power_values = [float(row["power_draw_w"]) for row in monitor.rows]
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    base.update(
        {
            "frames_processed": len(images) if return_code == 0 else 0,
            "runtime_seconds": elapsed,
            "seconds_per_frame": elapsed / len(images),
            "peak_vram_mib": max(memory_values) if memory_values else "",
            "gpu_utilization_mean_pct": (
                sum(gpu_values) / len(gpu_values) if gpu_values else ""
            ),
            "power_mean_w": sum(power_values) / len(power_values) if power_values else "",
            "refiner_invocation_count": log_text.count(
                "content completion by diffusion-vas"
            ),
            "temporary_disk_bytes": max(0, directory_size(output_dir) - before_disk),
            "output_bytes": directory_size(run_output) if run_output.exists() else 0,
            "status": "PASS" if return_code == 0 else "FAIL",
            "reason": "" if return_code == 0 else f"upstream process exit code {return_code}",
        }
    )
    write_csv(output_dir / "sam_body4d_benchmark.csv", base)
    print(json.dumps(base, ensure_ascii=False, indent=2))
    return 0 if return_code == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
