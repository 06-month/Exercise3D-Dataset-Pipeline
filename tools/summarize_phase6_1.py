#!/usr/bin/env python3
"""Build the redacted Phase 6-1 baseline/target-only comparison and gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PILOT_SEQUENCES = (
    "barbellrow_0000",
    "squat_0001",
    "pushup_0001",
    "benchpress_0003",
)
CAMERAS = ("cam1", "cam2", "cam3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-output-root", type=Path, required=True)
    parser.add_argument("--baseline-runtime-dir", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--selection-runtime-dir", type=Path, required=True)
    parser.add_argument("--target-output-root", type=Path, required=True)
    parser.add_argument("--target-runtime-dir", type=Path, required=True)
    parser.add_argument("--target-benchmark-dir", type=Path, required=True)
    parser.add_argument("--public-results-dir", type=Path, required=True)
    parser.add_argument("--dataset-frame-count", type=int, default=65_595)
    parser.add_argument(
        "--visual-qa-status", choices=("PASS", "REVIEW", "FAIL"), default="REVIEW"
    )
    parser.add_argument(
        "--background-misselection-count",
        type=int,
        default=-1,
        help="-1 means the manual review is incomplete",
    )
    parser.add_argument("--max-ambiguous-fraction", type=float, default=0.01)
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def numeric(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def preservation_manifest(baseline_root: Path, runtime: Path) -> dict[str, Any]:
    files: list[tuple[str, Path]] = []
    for sequence in PILOT_SEQUENCES:
        for camera in CAMERAS:
            directory = baseline_root / sequence / camera
            for name in ("poses_2d.npz", "bboxes.npz", "frames.csv", "metadata.json"):
                files.append((f"{sequence}/{camera}/{name}", directory / name))
    for name in (
        "batch_scaling.csv",
        "benchmark_summary.json",
        "gpu_utilization.csv",
        "sapiens2_benchmark.csv",
        "sapiens2_pilot_benchmark.csv",
        "pilot_gpu_utilization.csv",
        "pilot_qa.csv",
        "pilot_summary.json",
    ):
        files.append((f"runtime/{name}", runtime / name))
    records = []
    for label, path in files:
        if not path.exists():
            records.append({"file": label, "exists": False, "size_bytes": 0, "sha256": ""})
            continue
        records.append(
            {
                "file": label,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "workload": "ALL_DETECTIONS_BASELINE",
        "all_required_files_present": all(record["exists"] for record in records),
        "files": records,
    }


def gpu_summary(path: Path) -> dict[str, float | int | None]:
    if not path.exists():
        return {
            "sample_count": 0,
            "gpu_utilization_mean": None,
            "gpu_utilization_p90": None,
            "gpu_utilization_max": None,
            "memory_used_mib_max": None,
            "power_draw_w_mean": None,
            "power_draw_w_max": None,
        }
    rows = read_csv(path)
    result: dict[str, float | int | None] = {"sample_count": len(rows)}
    for field, prefix in (
        ("gpu_utilization_pct", "gpu_utilization"),
        ("memory_used_mib", "memory_used_mib"),
        ("power_draw_w", "power_draw_w"),
    ):
        values = [numeric(row.get(field), np.nan) for row in rows]
        values = [value for value in values if np.isfinite(value)]
        result[f"{prefix}_mean"] = float(np.mean(values)) if values else None
        result[f"{prefix}_p90"] = float(np.percentile(values, 90)) if values else None
        result[f"{prefix}_max"] = float(np.max(values)) if values else None
    return result


def baseline_outputs(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for sequence in PILOT_SEQUENCES:
        for camera in CAMERAS:
            output = root / sequence / camera
            required = [
                output / "poses_2d.npz",
                output / "bboxes.npz",
                output / "frames.csv",
                output / "metadata.json",
            ]
            missing = [path.name for path in required if not path.exists()]
            if missing:
                errors.append(f"{sequence}/{camera}:missing:{','.join(missing)}")
                continue
            try:
                with np.load(required[0], allow_pickle=False) as pose, np.load(
                    required[1], allow_pickle=False
                ) as bbox:
                    count = len(pose["frame_index"])
                    person_count = bbox["person_count"]
                    rows.append(
                        {
                            "sequence": sequence,
                            "camera": camera,
                            "frame_count": count,
                            "person_crop_count": int(person_count.sum()),
                            "mean_detected_persons_per_frame": float(person_count.mean()),
                            "multi_person_frame_count": int((person_count > 1).sum()),
                            "detector_fallback_count": int(bbox["detector_fallback"].sum()),
                            "output_preserved": True,
                        }
                    )
            except (OSError, KeyError, ValueError) as exc:
                errors.append(f"{sequence}/{camera}:invalid:{exc}")
    return rows, errors


def target_outputs(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    errors = []
    for sequence in PILOT_SEQUENCES:
        for camera in CAMERAS:
            output = root / sequence / camera
            required = [
                output / "poses_2d.npz",
                output / "bboxes.npz",
                output / "frames.csv",
                output / "metadata.json",
            ]
            missing = [path.name for path in required if not path.exists()]
            if missing:
                errors.append(f"{sequence}/{camera}:missing:{','.join(missing)}")
                continue
            metadata = read_json(required[3])
            qa = metadata.get("qa", {})
            rows.append(
                {
                    "sequence": sequence,
                    "camera": camera,
                    "frame_count": int(qa.get("frame_count", 0)),
                    "target_pose_count": int(qa.get("target_pose_count", 0)),
                    "target_ambiguous_count": int(qa.get("target_ambiguous_count", 0)),
                    "no_target_count": int(qa.get("no_target_count", 0)),
                    "forced_invalid_pose_count": int(qa.get("forced_ambiguous_pose_count", 0))
                    + int(qa.get("forced_no_target_pose_count", 0)),
                    "status": qa.get("status", "FAIL"),
                }
            )
    return rows, errors


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset_frame_count < 1 or not 0 <= args.max_ambiguous_fraction <= 1:
        raise RuntimeError("invalid dataset/gate limits")
    baseline_root = args.baseline_output_root.expanduser().resolve()
    baseline_runtime = args.baseline_runtime_dir.expanduser().resolve()
    selection_root = args.selection_root.expanduser().resolve()
    selection_runtime = args.selection_runtime_dir.expanduser().resolve()
    target_root = args.target_output_root.expanduser().resolve()
    target_runtime = args.target_runtime_dir.expanduser().resolve()
    target_benchmark = args.target_benchmark_dir.expanduser().resolve()
    public = args.public_results_dir.expanduser().resolve()

    baseline_rows, baseline_errors = baseline_outputs(baseline_root)
    target_rows, target_errors = target_outputs(target_root)
    selection_rows = read_csv(selection_runtime / "target_selection_pilot.csv")
    batch_rows = read_csv(baseline_runtime / "batch_scaling.csv")
    target_batch_rows = read_csv(target_benchmark / "target_only_batch_scaling.csv")
    baseline_summary = read_json(baseline_runtime / "pilot_summary.json")
    target_summary = read_json(target_runtime / "target_only_pilot_summary.json")
    selection_summary = read_json(selection_runtime / "target_selection_summary.json")
    equivalence_path = target_runtime / "target_only_equivalence.csv"
    equivalence_rows = read_csv(equivalence_path) if equivalence_path.exists() else []

    passing_batches = [row for row in batch_rows if row.get("status") == "PASS"]
    passing_target_batches = [
        row for row in target_batch_rows if row.get("status") == "PASS"
    ]
    if not passing_batches or not passing_target_batches:
        raise RuntimeError("baseline and target-only passing batches are required")
    baseline_best = max(passing_batches, key=lambda row: numeric(row["images_per_second"]))
    target_best = max(
        passing_target_batches, key=lambda row: numeric(row["images_per_second"])
    )
    baseline_frames = sum(row["frame_count"] for row in baseline_rows)
    baseline_crops = sum(row["person_crop_count"] for row in baseline_rows)
    selected_crops = int(selection_summary["target_only_sapiens_crops"])
    ambiguous = int(selection_summary["target_ambiguous_count"])
    no_target = int(selection_summary["no_target_count"])
    identity_switches = int(selection_summary["identity_switch_count"])
    baseline_elapsed = numeric(baseline_summary["total_elapsed_seconds"])
    target_elapsed = numeric(target_summary["total_elapsed_seconds"])
    baseline_frame_rate = baseline_frames / max(baseline_elapsed, 1e-12)
    baseline_crop_rate = baseline_crops / max(baseline_elapsed, 1e-12)
    target_frame_rate = sum(row["frame_count"] for row in target_rows) / max(target_elapsed, 1e-12)
    target_crop_rate = sum(row["target_pose_count"] for row in target_rows) / max(target_elapsed, 1e-12)

    detector_stages = sum(
        numeric(baseline_best.get(field))
        for field in (
            "detector_preprocess_seconds",
            "detector_forward_seconds",
            "detector_postprocess_seconds",
        )
    )
    detector_rate = numeric(baseline_best["images"]) / max(detector_stages, 1e-12)
    selector_seconds = sum(numeric(row.get("selection_elapsed_seconds")) for row in selection_rows)
    selector_frames = sum(int(row["frame_count"]) for row in selection_rows)
    selector_rate = selector_frames / max(selector_seconds, 1e-12)
    eligible_fraction = selected_crops / max(baseline_frames, 1)
    full_all_seconds = args.dataset_frame_count / max(baseline_frame_rate, 1e-12)
    full_target_seconds = (
        args.dataset_frame_count / max(detector_rate, 1e-12)
        + args.dataset_frame_count / max(selector_rate, 1e-12)
        + args.dataset_frame_count * eligible_fraction / max(target_crop_rate, 1e-12)
    )
    crop_reduction = 1.0 - selected_crops / max(baseline_crops, 1)
    runtime_reduction = 1.0 - full_target_seconds / max(full_all_seconds, 1e-12)
    pilot_gpu_path = baseline_runtime / "pilot_gpu_utilization.csv"
    if not pilot_gpu_path.exists():
        pilot_gpu_path = baseline_runtime / "pilot_gpu_utilization_recovery.csv"

    baseline_complete = (
        len(baseline_rows) == len(PILOT_SEQUENCES) * len(CAMERAS)
        and not baseline_errors
        and baseline_frames == int(baseline_summary["frame_count"])
    )
    selection_complete = len(selection_rows) == len(PILOT_SEQUENCES) * len(CAMERAS)
    target_complete = (
        len(target_rows) == len(PILOT_SEQUENCES) * len(CAMERAS)
        and not target_errors
        and all(row["status"] == "PASS" for row in target_rows)
        and all(row["forced_invalid_pose_count"] == 0 for row in target_rows)
    )
    ambiguity_fraction = (ambiguous + no_target) / max(baseline_frames, 1)
    benchmark_valid = all(
        row.get("numerically_equivalent", "").lower() == "true"
        for row in passing_target_batches
    )
    selected_pose_equivalent = (
        len(equivalence_rows) == len(PILOT_SEQUENCES) * len(CAMERAS)
        and all(row.get("status") == "PASS" for row in equivalence_rows)
    )
    preserved = preservation_manifest(baseline_root, baseline_runtime)
    atomic_write_text(
        baseline_runtime / "all_detections_preservation_manifest.json",
        json.dumps(preserved, indent=2) + "\n",
    )
    hard_failure = (
        not baseline_complete
        or not selection_complete
        or not target_complete
        or not benchmark_valid
        or not selected_pose_equivalent
        or not preserved["all_required_files_present"]
        or args.visual_qa_status == "FAIL"
        or identity_switches > 0
        or args.background_misselection_count > 0
    )
    manual_incomplete = (
        args.visual_qa_status != "PASS"
        or args.background_misselection_count < 0
        or ambiguity_fraction > args.max_ambiguous_fraction
    )
    if hard_failure:
        gate = "NO_GO"
    elif manual_incomplete:
        gate = "REVIEW_TARGET_SELECTION"
    else:
        gate = "GO_FULL_DATASET"

    baseline_public = []
    for row in baseline_rows:
        baseline_public.append({"workload": "ALL_DETECTIONS_BASELINE", **row})
    atomic_write_csv(
        public / "sapiens2_all_detections_baseline.csv",
        list(baseline_public[0]),
        baseline_public,
    )
    selection_fields = [
        "sequence",
        "camera",
        "frame_count",
        "total_person_candidates",
        "mean_detected_persons_per_frame",
        "all_detections_sapiens_crops",
        "target_only_sapiens_crops",
        "crop_reduction_fraction",
        "target_ambiguous_count",
        "no_target_count",
        "background_person_count",
        "multi_person_frame_count",
        "occlusion_risk_count",
        "detector_duplicate_candidate_count",
        "possible_reflection_candidate_count",
        "forward_backward_disagreement_count",
        "identity_switch_count",
        "target_bbox_center_jump_normalized_p95",
        "target_bbox_log_area_jump_p95",
        "target_visibility_transition_count",
        "target_gap_segment_count",
        "status",
    ]
    atomic_write_csv(
        public / "target_selection_pilot.csv",
        selection_fields,
        [{field: row.get(field, "") for field in selection_fields} for row in selection_rows],
    )
    target_batch_fields = [
        "batch_size",
        "status",
        "images",
        "person_crops",
        "elapsed_seconds",
        "images_per_second",
        "person_crops_per_second",
        "effective_seconds_per_image",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "pose_gpu_utilization_mean",
        "pose_gpu_utilization_p90",
        "pose_gpu_utilization_max",
        "pose_memory_used_mib_max",
        "pose_power_draw_w_mean",
        "pose_power_draw_w_max",
        "max_confident_xy_delta_vs_batch1_px",
        "max_confidence_delta_vs_batch1",
        "numerically_equivalent",
    ]
    atomic_write_csv(
        public / "sapiens2_target_only_batch_scaling.csv",
        target_batch_fields,
        [{field: row.get(field, "") for field in target_batch_fields} for row in target_batch_rows],
    )
    comparison = [
        {
            "workload": "ALL_DETECTIONS_BASELINE",
            "pilot_frames": baseline_frames,
            "pilot_person_crops": baseline_crops,
            "mean_detected_persons_per_frame": baseline_crops / max(baseline_frames, 1),
            "mean_sapiens_crops_per_frame": baseline_crops / max(baseline_frames, 1),
            "frames_per_second": baseline_frame_rate,
            "person_crops_per_second": baseline_crop_rate,
            "estimated_full_dataset_seconds": full_all_seconds,
        },
        {
            "workload": "TARGET_ONLY",
            "pilot_frames": baseline_frames,
            "pilot_person_crops": selected_crops,
            "mean_detected_persons_per_frame": baseline_crops / max(baseline_frames, 1),
            "mean_sapiens_crops_per_frame": selected_crops / max(baseline_frames, 1),
            "frames_per_second": target_frame_rate,
            "person_crops_per_second": target_crop_rate,
            "estimated_full_dataset_seconds": full_target_seconds,
        },
    ]
    atomic_write_csv(
        public / "phase6_1_runtime_comparison.csv", list(comparison[0]), comparison
    )

    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "full_dataset_inference_started": False,
        "gate": gate,
        "baseline": {
            "label": "ALL_DETECTIONS_BASELINE",
            "output_preserved": baseline_complete
            and preserved["all_required_files_present"],
            "preservation_manifest_complete": preserved[
                "all_required_files_present"
            ],
            "completed_batch_sizes": [int(row["batch_size"]) for row in passing_batches],
            "batch_records": passing_batches,
            "pilot_frame_count": baseline_frames,
            "total_sapiens_person_crops": baseline_crops,
            "mean_detected_persons_per_frame": baseline_crops / max(baseline_frames, 1),
            "frames_per_second": baseline_frame_rate,
            "person_crops_per_second": baseline_crop_rate,
            "gpu_monitor_file": pilot_gpu_path.name,
            "gpu": gpu_summary(pilot_gpu_path),
            "errors": baseline_errors,
        },
        "target_selection": {
            "camera_results": len(selection_rows),
            "target_only_crops": selected_crops,
            "ambiguous_frames": ambiguous,
            "no_target_frames": no_target,
            "ambiguity_fraction": ambiguity_fraction,
            "identity_switch_count": identity_switches,
            "background_misselection_count": args.background_misselection_count,
            "visual_qa_status": args.visual_qa_status,
            "crop_reduction_fraction": crop_reduction,
        },
        "target_only": {
            "best_batch_size": int(target_best["batch_size"]),
            "controlled_images_per_second": numeric(target_best["images_per_second"]),
            "controlled_person_crops_per_second": numeric(target_best["person_crops_per_second"]),
            "pilot_frames_per_second": target_frame_rate,
            "pilot_person_crops_per_second": target_crop_rate,
            "all_stable_batches_equivalent": benchmark_valid,
            "selected_pose_equivalent_to_all_detections": selected_pose_equivalent,
            "output_errors": target_errors,
        },
        "runtime": {
            "dataset_frame_count": args.dataset_frame_count,
            "all_detections_estimated_seconds": full_all_seconds,
            "target_only_estimated_seconds": full_target_seconds,
            "target_only_estimated_hours": full_target_seconds / 3600.0,
            "crop_reduction_fraction": crop_reduction,
            "runtime_reduction_fraction": runtime_reduction,
            "detector_frames_per_second": detector_rate,
            "selector_frames_per_second": selector_rate,
        },
        "acceptance": {
            "baseline_complete": baseline_complete,
            "selection_complete": selection_complete,
            "target_only_complete": target_complete,
            "target_only_benchmark_valid": benchmark_valid,
            "selected_pose_equivalence_valid": selected_pose_equivalent,
            "obvious_identity_switch_zero": identity_switches == 0,
            "manual_background_review_complete": args.background_misselection_count >= 0,
            "visual_qa_pass": args.visual_qa_status == "PASS",
        },
    }
    atomic_write_text(
        baseline_runtime / "phase6_1_acceptance.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_csv(
        public / "phase6_1_acceptance.csv",
        (
            "gate",
            "full_dataset_inference_started",
            "baseline_output_preserved",
            "pilot_frame_count",
            "all_detections_person_crops",
            "target_only_person_crops",
            "crop_reduction_fraction",
            "ambiguous_frames",
            "no_target_frames",
            "identity_switch_count",
            "best_target_batch_size",
            "target_only_estimated_hours",
            "runtime_reduction_fraction",
        ),
        [
            {
                "gate": gate,
                "full_dataset_inference_started": False,
                "baseline_output_preserved": baseline_complete
                and preserved["all_required_files_present"],
                "pilot_frame_count": baseline_frames,
                "all_detections_person_crops": baseline_crops,
                "target_only_person_crops": selected_crops,
                "crop_reduction_fraction": crop_reduction,
                "ambiguous_frames": ambiguous,
                "no_target_frames": no_target,
                "identity_switch_count": identity_switches,
                "best_target_batch_size": int(target_best["batch_size"]),
                "target_only_estimated_hours": full_target_seconds / 3600.0,
                "runtime_reduction_fraction": runtime_reduction,
            }
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if gate == "GO_FULL_DATASET" else 2


if __name__ == "__main__":
    raise SystemExit(main())
