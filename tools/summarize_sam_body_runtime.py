#!/usr/bin/env python3
"""Aggregate control/severe SAM modes and project full-dataset runtime."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODE_A = "A"
MODE_B = "B"
MODE_C = "C"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("sam_body_benchmark.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source_file"] = str(path.relative_to(root))
                rows.append(row)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    fieldnames = sorted({key for row in materialized for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise RuntimeError(f"missing numeric field {key} for mode {row.get('mode')}")
    return float(value)


def execution_seconds_per_frame(row: dict[str, str]) -> float:
    frames = number(row, "frames_processed")
    elapsed = number(row, "elapsed_wall_seconds")
    initialization = number(row, "model_initialization_seconds")
    if frames <= 0 or elapsed < initialization:
        raise RuntimeError("invalid elapsed/initialization/frame measurement")
    return (elapsed - initialization) / frames


def end_to_end_seconds_per_frame(row: dict[str, str]) -> float:
    return number(row, "elapsed_wall_seconds") / number(row, "frames_processed")


def result_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("candidate", ""), row.get("camera", ""), row.get("mode", ""))
        if key in index:
            raise RuntimeError(f"duplicate SAM benchmark row: {key}")
        index[key] = row
    return index


def compute_pass_summary(
    rows: list[dict[str, str]],
    control_key: tuple[str, str],
    severe_key: tuple[str, str],
    full_frame_count: int,
    severe_frame_fraction: float,
    selective_refinement_fraction: float,
) -> dict[str, Any]:
    if not 0 <= severe_frame_fraction <= 1:
        raise RuntimeError("severe frame fraction must be in [0,1]")
    if not 0 <= selective_refinement_fraction <= 1:
        raise RuntimeError("selective refinement fraction must be in [0,1]")
    index = result_index(rows)
    selected: dict[str, dict[str, dict[str, str]]] = {"control": {}, "severe": {}}
    for condition, key in (("control", control_key), ("severe", severe_key)):
        for mode in (MODE_A, MODE_B, MODE_C):
            full_key = (key[0], key[1], mode)
            if full_key not in index:
                raise RuntimeError(f"missing benchmark row: {full_key}")
            row = index[full_key]
            if row.get("status") != "PASS":
                raise RuntimeError(f"benchmark is not PASS: {full_key}")
            if int(float(row.get("persons_targeted", "0"))) != 1:
                raise RuntimeError(f"primary-target invariant failed: {full_key}")
            selected[condition][mode] = row

    rates: dict[str, dict[str, dict[str, float]]] = {}
    for condition in ("control", "severe"):
        rates[condition] = {}
        for mode in (MODE_A, MODE_B, MODE_C):
            row = selected[condition][mode]
            rates[condition][mode] = {
                "end_to_end_seconds_per_frame": end_to_end_seconds_per_frame(row),
                "execution_seconds_per_frame": execution_seconds_per_frame(row),
                "initialization_seconds": number(row, "model_initialization_seconds"),
                "peak_vram_mib": number(row, "peak_nvidia_vram_mib"),
                "refinement_model_seconds": number(row, "refinement_model_seconds"),
            }

    comparisons: dict[str, dict[str, float]] = {}
    for condition in ("control", "severe"):
        off = rates[condition][MODE_B]
        on = rates[condition][MODE_C]
        comparisons[condition] = {
            "refiner_on_off_end_to_end_ratio": on["end_to_end_seconds_per_frame"]
            / off["end_to_end_seconds_per_frame"],
            "refiner_on_off_execution_ratio": on["execution_seconds_per_frame"]
            / off["execution_seconds_per_frame"],
            "refiner_incremental_seconds_per_frame": max(
                0.0,
                on["execution_seconds_per_frame"] - off["execution_seconds_per_frame"],
            ),
        }

    severe = severe_frame_fraction
    control = 1.0 - severe
    base_b_seconds_per_frame = (
        control * rates["control"][MODE_B]["execution_seconds_per_frame"]
        + severe * rates["severe"][MODE_B]["execution_seconds_per_frame"]
    )
    incremental_seconds_per_frame = (
        control * comparisons["control"]["refiner_incremental_seconds_per_frame"]
        + severe * comparisons["severe"]["refiner_incremental_seconds_per_frame"]
    )
    initialization_b = max(
        rates["control"][MODE_B]["initialization_seconds"],
        rates["severe"][MODE_B]["initialization_seconds"],
    )
    initialization_c = max(
        rates["control"][MODE_C]["initialization_seconds"],
        rates["severe"][MODE_C]["initialization_seconds"],
    )

    best_seconds = (
        rates["control"][MODE_B]["execution_seconds_per_frame"] * full_frame_count
        + initialization_b
    )
    expected_seconds = (
        (
            base_b_seconds_per_frame
            + selective_refinement_fraction * incremental_seconds_per_frame
        )
        * full_frame_count
        + initialization_b
        + (initialization_c if selective_refinement_fraction > 0 else 0.0)
    )
    worst_seconds = (
        rates["severe"][MODE_C]["execution_seconds_per_frame"] * full_frame_count
        + initialization_c
    )
    base_a_control = (
        rates["control"][MODE_A]["execution_seconds_per_frame"] * full_frame_count
        + rates["control"][MODE_A]["initialization_seconds"]
    )
    base_a_severe = (
        rates["severe"][MODE_A]["execution_seconds_per_frame"] * full_frame_count
        + rates["severe"][MODE_A]["initialization_seconds"]
    )

    return {
        "status": "PASS",
        "full_frame_count": full_frame_count,
        "assumptions": {
            "severe_frame_fraction": severe_frame_fraction,
            "selective_refinement_fraction": selective_refinement_fraction,
            "model_lifecycle": "persistent model per mode; one measured initialization each",
            "best_case": "mode B completion off at control execution rate",
            "expected_case": "control/severe weighted mode B plus selective mode C increment",
            "worst_case": "mode C completion on at severe execution rate for every frame",
        },
        "rates": rates,
        "comparisons": comparisons,
        "occlusion_runtime_ratio": {
            mode: rates["severe"][mode]["execution_seconds_per_frame"]
            / rates["control"][mode]["execution_seconds_per_frame"]
            for mode in (MODE_A, MODE_B, MODE_C)
        },
        "projections": {
            "MODE_A_CONTROL_REFERENCE": {
                "seconds": base_a_control,
                "hours": base_a_control / 3600.0,
            },
            "MODE_A_SEVERE_REFERENCE": {
                "seconds": base_a_severe,
                "hours": base_a_severe / 3600.0,
            },
            "BEST_CASE": {"seconds": best_seconds, "hours": best_seconds / 3600.0},
            "EXPECTED_CASE": {
                "seconds": expected_seconds,
                "hours": expected_seconds / 3600.0,
            },
            "WORST_CASE": {
                "seconds": worst_seconds,
                "hours": worst_seconds / 3600.0,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-candidate", required=True)
    parser.add_argument("--control-camera", required=True)
    parser.add_argument("--severe-candidate", required=True)
    parser.add_argument("--severe-camera", required=True)
    parser.add_argument("--full-frame-count", type=int, default=65595)
    parser.add_argument("--severe-frame-fraction", type=float)
    parser.add_argument("--selective-refinement-fraction", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.results_root.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    if not rows:
        raise RuntimeError("no SAM benchmark CSV files found")
    atomic_write_csv(output_dir / "sam_mode_measurements.csv", rows)
    incomplete = [row for row in rows if row.get("status") != "PASS"]
    if incomplete:
        summary = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "status": "WAITING_CHECKPOINT_OR_RUNTIME",
            "measurement_count": len(rows),
            "pass_count": len(rows) - len(incomplete),
            "incomplete": [
                {
                    "candidate": row.get("candidate"),
                    "camera": row.get("camera"),
                    "mode": row.get("mode"),
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                }
                for row in incomplete
            ],
        }
        atomic_write_text(
            output_dir / "sam_runtime_summary.json",
            json.dumps(summary, indent=2) + "\n",
        )
        print(json.dumps(summary, indent=2))
        return 3
    if args.severe_frame_fraction is None or args.selective_refinement_fraction is None:
        raise RuntimeError(
            "PASS projections require measured severe and selective-refinement fractions"
        )
    summary = compute_pass_summary(
        rows,
        (args.control_candidate, args.control_camera),
        (args.severe_candidate, args.severe_camera),
        args.full_frame_count,
        args.severe_frame_fraction,
        args.selective_refinement_fraction,
    )
    summary["schema_version"] = 1
    summary["created_at_utc"] = utc_now()
    atomic_write_text(
        output_dir / "sam_runtime_summary.json", json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
