#!/usr/bin/env python3
"""Find bounded Mode C review clips from Mode B/body/geometry evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("cam1", "cam2", "cam3")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-prior-root", type=Path, required=True)
    parser.add_argument("--body-fit-root", type=Path, required=True)
    parser.add_argument("--triangulation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sam_mode_c_escalation.json",
    )
    parser.add_argument(
        "--canonical-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sapiens2_canonical_joints.json",
    )
    return parser


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def robust_threshold(values: np.ndarray, multiplier: float = 5.0) -> tuple[float, float, float]:
    selected = values[np.isfinite(values)]
    if not len(selected):
        return math.inf, math.nan, math.nan
    median = float(np.median(selected))
    mad = float(np.median(np.abs(selected - median)))
    threshold = median + multiplier * 1.4826 * max(mad, 1e-12)
    return threshold, median, mad


def bounded_clips(
    candidate: np.ndarray,
    severity: np.ndarray,
    padding: int,
    maximum_fraction: float,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    frame_count = len(candidate)
    limit = max(1, int(math.floor(frame_count * maximum_fraction)))
    selected = np.zeros(frame_count, dtype=np.bool_)
    seeds = np.flatnonzero(candidate)
    seeds = seeds[np.argsort(np.nan_to_num(severity[seeds], nan=0.0))[::-1]]
    for seed in seeds:
        start = max(0, int(seed) - padding)
        stop = min(frame_count, int(seed) + padding + 1)
        proposed = selected.copy()
        proposed[start:stop] = True
        if int(proposed.sum()) <= limit:
            selected = proposed
        elif not selected.any():
            half = limit // 2
            start = max(0, int(seed) - half)
            stop = min(frame_count, start + limit)
            start = max(0, stop - limit)
            selected[start:stop] = True
        if int(selected.sum()) >= limit:
            break
    clips = []
    indices = np.flatnonzero(selected)
    if len(indices):
        start = previous = int(indices[0])
        for value in indices[1:]:
            value = int(value)
            if value != previous + 1:
                clips.append({"start_frame_index": start, "end_frame_index": previous})
                start = value
            previous = value
        clips.append({"start_frame_index": start, "end_frame_index": previous})
    return selected, clips


def canonical_support(
    support: np.ndarray, canonical: dict[str, Any]
) -> tuple[np.ndarray, list[str]]:
    direct = np.asarray(
        [int(row["source_index"]) for row in canonical["direct"]], dtype=np.int32
    )
    names = [str(row["canonical"]) for row in canonical["direct"]]
    result = support[:, direct].astype(np.float32)
    for row in canonical["derived"]:
        left = names.index(row["inputs"][0])
        right = names.index(row["inputs"][1])
        derived = np.minimum(result[:, left], result[:, right])
        result = np.concatenate([result, derived[:, None]], axis=1)
        names.append(str(row["canonical"]))
    return result, names


def assess_sequence(args: argparse.Namespace, sequence: str) -> dict[str, Any]:
    policy = json.loads(args.policy_config.resolve().read_text(encoding="utf-8"))
    canonical = json.loads(args.canonical_config.resolve().read_text(encoding="utf-8"))
    with np.load(
        args.body_fit_root.resolve() / sequence / "body_fit.npz", allow_pickle=False
    ) as payload:
        body = {key: payload[key].copy() for key in payload.files}
    with np.load(
        args.triangulation_root.resolve() / sequence / "triangulated_3d.npz",
        allow_pickle=False,
    ) as payload:
        support, support_names = canonical_support(payload["supporting_views"], canonical)
    if list(body["joint_names"].astype(str)) != support_names:
        raise RuntimeError("canonical support and body joint conventions differ")
    padding = int(policy["clip_policy"]["padding_frames_each_side"])
    maximum_fraction = float(policy["clip_policy"]["maximum_sequence_fraction"])
    camera_results = []
    for camera_index, camera in enumerate(CAMERAS):
        with np.load(
            args.sam_prior_root.resolve() / sequence / camera / "sam_body_prior.npz",
            allow_pickle=False,
        ) as payload:
            prior = {key: payload[key].copy() for key in payload.files}
        source_index = body["sam_source_frame_index"][:, camera_index].astype(np.int32)
        reference_count = len(source_index)
        mapped = source_index >= 0
        occlusion = np.zeros(reference_count, dtype=np.bool_)
        missing = np.ones(reference_count, dtype=np.bool_)
        temporal = np.full(reference_count, np.nan, dtype=np.float64)
        valid_source = np.flatnonzero(prior["accepted_prior"])
        local = prior["canonical_local_3d"].astype(np.float64)
        source_temporal = np.full(len(local), np.nan, dtype=np.float64)
        consecutive = prior["accepted_prior"][1:] & prior["accepted_prior"][:-1]
        if consecutive.any():
            delta = np.linalg.norm(local[1:] - local[:-1], axis=-1)
            source_temporal[1:][consecutive] = np.nanmedian(delta[consecutive], axis=1)
        occlusion[mapped] = prior["occlusion_risk"][source_index[mapped]]
        missing[mapped] = ~prior["output_valid"][source_index[mapped]]
        temporal[mapped] = source_temporal[source_index[mapped]]
        temporal_threshold, temporal_median, temporal_mad = robust_threshold(
            source_temporal[valid_source]
        )
        temporal_outlier = temporal > temporal_threshold

        residual = np.nanmedian(
            body["aligned_prior_residual_sequence_gauge"][:, camera_index], axis=1
        )
        reference_length = float(
            json.loads(
                (args.body_fit_root.resolve() / sequence / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )["qa"]["anthropometry"]["reference_length_sequence_gauge"]
        )
        residual_normalized = residual / max(reference_length, 1e-12)
        residual_threshold, residual_median, residual_mad = robust_threshold(
            residual_normalized
        )
        low_geometry_support = np.nanmedian(support, axis=1) < 3
        alignment_outlier = (residual_normalized > residual_threshold) & low_geometry_support
        base_candidate = occlusion & (missing | temporal_outlier | alignment_outlier)
        severity = np.zeros(reference_count, dtype=np.float64)
        severity[missing] += 10.0
        severity += np.nan_to_num(
            temporal / max(temporal_threshold, 1e-12), nan=0.0
        )
        severity += np.nan_to_num(
            residual_normalized / max(residual_threshold, 1e-12), nan=0.0
        )
        selected, clips = bounded_clips(
            base_candidate, severity, padding, maximum_fraction
        )
        source_frames = sorted(set(int(value) for value in source_index[selected] if value >= 0))
        camera_results.append(
            {
                "camera": camera,
                "reference_frame_count": reference_count,
                "occlusion_reference_count": int(occlusion.sum()),
                "missing_signal_count": int((occlusion & missing).sum()),
                "temporal_outlier_signal_count": int((occlusion & temporal_outlier).sum()),
                "alignment_outlier_signal_count": int((occlusion & alignment_outlier).sum()),
                "base_candidate_count": int(base_candidate.sum()),
                "selected_reference_frame_count": int(selected.sum()),
                "selected_source_frame_count": len(source_frames),
                "selected_source_frame_indices": source_frames,
                "clips_reference_timeline": clips,
                "temporal_delta": {
                    "median": temporal_median,
                    "mad": temporal_mad,
                    "threshold_median_plus_5_scaled_mad": temporal_threshold,
                },
                "alignment_residual_normalized": {
                    "median": residual_median,
                    "mad": residual_mad,
                    "threshold_median_plus_5_scaled_mad": residual_threshold,
                },
            }
        )
    candidate_count = sum(
        row["selected_reference_frame_count"] for row in camera_results
    )
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence": sequence,
        "default_mode": "B",
        "mode_c_executed": False,
        "selected_reference_frame_count": candidate_count,
        "status": "REVIEW_MODE_C_CANDIDATE" if candidate_count else "PASS_MODE_B_FROZEN",
        "policy": policy,
        "cameras": camera_results,
    }
    atomic_text(
        args.output_root.resolve() / sequence / "mode_c_escalation.json",
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    )
    return result


def main() -> int:
    args = build_parser().parse_args()
    rows = []
    for sequence in args.sequences:
        result = assess_sequence(args, sequence)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence_count": len(rows),
        "pass_mode_b_frozen_count": sum(row["status"] == "PASS_MODE_B_FROZEN" for row in rows),
        "review_mode_c_candidate_count": sum(
            row["status"] == "REVIEW_MODE_C_CANDIDATE" for row in rows
        ),
        "selected_reference_frame_count": sum(
            row["selected_reference_frame_count"] for row in rows
        ),
        "mode_c_executed": False,
        "status": (
            "REVIEW" if any(row["status"] == "REVIEW_MODE_C_CANDIDATE" for row in rows) else "PASS"
        ),
    }
    atomic_text(
        args.output_root.resolve() / "mode_c_escalation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
