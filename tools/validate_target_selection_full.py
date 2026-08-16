#!/usr/bin/env python3
"""Validate full target-selection integrity without publishing private boxes."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("cam1", "cam2", "cam3")


def parse_list(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return result


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    fields = list(rows[0])
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def frame_dir(dataset_root: Path, sequence: str, camera: str) -> Path:
    exercise = sequence.rsplit("_", 1)[0]
    return dataset_root / "final_frame" / exercise / sequence / camera


def validate_camera(
    dataset_root: Path,
    detections_root: Path,
    selection_root: Path,
    sequence: str,
    camera: str,
) -> dict[str, Any]:
    source_names = np.asarray(
        sorted(path.name for path in frame_dir(dataset_root, sequence, camera).glob("*.jpg"))
    )
    with np.load(
        detections_root / sequence / camera / "bboxes.npz", allow_pickle=False
    ) as payload:
        detection = {key: payload[key].copy() for key in payload.files}
    with np.load(
        selection_root / sequence / camera / "target_selection.npz",
        allow_pickle=False,
    ) as payload:
        selected = {key: payload[key].copy() for key in payload.files}

    frame_count = len(source_names)
    target = selected["target_candidate_index"]
    present = target >= 0
    expected_background_count = selected["num_person_candidates"] - present.astype(
        selected["num_person_candidates"].dtype
    )
    checks = {
        "frame_count_matches_source": len(target) == frame_count,
        "frame_index_matches_source": np.array_equal(
            selected["frame_index"], np.arange(frame_count, dtype=np.int32)
        ),
        "frame_name_matches_source": np.array_equal(selected["frame_name"], source_names),
        "timestamps_finite": bool(
            len(selected["timestamp_pts_seconds"]) == frame_count
            and np.isfinite(selected["timestamp_pts_seconds"]).all()
        ),
        "candidate_counts_match_detector": np.array_equal(
            selected["num_person_candidates"], detection["person_count"]
        ),
        "candidate_offsets_match_detector": np.array_equal(
            selected["candidate_offsets"], detection["instance_offsets"]
        ),
        "candidate_boxes_lossless": np.array_equal(
            selected["all_person_detections_xyxy"], detection["all_bboxes_xyxy"]
        ),
        "candidate_scores_lossless": np.array_equal(
            selected["all_person_detection_scores"], detection["all_bbox_scores"]
        ),
        "target_indices_in_bounds": bool(
            np.all(target[present] < selected["num_person_candidates"][present])
        ),
        "no_forced_abstention": bool(
            not np.any(present & (selected["target_ambiguous"] | selected["no_target"]))
        ),
        "accepted_status_is_target": bool(
            np.all(selected["target_status"][present] == "TARGET")
        ),
        "background_counts_consistent": np.array_equal(
            selected["background_person_count"], expected_background_count
        ),
        "background_offsets_consistent": bool(
            len(selected["background_instance_offsets"]) == frame_count + 1
            and selected["background_instance_offsets"][0] == 0
            and selected["background_instance_offsets"][-1]
            == len(selected["background_bboxes_xyxy"])
            and np.array_equal(
                np.diff(selected["background_instance_offsets"]),
                selected["background_person_count"],
            )
        ),
    }
    errors = sorted(key for key, value in checks.items() if not value)
    return {
        "sequence": sequence,
        "camera": camera,
        "frame_count": frame_count,
        "person_candidate_count": int(selected["num_person_candidates"].sum()),
        "target_crop_count": int(present.sum()),
        "target_ambiguous_count": int(selected["target_ambiguous"].sum()),
        "no_target_count": int(selected["no_target"].sum()),
        "background_person_count": int(selected["background_person_count"].sum()),
        "occlusion_risk_count": int(selected["occlusion_risk"].sum()),
        "forward_backward_disagreement_count": int(
            (
                (selected["forward_candidate_index"] != selected["backward_candidate_index"])
                & ~selected["no_target"]
            ).sum()
        ),
        "identity_switch_risk_count": int(selected["identity_switch_risk"].sum()),
        "integrity_status": "PASS" if not errors else "FAIL",
        "errors": ";".join(errors),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detections-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--sequences", type=parse_list, required=True)
    parser.add_argument("--cameras", type=parse_list, default=list(CAMERAS))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        validate_camera(
            args.dataset_root.resolve(),
            args.detections_root.resolve(),
            args.selection_root.resolve(),
            sequence,
            camera,
        )
        for sequence in args.sequences
        for camera in args.cameras
    ]
    integrity_failures = sum(row["integrity_status"] != "PASS" for row in rows)
    identity_risks = sum(row["identity_switch_risk_count"] for row in rows)
    disagreements = sum(row["forward_backward_disagreement_count"] for row in rows)
    if integrity_failures:
        gate = "NO_GO"
    elif identity_risks or disagreements:
        gate = "REVIEW_TARGET_SELECTION"
    else:
        gate = "GO_FULL_DATASET"
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": gate,
        "camera_count": len(rows),
        "frame_count": int(sum(row["frame_count"] for row in rows)),
        "person_candidate_count": int(sum(row["person_candidate_count"] for row in rows)),
        "target_crop_count": int(sum(row["target_crop_count"] for row in rows)),
        "target_ambiguous_count": int(sum(row["target_ambiguous_count"] for row in rows)),
        "no_target_count": int(sum(row["no_target_count"] for row in rows)),
        "background_person_count": int(sum(row["background_person_count"] for row in rows)),
        "occlusion_risk_count": int(sum(row["occlusion_risk_count"] for row in rows)),
        "forward_backward_disagreement_count": disagreements,
        "identity_switch_risk_count": identity_risks,
        "integrity_failure_count": integrity_failures,
        "privacy": "aggregate only; private coordinate payload remains under ignored outputs",
    }
    atomic_csv(args.output_dir / "target_selection_validation.csv", rows)
    atomic_text(
        args.output_dir / "target_selection_validation_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if gate == "GO_FULL_DATASET" else 2


if __name__ == "__main__":
    raise SystemExit(main())
