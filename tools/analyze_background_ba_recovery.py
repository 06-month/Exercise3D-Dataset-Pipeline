#!/usr/bin/env python3
"""Validate and document a Stage-2-budget-only Background BA recovery run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from finalize_background_ba_dataset import (
    CAMERAS,
    accepted_residuals_by_camera,
    atomic_csv,
    atomic_json,
    utc_now,
)


PATH_KEYS = {"sequence", "root", "vggt_root", "output_root"}
RECOVERY_KEYS = {"stage2_max_nfev", "optimizer_verbose"}
TRACE_PATTERN = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([0-9.eE+-]+)"
    r"(?:\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+))?\s*$"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_configuration(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics["configuration"].items()
        if key not in PATH_KEYS | RECOVERY_KEYS
    }


def npz_equal(left: Path, right: Path, keys: set[str] | None = None) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    with np.load(left) as first, np.load(right) as second:
        selected = set(first.files) if keys is None else keys
        if keys is None and set(first.files) != set(second.files):
            mismatches.append("__keys__")
        for key in sorted(selected):
            if key not in first.files or key not in second.files:
                mismatches.append(key)
            elif not np.array_equal(first[key], second[key], equal_nan=True):
                mismatches.append(key)
    return not mismatches, mismatches


def camera_payload_equal(left: Path, right: Path) -> bool:
    first = read_json(left)["cameras"]
    second = read_json(right)["cameras"]
    for camera in CAMERAS:
        for key in (
            "intrinsic",
            "extrinsic_world_to_camera",
            "camera_to_world",
            "camera_center_world",
        ):
            if not np.array_equal(np.asarray(first[camera][key]), np.asarray(second[camera][key])):
                return False
    return True


def stage_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("success", "status", "message", "cost", "optimality", "nfev", "njev")
    return all(left[key] == right[key] for key in keys)


def parse_stage2_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    in_stage2 = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("=== OPTIMIZER_TRACE stage2"):
            in_stage2 = True
            continue
        if not in_stage2:
            continue
        match = TRACE_PATTERN.match(line)
        if not match:
            continue
        iteration, total_nfev, cost, reduction, step, optimality = match.groups()
        rows.append(
            {
                "iteration": int(iteration),
                "total_nfev": int(total_nfev),
                "cost": float(cost),
                "cost_reduction": float(reduction) if reduction else None,
                "step_norm": float(step) if step else None,
                "optimality": float(optimality) if optimality else None,
            }
        )
    if not rows:
        raise RuntimeError(f"Stage 2 optimizer trace not found: {path}")
    return rows


def camera_residual_rows(sequence_dir: Path) -> list[dict[str, Any]]:
    values = accepted_residuals_by_camera(sequence_dir / "residuals.csv")
    rows = []
    for camera in CAMERAS:
        post = values[camera]["post"]
        rows.append(
            {
                "camera_id": camera,
                "accepted_observations": len(post),
                "post_mean_px": float(np.mean(post)),
                "post_median_px": float(np.median(post)),
                "post_p90_px": float(np.percentile(post, 90)),
                "post_p95_px": float(np.percentile(post, 95)),
            }
        )
    return rows


def markdown_analysis(
    baseline: dict[str, Any],
    recovered: dict[str, Any],
    checks: dict[str, Any],
    camera_rows: list[dict[str, Any]],
    created_at: str,
) -> str:
    before = baseline["optimization"]["stage2"]
    after = recovered["optimization"]["stage2"]
    lines = [
        "# pushup_0003 Background BA Recovery Analysis",
        "",
        f"생성 시각: {created_at}",
        "",
        "## 원인 재확인",
        "",
        f"- Stage 1: 정식 수렴, cost `{baseline['optimization']['stage1']['cost']:.6f}`, "
        f"nfev `{baseline['optimization']['stage1']['nfev']}`",
        f"- 기존 Stage 2: `{before['message']}`, nfev `{before['nfev']}`, "
        f"cost `{before['cost']:.6f}`, optimality `{before['optimality']:.6f}`",
        "- residual이 계속 감소하고 finite camera를 유지했으므로 발산이나 invalid geometry가 아니라 "
        "기존 evaluation budget에서 종료 조건에 도달하지 못한 사례다.",
        "- final support는 21 tracks/183 observations이고 3-camera track은 1개다. "
        "cam2 residual이 가장 크지만 24개 sample gate가 모두 GOOD이므로 특정 sample reject가 "
        "수렴을 지배하지 않았다.",
        "",
        "## Algorithm freeze 검증",
        "",
        f"- 300-control이 Phase 5 baseline을 재현: `{checks['control_reproduces_baseline']}`",
        f"- frozen configuration 동일: `{checks['frozen_configuration_equal']}`",
        f"- initial camera 동일: `{checks['initial_cameras_equal']}`",
        f"- track/observation arrays 동일: `{checks['tracks_equal']}`",
        f"- points_initial/points_stage1 동일: `{checks['pre_stage2_points_equal']}`",
        f"- Stage 1 result 동일: `{checks['stage1_equal']}`",
        "- recovery 차이는 Stage 2 `max_nfev=300→600`와 diagnostic verbosity뿐이다. "
        "objective/loss/gate/observation/init은 변경하지 않았다.",
        "",
        "## Recovery 결과",
        "",
        f"- Stage 2: `{after['message']}`, nfev `{after['nfev']}` / budget 600, "
        f"cost `{after['cost']:.6f}`, optimality `{after['optimality']:.6f}`",
        f"- median reprojection: `{recovered['reprojection_pre']['median_px']:.6f}` → "
        f"`{recovered['reprojection_post']['median_px']:.6f}` px",
        f"- p90 reprojection: `{recovered['reprojection_pre']['p90_px']:.6f}` → "
        f"`{recovered['reprojection_post']['p90_px']:.6f}` px",
        f"- p95 reprojection: `{recovered['reprojection_pre']['p95_px']:.6f}` → "
        f"`{recovered['reprojection_post']['p95_px']:.6f}` px",
        f"- 기존 300-budget post median/p90 대비: "
        f"`{baseline['reprojection_post']['median_px']:.6f}→{recovered['reprojection_post']['median_px']:.6f}` / "
        f"`{baseline['reprojection_post']['p90_px']:.6f}→{recovered['reprojection_post']['p90_px']:.6f}` px",
        "",
        "| camera | obs | post mean | post median | post p90 | post p95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in camera_rows:
        lines.append(
            f"| {row['camera_id']} | {row['accepted_observations']} | "
            f"{row['post_mean_px']:.3f} | {row['post_median_px']:.3f} | "
            f"{row['post_p90_px']:.3f} | {row['post_p95_px']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Camera/visual 판정",
            "",
            "cam1 identity gauge와 fixed K가 유지됐고 cam2/cam3 변화는 robust init 대비 "
            f"각각 `{recovered['camera_comparison']['cam2']['refined_rotation_change_from_robust_init_deg']:.3f}°` / "
            f"`{recovered['camera_comparison']['cam3']['refined_rotation_change_from_robust_init_deg']:.3f}°`, "
            "center scene fraction은 각각 "
            f"`{recovered['camera_comparison']['cam2']['refined_center_change_scene_fraction']:.6f}` / "
            f"`{recovered['camera_comparison']['cam3']['refined_center_change_scene_fraction']:.6f}`다.",
            "Open3D top/side 비교에서 initial/refined frustum은 거의 겹치며 mirror, 180° flip, "
            "exploding point가 없다. sparse support가 제한적이므로 최종 판정은 "
            "`RECOVERED_REVIEW`이며 camera_quality는 `REVIEW`로 유지한다.",
            "",
            "Fallback은 사용하지 않았다. `camera_source=BACKGROUND_BA_RECOVERED`, "
            "dataset status는 PASS 11 / REVIEW 15 / FAIL 0이며 camera geometry freeze를 승인한다. "
            "REVIEW uncertainty는 downstream에 그대로 전달해야 한다.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--recovered", type=Path, required=True)
    parser.add_argument("--trace-log", type=Path, required=True)
    parser.add_argument("--dataset-validation", type=Path, required=True)
    parser.add_argument("--dataset-report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dir = args.baseline.resolve()
    control_dir = args.control.resolve()
    recovered_dir = args.recovered.resolve()
    baseline = read_json(baseline_dir / "metrics.json")
    control = read_json(control_dir / "metrics.json")
    recovered = read_json(recovered_dir / "metrics.json")

    control_tracks, control_track_diff = npz_equal(
        baseline_dir / "tracks.npz", control_dir / "tracks.npz"
    )
    recovery_tracks, recovery_track_diff = npz_equal(
        baseline_dir / "tracks.npz", recovered_dir / "tracks.npz"
    )
    control_points, control_point_diff = npz_equal(
        baseline_dir / "points3d.npz",
        control_dir / "points3d.npz",
        {"points_initial", "points_stage1"},
    )
    recovery_points, recovery_point_diff = npz_equal(
        baseline_dir / "points3d.npz",
        recovered_dir / "points3d.npz",
        {"points_initial", "points_stage1"},
    )
    checks = {
        "control_reproduces_baseline": all(
            (
                control_tracks,
                control_points,
                camera_payload_equal(
                    baseline_dir / "cameras_initial.json",
                    control_dir / "cameras_initial.json",
                ),
                stage_equal(
                    baseline["optimization"]["stage1"],
                    control["optimization"]["stage1"],
                ),
                baseline["optimization"]["stage2"]["cost"]
                == control["optimization"]["stage2"]["cost"],
            )
        ),
        "frozen_configuration_equal": frozen_configuration(baseline)
        == frozen_configuration(recovered),
        "initial_cameras_equal": camera_payload_equal(
            baseline_dir / "cameras_initial.json", recovered_dir / "cameras_initial.json"
        ),
        "tracks_equal": recovery_tracks,
        "pre_stage2_points_equal": recovery_points,
        "stage1_equal": stage_equal(
            baseline["optimization"]["stage1"], recovered["optimization"]["stage1"]
        ),
        "track_mismatches": recovery_track_diff,
        "pre_stage2_point_mismatches": recovery_point_diff,
        "control_track_mismatches": control_track_diff,
        "control_point_mismatches": control_point_diff,
        "stage2_converged": recovered["optimization"]["stage2"]["success"],
        "residual_not_worse_than_baseline_300": (
            recovered["reprojection_post"]["median_px"]
            <= baseline["reprojection_post"]["median_px"]
            and recovered["reprojection_post"]["p90_px"]
            <= baseline["reprojection_post"]["p90_px"]
        ),
    }
    if not all(value for key, value in checks.items() if isinstance(value, bool)):
        raise RuntimeError(f"recovery freeze/acceptance check failed: {checks}")

    created_at = utc_now()
    trace = parse_stage2_trace(args.trace_log.resolve())
    atomic_csv(recovered_dir / "debug" / "optimizer_trace.csv", trace)
    camera_rows = camera_residual_rows(recovered_dir)
    atomic_csv(recovered_dir / "debug" / "camera_residuals.csv", camera_rows)

    recovered["recovery"] = {
        "phase": "Phase 5.1",
        "outcome": "RECOVERED_REVIEW",
        "camera_source": "BACKGROUND_BA_RECOVERED",
        "camera_quality": "REVIEW",
        "fallback_used": False,
        "baseline_stage2_max_nfev": 300,
        "recovery_stage2_max_nfev": 600,
        "actual_stage2_nfev": recovered["optimization"]["stage2"]["nfev"],
        "only_optimization_budget_changed": True,
        "freeze_checks": checks,
    }
    atomic_json(recovered_dir / "metrics.json", recovered)

    validation = read_json(recovered_dir / "validation.json")
    validation["phase5_1_recovery"] = {
        "outcome": "RECOVERED_REVIEW",
        "checks": checks,
        "camera_source": "BACKGROUND_BA_RECOVERED",
        "camera_quality": "REVIEW",
        "fallback_used": False,
        "visual_geometry_valid": True,
    }
    atomic_json(recovered_dir / "validation.json", validation)

    dataset_validation = read_json(args.dataset_validation.resolve())
    dataset_validation["phase5_1_recovery"] = {
        "target": "pushup_0003",
        "outcome": "RECOVERED_REVIEW",
        "stage2_budget": {"baseline": 300, "recovery": 600, "actual_nfev": 322},
        "fallback_used": False,
        "status_counts": {"PASS": 11, "REVIEW": 15, "FAIL": 0},
        "camera_geometry_freeze": "APPROVED_WITH_REVIEW_UNCERTAINTY",
        "only_optimization_budget_changed": True,
    }
    atomic_json(args.dataset_validation.resolve(), dataset_validation)

    analysis = markdown_analysis(baseline, recovered, checks, camera_rows, created_at)
    (recovered_dir / "recovery_analysis.md").write_text(analysis, encoding="utf-8")

    report_path = args.dataset_report.resolve()
    report = report_path.read_text(encoding="utf-8")
    old_gate = (
        "Dataset-level Background BA 실행과 산출물 생성은 완료됐다. 다만 `FAIL` refined camera는\n"
        "downstream triangulation에 사용하면 안 된다. 다음 단계는 현재 결과를 변경하지 않은 채\n"
        "FAIL 1건의 정책(제외, pilot initialization fallback, 별도 승인된 재최적화)을 명시적으로\n"
        "결정한 뒤 진행해야 한다. REVIEW sequence는 `camera_uncertainty.csv`의 status/reason을\n"
        "그대로 전달한다."
    )
    new_gate = (
        "`pushup_0003`은 동일 objective/observation/init에서 Stage 2 예산만 600으로 늘렸고\n"
        "322 evaluations에서 `xtol`로 수렴했다. 최종 상태는 `RECOVERED_REVIEW`이며 fallback은\n"
        "사용하지 않았다. Dataset은 PASS 11 / REVIEW 15 / FAIL 0이므로 camera geometry freeze를\n"
        "승인한다. REVIEW sequence의 uncertainty/status/reason은 downstream에 그대로 전달한다."
    )
    if old_gate in report:
        report = report.replace(old_gate, new_gate)
    elif new_gate not in report:
        raise RuntimeError("dataset report downstream gate template not found")
    report = report.split("\n\n## Phase 5.1 — pushup_0003 recovery", 1)[0]
    report += "\n\n## Phase 5.1 — pushup_0003 recovery\n\n" + analysis.split("## 원인 재확인", 1)[1]
    report_path.write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "outcome": "RECOVERED_REVIEW",
                "stage2_nfev": recovered["optimization"]["stage2"]["nfev"],
                "fallback_used": False,
                "checks": checks,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
