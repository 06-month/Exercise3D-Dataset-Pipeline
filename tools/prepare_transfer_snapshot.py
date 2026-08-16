#!/usr/bin/env python3
"""Prepare a resumable Windows transfer snapshot after a durable freeze gate.

This tool is deliberately CPU/GPU-light while it waits: it reads only the
atomic predeadline-checkpoint state.  Once an integrity-verified,
``freeze_eligible`` checkpoint reaches the requested sequence count, it takes a
single metadata-only inventory of transfer roots, records current durable
pipeline state, and writes atomic JSON/Markdown manifests.  It never launches,
signals, suspends, or restarts inference and it never hashes or compresses live
payloads.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable

try:
    from tools.run_autonomous_supervisor_watchdog import acquire_singleton_lock
except ModuleNotFoundError:
    from run_autonomous_supervisor_watchdog import acquire_singleton_lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
CAMERAS = ("cam1", "cam2", "cam3")
CHUNK_PATTERN = re.compile(r"^chunk_(\d+)_(\d+)\.npz$")

CRITICAL_RELATIVE_PATHS = (
    "outputs/private_dataset_freeze",
    "outputs/body_fit_full",
    "outputs/triangulation_final",
    "outputs/sam_body_prior_full",
    "outputs/runtime",
    ".runtime",
    "HANDOFF.md",
    "docs/plan.md",
    "docs/process.md",
)
RESUME_RELATIVE_PATHS = (
    "outputs/sapiens2_target_only_full",
    "outputs/sam_body4d_full",
    "outputs/target_selection_full",
    "outputs/background_ba",
    "outputs/camera_observation_recovery",
    "outputs/triangulation_full_initial",
    "outputs/sam_mode_c_review_full",
)
GENERATED_MANIFEST_NAMES = {"transfer_manifest.json", "TRANSFER_MANIFEST.md"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def is_transient_name(name: str) -> bool:
    """Return whether a pathname is non-durable transfer noise."""
    lowered = name.lower()
    return bool(
        name in GENERATED_MANIFEST_NAMES
        or ".tmp" in lowered
        or lowered.endswith((".partial", ".part", ".lock"))
        or (lowered.startswith(".") and lowered.endswith(".inprogress"))
        or lowered == ".rsync-partial"
    )


def scan_finalized_path(path: Path, *, label: str) -> dict[str, Any]:
    """Inventory regular finalized files with one metadata-only traversal."""
    started = utc_now()
    file_count = 0
    logical_bytes = 0
    directory_count = 0
    transient_files = 0
    transient_bytes = 0
    symlinks_excluded = 0
    errors: list[str] = []

    def exclude_entry(entry: os.DirEntry[str]) -> None:
        nonlocal transient_files, transient_bytes
        transient_files += 1
        try:
            if entry.is_file(follow_symlinks=False):
                transient_bytes += entry.stat(follow_symlinks=False).st_size
        except OSError:
            pass

    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "file_count": 0,
            "logical_bytes": 0,
            "gib": 0.0,
            "directory_count": 0,
            "transient_files_excluded": 0,
            "transient_bytes_excluded": 0,
            "symlinks_excluded": 0,
            "scan_errors": [],
            "scan_started_at_utc": started.isoformat(),
            "scan_finished_at_utc": utc_now().isoformat(),
        }
    if path.is_symlink():
        symlinks_excluded = 1
    elif path.is_file():
        if is_transient_name(path.name):
            transient_files = 1
            transient_bytes = path.stat().st_size
        else:
            file_count = 1
            logical_bytes = path.stat().st_size
    elif path.is_dir():
        stack = [path]
        while stack:
            directory = stack.pop()
            directory_count += 1
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        try:
                            if entry.is_symlink():
                                symlinks_excluded += 1
                            elif is_transient_name(entry.name):
                                exclude_entry(entry)
                            elif entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                stat = entry.stat(follow_symlinks=False)
                                file_count += 1
                                logical_bytes += stat.st_size
                        except OSError as error:
                            errors.append(f"{entry.path}:{type(error).__name__}")
            except OSError as error:
                errors.append(f"{directory}:{type(error).__name__}")
    finished = utc_now()
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "file_count": file_count,
        "logical_bytes": logical_bytes,
        "gib": logical_bytes / (1024**3),
        "directory_count": directory_count,
        "transient_files_excluded": transient_files,
        "transient_bytes_excluded": transient_bytes,
        "symlinks_excluded": symlinks_excluded,
        "scan_errors": errors,
        "scan_started_at_utc": started.isoformat(),
        "scan_finished_at_utc": finished.isoformat(),
    }


def aggregate_inventory(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    return {
        "path_count": len(values),
        "existing_path_count": sum(bool(row.get("exists")) for row in values),
        "file_count": sum(int(row.get("file_count", 0)) for row in values),
        "logical_bytes": sum(int(row.get("logical_bytes", 0)) for row in values),
        "gib": sum(int(row.get("logical_bytes", 0)) for row in values) / (1024**3),
        "transient_files_excluded": sum(
            int(row.get("transient_files_excluded", 0)) for row in values
        ),
        "symlinks_excluded": sum(
            int(row.get("symlinks_excluded", 0)) for row in values
        ),
        "scan_error_count": sum(len(row.get("scan_errors", [])) for row in values),
    }


def checkpoint_gate(
    state: dict[str, Any], output_root: Path, minimum_ready: int
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the already verified follower checkpoint without rehashing."""
    reasons: list[str] = []
    best = state.get("best_checkpoint")
    if not isinstance(best, dict):
        return None, ["best checkpoint is unavailable"]
    try:
        count = int(best.get("completed_sequence_count", -1))
        file_count = int(best.get("file_count", -1))
        byte_count = int(best.get("total_payload_bytes", -1))
        verified_files = int(best.get("verified_file_count", -2))
        verified_bytes = int(best.get("verified_payload_bytes", -2))
    except (TypeError, ValueError):
        return None, ["best checkpoint counts are invalid"]
    build_id = str(best.get("build_id") or "")
    sequences = best.get("sequences")
    if count < minimum_ready:
        reasons.append(f"checkpoint count {count} is below {minimum_ready}")
    if not build_id or not isinstance(sequences, list) or len(sequences) != count:
        reasons.append("checkpoint build ID or sequence list is invalid")
    if not bool(best.get("integrity_verified")):
        reasons.append("checkpoint follower integrity verification is not PASS")
    if not bool(best.get("freeze_eligible")):
        reasons.append("checkpoint is not freeze eligible")
    if file_count < 0 or file_count != verified_files:
        reasons.append("checkpoint verified file count differs")
    if byte_count < 0 or byte_count != verified_bytes:
        reasons.append("checkpoint verified byte count differs")
    build_root = output_root / build_id
    manifest_path = build_root / "dataset_manifest.json"
    manifest = read_json(manifest_path)
    if not build_root.is_dir() or build_root.name.startswith("."):
        reasons.append("final immutable checkpoint directory is missing")
    try:
        manifest_ok = bool(
            manifest.get("build_id") == build_id
            and manifest.get("freeze_contract_version") == 2
            and manifest.get("requested_sequences") == sequences
            and int(manifest.get("sequence_count", -1)) == count
            and int(manifest.get("fail_count", -1)) == 0
            and int(manifest.get("incomplete_count", -1)) == 0
            and int(manifest.get("file_count", -1)) == file_count
            and int(manifest.get("total_payload_bytes", -1)) == byte_count
            and bool(manifest.get("freeze_eligible"))
        )
    except (TypeError, ValueError):
        manifest_ok = False
    if not manifest_ok:
        reasons.append("final checkpoint manifest disagrees with verified follower state")
    if reasons:
        return None, reasons
    return {
        "build_id": build_id,
        "path": str(build_root),
        "manifest_path": str(manifest_path),
        "created_at_utc": manifest.get("created_at_utc"),
        "sequence_count": count,
        "sequences": [str(sequence) for sequence in sequences],
        "file_count": file_count,
        "total_payload_bytes": byte_count,
        "total_payload_gib": byte_count / (1024**3),
        "integrity": "PASS_REUSED_FOLLOWER_VERIFICATION",
        "integrity_verified": True,
        "freeze_eligible": True,
        "fail_count": 0,
        "incomplete_count": 0,
        "git_commit_at_export": manifest.get("git_commit"),
    }, []


def durable_sapiens_progress(pose_root: Path, selection_root: Path, sequence: str) -> dict[str, Any]:
    cameras: list[dict[str, Any]] = []
    for camera in CAMERAS:
        output = pose_root / sequence / camera
        selection = read_json(selection_root / sequence / camera / "summary.json")
        expected = int(selection.get("target_only_sapiens_crops", 0) or 0)
        ranges: list[tuple[int, int]] = []
        for path in (output / "chunks").glob("chunk_*.npz"):
            match = CHUNK_PATTERN.fullmatch(path.name)
            if match:
                ranges.append((int(match.group(1)), int(match.group(2))))
        ranges.sort()
        durable = sum(max(0, end - start) for start, end in ranges)
        metadata = read_json(output / "metadata.json")
        complete = bool(metadata.get("qa", {}).get("status") == "PASS")
        cameras.append(
            {
                "camera": camera,
                "expected_target_crops": expected,
                "durable_finalized_chunk_count": len(ranges),
                "durable_target_crops": durable,
                "camera_complete": complete,
            }
        )
    return {
        "completed_cameras": sum(row["camera_complete"] for row in cameras),
        "durable_target_crops": sum(row["durable_target_crops"] for row in cameras),
        "expected_target_crops": sum(row["expected_target_crops"] for row in cameras),
        "cameras": cameras,
    }


def durable_sam_progress(sam_root: Path, selection_root: Path, sequence: str) -> dict[str, Any]:
    cameras: list[dict[str, Any]] = []
    for camera in CAMERAS:
        output = sam_root / sequence / camera
        selection = read_json(selection_root / sequence / camera / "summary.json")
        expected = int(selection.get("frame_count", 0) or 0)
        numeric = output / "mode_b_private_output" / "mhr_numeric" / "1"
        durable = sum(1 for path in numeric.glob("*.npz") if path.is_file())
        benchmark_rows: list[dict[str, str]] = []
        benchmark = output / "sam_body_benchmark.csv"
        try:
            with benchmark.open(newline="", encoding="utf-8") as handle:
                benchmark_rows = list(csv.DictReader(handle))
        except OSError:
            pass
        complete = bool(
            len(benchmark_rows) == 1
            and benchmark_rows[0].get("status") == "PASS"
            and (output / "run_provenance.json").is_file()
        )
        cameras.append(
            {
                "camera": camera,
                "expected_source_frames": expected,
                "durable_numeric_prior_frames": durable,
                "camera_complete": complete,
            }
        )
    return {
        "completed_cameras": sum(row["camera_complete"] for row in cameras),
        "durable_numeric_prior_frames": sum(
            row["durable_numeric_prior_frames"] for row in cameras
        ),
        "expected_source_frames": sum(row["expected_source_frames"] for row in cameras),
        "cameras": cameras,
    }


def metadata_status(path: Path, *keys: str) -> str | None:
    payload = read_json(path)
    qa = payload.get("qa", {}) if isinstance(payload.get("qa"), dict) else {}
    for key in keys:
        value = qa.get(key) if key in qa else payload.get(key)
        if value:
            return str(value)
    return None


def incomplete_sequence_state(
    *,
    sequence: str,
    checkpoint_sequences: set[str],
    pose_root: Path,
    selection_root: Path,
    triangulation_root: Path,
    sam_root: Path,
    sam_prior_root: Path,
    body_fit_root: Path,
    mode_c_root: Path,
    quality_root: Path,
    dashboard: dict[str, Any],
) -> dict[str, Any]:
    pose = durable_sapiens_progress(pose_root, selection_root, sequence)
    sam = durable_sam_progress(sam_root, selection_root, sequence)
    triangulation = metadata_status(
        triangulation_root / sequence / "metadata.json",
        "quality_status",
        "status",
        "schema_status",
    )
    body = metadata_status(body_fit_root / sequence / "metadata.json", "status")
    mode_c = metadata_status(mode_c_root / sequence / "mode_c_escalation.json", "status")
    quality = metadata_status(quality_root / sequence / "metadata.json", "sequence_status")
    prior_cameras = sum(
        (sam_prior_root / sequence / camera / "sam_body_prior.npz").is_file()
        and (sam_prior_root / sequence / camera / "metadata.json").is_file()
        for camera in CAMERAS
    )
    if pose["completed_cameras"] < len(CAMERAS):
        stage = "SAPIENS2_TARGET_ONLY"
    elif not triangulation:
        stage = "PHASE7_TRIANGULATION"
    elif sam["completed_cameras"] < len(CAMERAS):
        stage = "SAM_MODE_B"
    elif prior_cameras < len(CAMERAS):
        stage = "SAM_PRIOR_CONSOLIDATION"
    elif not body:
        stage = "BODY_FIT"
    elif not mode_c:
        stage = "MODE_C_ASSESSMENT"
    elif quality not in {"PASS", "REVIEW"}:
        stage = "QUALITY_CONTROL"
    elif sequence not in checkpoint_sequences:
        stage = "FREEZE_READINESS_OR_CHECKPOINT"
    else:
        stage = "COMPLETE"
    active = bool(
        dashboard.get("sapiens", {}).get("current_sequence") == sequence
        or dashboard.get("sam", {}).get("current_sequence") == sequence
        or dashboard.get("supervisor", {}).get("active_sequence") == sequence
    )
    return {
        "sequence": sequence,
        "completed": sequence in checkpoint_sequences,
        "current_stage": stage,
        "active_now": active,
        "sapiens": pose,
        "triangulation_status": triangulation or "NOT_STARTED",
        "sam_mode_b": sam,
        "sam_prior_completed_cameras": prior_cameras,
        "body_fit_status": body or "NOT_STARTED",
        "mode_c_assessment_status": mode_c or "NOT_STARTED",
        "quality_status": quality or "NOT_STARTED",
        "partial_outputs_preserved": True,
    }


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, timeout=20
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=20
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


def checkpoint_identifiers(checkpoint_integrity: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    try:
        with checkpoint_integrity.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        pass
    valid = [row for row in rows if row.get("status") == "PASS"]
    return {
        "integrity_manifest": str(checkpoint_integrity),
        "payload_count": len(rows),
        "pass_count": len(valid),
        "total_bytes": sum(int(row.get("bytes", 0) or 0) for row in valid),
        "hashes_reused_not_recomputed": True,
    }


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def rsync_command(relative_paths: Iterable[str], *, bwlimit_kib: int) -> str:
    sources = " \\\n  ".join(
        f'"${{EX3D_SERVER}}:${{EX3D_REMOTE}}/./{path}"' for path in relative_paths
    )
    return f"""rsync -rltvh --info=progress2 --human-readable \\
  --partial --partial-dir=.rsync-partial --delay-updates \\
  --bwlimit={bwlimit_kib} --protect-args \\
  --rsync-path='ionice -c2 -n7 nice -n 19 rsync' \\
  --exclude='*.tmp' --exclude='*.tmp.*' --exclude='*.partial' \\
  --exclude='*.part' --exclude='*.lock' --exclude='.*.inprogress/***' \\
  --exclude='.rsync-partial/***' --relative \\
  -e \"ssh -p ${{EX3D_SSH_PORT}}\" \\
  {sources} \\
  \"${{EX3D_LOCAL}}/\""""


def transfer_commands(args: argparse.Namespace) -> dict[str, str]:
    setup = f"""export EX3D_REMOTE_USER={shell_quote(args.ssh_user)}
export EX3D_REMOTE_HOST={shell_quote(args.ssh_host)}
export EX3D_SSH_PORT={shell_quote(str(args.ssh_port))}
export EX3D_SERVER=\"${{EX3D_REMOTE_USER}}@${{EX3D_REMOTE_HOST}}\"
export EX3D_REMOTE={shell_quote(str(PROJECT_ROOT))}
export EX3D_LOCAL={shell_quote(args.windows_local_root)}
mkdir -p \"${{EX3D_LOCAL}}\""""
    critical = tuple(CRITICAL_RELATIVE_PATHS) + (
        ".runtime/transfer_manifest.json",
        ".runtime/TRANSFER_MANIFEST.md",
    )
    final = critical + tuple(RESUME_RELATIVE_PATHS)
    return {
        "wsl_setup": setup,
        "critical_sync": rsync_command(critical, bwlimit_kib=args.bwlimit_kib),
        "full_resume_sync": rsync_command(
            RESUME_RELATIVE_PATHS, bwlimit_kib=args.bwlimit_kib
        ),
        "final_incremental_sync": rsync_command(final, bwlimit_kib=args.bwlimit_kib),
        "policy": (
            "No --delete; finalized paths only; remote rsync runs at low CPU/I/O priority; "
            "default rsync size+mtime checks skip unchanged files; rerun final sync after generation."
        ),
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    checkpoint = manifest["checkpoint"]
    sizes = manifest["transfer_sizes"]
    incomplete = manifest["incomplete_sequences"]
    commands = manifest["windows_transfer"]["commands"]
    lines = [
        "# Exercise3D Transfer Manifest",
        "",
        "STATUS: TRANSFER_READY",
        "",
        f"- Snapshot KST: `{manifest['snapshot_at_kst']}`",
        f"- Git HEAD: `{manifest['git']['head']}`",
        f"- Freeze-ready: **{checkpoint['sequence_count']}/{manifest['sequence_count']}**",
        f"- Build: `{checkpoint['build_id']}`",
        f"- Checkpoint payload: {checkpoint['total_payload_bytes']:,} bytes ({checkpoint['total_payload_gib']:.3f} GiB), {checkpoint['file_count']:,} files",
        f"- Integrity: `{checkpoint['integrity']}`",
        f"- freeze_eligible: `{str(checkpoint['freeze_eligible']).lower()}`",
        "- Generation stopped for transfer: **NO**",
        "",
        "## Freeze-ready sequences",
        "",
    ]
    lines.extend(f"- `{sequence}`" for sequence in checkpoint["sequences"])
    lines.extend(["", "## Incomplete sequences", ""])
    for row in incomplete:
        pose = row["sapiens"]
        sam = row["sam_mode_b"]
        lines.extend(
            [
                f"### `{row['sequence']}`",
                "",
                f"- Current stage: `{row['current_stage']}`",
                f"- Active now: `{str(row['active_now']).lower()}`",
                f"- Sapiens durable: {pose['completed_cameras']}/3 cameras, {pose['durable_target_crops']:,}/{pose['expected_target_crops']:,} crops",
                f"- SAM durable: {sam['completed_cameras']}/3 cameras, {sam['durable_numeric_prior_frames']:,}/{sam['expected_source_frames']:,} numeric frames",
                f"- Triangulation: `{row['triangulation_status']}`",
                f"- Body fit: `{row['body_fit_status']}`",
                f"- Quality: `{row['quality_status']}`",
                "- Partial/finalized outputs preserved: `true`",
                "",
            ]
        )
    lines.extend(
        [
            "## Transfer sizes",
            "",
            f"- Critical backup: **{sizes['critical']['logical_bytes']:,} bytes ({sizes['critical']['gib']:.3f} GiB), {sizes['critical']['file_count']:,} files**",
            f"- Resume intermediates: **{sizes['resume']['logical_bytes']:,} bytes ({sizes['resume']['gib']:.3f} GiB), {sizes['resume']['file_count']:,} files**",
            f"- Full resumable backup (critical + intermediates): **{sizes['full_resume']['logical_bytes']:,} bytes ({sizes['full_resume']['gib']:.3f} GiB), {sizes['full_resume']['file_count']:,} files**",
            f"- Optional/excluded: **{sizes['optional']['logical_bytes']:,} bytes ({sizes['optional']['gib']:.3f} GiB), {sizes['optional']['file_count']:,} files**",
            "- Transfer manifests themselves are intentionally excluded from the self-referential size totals.",
            "- Live-tree totals are one-pass finalized-file metadata snapshots; immutable checkpoint integrity uses the existing verified manifest.",
            "",
            "### Critical paths",
            "",
            "| Path | Bytes | GiB | Files | Transient excluded |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in manifest["inventories"]["critical"]:
        lines.append(
            f"| `{row['path']}` | {row['logical_bytes']:,} | {row['gib']:.3f} | {row['file_count']:,} | {row['transient_files_excluded']:,} |"
        )
    lines.extend(
        [
            "",
            "### Resume paths",
            "",
            "| Path | Bytes | GiB | Files | Transient excluded |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in manifest["inventories"]["resume"]:
        lines.append(
            f"| `{row['path']}` | {row['logical_bytes']:,} | {row['gib']:.3f} | {row['file_count']:,} | {row['transient_files_excluded']:,} |"
        )
    lines.extend(
        [
            "",
            "### Optional/excluded paths",
            "",
            "| Path | Bytes | GiB | Files |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in manifest["inventories"]["optional"]:
        lines.append(
            f"| `{row['path']}` | {row['logical_bytes']:,} | {row['gib']:.3f} | {row['file_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Windows WSL + rsync",
            "",
            "Replace the explicit placeholders before running from WSL. The server endpoint could not be inferred safely from this container.",
            "",
            "### Setup",
            "",
            "```bash",
            commands["wsl_setup"],
            "```",
            "",
            "### Stage A — critical sync",
            "",
            "```bash",
            commands["critical_sync"],
            "```",
            "",
            "### Stage B — full resume intermediates",
            "",
            "```bash",
            commands["full_resume_sync"],
            "```",
            "",
            "### Final incremental sync",
            "",
            "```bash",
            commands["final_incremental_sync"],
            "```",
            "",
            "Do not use `--delete`. Re-run the final command after generation and deadline snapshot complete; rsync transfers only changed/new finalized files.",
            "",
        ]
    )
    return "\n".join(lines)


def build_manifest(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    snapshot = utc_now()
    dashboard = read_json(args.dashboard_state.resolve())
    handoff = read_json(args.handoff_state.resolve())
    quality = read_json(args.quality_state.resolve())
    checkpoint_state = read_json(args.checkpoint_state.resolve())
    deadline = read_json(args.deadline_state.resolve())
    frozen_order = [str(value) for value in handoff.get("completed", [])] + [
        str(value) for value in handoff.get("remaining", [])
    ]
    frozen_order = list(dict.fromkeys(frozen_order))
    if len(frozen_order) != args.sequence_count:
        command = handoff.get("resume_commands", {}).get("run_autonomous_generation.py", "")
        try:
            argv = shlex.split(str(command))
            frozen_order = argv[argv.index("--sequences") + 1].split(",")
        except (ValueError, IndexError):
            raise RuntimeError("cannot recover the exact frozen sequence order")
    checkpoint_set = set(checkpoint["sequences"])
    incomplete_names = [sequence for sequence in frozen_order if sequence not in checkpoint_set]
    incomplete = [
        incomplete_sequence_state(
            sequence=sequence,
            checkpoint_sequences=checkpoint_set,
            pose_root=args.pose_root.resolve(),
            selection_root=args.selection_root.resolve(),
            triangulation_root=args.triangulation_root.resolve(),
            sam_root=args.sam_root.resolve(),
            sam_prior_root=args.sam_prior_root.resolve(),
            body_fit_root=args.body_fit_root.resolve(),
            mode_c_root=args.mode_c_root.resolve(),
            quality_root=args.quality_root.resolve(),
            dashboard=dashboard,
        )
        for sequence in incomplete_names
    ]

    critical = [
        scan_finalized_path(PROJECT_ROOT / relative, label=relative)
        for relative in CRITICAL_RELATIVE_PATHS
    ]
    resume = [
        scan_finalized_path(PROJECT_ROOT / relative, label=relative)
        for relative in RESUME_RELATIVE_PATHS
    ]
    optional = [
        scan_finalized_path(path.resolve(), label=label)
        for label, path in args.optional_path
    ]
    critical_total = aggregate_inventory(critical)
    resume_total = aggregate_inventory(resume)
    optional_total = aggregate_inventory(optional)
    full_resume = {
        key: critical_total[key] + resume_total[key]
        for key in ("file_count", "logical_bytes", "transient_files_excluded", "symlinks_excluded", "scan_error_count")
    }
    full_resume.update(
        {
            "path_count": critical_total["path_count"] + resume_total["path_count"],
            "existing_path_count": critical_total["existing_path_count"]
            + resume_total["existing_path_count"],
            "gib": full_resume["logical_bytes"] / (1024**3),
        }
    )
    sapiens_config = read_json(PROJECT_ROOT / "configs/sapiens2_pose_5b_environment.json")
    pose_model = sapiens_config.get("pose_model", {})
    active_processes = handoff.get("active_processes", [])
    if not isinstance(active_processes, list):
        active_processes = []
    process_rows = [
        {
            key: process.get(key)
            for key in ("pid", "command", "argv", "cwd", "process_dir_ctime_utc")
        }
        for process in active_processes
        if isinstance(process, dict)
    ]
    manifest = {
        "schema_version": 1,
        "status": "TRANSFER_READY",
        "snapshot_at_utc": snapshot.isoformat(),
        "snapshot_at_kst": snapshot.astimezone(KST).isoformat(),
        "sequence_count": args.sequence_count,
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "origin_head": git_output(
                "rev-parse", "origin/agent/phase-5-1-pushup-0003-recovery"
            ),
            "status_porcelain": git_output("status", "--short"),
        },
        "checkpoint": checkpoint,
        "freeze_ready_sequences": checkpoint["sequences"],
        "incomplete_sequences": incomplete,
        "durable_progress": {
            "sapiens": dashboard.get("sapiens", {}),
            "sam": dashboard.get("sam", {}),
            "triangulation": dashboard.get("triangulation", {}),
            "body_fit": dashboard.get("body_fit", {}),
            "quality_control": dashboard.get("quality_control", {}),
        },
        "runtime": {
            "dashboard_updated_at": dashboard.get("updated_at"),
            "overall_status": dashboard.get("overall_status"),
            "attention_required": dashboard.get("attention_required"),
            "attention_reasons": dashboard.get("attention_reasons", []),
            "current_operational_event": dashboard.get("current_operational_event"),
            "last_completed_event": dashboard.get("last_completed_event"),
            "deadline": dashboard.get("deadline", {}),
            "gpu": dashboard.get("gpu", {}),
            "disk": dashboard.get("disk", {}),
            "quality_follower": {
                key: quality.get(key)
                for key in (
                    "updated_at_utc",
                    "status",
                    "completed_sequence_count",
                    "freeze_readiness",
                    "last_event",
                )
            },
            "checkpoint_follower": checkpoint_state,
            "deadline_snapshot": deadline,
            "active_processes": process_rows,
            "generation_continues": True,
            "processes_signaled_or_restarted_for_transfer": False,
            "gpu_work_added_for_transfer": False,
        },
        "models": {
            "sapiens2": {
                "model_id": pose_model.get("model_id"),
                "checkpoint_filename": pose_model.get("checkpoint_filename"),
                "checkpoint_bytes": pose_model.get("size_bytes"),
                "checkpoint_sha256": pose_model.get("sha256"),
                "official_repository_commit": sapiens_config.get(
                    "official_repository_commit"
                ),
                "flip_test": pose_model.get("flip_test"),
            },
            "sam_body4d": {
                "mode": "B",
                "mode_c_policy": "SELECTIVE_ESCALATION_ONLY",
                "sam_body4d_repository_commit": git_revision(args.sam_body4d_root.resolve()),
                "sam_3d_body_repository_commit": git_revision(
                    args.sam_3d_body_root.resolve()
                ),
                "checkpoint_identity": checkpoint_identifiers(
                    args.checkpoint_integrity.resolve()
                ),
            },
        },
        "policies": {
            "target_identity": (
                "primary exercise subject only; bidirectional temporal tracking and cross-view QA; "
                "background persons excluded; NO_TARGET/TARGET_AMBIGUOUS abstention preserved"
            ),
            "sapiens": "Sapiens2-5B target-only; completed atomic chunks reused",
            "sam": (
                "Mode B frozen default; Mode C selective escalation only; accepted prior equals "
                "output_valid AND target_valid; abstention outputs never accepted"
            ),
            "quality": "PASS/REVIEW/FAIL provenance preserved; no deadline threshold relaxation",
            "privacy": "raw/private payload excluded from Git",
        },
        "resume": {
            "policy": handoff.get("resume_policy"),
            "semantics": (
                "Validate PASS metadata/schema/source binding; skip finalized complete cameras/chunks; "
                "preserve durable partial output; rerun only incomplete/corrupt work; singleton locks and "
                "orphan guards remain authoritative; never force a background target."
            ),
            "exact_commands": handoff.get("resume_commands", {}),
            "transient_exclusions": [
                "names containing .tmp",
                "*.partial",
                "*.part",
                "*.lock",
                ".*.inprogress directories",
                ".rsync-partial",
                "symlinks",
            ],
            "live_output_policy": (
                "Atomic finalized pathnames are transferable. Files created after the scan are picked "
                "up by the final incremental rsync; no source deletion and no --delete."
            ),
        },
        "inventories": {
            "critical": critical,
            "resume": resume,
            "optional": optional,
        },
        "transfer_sizes": {
            "critical": critical_total,
            "resume": resume_total,
            "full_resume": full_resume,
            "optional": optional_total,
            "optional_excluded_from_full_resume_total": True,
            "generated_transfer_manifests_excluded_from_totals": True,
        },
        "windows_transfer": {
            "recommended_method": "Windows WSL + rsync",
            "recommended_local_layout": args.windows_local_root,
            "server_repository_path": str(PROJECT_ROOT),
            "endpoint_inference": (
                "SSH username/host/port are placeholders because this container does not expose the "
                "user's externally routable SSH endpoint safely."
            ),
            "commands": transfer_commands(args),
            "two_stage_policy": {
                "stage_a": "sync critical group immediately after this manifest",
                "stage_b": "sync large resume intermediates at low priority while generation continues",
                "final": "rerun combined incremental sync after generation/deadline snapshot",
            },
        },
        "measurement": {
            "method": "single os.scandir/lstat metadata pass; no payload hashing or compression",
            "logical_bytes": "sum of st_size for finalized regular files",
            "checkpoint_integrity": "reused follower byte/hash verification and immutable manifest",
            "live_tree_consistency": (
                "point-in-time best effort for still-growing intermediate roots; finalized atomic files only"
            ),
        },
    }
    if any(
        row["scan_errors"]
        for group in manifest["inventories"].values()
        for row in group
    ):
        raise RuntimeError("one or more transfer inventory metadata scans failed")
    return manifest


def parse_optional_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("optional path must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("optional path must be LABEL=PATH")
    return label, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime/predeadline_checkpoint_follower_state.json",
    )
    parser.add_argument(
        "--dashboard-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime/dashboard_state.json",
    )
    parser.add_argument(
        "--handoff-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime/handoff_state.json",
    )
    parser.add_argument(
        "--quality-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime/quality_follower_state.json",
    )
    parser.add_argument(
        "--deadline-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime/deadline_snapshot_state.json",
    )
    parser.add_argument(
        "--runtime-state",
        type=Path,
        default=PROJECT_ROOT / ".runtime/transfer_snapshot_state.json",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / ".runtime/transfer_manifest.json",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / ".runtime/TRANSFER_MANIFEST.md",
    )
    parser.add_argument(
        "--instance-lock",
        type=Path,
        default=PROJECT_ROOT / ".runtime/transfer_snapshot.lock",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs/private_dataset_freeze"
    )
    parser.add_argument(
        "--selection-root", type=Path, default=PROJECT_ROOT / "outputs/target_selection_full"
    )
    parser.add_argument(
        "--pose-root", type=Path, default=PROJECT_ROOT / "outputs/sapiens2_target_only_full"
    )
    parser.add_argument(
        "--triangulation-root", type=Path, default=PROJECT_ROOT / "outputs/triangulation_final"
    )
    parser.add_argument(
        "--sam-root", type=Path, default=PROJECT_ROOT / "outputs/sam_body4d_full"
    )
    parser.add_argument(
        "--sam-prior-root", type=Path, default=PROJECT_ROOT / "outputs/sam_body_prior_full"
    )
    parser.add_argument(
        "--body-fit-root", type=Path, default=PROJECT_ROOT / "outputs/body_fit_full"
    )
    parser.add_argument(
        "--mode-c-root", type=Path, default=PROJECT_ROOT / "outputs/sam_mode_c_review_full"
    )
    parser.add_argument(
        "--quality-root", type=Path, default=PROJECT_ROOT / "outputs/quality_control_full"
    )
    parser.add_argument("--sam-body4d-root", type=Path, required=True)
    parser.add_argument("--sam-3d-body-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-integrity",
        type=Path,
        default=PROJECT_ROOT / "metadata/results/sam_body4d_checkpoint_integrity.csv",
    )
    parser.add_argument(
        "--optional-path", action="append", type=parse_optional_path, default=[]
    )
    parser.add_argument("--minimum-ready", type=int, default=24)
    parser.add_argument("--sequence-count", type=int, default=26)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--ssh-user", default="<SSH_USERNAME>")
    parser.add_argument("--ssh-host", default="<SERVER_HOST_OR_IP>")
    parser.add_argument("--ssh-port", default="<SSH_PORT>")
    parser.add_argument(
        "--windows-local-root", default="/mnt/d/Exercise3D-Dataset-Pipeline"
    )
    parser.add_argument("--bwlimit-kib", type=int, default=30000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.minimum_ready <= 0
        or args.minimum_ready > args.sequence_count
        or args.poll_seconds <= 0
        or args.bwlimit_kib <= 0
    ):
        raise RuntimeError("invalid transfer snapshot gate or polling configuration")
    singleton: BinaryIO | None = acquire_singleton_lock(args.instance_lock.resolve())
    if singleton is None:
        print("transfer snapshot waiter is already running", flush=True)
        return 3
    last_count: int | None = None
    while True:
        state = read_json(args.checkpoint_state.resolve())
        checkpoint, reasons = checkpoint_gate(
            state, args.output_root.resolve(), args.minimum_ready
        )
        best = state.get("best_checkpoint", {})
        try:
            count = int(best.get("completed_sequence_count", 0))
        except (TypeError, ValueError):
            count = 0
        if checkpoint is not None:
            atomic_json(
                args.runtime_state.resolve(),
                {
                    "schema_version": 1,
                    "updated_at_utc": utc_now().isoformat(),
                    "status": "INVENTORY_RUNNING",
                    "pid": os.getpid(),
                    "gpu_work": False,
                    "minimum_ready": args.minimum_ready,
                    "checkpoint": checkpoint,
                    "policy": "metadata-only inventory; never signal or launch generation",
                },
            )
            manifest = build_manifest(args, checkpoint)
            atomic_json(args.json_output.resolve(), manifest)
            atomic_text(args.markdown_output.resolve(), render_markdown(manifest))
            atomic_json(
                args.runtime_state.resolve(),
                {
                    "schema_version": 1,
                    "updated_at_utc": utc_now().isoformat(),
                    "status": "TRANSFER_READY",
                    "pid": os.getpid(),
                    "gpu_work": False,
                    "minimum_ready": args.minimum_ready,
                    "checkpoint": checkpoint,
                    "json_manifest": str(args.json_output.resolve()),
                    "markdown_manifest": str(args.markdown_output.resolve()),
                    "critical_bytes": manifest["transfer_sizes"]["critical"][
                        "logical_bytes"
                    ],
                    "full_resume_bytes": manifest["transfer_sizes"]["full_resume"][
                        "logical_bytes"
                    ],
                    "optional_bytes": manifest["transfer_sizes"]["optional"][
                        "logical_bytes"
                    ],
                    "generation_continues": True,
                    "processes_signaled_or_restarted": False,
                    "last_event": "TRANSFER_MANIFEST_MATERIALIZED",
                },
            )
            print(
                json.dumps(
                    {
                        "status": "TRANSFER_READY",
                        "build_id": checkpoint["build_id"],
                        "sequence_count": checkpoint["sequence_count"],
                        "json_manifest": str(args.json_output.resolve()),
                        "markdown_manifest": str(args.markdown_output.resolve()),
                    }
                ),
                flush=True,
            )
            return 0
        if not args.wait:
            print(
                json.dumps(
                    {
                        "status": "WAITING_FOR_DURABLE_CHECKPOINT",
                        "completed_sequence_count": count,
                        "minimum_ready": args.minimum_ready,
                        "reasons": reasons,
                    }
                ),
                flush=True,
            )
            return 4
        if count != last_count:
            atomic_json(
                args.runtime_state.resolve(),
                {
                    "schema_version": 1,
                    "updated_at_utc": utc_now().isoformat(),
                    "status": "WAITING_FOR_DURABLE_CHECKPOINT",
                    "pid": os.getpid(),
                    "gpu_work": False,
                    "completed_sequence_count": count,
                    "minimum_ready": args.minimum_ready,
                    "reasons": reasons,
                    "policy": (
                        "read atomic checkpoint state only; perform one metadata inventory after gate; "
                        "never signal or launch generation"
                    ),
                },
            )
            print(
                json.dumps(
                    {
                        "status": "WAITING_FOR_DURABLE_CHECKPOINT",
                        "completed_sequence_count": count,
                        "minimum_ready": args.minimum_ready,
                    }
                ),
                flush=True,
            )
            last_count = count
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
