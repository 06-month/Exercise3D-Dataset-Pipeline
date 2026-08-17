#!/usr/bin/env python3
"""공개 Git commit 후보에서 private/large payload를 보수적으로 차단한다."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_BYTES = 5 * 1024 * 1024
PUBLIC_SHOWCASE_PATHS = {
    Path("docs/assets/showcase/benchpress_0004_mhr_mesh.mp4"),
    Path("docs/assets/showcase/deadlift_0001_mhr_mesh.mp4"),
    Path("docs/assets/showcase/barbellrow_0003_mhr_mesh.mp4"),
    Path("docs/assets/showcase/latpulldown_0003_mhr_mesh.mp4"),
    Path("docs/assets/showcase/squat_0002_mhr_mesh.mp4"),
    Path("docs/assets/showcase/pushup_0000_mhr_mesh.mp4"),
    Path("docs/assets/showcase/benchpress_0004_mhr_mesh.gif"),
    Path("docs/assets/showcase/deadlift_0001_mhr_mesh.gif"),
    Path("docs/assets/showcase/barbellrow_0003_mhr_mesh.gif"),
    Path("docs/assets/showcase/latpulldown_0003_mhr_mesh.gif"),
    Path("docs/assets/showcase/squat_0002_mhr_mesh.gif"),
    Path("docs/assets/showcase/pushup_0000_mhr_mesh.gif"),
}
FORBIDDEN_SUFFIXES = {
    ".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".wav", ".mp3", ".m4a",
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff", ".gif",
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine",
    ".npz", ".npy", ".ply", ".pcd", ".obj", ".gpx", ".kml", ".kmz",
}
FORBIDDEN_PARTS = {
    "origin", "_replaced_originals", "synced_video", "final_frame", "working_frames",
    "checkpoints", "weights", "pretrained", "private_data", "pseudo_labels",
}
TEXT_PATTERNS = {
    "절대 workspace 경로": re.compile(r"/workspace/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[opusr]_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}


def tracked_paths(staged: bool) -> list[Path]:
    if staged:
        command = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
    else:
        command = ["git", "ls-files", "-z"]
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True)
    return [PROJECT_ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def inspect(path: Path) -> list[str]:
    relative = path.relative_to(PROJECT_ROOT)
    errors: list[str] = []
    if path.is_symlink():
        errors.append(f"symlink 금지: {relative}")
        return errors
    if any(part in FORBIDDEN_PARTS for part in relative.parts):
        errors.append(f"private 경로 금지: {relative}")
    public_showcase = relative in PUBLIC_SHOWCASE_PATHS
    if path.suffix.lower() in FORBIDDEN_SUFFIXES and not public_showcase:
        errors.append(f"payload 확장자 금지: {relative}")
    if path.exists() and path.stat().st_size > MAX_PUBLIC_BYTES:
        errors.append(f"5 MiB 초과 파일 금지: {relative} ({path.stat().st_size} bytes)")
    if relative == Path("tools/check_publication_safety.py") or public_showcase:
        return errors
    if not path.exists() or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        return errors
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        errors.append(f"검토되지 않은 binary 파일: {relative}")
        return errors
    for label, pattern in TEXT_PATTERNS.items():
        if pattern.search(content):
            errors.append(f"{label} 발견: {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="staged 파일 대신 전체 tracked 파일 검사")
    args = parser.parse_args()
    paths = tracked_paths(staged=not args.all)
    errors = [error for path in paths for error in inspect(path)]
    if errors:
        print("PUBLICATION SAFETY: FAIL")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"PUBLICATION SAFETY: PASS ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
