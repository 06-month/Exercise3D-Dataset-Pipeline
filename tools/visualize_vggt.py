#!/usr/bin/env python3
"""Open3D viewer for immutable VGGT-Omega outputs and Phase 4 BA overlays.

The viewer reads existing Phase 3 geometry and synchronized videos. It never
changes poses, point maps, depth, confidence, source video, or working frames.
Optional screenshots, PLY files, and stats default to
``outputs/vggt/visualization``. ``--debug-root`` can redirect them to a separate
debug directory, while immutable dataset input trees remain write-protected.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT)
).expanduser()
CAMERA_COLORS = {
    "cam1": (0.95, 0.16, 0.16),
    "cam2": (0.15, 0.82, 0.28),
    "cam3": (0.18, 0.38, 0.98),
}
TIME_COLORS = (
    (0.95, 0.10, 0.10),
    (1.00, 0.52, 0.05),
    (0.95, 0.90, 0.08),
    (0.12, 0.82, 0.22),
    (0.05, 0.82, 0.88),
    (0.12, 0.32, 0.98),
    (0.55, 0.18, 0.95),
    (0.95, 0.18, 0.70),
)
PRESET_TO_PERCENTILE = {"all": 0.0, "top75": 25.0, "top50": 50.0, "top25": 75.0}


def ensure_open3d_runtime() -> None:
    """Re-exec in an existing local Open3D environment when necessary."""
    try:
        import open3d  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    if os.environ.get("EXERCISE3D_OPEN3D_REEXEC") == "1":
        raise RuntimeError("Open3D is unavailable in the selected runtime")
    configured = os.environ.get("EXERCISE3D_OPEN3D_PYTHON")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            Path(sys.prefix) / "envs" / "c4g-fresh" / "bin" / "python",
            Path(sys.prefix).parent / "envs" / "c4g-fresh" / "bin" / "python",
        ]
    )
    runtime = next((path for path in candidates if path.is_file()), None)
    if runtime is None:
        raise RuntimeError(
            "Open3D is unavailable. Set EXERCISE3D_OPEN3D_PYTHON to an existing local Python runtime."
        )
    conda_root = runtime.parents[3]
    env = os.environ.copy()
    env["EXERCISE3D_OPEN3D_REEXEC"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    library_path = str(conda_root / "lib")
    if env.get("LD_LIBRARY_PATH"):
        library_path += os.pathsep + env["LD_LIBRARY_PATH"]
    env["LD_LIBRARY_PATH"] = library_path
    os.execve(str(runtime), [str(runtime), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def prepare_headless_environment() -> None:
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    if not os.environ.get("XDG_RUNTIME_DIR"):
        runtime_dir = Path(tempfile.mkdtemp(prefix="exercise3d-open3d-runtime-"))
        runtime_dir.chmod(0o700)
        os.environ["XDG_RUNTIME_DIR"] = str(runtime_dir)
        atexit.register(shutil.rmtree, runtime_dir, True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_sequence(output_root: Path, sequence_id: str) -> Path:
    candidates = sorted(
        path.parent
        for path in output_root.glob(f"*/*/{sequence_id}/metadata.json")
        if path.parent.name == sequence_id
    )
    if not candidates:
        raise RuntimeError(f"sequence not found below {output_root}: {sequence_id}")
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous sequence id {sequence_id}: {candidates}")
    return candidates[0]


def safe_debug_path(debug_root: Path, value: str | None, expected_suffix: str) -> Path | None:
    if value is None:
        return None
    requested = Path(value)
    target = requested.resolve() if requested.is_absolute() else (debug_root / requested).resolve()
    debug_resolved = debug_root.resolve()
    if target != debug_resolved and debug_resolved not in target.parents:
        raise RuntimeError(f"debug output must remain below {debug_root}: {target}")
    if target.suffix.lower() != expected_suffix:
        raise RuntimeError(f"expected {expected_suffix} output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def load_background_ba(ba_root: Path, sequence_id: str) -> dict[str, Any]:
    import numpy as np

    sequence_dir = ba_root / sequence_id
    required = [
        sequence_dir / "cameras_initial.json",
        sequence_dir / "cameras_refined.json",
        sequence_dir / "points3d.npz",
        sequence_dir / "metrics.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"background BA comparison inputs missing: {missing}")
    initial = json.loads(required[0].read_text(encoding="utf-8"))
    refined = json.loads(required[1].read_text(encoding="utf-8"))
    metrics = json.loads(required[3].read_text(encoding="utf-8"))
    if initial["sequence"] != sequence_id or refined["sequence"] != sequence_id:
        raise RuntimeError(f"background BA sequence mismatch: {sequence_dir}")
    with np.load(required[2]) as archive:
        points_initial = np.asarray(archive["points_initial"], dtype=np.float64)
        points_refined = np.asarray(archive["points_refined"], dtype=np.float64)
        accepted = np.asarray(archive["accepted_track_mask"], dtype=bool)
    if points_initial.shape != points_refined.shape or points_initial.shape[1:] != (3,):
        raise RuntimeError(f"invalid background BA point shape: {sequence_dir}")
    if accepted.shape != (len(points_initial),):
        raise RuntimeError(f"invalid background BA point mask: {sequence_dir}")
    transform = np.asarray(initial["source_vggt_world_to_ba_world"], dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise RuntimeError(f"invalid VGGT-to-BA gauge transform: {sequence_dir}")
    return {
        "sequence_dir": sequence_dir,
        "initial": initial,
        "refined": refined,
        "metrics": metrics,
        "points_initial": points_initial,
        "points_refined": points_refined,
        "accepted": accepted,
        "source_to_display": transform,
    }


def load_geometry(camera_dir: Path) -> dict[str, Any]:
    import numpy as np

    with np.load(camera_dir / "poses.npz") as archive:
        pose = {key: np.array(archive[key]) for key in archive.files}
    with np.load(camera_dir / "pointmap.npz") as archive:
        pointmap = np.array(archive["world_points_from_depth"])
        point_timestamps = np.array(archive["timestamps_sec"])
    with np.load(camera_dir / "depth.npz") as archive:
        depth = np.array(archive["depth"])
        depth_timestamps = np.array(archive["timestamps_sec"])
    with np.load(camera_dir / "confidence.npz") as archive:
        confidence = np.array(archive["depth_confidence"])
        confidence_timestamps = np.array(archive["timestamps_sec"])
    frames = read_csv(camera_dir / "frames.csv")
    count = len(frames)
    expected = {
        "pose_encoding": (count, 9),
        "extrinsics_world_to_camera": (count, 3, 4),
        "camera_to_world": (count, 4, 4),
        "intrinsics": (count, 3, 3),
    }
    for key, shape in expected.items():
        if pose[key].shape != shape:
            raise RuntimeError(f"{camera_dir}/{key}: expected {shape}, got {pose[key].shape}")
    height, width = pointmap.shape[1:3]
    if pointmap.shape != (count, height, width, 3):
        raise RuntimeError(f"invalid pointmap shape: {pointmap.shape}")
    if depth.shape != (count, height, width, 1):
        raise RuntimeError(f"invalid depth shape: {depth.shape}")
    if confidence.shape != (count, height, width):
        raise RuntimeError(f"invalid confidence shape: {confidence.shape}")
    timestamps = pose["timestamps_sec"]
    if not (
        np.allclose(timestamps, point_timestamps)
        and np.allclose(timestamps, depth_timestamps)
        and np.allclose(timestamps, confidence_timestamps)
    ):
        raise RuntimeError(f"timestamp arrays disagree: {camera_dir}")
    arrays = [
        pose["pose_encoding"], pose["extrinsics_world_to_camera"], pose["camera_to_world"],
        pose["intrinsics"], pointmap, depth, confidence,
    ]
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"non-finite VGGT values: {camera_dir}")
    if not (depth > 0).all() or not (confidence >= 1.0).all():
        raise RuntimeError(f"invalid depth/confidence values: {camera_dir}")
    extrinsic4 = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
    extrinsic4[:, :3, :4] = pose["extrinsics_world_to_camera"]
    inverse_error = float(np.max(np.abs(extrinsic4 @ pose["camera_to_world"] - np.eye(4))))
    if inverse_error > 1e-4:
        raise RuntimeError(f"world/camera inverse error {inverse_error}: {camera_dir}")
    rotation = pose["extrinsics_world_to_camera"][:, :3, :3]
    translation = pose["extrinsics_world_to_camera"][:, :3, 3]
    center_from_extrinsic = -np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), translation)
    center_error = float(
        np.max(np.abs(center_from_extrinsic - pose["camera_to_world"][:, :3, 3]))
    )
    if center_error > 1e-4:
        raise RuntimeError(f"camera-center conversion error {center_error}: {camera_dir}")
    rotation_orthogonality_error = float(
        np.max(np.abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3)))
    )
    rotation_determinants = np.linalg.det(rotation)
    if rotation_orthogonality_error > 1e-3 or not np.allclose(
        rotation_determinants, 1.0, atol=1e-3
    ):
        raise RuntimeError(f"invalid camera rotations: {camera_dir}")

    # Verify a sparse grid against the documented depth+K+camera_to_world formula.
    sample_y = np.unique(np.linspace(0, height - 1, 7, dtype=np.int64))
    sample_x = np.unique(np.linspace(0, width - 1, 7, dtype=np.int64))
    grid_y, grid_x = np.meshgrid(sample_y, sample_x, indexing="ij")
    pointmap_error = 0.0
    for index in range(count):
        intrinsic = pose["intrinsics"][index]
        z = depth[index, grid_y, grid_x, 0]
        camera_points = np.stack(
            [
                (grid_x - intrinsic[0, 2]) / intrinsic[0, 0] * z,
                (grid_y - intrinsic[1, 2]) / intrinsic[1, 1] * z,
                z,
            ],
            axis=-1,
        )
        camera_to_world = pose["camera_to_world"][index]
        reconstructed = (
            camera_points @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
        )
        error = np.max(np.abs(reconstructed - pointmap[index, grid_y, grid_x]))
        pointmap_error = max(pointmap_error, float(error))
    if pointmap_error > 1e-4:
        raise RuntimeError(f"depth/pointmap reconstruction error {pointmap_error}: {camera_dir}")
    for index, row in enumerate(frames):
        if int(row["source_frame_index"]) != int(pose["source_frame_indices"][index]):
            raise RuntimeError(f"frame index provenance mismatch: {camera_dir}, row {index}")
        if abs(float(row["source_packet_pts_sec"]) - float(timestamps[index])) > 1e-6:
            raise RuntimeError(f"PTS provenance mismatch: {camera_dir}, row {index}")
    return {
        **pose,
        "pointmap": pointmap,
        "depth": depth,
        "confidence": confidence,
        "frames": frames,
        "height": height,
        "width": width,
        "inverse_error": inverse_error,
        "center_error": center_error,
        "rotation_orthogonality_error": rotation_orthogonality_error,
        "rotation_determinants": rotation_determinants,
        "pointmap_error": pointmap_error,
    }


def decode_selected_rgb(video_path: Path, indices: list[int]) -> dict[int, Any]:
    import cv2

    targets = set(indices)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    output = {}
    index = 0
    maximum = max(targets)
    try:
        while index <= maximum:
            ok, frame = capture.read()
            if not ok:
                break
            if index in targets:
                output[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    missing = targets - output.keys()
    if missing:
        raise RuntimeError(f"failed RGB decode {sorted(missing)}: {video_path}")
    return output


def preprocess_rgb(rgb: Any, frame: dict[str, str], expected_shape: tuple[int, int]) -> Any:
    import numpy as np
    from PIL import Image

    left, top = int(frame["crop_left"]), int(frame["crop_top"])
    right, bottom = int(frame["crop_right"]), int(frame["crop_bottom"])
    image = Image.fromarray(rgb).crop((left, top, right, bottom))
    image = image.resize(
        (int(frame["resized_width"]), int(frame["resized_height"])), Image.Resampling.BICUBIC
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    pad_width = (
        (int(frame["pad_top"]), int(frame["pad_bottom"])),
        (int(frame["pad_left"]), int(frame["pad_right"])),
        (0, 0),
    )
    if any(value for pair in pad_width[:2] for value in pair):
        array = np.pad(array, pad_width, mode="constant", constant_values=1.0)
    if array.shape[:2] != expected_shape:
        raise RuntimeError(f"RGB/model shape mismatch: {array.shape[:2]} vs {expected_shape}")
    return array


def confidence_colors(values: Any, lower: float, upper: float) -> Any:
    import numpy as np

    normalized = np.clip((values - lower) / max(upper - lower, 1e-9), 0.0, 1.0)
    # Blue -> cyan -> yellow -> red, deliberately dependency-free.
    red = np.clip(3.0 * normalized - 1.0, 0.0, 1.0)
    green = np.clip(1.5 - np.abs(3.0 * normalized - 1.5), 0.0, 1.0)
    blue = np.clip(2.0 - 3.0 * normalized, 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


def time_color(camera: str, sample_index: int, sample_count: int) -> Any:
    import numpy as np

    del camera, sample_count
    return np.asarray(TIME_COLORS[sample_index % len(TIME_COLORS)], dtype=np.float64)


def camera_color(camera: str, sample_index: int, sample_count: int) -> Any:
    import numpy as np

    base = np.asarray(CAMERA_COLORS[camera], dtype=np.float64)
    factor = 0.55 + 0.45 * (sample_index + 1) / max(sample_count, 1)
    return np.clip(base * factor, 0.0, 1.0)


def create_frustum(
    open3d: Any,
    intrinsic: Any,
    camera_to_world: Any,
    width: int,
    height: int,
    scale: float,
    color: Any,
) -> Any:
    import numpy as np

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    pixels = [(0.0, 0.0), (width - 1.0, 0.0), (width - 1.0, height - 1.0), (0.0, height - 1.0)]
    camera_points = [np.zeros(3, dtype=np.float64)]
    for u, v in pixels:
        camera_points.append(np.array([(u - cx) / fx * scale, (v - cy) / fy * scale, scale]))
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    world_points = [rotation @ point + translation for point in camera_points]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(np.asarray(world_points))
    line_set.lines = open3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    line_set.colors = open3d.utility.Vector3dVector(
        np.repeat(np.asarray(color, dtype=np.float64)[None], len(lines), axis=0)
    )
    return line_set


def create_track(open3d: Any, centers: Any, color: Any) -> Any | None:
    import numpy as np

    if len(centers) < 2:
        return None
    lines = np.stack([np.arange(len(centers) - 1), np.arange(1, len(centers))], axis=1)
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(np.asarray(centers, dtype=np.float64))
    line_set.lines = open3d.utility.Vector2iVector(lines.astype(np.int32))
    line_set.colors = open3d.utility.Vector3dVector(
        np.repeat(np.asarray(color, dtype=np.float64)[None], len(lines), axis=0)
    )
    return line_set


def robust_scene_bounds(points: Any) -> tuple[Any, Any, float]:
    import numpy as np

    if len(points) == 0:
        raise RuntimeError("no points remain after filtering")
    stride = max(1, len(points) // 200_000)
    sample = points[::stride]
    lower, upper = np.percentile(sample, [2, 98], axis=0)
    center = (lower + upper) / 2.0
    radius = float(np.linalg.norm(upper - lower) / 2.0)
    return lower, upper, max(radius, 1e-3)


def normalized(vector: Any, fallback: Any) -> Any:
    import numpy as np

    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else np.asarray(fallback, dtype=np.float64)


def view_parameters(points: Any, cameras: list[dict[str, Any]], view: str) -> tuple[Any, Any, Any]:
    import numpy as np

    _, _, radius = robust_scene_bounds(points)
    point_stride = max(1, len(points) // 200_000)
    look_at = np.median(points[::point_stride], axis=0)
    forwards, ups = [], []
    for item in cameras:
        rotation = item["camera_to_world"][:3, :3]
        forwards.append(rotation @ np.array([0.0, 0.0, 1.0]))
        ups.append(rotation @ np.array([0.0, -1.0, 0.0]))
    forward = normalized(np.mean(forwards, axis=0) if forwards else [0, 0, 1], [0, 0, 1])
    up = normalized(np.mean(ups, axis=0) if ups else [0, 1, 0], [0, 1, 0])
    right = normalized(np.cross(forward, up), [1, 0, 0])
    up = normalized(np.cross(right, forward), [0, 1, 0])
    if view == "top":
        eye = look_at + up * radius * 2.7 - forward * radius * 0.15
        render_up = forward
    elif view == "side":
        eye = look_at + right * radius * 2.7 + up * radius * 0.25
        render_up = up
    elif view == "front":
        eye = look_at - forward * radius * 2.7 + up * radius * 0.15
        render_up = up
    else:
        eye = look_at - forward * radius * 2.1 + right * radius * 1.2 + up * radius * 0.65
        render_up = up
    return look_at, eye, normalized(render_up, [0, 1, 0])


def make_material(open3d: Any, shader: str, point_size: float = 3.0, line_width: float = 2.0) -> Any:
    material = open3d.visualization.rendering.MaterialRecord()
    material.shader = shader
    material.point_size = point_size
    material.line_width = line_width
    return material


def render_screenshot(
    open3d: Any,
    path: Path,
    geometries: list[dict[str, Any]],
    points: Any,
    camera_items: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    renderer = open3d.visualization.rendering.OffscreenRenderer(args.width, args.height)
    background = [0.035, 0.035, 0.045, 1.0] if args.background == "dark" else [1.0, 1.0, 1.0, 1.0]
    renderer.scene.set_background(background)
    point_material = make_material(open3d, "defaultUnlit", point_size=args.point_size)
    line_material = make_material(open3d, "unlitLine", line_width=2.5)
    mesh_material = make_material(open3d, "defaultLit")
    for item in geometries:
        geometry = item["geometry"]
        if isinstance(geometry, open3d.geometry.PointCloud):
            material = point_material
        elif isinstance(geometry, open3d.geometry.LineSet):
            material = line_material
        else:
            material = mesh_material
        renderer.scene.add_geometry(item["name"], geometry, material)
    look_at, eye, up = view_parameters(points, camera_items, args.view)
    renderer.setup_camera(55.0, look_at, eye, up)
    image = renderer.render_to_image()
    if not open3d.io.write_image(str(path), image, 9):
        raise RuntimeError(f"Open3D failed to write screenshot: {path}")


def interactive_view(open3d: Any, geometries: list[dict[str, Any]], title: str) -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError(
            "No display server is available. Use --headless with --save-screenshot, "
            "or run through X11/Wayland forwarding for interactive mode."
        )
    open3d.visualization.draw(
        geometries,
        title=title,
        show_ui=True,
        show_skybox=False,
        bg_color=(0.035, 0.035, 0.045, 1.0),
        point_size=3,
        line_width=2,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="Sequence id such as squat_0001")
    parser.add_argument(
        "--dataset-root", "--root", dest="root", type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="private dataset root (or EXERCISE3D_DATASET_ROOT)",
    )
    parser.add_argument("--vggt-output", type=Path, default=None)
    parser.add_argument("--compare-background-ba", action="store_true")
    parser.add_argument("--background-ba-root", type=Path, default=None)
    parser.add_argument(
        "--debug-root", type=Path, default=None,
        help="debug output root; must not overlap immutable dataset inputs",
    )
    parser.add_argument("--camera", action="append", choices=tuple(CAMERA_COLORS), default=[])
    parser.add_argument(
        "--frame-index", type=int, default=None,
        help="VGGT point-map sample 0..7; use --frustum-mode selected for its pose only",
    )
    parser.add_argument("--show-cameras", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-camera-tracks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-pointcloud", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-axis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-ba-initial-points", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show-ba-refined-points", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frustum-mode", choices=("all", "selected", "first"), default="all")
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--confidence-preset", choices=tuple(PRESET_TO_PERCENTILE), default=None)
    parser.add_argument("--confidence-scope", choices=("frame", "sequence"), default="frame")
    parser.add_argument("--color-mode", choices=("rgb", "confidence", "camera", "time"), default="rgb")
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--frustum-scale", type=float, default=None)
    parser.add_argument("--view", choices=("auto", "front", "top", "side"), default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--background", choices=("dark", "light"), default="dark")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-screenshot", default=None, help="PNG path relative to the debug root")
    parser.add_argument("--export-ply", default=None, help="PLY path relative to the debug root")
    parser.add_argument("--stats-json", default=None, help="JSON path relative to the debug root")
    parser.add_argument("--quiet", action="store_true", help="Print only a compact summary")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ensure_open3d_runtime()
    args = parse_args(argv)
    args.root = args.root.resolve()
    output_root = (args.vggt_output or args.root / "outputs" / "vggt").resolve()
    if args.debug_root is None:
        debug_root = output_root / "visualization"
    else:
        debug_root = (
            args.debug_root.resolve()
            if args.debug_root.is_absolute()
            else (args.root / args.debug_root).resolve()
        )
        immutable_roots = [
            (args.root / name).resolve()
            for name in ("origin", "synced_video", "final_frame")
        ]
        if any(debug_root == item or item in debug_root.parents for item in immutable_roots):
            raise RuntimeError(f"--debug-root overlaps immutable dataset input: {debug_root}")
    screenshot_path = safe_debug_path(debug_root, args.save_screenshot, ".png")
    ply_path = safe_debug_path(debug_root, args.export_ply, ".ply")
    stats_path = safe_debug_path(debug_root, args.stats_json, ".json")
    if args.headless or screenshot_path is not None:
        prepare_headless_environment()

    import numpy as np
    import open3d as o3d

    if not 0.0 <= args.confidence_percentile <= 100.0:
        raise RuntimeError("--confidence-percentile must be in [0,100]")
    percentile = (
        PRESET_TO_PERCENTILE[args.confidence_preset]
        if args.confidence_preset is not None
        else args.confidence_percentile
    )
    if args.voxel_size < 0 or args.max_points < 1:
        raise RuntimeError("voxel size must be nonnegative and max points must be positive")

    sequence_dir = resolve_sequence(output_root, args.sequence)
    sequence_metadata = json.loads((sequence_dir / "metadata.json").read_text(encoding="utf-8"))
    background_ba = None
    source_to_display = np.eye(4, dtype=np.float64)
    if args.compare_background_ba:
        ba_root = (args.background_ba_root or args.root / "outputs" / "background_ba").resolve()
        background_ba = load_background_ba(ba_root, args.sequence)
        source_to_display = background_ba["source_to_display"]
    cameras = args.camera or list(CAMERA_COLORS)
    loaded = {camera: load_geometry(sequence_dir / camera) for camera in cameras}
    sample_count = next(iter(loaded.values()))["pointmap"].shape[0]
    if args.frame_index is not None and not 0 <= args.frame_index < sample_count:
        raise RuntimeError(f"--frame-index must be in [0,{sample_count - 1}]")
    selected_indices = (
        [args.frame_index] if args.frame_index is not None else list(range(sample_count))
    )

    confidence_values = []
    for camera in cameras:
        data = loaded[camera]
        for index in selected_indices:
            confidence_values.append(data["confidence"][index].reshape(-1))
    global_confidence = np.concatenate(confidence_values)
    global_threshold = float(np.percentile(global_confidence, percentile))
    global_color_low, global_color_high = np.percentile(global_confidence, [5, 95])

    chunks = []
    color_chunks = []
    frame_stats = []
    camera_items = []
    for camera in cameras:
        data = loaded[camera]
        frame_rows = data["frames"]
        source_indices = [int(frame_rows[index]["source_frame_index"]) for index in selected_indices]
        video_path = args.root / frame_rows[0]["source_video"]
        decoded = decode_selected_rgb(video_path, source_indices) if args.color_mode == "rgb" else {}
        for index in selected_indices:
            points = data["pointmap"][index].reshape(-1, 3)
            if background_ba is not None:
                points = (
                    points @ source_to_display[:3, :3].T + source_to_display[:3, 3]
                )
            confidence = data["confidence"][index].reshape(-1)
            valid = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
            threshold = (
                global_threshold
                if args.confidence_scope == "sequence"
                else float(np.percentile(confidence[valid], percentile))
            )
            valid &= confidence >= threshold
            if args.color_mode == "rgb":
                row = frame_rows[index]
                rgb = preprocess_rgb(
                    decoded[int(row["source_frame_index"])], row, (data["height"], data["width"])
                ).reshape(-1, 3)
                colors = rgb[valid]
            elif args.color_mode == "confidence":
                colors = confidence_colors(confidence[valid], float(global_color_low), float(global_color_high))
            elif args.color_mode == "camera":
                colors = np.repeat(np.asarray(CAMERA_COLORS[camera])[None], int(valid.sum()), axis=0)
            else:
                colors = np.repeat(time_color(camera, index, sample_count)[None], int(valid.sum()), axis=0)
            chunks.append(points[valid].astype(np.float64, copy=False))
            color_chunks.append(colors.astype(np.float64, copy=False))
            frame_stats.append(
                {
                    "camera_id": camera,
                    "sample_index": index,
                    "timestamp_sec": float(data["timestamps_sec"][index]),
                    "source_frame_index": int(frame_rows[index]["source_frame_index"]),
                    "confidence_threshold": threshold,
                    "points_before_filter": len(points),
                    "points_after_confidence": int(valid.sum()),
                    "retained_fraction": float(valid.mean()),
                }
            )
        for index in range(sample_count):
            camera_items.append(
                {
                    "camera_id": camera,
                    "sample_index": index,
                    "timestamp_sec": float(data["timestamps_sec"][index]),
                    "camera_to_world": source_to_display @ data["camera_to_world"][index],
                    "intrinsic": data["intrinsics"][index],
                    "width": data["width"],
                    "height": data["height"],
                }
            )

    points = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3))
    colors = np.concatenate(color_chunks, axis=0) if color_chunks else np.empty((0, 3))
    before_downsample = len(points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if args.voxel_size > 0:
        point_cloud = point_cloud.voxel_down_sample(args.voxel_size)
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    after_voxel = len(points)
    if len(points) > args.max_points:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(len(points), size=args.max_points, replace=False))
        points, colors = points[keep], colors[keep]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if len(points) == 0 and args.show_pointcloud:
        raise RuntimeError("no point cloud remains after confidence filtering")

    lower, upper, radius = robust_scene_bounds(points if len(points) else np.zeros((1, 3)))
    frustum_scale = args.frustum_scale or radius * 0.10
    geometries = []
    if args.show_pointcloud:
        geometries.append({"name": "VGGT point cloud", "geometry": point_cloud})
    visible_camera_items = []
    if args.show_cameras:
        for item in camera_items:
            if args.frustum_mode == "first" and item["sample_index"] != 0:
                continue
            if args.frustum_mode == "selected" and item["sample_index"] not in selected_indices:
                continue
            color = camera_color(item["camera_id"], item["sample_index"], sample_count)
            frustum = create_frustum(
                o3d, item["intrinsic"], item["camera_to_world"], item["width"], item["height"],
                frustum_scale, color,
            )
            name = f"{item['camera_id']} t={item['timestamp_sec']:.3f}s"
            geometries.append({"name": name, "geometry": frustum})
            visible_camera_items.append(item)
        if args.show_camera_tracks:
            for camera in cameras:
                items = [item for item in camera_items if item["camera_id"] == camera]
                centers = np.stack([item["camera_to_world"][:3, 3] for item in items])
                track = create_track(o3d, centers, CAMERA_COLORS[camera])
                if track is not None:
                    geometries.append({"name": f"{camera} temporal pose track", "geometry": track})
    if args.show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=radius * 0.18, origin=(0, 0, 0))
        geometries.append({"name": "VGGT world XYZ", "geometry": axis})
    if background_ba is not None:
        ba_initial = background_ba["points_initial"][background_ba["accepted"]]
        ba_refined = background_ba["points_refined"][background_ba["accepted"]]
        if args.show_ba_initial_points and len(ba_initial):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(ba_initial)
            cloud.colors = o3d.utility.Vector3dVector(
                np.repeat(np.asarray([[0.65, 0.65, 0.65]]), len(ba_initial), axis=0)
            )
            geometries.append({"name": "BA initial sparse background", "geometry": cloud})
        if args.show_ba_refined_points and len(ba_refined):
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(ba_refined)
            cloud.colors = o3d.utility.Vector3dVector(
                np.repeat(np.asarray([[1.0, 0.82, 0.08]]), len(ba_refined), axis=0)
            )
            geometries.append({"name": "BA refined sparse background", "geometry": cloud})
        for camera in cameras:
            camera_data = loaded[camera]
            initial_camera = background_ba["initial"]["cameras"][camera]
            refined_camera = background_ba["refined"]["cameras"][camera]
            for label, payload, color in (
                ("physical init", initial_camera, (0.72, 0.72, 0.72)),
                ("refined", refined_camera, CAMERA_COLORS[camera]),
            ):
                frustum = create_frustum(
                    o3d,
                    np.asarray(payload["intrinsic"], dtype=np.float64),
                    np.asarray(payload["camera_to_world"], dtype=np.float64),
                    camera_data["width"], camera_data["height"], frustum_scale * 1.15, color,
                )
                geometries.append(
                    {"name": f"BA {label} {camera}", "geometry": frustum}
                )

    if ply_path is not None:
        if not o3d.io.write_point_cloud(str(ply_path), point_cloud, write_ascii=False, compressed=False):
            raise RuntimeError(f"Open3D failed to export PLY: {ply_path}")
        reloaded = o3d.io.read_point_cloud(str(ply_path))
        if len(reloaded.points) != len(point_cloud.points):
            raise RuntimeError(f"exported PLY reload count mismatch: {ply_path}")

    pose_summary = {}
    transform_sanity = {}
    for camera in cameras:
        data = loaded[camera]
        items = [item for item in camera_items if item["camera_id"] == camera]
        centers = np.stack([item["camera_to_world"][:3, 3] for item in items])
        rotations = np.stack([item["camera_to_world"][:3, :3] for item in items])
        # Project only for the diagnostic angle calculation; displayed/stored geometry stays raw.
        proper_rotations = []
        for rotation in rotations:
            u, _, vt = np.linalg.svd(rotation)
            proper = u @ vt
            if np.linalg.det(proper) < 0:
                u[:, -1] *= -1
                proper = u @ vt
            proper_rotations.append(proper)
        proper_rotations = np.stack(proper_rotations)
        first = proper_rotations[0]
        angles = []
        for rotation in proper_rotations:
            cosine = np.clip((np.trace(rotation.T @ first) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(float(np.degrees(np.arccos(cosine))))
        pose_summary[camera] = {
            "center_distance_from_first": np.linalg.norm(centers - centers[0], axis=1).tolist(),
            "rotation_deg_from_first": angles,
            "timestamps_sec": [item["timestamp_sec"] for item in items],
        }
        transform_sanity[camera] = {
            "world_to_camera_inverse_max_abs_error": data["inverse_error"],
            "camera_center_formula_max_abs_error": data["center_error"],
            "rotation_orthogonality_max_abs_error": data["rotation_orthogonality_error"],
            "rotation_determinant_min": float(data["rotation_determinants"].min()),
            "rotation_determinant_max": float(data["rotation_determinants"].max()),
            "sparse_depth_to_pointmap_max_abs_error": data["pointmap_error"],
        }

    stats = {
        "schema_version": 1,
        "sequence": args.sequence,
        "sequence_path": str(sequence_dir),
        "initialization_only": True,
        "geometry_modified": False,
        "coordinate_convention": (
            "cam1-reference BA world; raw VGGT transformed by stored gauge"
            if background_ba is not None
            else "raw VGGT OpenCV world; frustums use stored camera_to_world inverse"
        ),
        "camera_legend": CAMERA_COLORS,
        "background_ba_legend": (
            {
                "vggt_sample_frustums": "camera color",
                "robust_physical_initial_frustums": [0.72, 0.72, 0.72],
                "refined_physical_frustums": "camera color",
                "initial_sparse_points": [0.65, 0.65, 0.65],
                "refined_sparse_points": [1.0, 0.82, 0.08],
            }
            if background_ba is not None else None
        ),
        "selected_cameras": cameras,
        "selected_frame_indices": selected_indices,
        "frustum_mode": args.frustum_mode,
        "confidence": {
            "score_semantics": "1+exp(logit) ranking score, not probability",
            "percentile_removed": percentile,
            "scope": args.confidence_scope,
            "global_threshold": global_threshold,
        },
        "color_mode": args.color_mode,
        "point_counts": {
            "after_confidence_before_downsampling": before_downsample,
            "after_voxel_downsampling": after_voxel,
            "final": len(point_cloud.points),
        },
        "voxel_size": args.voxel_size,
        "max_points": args.max_points,
        "robust_bounds_p02": lower.tolist(),
        "robust_bounds_p98": upper.tolist(),
        "frustum_scale": frustum_scale,
        "frame_stats": frame_stats,
        "pose_summary": pose_summary,
        "transform_sanity": transform_sanity,
        "temporal_qa_classification": sequence_metadata["sequence_status"].get(
            "temporal_qa_classification", "UNKNOWN"
        ),
        "outputs": {
            "screenshot": str(screenshot_path) if screenshot_path else None,
            "ply": str(ply_path) if ply_path else None,
        },
        "background_ba_comparison": (
            {
                "sequence_path": str(background_ba["sequence_dir"]),
                "acceptance": background_ba["metrics"]["acceptance"],
                "accepted_sparse_points": int(background_ba["accepted"].sum()),
                "source_vggt_world_to_ba_world": source_to_display.tolist(),
            }
            if background_ba is not None else None
        ),
    }
    if stats_path is not None:
        stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    if screenshot_path is not None:
        render_screenshot(o3d, screenshot_path, geometries, points, visible_camera_items or camera_items, args)

    print(f"Sequence: {args.sequence}")
    if background_ba is not None:
        print("Coordinates: cam1-reference BA world; raw VGGT geometry transformed by stored gauge.")
    else:
        print("Coordinates: VGGT OpenCV world; extrinsic is world->camera [R|t]; frustums use camera_to_world.")
    print("Scale: arbitrary sequence-relative, non-metric. Confidence: percentile ranking, not probability.")
    print("Frustum colors: cam1=red, cam2=green, cam3=blue; later samples are brighter.")
    if background_ba is not None:
        print("BA overlay: robust physical init=gray; refined pose=camera color; refined sparse points=yellow.")
    if args.color_mode == "time":
        print("Point time colors: sample 0..7 = red, orange, yellow, green, cyan, blue, violet, magenta.")
    if args.quiet:
        print(
            f"Points: {stats['point_counts']['final']} | removed percentile: {percentile:g} | "
            f"screenshot: {screenshot_path or 'none'} | PLY: {ply_path or 'none'}"
        )
    else:
        print(json.dumps(stats, indent=2))

    if not args.headless:
        title = (
            f"VGGT vs fixed-camera Background BA — {args.sequence}"
            if background_ba is not None else f"VGGT initialization — {args.sequence}"
        )
        interactive_view(o3d, geometries, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
