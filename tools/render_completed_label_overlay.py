#!/usr/bin/env python3
"""Render a privacy-local three-view overlay from completed Exercise3D labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--pipeline-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--mesh-overlay", action="store_true")
    return parser.parse_args()


def load_camera(root: Path, sequence: str, camera: str) -> dict[str, object]:
    pose_dir = root / "outputs/sapiens2_target_only_full" / sequence / camera
    prior_dir = root / "outputs/sam_body_prior_full" / sequence / camera
    target_dir = root / "outputs/target_selection_full" / sequence / camera
    mask_dir = (
        root / "outputs/sam_body4d_full" / sequence / camera
        / "mode_b_private_output/masks"
    )
    render_dir = (
        root / "outputs/sam_body4d_full" / sequence / camera
        / "mode_b_private_output/rendered_frames_individual/1"
    )
    with (pose_dir / "metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    pose_npz = np.load(pose_dir / "poses_2d.npz")
    prior_npz = np.load(prior_dir / "sam_body_prior.npz")
    target_npz = np.load(target_dir / "target_selection.npz")
    return {
        "pose": pose_npz,
        "prior": prior_npz,
        "target": target_npz,
        "names": metadata["keypoint_names"],
        "mask_dir": mask_dir,
        "render_dir": render_dir,
    }


def draw_panel(
    image: np.ndarray,
    data: dict[str, object],
    frame_index: int,
    camera: str,
    confidence_threshold: float,
    mesh_overlay: bool,
) -> np.ndarray:
    pose = data["pose"]
    prior = data["prior"]
    target = data["target"]
    names = data["names"]
    mask_path = data["mask_dir"] / f"{frame_index:08d}.png"

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    accepted = bool(prior["accepted_prior"][frame_index])
    if mesh_overlay and accepted:
        render_path = data["render_dir"] / f"{frame_index:08d}_1.jpg"
        mesh = cv2.imread(str(render_path))
        if mesh is None or mesh.shape != image.shape:
            raise RuntimeError(f"missing or invalid mesh render: {render_path}")
        foreground = np.min(mesh, axis=2) < 242
        foreground = cv2.morphologyEx(
            foreground.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        ).astype(bool)
        image[foreground] = cv2.addWeighted(
            image[foreground], 0.32, mesh[foreground], 0.68, 0
        )
    elif mask is not None and accepted:
        selected = mask > 0
        tint = np.zeros_like(image)
        tint[:, :, 1] = 220
        image[selected] = cv2.addWeighted(image[selected], 0.58, tint[selected], 0.42, 0)

    bbox = prior["target_bbox_xyxy"][frame_index]
    if np.isfinite(bbox).all():
        x1, y1, x2, y2 = np.rint(bbox).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 3)

    if not mesh_overlay:
        xy = pose["keypoints_xy"][frame_index]
        score = pose["confidence"][frame_index]
        valid = pose["valid_mask"][frame_index] & np.isfinite(xy).all(axis=1)
        name_to_id = {name: index for index, name in enumerate(names)}
        for left, right in EDGES:
            a, b = name_to_id.get(left), name_to_id.get(right)
            if a is None or b is None or not valid[a] or not valid[b]:
                continue
            if score[a] < confidence_threshold or score[b] < confidence_threshold:
                continue
            pa, pb = tuple(np.rint(xy[a]).astype(int)), tuple(np.rint(xy[b]).astype(int))
            cv2.line(image, pa, pb, (255, 80, 40), 4, cv2.LINE_AA)
        for name in {name for edge in EDGES for name in edge}:
            index = name_to_id.get(name)
            if index is not None and valid[index] and score[index] >= confidence_threshold:
                point = tuple(np.rint(xy[index]).astype(int))
                cv2.circle(image, point, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(image, point, 7, (255, 80, 40), 2, cv2.LINE_AA)

    status = "ACCEPTED" if accepted else str(target["target_status"][frame_index])
    cv2.rectangle(image, (0, 0), (image.shape[1], 54), (12, 12, 12), -1)
    cv2.putText(image, f"{camera}  {status}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main() -> None:
    args = arguments()
    root = args.pipeline_root.resolve()
    cameras = ("cam1", "cam2", "cam3")
    frame_dirs = {camera: args.frames_root / args.sequence / camera for camera in cameras}
    frame_paths = {camera: sorted(frame_dirs[camera].glob("*.jpg")) for camera in cameras}
    counts = {camera: len(paths) for camera, paths in frame_paths.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise RuntimeError(f"camera frame counts do not match: {counts}")
    data = {camera: load_camera(root, args.sequence, camera) for camera in cameras}
    frame_count = next(iter(counts.values()))
    for camera in cameras:
        if data[camera]["pose"]["keypoints_xy"].shape[0] != frame_count:
            raise RuntimeError(f"pose/frame mismatch for {camera}")

    pts = data["cam1"]["pose"]["timestamp_pts_seconds"]
    delta = np.diff(pts)
    fps = float(1.0 / np.median(delta[delta > 0]))
    sample = cv2.imread(str(frame_paths["cam1"][0]))
    if sample is None:
        raise RuntimeError("cannot decode first frame")
    panel_height = round(sample.shape[0] * args.panel_width / sample.shape[1])
    output_size = (args.panel_width * 3, panel_height + 48)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")
    try:
        for index in range(frame_count):
            panels = []
            for camera in cameras:
                image = cv2.imread(str(frame_paths[camera][index]))
                if image is None:
                    raise RuntimeError(f"cannot decode {frame_paths[camera][index]}")
                image = draw_panel(
                    image, data[camera], index, camera, args.confidence, args.mesh_overlay
                )
                panels.append(cv2.resize(image, (args.panel_width, panel_height),
                                         interpolation=cv2.INTER_AREA))
            canvas = np.vstack((np.hstack(panels), np.zeros((48, output_size[0], 3), np.uint8)))
            legend = ("orange=MHR mesh  cyan=target box" if args.mesh_overlay else
                      "green=SAM mask  cyan=target box  blue=pose")
            cv2.putText(canvas, f"{args.sequence} | frame {index + 1:04d}/{frame_count} | {legend}",
                        (18, panel_height + 33), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (235, 235, 235), 2, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
        for camera_data in data.values():
            camera_data["pose"].close()
            camera_data["prior"].close()
            camera_data["target"].close()
    print(f"wrote {args.output} ({frame_count} frames, {fps:.6f} fps, {output_size[0]}x{output_size[1]})")


if __name__ == "__main__":
    main()
