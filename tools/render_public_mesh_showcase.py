#!/usr/bin/env python3
"""Build a public-safe three-view video from mesh-only render frames.

This tool deliberately has no source-RGB input. It accepts only the white-background
MHR render directory produced by the private SAM-Body4D stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("cam1", "cam2", "cam3")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--mesh-render-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--panel-width", type=int, default=240)
    parser.add_argument("--panel-height", type=int, default=426)
    return parser.parse_args()


def render_directory(root: Path, sequence: str, camera: str) -> Path:
    return (
        root / sequence / camera / "mode_b_private_output"
        / "rendered_frames_individual" / "1"
    )


def main() -> None:
    args = arguments()
    if min(args.source_fps, args.frame_step, args.panel_width, args.panel_height) <= 0:
        raise ValueError("fps, frame-step, panel-width, and panel-height must be positive")

    paths = {
        camera: sorted(render_directory(args.mesh_render_root, args.sequence, camera).glob("*.jpg"))
        for camera in CAMERAS
    }
    counts = {camera: len(items) for camera, items in paths.items()}
    if not counts["cam1"] or len(set(counts.values())) != 1:
        raise RuntimeError(f"mesh-render camera counts do not match: {counts}")

    sample = cv2.imread(str(paths["cam1"][0]))
    if sample is None:
        raise RuntimeError("cannot decode the first mesh render")
    panel_height = args.panel_height
    footer_height = 44
    output_size = (args.panel_width * len(CAMERAS), panel_height + footer_height)
    output_fps = args.source_fps / args.frame_step
    selected_indices = range(0, counts["cam1"], args.frame_step)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, output_size
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer")

    written = 0
    try:
        for index in selected_indices:
            panels = []
            for camera in CAMERAS:
                image = cv2.imread(str(paths[camera][index]))
                if image is None:
                    raise RuntimeError(f"cannot decode {paths[camera][index]}")
                scale = min(args.panel_width / image.shape[1], panel_height / image.shape[0])
                resized = cv2.resize(
                    image,
                    (round(image.shape[1] * scale), round(image.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
                panel = np.full((panel_height, args.panel_width, 3), 255, np.uint8)
                x = (args.panel_width - resized.shape[1]) // 2
                y = (panel_height - resized.shape[0]) // 2
                panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                cv2.rectangle(panel, (0, 0), (args.panel_width, 34), (250, 250, 250), -1)
                cv2.putText(
                    panel, camera, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (45, 45, 45), 1, cv2.LINE_AA,
                )
                panels.append(panel)
            canvas = np.full((output_size[1], output_size[0], 3), 255, np.uint8)
            canvas[:panel_height] = np.hstack(panels)
            cv2.putText(
                canvas,
                f"{args.sequence} | MHR mesh-only preview | frame {index + 1}/{counts['cam1']}",
                (14, panel_height + 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (45, 45, 45), 1, cv2.LINE_AA,
            )
            writer.write(canvas)
            written += 1
    finally:
        writer.release()

    print(
        f"wrote {args.output} ({written} frames, {output_fps:.3f} fps, "
        f"{output_size[0]}x{output_size[1]}; source RGB was not read)"
    )


if __name__ == "__main__":
    main()
