#!/usr/bin/env python3
"""출력 프레임이 원본의 몇 번째 프레임에서 왔는지 연속으로 추적한다.

뷰 01·03 은 30fps, 뷰 02 는 60fps 인데 출력은 전부 30fps 다.
그래서 60fps 쪽에서 프레임이 버려졌는지(drop), 30fps 쪽에서 같은 프레임이
두 번 쓰였는지(dup) 를 추측이 아니라 픽셀 매칭으로 확인한다.

  원본 프레임 인덱스가 +2 씩 증가 → 60fps 에서 한 장 걸러 버림
  +1 씩 증가                    → 1:1 통과
  같은 값 반복                   → 중복 사용(dup)
  +3 이상 점프                   → 건너뜀(drop)

  python3 check_frame_mapping.py --set deadlift_0000 --start 100 --count 16
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

W, H = 96, 54
PTS_RE = re.compile(r"pts_time:([0-9.]+)")

# 실제 데이터는 저장소 밖에 둘 수 있다. 환경변수나 CLI로 경로를 주입한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT / "data")
).expanduser()


def decode(path: Path, ss: float | None, n: int, copyts: bool = False):
    cmd = ["ffmpeg", "-v", "info", "-nostats"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.6f}"]
    if copyts:
        cmd += ["-copyts"]
    cmd += ["-i", str(path), "-frames:v", str(n),
            "-vf", f"scale={W}:{H},format=gray,showinfo", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True, check=True)
    k = len(p.stdout) // (W * H)
    fr = np.frombuffer(p.stdout[: k * W * H], dtype=np.uint8).reshape(k, H * W).astype(np.float32)
    pts = np.array([float(m) for m in PTS_RE.findall(p.stderr.decode("utf-8", "replace"))])
    m = min(len(fr), len(pts))
    return fr[:m], pts[:m]


def main() -> None:
    ap = argparse.ArgumentParser(description="출력→원본 프레임 대응 추적")
    ap.add_argument(
        "--dataset-root", "--out", dest="out", default=str(DEFAULT_DATASET_ROOT)
    )
    ap.add_argument(
        "--source-root", "--root", dest="root",
        default=str(DEFAULT_DATASET_ROOT / "origin"),
    )
    ap.add_argument("--set", dest="set_id", required=True)
    ap.add_argument("--start", type=int, default=100, help="검사 시작 출력 프레임")
    ap.add_argument("--count", type=int, default=16, help="연속으로 볼 프레임 수")
    args = ap.parse_args()

    root = Path(args.root)
    setdir = next(Path(args.out).glob(f"synced_video/*/{args.set_id}"))
    sync = json.loads((setdir / "sync.json").read_text())
    fps_out = sync["fps"]

    print(f"{args.set_id}: 출력 프레임 {args.start}~{args.start + args.count - 1} "
          f"({fps_out}fps 출력)\n")
    for c in sync["clips"]:
        cam, src = c["cam"], root / c["source"]
        src_fps = c["source_fps"]

        # 출력에서 연속 프레임
        outf, _ = decode(setdir / f"cam{cam}.mp4", args.start / fps_out, args.count)
        # 원본에서 그 구간을 넉넉히 (양쪽 여유 0.2초)
        lo = max(0.0, c["cut_start_sec"] + args.start / fps_out - 0.2)
        span = args.count / fps_out + 0.4
        srcf, spts = decode(src, lo, int(span * src_fps) + 4, copyts=True)

        # 원본 "몇 번째 프레임"은 pts × fps 로 계산하면 안 된다. 실제 레이트가
        # 59.973fps 같은 값이라 정수에 안 떨어져 반올림이 흔들린다.
        # 디코딩한 원본 창 안에서의 순번(j)을 그대로 쓰는 게 정확하다.
        idx, times = [], []
        for f in outf:
            j = int(np.argmin(np.abs(srcf - f).mean(axis=1)))
            idx.append(j)                                 # 원본 창 안에서의 순번
            times.append(spts[j])

        steps = np.diff(idx)
        dup = int((steps == 0).sum())
        skips = Counter(int(s) for s in steps)
        interval = np.diff(times)
        print(f"  cam{cam}  원본 {src_fps:g}fps  {src.name}")
        print(f"    원본 프레임 순번: {idx}")
        print(f"    증가폭: {dict(sorted(skips.items()))}   중복(dup) {dup}회")
        print(f"    실제 시간 간격 평균 {interval.mean() * 1000:.2f}ms "
              f"(30fps 목표 33.33ms)\n")


if __name__ == "__main__":
    main()
