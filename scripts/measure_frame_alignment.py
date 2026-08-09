#!/usr/bin/env python3
"""출력 프레임이 "정말로" 같은 순간인지 프레임 단위로 실측한다.

지금까지의 `residual` 은 오디오 컷 지점의 정확도다. 그런데 최종 산출물은 이미지라서,
컷이 아무리 정확해도 **원본 프레임 격자**가 남긴 양자화 오차가 더해진다.

  - 원본은 CFR 이라 프레임이 0, 1/fps, 2/fps ... 에만 존재한다
  - 컷 시작 시각은 임의의 소수점이므로, 출력 프레임이 원하는 시각과 정확히
    일치하는 원본 프레임은 대개 없다. ffmpeg 는 가장 가까운 것을 고른다
  - 그 오차는 30fps 원본에서 최대 ±1/60초(16.7ms), 60fps 원본에서 ±1/120초(8.3ms)

이 스크립트는 추론하지 않고 실제로 잰다.

  1) 출력 mp4 의 프레임 k 를 디코딩
  2) 원본에서 그 근처 프레임들을 전부 디코딩
  3) 픽셀 차이가 최소인 원본 프레임을 찾는다 = 출력 프레임의 진짜 출처
  4) 그 원본 프레임의 실제 촬영 시각을 사건 좌표계로 환산해서 카메라끼리 비교

  python3 measure_frame_alignment.py --sets deadlift_0000 pushup_0003 --frames 0 100 300
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

W, H = 96, 54          # 매칭용 축소 해상도 (내용 식별에는 충분)
SEARCH_SEC = 0.30      # 예상 위치 ±이 범위의 원본 프레임을 후보로

# 실제 데이터는 저장소 밖에 둘 수 있다. 환경변수나 CLI로 경로를 주입한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT / "data")
).expanduser()


PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def decode_gray(path: Path, ss: float | None, n_frames: int,
                copyts: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """축소 흑백 프레임과 **그 프레임들의 PTS** 를 한 번의 디코딩에서 함께 받는다.

    픽셀과 시각을 따로 두 번 호출하면 안 된다. ffprobe 의 -read_intervals 는 앞쪽
    키프레임부터 돌려주고 ffmpeg -ss 는 정확 시크라, 두 배열의 시작점이 어긋난다.
    showinfo 를 필터체인에 끼워 같은 프레임의 PTS 를 stderr 로 받는다.
    copyts=True 면 PTS 가 원본 타임라인 기준으로 유지된다.
    """
    cmd = ["ffmpeg", "-v", "info", "-nostats"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.6f}"]
    if copyts:
        cmd += ["-copyts"]
    cmd += ["-i", str(path), "-frames:v", str(n_frames),
            "-vf", f"scale={W}:{H},format=gray,showinfo",
            "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    n = len(proc.stdout) // (W * H)
    frames = np.frombuffer(proc.stdout[: n * W * H], dtype=np.uint8)
    frames = frames.reshape(n, H * W).astype(np.float32)
    pts = np.array([float(m) for m in PTS_RE.findall(proc.stderr.decode("utf-8", "replace"))])
    m = min(len(frames), len(pts))
    return frames[:m], pts[:m]


def measure(setdir: Path, root: Path, frame_idxs: list[int] | None,
            fracs: list[float] | None) -> list[dict]:
    sync = json.loads((setdir / "sync.json").read_text())
    out_fps = sync["fps"]
    n = sync["n_frames"]
    # 드리프트를 보려면 클립 전체에 고르게 퍼진 지점을 봐야 한다.
    idxs = list(frame_idxs) if frame_idxs else []
    if fracs:
        idxs += [min(n - 1, max(0, int(round(f * (n - 1))))) for f in fracs]
    rows = []
    for k in sorted(set(idxs)):
        if k >= n:
            continue
        rec = {"frame_idx": k, "cams": {}}
        for c in sync["clips"]:
            cam, src = c["cam"], root / c["source"]
            want = c["cut_start_sec"] + k / out_fps       # 이 출력 프레임이 노리는 원본 시각

            # 출력 프레임 k 한 장
            got, _ = decode_gray(setdir / f"cam{cam}.mp4", ss=k / out_fps, n_frames=1)
            if len(got) == 0:
                continue
            got = got[0]

            # 원본 후보 프레임들 + 원본 타임라인 기준 실제 시각
            lo = max(0.0, want - SEARCH_SEC)
            n_cand = int(2 * SEARCH_SEC * c["source_fps"]) + 2
            cand, times = decode_gray(src, ss=lo, n_frames=n_cand, copyts=True)
            if len(cand) == 0:
                continue

            j = int(np.argmin(np.abs(cand - got).mean(axis=1)))
            err = float(np.abs(cand[j] - got).mean())
            true_t = float(times[j])
            rec["cams"][cam] = {
                "wanted_src_sec": round(want, 6),
                "actual_src_sec": round(true_t, 6),
                "quantize_ms": round((true_t - want) * 1000, 2),
                # 사건 좌표계로 환산 = 카메라끼리 직접 비교 가능한 절대 시각
                "event_sec": round(true_t - c["offset_sec"], 6),
                "match_err": round(err, 2),
            }
        if len(rec["cams"]) == 3:
            ev = [v["event_sec"] for v in rec["cams"].values()]
            rec["spread_ms"] = round((max(ev) - min(ev)) * 1000, 2)
            rec["spread_frames"] = round((max(ev) - min(ev)) * out_fps, 3)
            rec["max_match_err"] = max(v["match_err"] for v in rec["cams"].values())
        rows.append(rec)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="프레임 단위 정합 실측")
    ap.add_argument(
        "--dataset-root", "--out", dest="out", default=str(DEFAULT_DATASET_ROOT)
    )
    ap.add_argument(
        "--source-root", "--root", dest="root",
        default=str(DEFAULT_DATASET_ROOT / "origin"),
    )
    ap.add_argument("--sets", nargs="*", help="검사할 set_id (기본: 전부)")
    ap.add_argument("--frames", nargs="*", type=int, help="검사할 프레임 인덱스(절대)")
    ap.add_argument("--at", nargs="*", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="클립 길이 대비 상대 위치 (드리프트 확인용)")
    ap.add_argument("--json", help="결과를 JSON 으로 저장할 경로")
    args = ap.parse_args()

    out, root = Path(args.out), Path(args.root)
    setdirs = sorted(p.parent for p in out.glob("synced_video/*/*/sync.json"))
    if args.sets:
        setdirs = [d for d in setdirs if d.name in args.sets]

    print(f"{'set':18s} {'frame':>5s} {'cam1 양자화':>11s} {'cam2':>8s} {'cam3':>8s} "
          f"{'뷰간 편차':>10s} {'프레임':>7s} {'매칭오차':>8s}")
    all_rows, spreads = {}, []
    for d in setdirs:
        rows = measure(d, root, args.frames, args.at)
        all_rows[d.name] = rows
        for r in rows:
            if "spread_ms" not in r:
                continue
            q = {c: r["cams"][c]["quantize_ms"] for c in (1, 2, 3)}
            spreads.append(r["spread_ms"])
            print(f"{d.name:18s} {r['frame_idx']:5d} {q[1]:9.2f}ms {q[2]:6.2f}ms "
                  f"{q[3]:6.2f}ms {r['spread_ms']:8.2f}ms {r['spread_frames']:7.3f} "
                  f"{r['max_match_err']:8.2f}")

    if spreads:
        a = np.array(spreads)
        print(f"\n표본 {len(a)}개 — 뷰 간 실제 촬영시각 편차")
        print(f"  평균 {a.mean():.2f}ms / 중앙값 {np.median(a):.2f}ms / 최대 {a.max():.2f}ms")
        print(f"  30fps 한 프레임(33.33ms) 대비 최대 {a.max() / 33.333:.2f} 프레임")
        for th, lbl in ((33.333, "1 프레임"), (16.667, "0.5 프레임")):
            print(f"  {lbl} 이내: {(a <= th).sum()}/{len(a)} ({(a <= th).mean() * 100:.1f}%)")
        # 세트 안에서 처음 지점과 마지막 지점의 편차 변화 = 클럭 드리프트 신호
        drift = []
        for sid, rows in all_rows.items():
            ok = [r for r in rows if "spread_ms" in r]
            if len(ok) >= 2:
                drift.append((sid, ok[0]["spread_ms"], ok[-1]["spread_ms"],
                              ok[-1]["frame_idx"] / 30.0))
        if drift:
            worst = sorted(drift, key=lambda x: abs(x[2] - x[1]))[-5:]
            print("\n처음 → 마지막 지점 편차 변화 (클럭 드리프트, 변화 큰 순 5개)")
            for sid, a0, a1, tsec in reversed(worst):
                rate = (a1 - a0) / tsec if tsec > 0 else 0.0
                print(f"  {sid:18s} {a0:6.2f}ms → {a1:6.2f}ms  ({tsec:5.1f}s 동안, "
                      f"{rate:+.3f} ms/s)")

    if args.json:
        Path(args.json).write_text(json.dumps(all_rows, indent=2, ensure_ascii=False),
                                   encoding="utf-8")


if __name__ == "__main__":
    main()
