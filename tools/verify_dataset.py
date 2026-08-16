#!/usr/bin/env python3
"""완성된 데이터셋 검수.

  1) manifest 의 모든 경로가 실제로 존재하는지 전수 검사
  2) 세트별 프레임 수 / split 누수 / 이미지 크기 확인
  3) 랜덤 트리플렛을 가로로 붙인 대조 시트를 만들어 육안 확인용으로 저장

  python3 verify_dataset.py --sheets 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# 실제 데이터는 저장소 밖에 둘 수 있다. 환경변수나 CLI로 경로를 주입한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT / "data")
).expanduser()


def contact_sheet(root: Path, row: dict, dst: Path, width: int = 640) -> None:
    """cam1|cam2|cam3 를 가로로 붙이고 sample_id 를 얹은 확인용 이미지."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for cam in ("cam1", "cam2", "cam3"):
        cmd += ["-i", str(root / row[cam])]
    label = f"{row['sample_id']}  t={float(row['time_sec']):.3f}s  {row['split']}/{row['qc']}"
    parts = [f"[{i}:v]scale={width}:-2,setsar=1,"
             f"drawtext=fontfile={FONT}:text='CAM {i + 1}':x=10:y=10:fontsize=26:"
             f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6[v{i}]"
             for i in range(3)]
    filt = (";".join(parts) + ";[v0][v1][v2]hstack=inputs=3,"
            f"drawtext=fontfile={FONT}:text='{label}':x=10:y=h-40:fontsize=24:"
            f"fontcolor=yellow:box=1:boxcolor=black@0.7:boxborderw=8[out]")
    cmd += ["-filter_complex", filt, "-map", "[out]", "-frames:v", "1", str(dst)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exercise3D 데이터셋 검수")
    ap.add_argument(
        "--dataset-root", "--out", dest="out", default=str(DEFAULT_DATASET_ROOT)
    )
    ap.add_argument("--sheets", type=int, default=8, help="만들 대조 시트 장수")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.out)
    with (root / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"manifest: {len(rows):,} 트리플렛")

    # --- 1) 경로 전수 검사 ---------------------------------------------- #
    missing = [r["sample_id"] for r in rows
               for cam in ("cam1", "cam2", "cam3") if not (root / r[cam]).is_file()]
    print(f"[경로] 누락 {len(missing)}건" + (f"  예: {missing[:5]}" if missing else "  ✓"))

    # --- 2) 구조 검사 --------------------------------------------------- #
    per_set: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_set[r["set_id"]].append(r)

    bad_idx, split_leak = [], []
    for sid, rs in per_set.items():
        idx = sorted(int(r["frame_idx"]) for r in rs)
        if idx != list(range(len(idx))):
            bad_idx.append(sid)
        if len({r["split"] for r in rs}) != 1:
            split_leak.append(sid)
    print(f"[프레임 인덱스] 0..N-1 연속하지 않은 세트 {len(bad_idx)}개"
          + (f" {bad_idx}" if bad_idx else "  ✓"))
    print(f"[split 누수] 한 세트가 여러 split 에 걸친 경우 {len(split_leak)}개"
          + (f" {split_leak}" if split_leak else "  ✓"))

    by_split = Counter(r["split"] for r in rows)
    sets_by_split = Counter(rs[0]["split"] for rs in per_set.values())
    total = sum(by_split.values())
    print(f"[split] 세트 {dict(sets_by_split)}")
    for k in ("train", "val", "test", "review"):
        if by_split.get(k):
            print(f"         {k:6s} {by_split[k]:7,} 트리플렛 ({by_split[k] / total * 100:.1f}%)")
    print(f"[qc]    {dict(Counter(r['qc'] for r in rows))}")

    # 종목이 split 마다 몇 개나 들어갔는지
    ex_by_split: dict[str, set] = defaultdict(set)
    for rs in per_set.values():
        ex_by_split[rs[0]["split"]].add(rs[0]["exercise"])
    for k in ("train", "val", "test"):
        if k in ex_by_split:
            print(f"[종목] {k:6s} {len(ex_by_split[k])}종목 {sorted(ex_by_split[k])}")

    # --- 3) 이미지 크기 표본 검사 --------------------------------------- #
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(30, len(rows)))
    sizes = Counter()
    for r in sample:
        for cam in ("cam1", "cam2", "cam3"):
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
                 str(root / r[cam])], capture_output=True, text=True, check=True)
            sizes[out.stdout.strip()] += 1
    print(f"[이미지 크기] 표본 {sum(sizes.values())}장 → {dict(sizes)}")

    # --- 4) 대조 시트 ---------------------------------------------------- #
    sheets = root / "check_sheets"
    sheets.mkdir(exist_ok=True)
    for p in sheets.glob("*.jpg"):
        p.unlink()
    picks = []
    for sid in rng.sample(sorted(per_set), min(args.sheets, len(per_set))):
        picks.append(rng.choice(per_set[sid]))
    for r in picks:
        contact_sheet(root, r, sheets / f"{r['sample_id']}.jpg")
    print(f"[대조 시트] {len(picks)}장 → {sheets}/")

    ok = not missing and not bad_idx and not split_leak
    print("\n" + ("검수 통과" if ok else "검수 실패 — 위 항목 확인 필요"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
