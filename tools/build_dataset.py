#!/usr/bin/env python3
"""Exercise3D 3시점 영상 → 싱크 → 프레임 → 트리플렛 데이터셋 배치 빌더.

세트 = (종목, 테이크) 하나에 뷰 01/02/03 세 영상이 대응한다.
싱크 자체는 sync_videos.py 의 검증된 함수를 그대로 쓴다.

  python3 build_dataset.py --stage all
  python3 build_dataset.py --stage sync --only deadlift_0000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import sync_videos as sv

VIEW_RE = re.compile(r"^(?P<exercise>.+)_(?P<view>\d{2})_(?P<take>\d{4})$", re.I)
OUT_FPS = 30
SYNC_DIR = "synced_video"      # 산출물 루트 안의 싱크 영상 폴더
FRAME_DIR = "final_frame"     # 산출물 루트 안의 프레임 폴더

# 실제 데이터는 저장소 밖에 둘 수 있다. 환경변수나 CLI로 경로를 주입한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_ROOT = Path(
    os.environ.get("EXERCISE3D_DATASET_ROOT", PROJECT_ROOT / "data")
).expanduser()

RESIDUAL_PASS_MS = 15.0     # 잔여 오프셋 (30fps 반 프레임 = 16.7ms)
RESIDUAL_WARN_MS = 40.0
SPREAD_PASS_MS = 20.0       # 정렬 후 뷰 간 박수 봉우리 편차 (온셋 해상도 5ms × 4칸)
MOTION_TOL_MS = 67.0        # 영상 모션 교차검증 허용 오차 (30fps 2프레임)
MOTION_W, MOTION_H = 64, 36
MOTION_REL = 0.85           # 모션 상관 최상위 봉우리의 이 비율 이상을 후보로 인정


@dataclass
class EncOpts:
    """sync_videos.cut / make_preview 가 기대하는 인터페이스."""
    fps: float = OUT_FPS
    crf: int = 18
    preset: str = "veryfast"
    vcodec: str = "libx264"
    preview_width: int = 1920


# --------------------------------------------------------------------------- #
# 세트 탐색
# --------------------------------------------------------------------------- #
def discover_sets(root: Path) -> list[dict]:
    """파일명 {종목}_{뷰}_{테이크} 를 파싱해 세트로 묶는다."""
    groups: dict[tuple[str, str], dict[int, Path]] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in sv.VIDEO_EXTS:
            continue
        m = VIEW_RE.match(p.stem)
        if not m:
            print(f"  [skip] 이름 규칙에 맞지 않음: {p}")
            continue
        key = (m["exercise"].lower(), m["take"])
        groups.setdefault(key, {})[int(m["view"])] = p

    sets, bad = [], []
    for (exercise, take), views in sorted(groups.items()):
        if sorted(views) != [1, 2, 3]:
            bad.append(f"{exercise}_{take}: 뷰 {sorted(views)} (3개가 아님)")
            continue
        sets.append({"set_id": f"{exercise}_{take}", "exercise": exercise,
                     "take": take, "views": {k: views[k] for k in (1, 2, 3)}})
    for b in bad:
        print(f"  [warn] {b}")
    return sets


# --------------------------------------------------------------------------- #
# 1단계: 싱크
# --------------------------------------------------------------------------- #
def analyze_set(views: dict[int, Path], max_offset: float,
                anchor: str, margin: float) -> tuple[list[dict], dict]:
    """뷰 3개의 오디오를 분석해 오프셋과 컷 구간을 구한다. cam1 이 기준."""
    hop = int(round(sv.SR * sv.COARSE_HOP_MS / 1000.0))
    clips = []
    for view in (1, 2, 3):
        path = views[view]
        meta = sv.probe(path)
        audio = sv.load_audio(path)
        onset = sv.spectral_flux(audio, sv.COARSE_FRAME, hop)
        clips.append({
            "cam": view, "path": path, "audio": audio, "onset": onset,
            "clap_at": sv.find_clap(onset, sv.SR / hop),
            "duration": meta["duration"], "fps": meta["fps"],
            "width": meta["width"], "height": meta["height"],
        })

    ref = clips[0]
    for i, c in enumerate(clips):
        if i == 0:
            c["offset"], c["confidence"] = 0.0, float("inf")
        else:
            c["offset"], c["confidence"] = sv.estimate_offset(ref, c, max_offset)
    return clips, sv.plan_window(clips, anchor, margin)


def measure_residual(clips: list[dict], lo: float, hi: float) -> list[float]:
    """실제 컷 타임스탬프로 원본에서 오디오만 다시 뽑아 잔여 오프셋을 잰다.

    ffmpeg 의 시크 정확도까지 포함해서 검증되므로, 이 값이 최종 지표다.
    구간은 박수를 포함하는 최대 겹침 구간(lo~hi)을 쓴다.
    """
    hop = int(round(sv.SR * sv.COARSE_HOP_MS / 1000.0))
    cut = []
    for c in clips:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{lo + c['offset']:.6f}",
             "-i", str(c["path"]), "-t", f"{hi - lo:.6f}",
             "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sv.SR), "-f", "f32le", "-"],
            capture_output=True, check=True)
        a = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
        peak = float(np.max(np.abs(a)))
        a = a / peak if peak > 0 else a
        onset = sv.spectral_flux(a, sv.COARSE_FRAME, hop)
        cut.append({"audio": a, "onset": onset,
                    "clap_at": sv.find_clap(onset, sv.SR / hop)})

    residuals = [0.0]
    for c in cut[1:]:
        off, _ = sv.estimate_offset(cut[0], c, 3.0)
        residuals.append(off * 1000.0)
    return residuals


def motion_signal(path: Path) -> np.ndarray:
    """저해상도 흑백 프레임차 에너지. 오디오와 완전히 독립적인 움직임 신호."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-vf", f"fps={OUT_FPS},scale={MOTION_W}:{MOTION_H},format=gray",
         "-f", "rawvideo", "-"], capture_output=True, check=True)
    v = np.frombuffer(proc.stdout, dtype=np.uint8).astype(np.float32)
    v = v[: (len(v) // (MOTION_W * MOTION_H)) * MOTION_W * MOTION_H]
    v = v.reshape(-1, MOTION_H * MOTION_W)
    d = np.abs(np.diff(v, axis=0)).mean(axis=1)
    return sv.normalize(d)


def motion_check(clips: list[dict], max_offset: float) -> list[tuple[float, float]]:
    """오디오 오프셋을 영상 움직임으로 교차검증한다.

    팔굽혀펴기처럼 반복 동작이 있으면 모션 상관이 한 렙 밀려(aliasing) 잡히므로
    최댓값 하나만 비교하면 안 된다. 대신 "오디오 값이 모션 상관의 강한 봉우리
    중 하나와 일치하는가"를 본다. 일치하면 두 신호가 서로를 확증한 것이다.

    반환: 뷰마다 (가장 가까운 강한 봉우리까지의 거리 ms, 오디오 위치의 상대점수).
    두 번째 값은 WARN 이 떴을 때 "완전히 어긋난 것"인지 "봉우리는 있는데 기준에
    조금 못 미친 것"인지 구분하려고 같이 기록한다.
    """
    sig = [motion_signal(c["path"]) for c in clips]
    out = [(0.0, 1.0)]
    for i, c in enumerate(clips[1:], start=1):
        a, b = sig[i], sig[0]
        # 겹치는 구간이 짧은 시간차는 상관값이 몇 샘플만으로 튀어 오르므로 제외한다.
        # (정규화 없는 상호상관의 전형적인 가장자리 인공물)
        limit = int(min(round(max_offset * OUT_FPS), 0.5 * min(len(a), len(b))))
        if limit < 2:
            out.append((float("inf"), 0.0))
            continue
        corr = sv.xcorr(a, b)
        lags = np.arange(-limit, limit + 1)
        overlap = np.minimum(len(a), len(b)) - np.abs(lags)   # 시간차별 겹침 샘플 수
        vals = corr[lags % len(corr)] / np.maximum(overlap, 1)
        top = float(vals.max())
        if top <= 0:
            out.append((float("inf"), 0.0))
            continue
        cand = lags[vals >= top * MOTION_REL] / OUT_FPS       # 강한 봉우리 후보들
        dist = float(np.min(np.abs(cand - c["offset"]))) * 1000.0
        at_audio = int(np.clip(round(c["offset"] * OUT_FPS) + limit, 0, len(vals) - 1))
        out.append((dist, float(vals[at_audio]) / top))
    return out


def sync_one(job: dict) -> dict:
    """세트 하나를 싱크해서 cam1/2/3.mp4 + preview.mp4 + sync.json 을 만든다."""
    t0 = time.time()
    sv.QUIET = True                       # 워커 프로세스에서 ffmpeg 진행률 출력 억제
    s, outdir, args = job["set"], Path(job["outdir"]), job["args"]
    opts = EncOpts(fps=OUT_FPS, crf=args["crf"], preset=args["preset"])
    result = {"set_id": s["set_id"], "exercise": s["exercise"], "take": s["take"]}

    try:
        clips, plan = analyze_set(s["views"], args["max_offset"],
                                  args["anchor"], args["margin"])
        dur = plan["end"] - plan["start"]
        if dur < args["min_duration"]:
            raise RuntimeError(
                f"공통 구간이 {dur:.2f}s 뿐입니다 (최소 {args['min_duration']}s)")

        residuals = measure_residual(clips, plan["lo"], plan["hi"])
        motion = motion_check(clips, args["max_offset"])
        max_res = max(abs(r) for r in residuals)
        max_motion = max(d for d, _ in motion)
        min_motion_score = min(sc for _, sc in motion)
        spread = plan["peak_spread_ms"]

        # 오디오 근거(잔여 오프셋 + 뷰 간 박수 편차)가 1차 판정, 영상 모션이 교차검증.
        audio_ok = max_res <= RESIDUAL_PASS_MS and spread <= SPREAD_PASS_MS
        status = ("PASS" if audio_ok and max_motion <= MOTION_TOL_MS
                  else "WARN" if audio_ok or max_res <= RESIDUAL_WARN_MS
                  else "FAIL")

        outdir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for c in clips:
            dst = outdir / f"cam{c['cam']}.mp4"
            sv.cut(c["path"], dst, plan["start"] + c["offset"], dur, opts)
            outputs.append(dst)
        sv.make_preview(clips, outdir / "preview.mp4",
                        plan["lo"], plan["hi"] - plan["lo"], opts)

        n_frames = min(count_frames(p) for p in outputs)
        result.update({
            "status": status, "anchor": plan["anchor"],
            "duration_sec": round(dur, 6), "fps": OUT_FPS, "n_frames": n_frames,
            "overlap_with_clap_sec": round(plan["hi"] - plan["lo"], 6),
            "head_clap_sec": (None if plan["head_clap"] is None
                              else round(plan["head_clap"], 4)),
            "tail_clap_sec": (None if plan["tail_clap"] is None
                              else round(plan["tail_clap"], 4)),
            "aligned_onset_peaks": plan["peaks"],
            "max_residual_ms": round(max_res, 3),
            "peak_spread_ms": round(spread, 3),
            "max_motion_check_ms": round(max_motion, 3),
            "min_motion_score": round(min_motion_score, 3),
            "clips": [{
                "cam": c["cam"], "source": str(c["path"].relative_to(job["root"])),
                "source_fps": c["fps"], "source_duration_sec": round(c["duration"], 3),
                "offset_sec": round(c["offset"], 6),
                "cut_start_sec": round(plan["start"] + c["offset"], 6),
                "loudest_transient_sec": round(c["clap_at"], 4),
                "confidence": None if c["confidence"] == float("inf")
                              else round(c["confidence"], 3),
                "residual_ms": round(residuals[i], 3),
                "motion_check_ms": round(motion[i][0], 3),
                "motion_score": round(motion[i][1], 3),
            } for i, c in enumerate(clips)],
            "elapsed_sec": round(time.time() - t0, 1),
        })
        (outdir / "sync.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:                                  # noqa: BLE001
        result.update({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc(limit=3),
                       "elapsed_sec": round(time.time() - t0, 1)})
    return result


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip().rstrip(",")
    return int(out) if out.isdigit() else 0


# --------------------------------------------------------------------------- #
# 3단계: 프레임 추출
# --------------------------------------------------------------------------- #
def extract_one(job: dict) -> dict:
    """세트 하나의 세 영상을 JPEG 프레임으로 뽑고, 개수를 min 에 맞춘다."""
    t0 = time.time()
    set_id, syncdir, framedir = job["set_id"], Path(job["syncdir"]), Path(job["framedir"])
    long_side, quality = job["long_side"], job["quality"]
    # 종목마다 가로/세로 촬영이 섞여 있다 (벤치프레스·푸시업은 가로, 나머지는 세로).
    # 높이 기준으로 줄이면 세로 영상만 폭이 406px 로 뭉개지므로 **긴 변** 기준으로 맞춘다.
    scale = (f"scale=w={long_side}:h={long_side}"
             ":force_original_aspect_ratio=decrease:force_divisible_by=2")
    try:
        counts = {}
        for cam in (1, 2, 3):
            camdir = framedir / f"cam{cam}"
            if camdir.exists():
                shutil.rmtree(camdir)
            camdir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", str(syncdir / f"cam{cam}.mp4"),
                 "-vf", f"fps={OUT_FPS},{scale}",
                 "-q:v", str(quality), "-start_number", "0",
                 str(camdir / "%06d.jpg")], check=True)
            counts[cam] = len(list(camdir.glob("*.jpg")))

        n = min(counts.values())
        for cam in (1, 2, 3):                    # 초과분 삭제해 세 뷰를 동일 길이로
            for i in range(n, counts[cam]):
                (framedir / f"cam{cam}" / f"{i:06d}.jpg").unlink(missing_ok=True)
        return {"set_id": set_id, "status": "OK", "n_frames": n,
                "raw_counts": counts, "elapsed_sec": round(time.time() - t0, 1)}
    except Exception as exc:                                  # noqa: BLE001
        return {"set_id": set_id, "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_sec": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# 4단계: split + manifest
# --------------------------------------------------------------------------- #
def assign_splits(sets: list[dict], seed: int,
                  val_frac: float = 0.15, test_frac: float = 0.15) -> dict[str, str]:
    """세트 단위 분할. 프레임 단위로 섞으면 인접 프레임이 거의 같아서 누수가 심하다.

    종목마다 val/test 를 따로 떼면 세트가 종목당 4~5개뿐이라 train 이 50%까지
    떨어진다. 그래서 전체 비율(70/15/15)을 맞추되, val/test 로 뽑히는 세트가
    한 종목에 몰리지 않도록 종목을 번갈아 가며 고른다.
    """
    rng = random.Random(seed)
    pools: dict[str, list[str]] = {}
    for s in sets:
        pools.setdefault(s["exercise"], []).append(s["set_id"])
    for ids in pools.values():
        ids.sort()
        rng.shuffle(ids)

    cycle = sorted(pools)
    rng.shuffle(cycle)
    order, i = [], 0
    while any(pools.values()):                     # 종목 라운드로빈으로 순서를 만든다
        ex = cycle[i % len(cycle)]
        if pools[ex]:
            order.append(pools[ex].pop())
        i += 1

    n = len(order)
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    n_val = max(1, round(n * val_frac)) if n >= 3 else 0
    return {sid: ("test" if j < n_test
                  else "val" if j < n_test + n_val else "train")
            for j, sid in enumerate(order)}


def build_manifest(out: Path, sets: list[dict], sync_results: dict,
                   frame_results: dict, seed: int, do_split: bool = True) -> dict:
    usable = [s for s in sets
              if sync_results.get(s["set_id"], {}).get("status") in ("PASS", "WARN")
              and frame_results.get(s["set_id"], {}).get("status") == "OK"]
    # 분할을 끄면 전부 "all" 로 둔다. 나중에 --stage manifest 만 다시 돌리면 붙는다.
    splits = assign_splits(usable, seed) if do_split else {s["set_id"]: "all" for s in usable}

    rows = []
    for s in sorted(sets, key=lambda x: x["set_id"]):
        sid = s["set_id"]
        sync = sync_results.get(sid, {})
        frames = frame_results.get(sid, {})
        if sync.get("status") not in ("PASS", "WARN", "FAIL") or frames.get("status") != "OK":
            continue
        n = frames["n_frames"]
        split = splits.get(sid, "review")        # FAIL 세트는 review 로 격리
        rel = f"{FRAME_DIR}/{s['exercise']}/{sid}"
        for i in range(n):
            rows.append({
                "sample_id": f"{sid}_{i:06d}",
                "exercise": s["exercise"], "take": s["take"], "set_id": sid,
                "frame_idx": i, "time_sec": round(i / OUT_FPS, 6),
                "cam1": f"{rel}/cam1/{i:06d}.jpg",
                "cam2": f"{rel}/cam2/{i:06d}.jpg",
                "cam3": f"{rel}/cam3/{i:06d}.jpg",
                "split": split, "qc": sync.get("status", "?"),
                "sync_residual_ms": sync.get("max_residual_ms"),
            })

    fields = list(rows[0].keys()) if rows else []
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_split: dict[str, int] = {}
    by_exercise: dict[str, int] = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
        by_exercise[r["exercise"]] = by_exercise.get(r["exercise"], 0) + 1
    return {"n_triplets": len(rows), "n_images": len(rows) * 3,
            "by_split": by_split, "by_exercise": by_exercise,
            "set_splits": splits}


def write_qc_report(out: Path, sync_results: dict) -> None:
    lines = [
        "# 싱크 QC 리포트", "",
        "30fps 한 프레임 = 33.3ms. 아래 수치는 모두 이보다 작아야 프레임 라벨이 정확하다.", "",
        "| 지표 | 뜻 | 기준 |",
        "|---|---|---|",
        f"| `residual` | 실제 컷 타임스탬프로 원본 오디오를 다시 잘라 잰 잔여 오프셋. "
        f"ffmpeg 시크 오차 포함 | ≤{RESIDUAL_PASS_MS:g}ms |",
        f"| `spread` | 정렬 후 세 뷰의 박수 봉우리가 벌어진 정도 | ≤{SPREAD_PASS_MS:g}ms |",
        f"| `motion` | 영상 프레임차 에너지로 오프셋을 다시 재서 오디오와 비교 (오디오와 무관한 독립 검증) "
        f"| ≤{MOTION_TOL_MS:g}ms |", "",
        "판정: **PASS** = 셋 다 통과 / **WARN** = 오디오는 통과했지만 모션 교차검증이 어긋남 "
        "/ **FAIL** = 오디오 근거가 부족", "",
        "`anchor` 는 박수를 어디서 찾았는지: `head` 앞, `tail` 뒤, `both` 양쪽, `none` 못 찾음.", "",
        "`motion score` 는 오디오가 가리키는 지점에서의 모션 상관 상대점수. "
        "1.0 이면 모션 최고점과 정확히 일치, 0.85 이상이면 통과.", "",
        "| 세트 | 판정 | anchor | 길이(s) | 프레임 | residual(ms) | spread(ms) | motion(ms) | motion score | cam2 offset | cam3 offset |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid in sorted(sync_results):
        r = sync_results[sid]
        if r.get("status") == "ERROR":
            lines.append(f"| {sid} | ERROR | | | | | | | | | {r.get('error', '')} |")
            continue
        offs = {c["cam"]: c["offset_sec"] for c in r["clips"]}
        lines.append(
            f"| {sid} | {r['status']} | {r['anchor']} | {r['duration_sec']:.2f} | "
            f"{r['n_frames']} | {r['max_residual_ms']:.2f} | {r['peak_spread_ms']:.1f} | "
            f"{r['max_motion_check_ms']:.1f} | {r['min_motion_score']:.2f} | "
            f"{offs[2]:+.3f} | {offs[3]:+.3f} |")
    (out / "qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "qc_report.json").write_text(
        json.dumps(sync_results, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
def load_existing(out: Path, sets: list[dict]) -> dict:
    results = {}
    for s in sets:
        f = out / SYNC_DIR / s["exercise"] / s["set_id"] / "sync.json"
        if f.exists():
            results[s["set_id"]] = json.loads(f.read_text(encoding="utf-8"))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Exercise3D 다시점 데이터셋 빌더")
    ap.add_argument(
        "--source-root", "--root", dest="root",
        default=str(DEFAULT_DATASET_ROOT / "origin"), help="원본 영상 루트",
    )
    ap.add_argument(
        "--dataset-root", "--out", dest="out",
        default=str(DEFAULT_DATASET_ROOT), help="산출물 루트",
    )
    ap.add_argument("--stage", default="all",
                    choices=["all", "scan", "sync", "verify", "frames", "manifest"])
    ap.add_argument("--only", nargs="*", help="특정 set_id 만 처리")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="제외할 set_id (예: barbellrow_0000)")
    ap.add_argument("--jobs", type=int, default=4, help="동시에 처리할 세트 수")
    ap.add_argument("--anchor", choices=["auto", "head", "tail"], default="auto")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--max-offset", type=float, default=20.0)
    ap.add_argument("--min-duration", type=float, default=3.0,
                    help="이보다 짧은 공통 구간은 실패 처리")
    ap.add_argument("--frame-long-side", type=int, default=1280,
                    help="프레임 이미지의 긴 변 픽셀 (가로/세로 촬영이 섞여 있어 긴 변 기준)")
    ap.add_argument("--frame-quality", type=int, default=2, help="JPEG q:v (낮을수록 고품질)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-split", action="store_true",
                    help="train/val/test 로 나누지 않고 split 을 전부 all 로 둔다")
    ap.add_argument("--force", action="store_true", help="이미 만들어진 산출물도 다시 생성")
    args = ap.parse_args()

    sv.check_tools()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[scan] {root}")
    sets = discover_sets(root)
    if args.only:
        sets = [s for s in sets if s["set_id"] in args.only]
    if args.exclude:
        dropped = [s["set_id"] for s in sets if s["set_id"] in args.exclude]
        sets = [s for s in sets if s["set_id"] not in args.exclude]
        print(f"  제외: {dropped}")
    print(f"  세트 {len(sets)}개 / 영상 {len(sets) * 3}개")
    for s in sets:
        print(f"    {s['set_id']}: " + ", ".join(p.name for p in s["views"].values()))
    if args.stage == "scan":
        return

    stages = (["sync", "frames", "manifest"] if args.stage == "all"
              else ["sync"] if args.stage == "verify" else [args.stage])

    sync_results = load_existing(out, sets)

    # ---------------- 1·2단계: 싱크 + 검증 ---------------- #
    if "sync" in stages:
        todo = [s for s in sets
                if args.force or sync_results.get(s["set_id"], {}).get("status")
                not in ("PASS", "WARN", "FAIL")]
        print(f"\n[sync] {len(todo)}개 처리 (건너뜀 {len(sets) - len(todo)}개), 동시 {args.jobs}")
        jobs = [{"set": s, "root": str(root),
                 "outdir": str(out / SYNC_DIR / s["exercise"] / s["set_id"]),
                 "args": {"crf": args.crf, "preset": args.preset,
                          "max_offset": args.max_offset, "anchor": args.anchor,
                          "margin": args.margin, "min_duration": args.min_duration}}
                for s in todo]
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(sync_one, j): j["set"]["set_id"] for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                sync_results[r["set_id"]] = r
                done += 1
                if r["status"] == "ERROR":
                    print(f"  ({done}/{len(jobs)}) {r['set_id']:22s} ERROR  {r['error']}")
                else:
                    print(f"  ({done}/{len(jobs)}) {r['set_id']:22s} {r['status']:4s} "
                          f"{r['anchor']:4s} {r['duration_sec']:6.2f}s "
                          f"{r['n_frames']:5d}f  잔여 {r['max_residual_ms']:5.2f}ms "
                          f"편차 {r['peak_spread_ms']:5.1f}ms "
                          f"모션 {r['max_motion_check_ms']:6.1f}ms/{r['min_motion_score']:.2f} "
                          f"({r['elapsed_sec']:.0f}s)")

        write_qc_report(out, sync_results)
        tally: dict[str, int] = {}
        for r in sync_results.values():
            tally[r["status"]] = tally.get(r["status"], 0) + 1
        print(f"  판정: {tally}")

        previews = out / "previews"
        previews.mkdir(exist_ok=True)
        for s in sets:
            src = out / SYNC_DIR / s["exercise"] / s["set_id"] / "preview.mp4"
            link = previews / f"{s['set_id']}.mp4"
            if src.exists():
                link.unlink(missing_ok=True)
                link.symlink_to(Path("..") / src.relative_to(out))

    if args.stage == "verify":
        return

    # ---------------- 3단계: 프레임 ---------------- #
    frame_results = {}
    fr_path = out / "frame_results.json"
    if fr_path.exists():
        frame_results = json.loads(fr_path.read_text(encoding="utf-8"))

    if "frames" in stages:
        eligible = [s for s in sets
                    if sync_results.get(s["set_id"], {}).get("status")
                    in ("PASS", "WARN", "FAIL")]
        todo = [s for s in eligible
                if args.force or frame_results.get(s["set_id"], {}).get("status") != "OK"]
        print(f"\n[frames] {len(todo)}개 처리 (건너뜀 {len(eligible) - len(todo)}개), "
              f"긴 변 {args.frame_long_side}px q={args.frame_quality}")
        jobs = [{"set_id": s["set_id"],
                 "syncdir": str(out / SYNC_DIR / s["exercise"] / s["set_id"]),
                 "framedir": str(out / FRAME_DIR / s["exercise"] / s["set_id"]),
                 "long_side": args.frame_long_side, "quality": args.frame_quality}
                for s in todo]
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(extract_one, j): j["set_id"] for j in jobs}
            for fut in as_completed(futures):
                r = fut.result()
                frame_results[r["set_id"]] = r
                done += 1
                if r["status"] == "OK":
                    print(f"  ({done}/{len(jobs)}) {r['set_id']:22s} {r['n_frames']:5d}f "
                          f"×3  raw={list(r['raw_counts'].values())}  ({r['elapsed_sec']:.0f}s)")
                else:
                    print(f"  ({done}/{len(jobs)}) {r['set_id']:22s} ERROR  {r['error']}")
        fr_path.write_text(json.dumps(frame_results, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    # ---------------- 4단계: manifest ---------------- #
    if "manifest" in stages:
        print("\n[manifest]")
        stats = build_manifest(out, sets, sync_results, frame_results,
                               args.seed, do_split=not args.no_split)
        meta = {
            "name": "Exercise3D multi-view synchronized frame dataset",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_root": str(root),
            "label_definition":
                "같은 sample_id 를 가진 cam1/cam2/cam3 이미지 3장은 "
                "동일한 순간을 서로 다른 각도에서 촬영한 것이다.",
            "sync_method":
                "박수 트랜지언트 상호상관 (온셋 엔벨로프 5ms → 박수 주변 0.25ms 정밀 보정)",
            "fps": OUT_FPS,
            "frame_long_side": args.frame_long_side,
            "jpeg_quality": args.frame_quality,
            "split_unit": ("나누지 않음 (split=all). build_dataset.py --stage manifest 로 언제든 분할 가능"
                           if args.no_split else
                           "set (종목_테이크). 프레임 단위 분할은 누수 때문에 금지"),
            "split_seed": None if args.no_split else args.seed,
            "qc_thresholds_ms": {"residual_pass": RESIDUAL_PASS_MS,
                                 "residual_warn": RESIDUAL_WARN_MS,
                                 "peak_spread_pass": SPREAD_PASS_MS,
                                 "motion_cross_check": MOTION_TOL_MS},
            "n_sets_total": len(sets),
            "n_sets_by_status": {
                k: sum(1 for r in sync_results.values() if r.get("status") == k)
                for k in sorted({r.get("status") for r in sync_results.values()})},
            **stats,
        }
        (out / "dataset.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  트리플렛 {stats['n_triplets']:,}개 / 이미지 {stats['n_images']:,}장")
        print(f"  split: {stats['by_split']}")
        print(f"  종목별: {stats['by_exercise']}")

    print(f"\n완료 → {out}")


if __name__ == "__main__":
    main()
