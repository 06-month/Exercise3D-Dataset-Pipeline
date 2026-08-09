#!/usr/bin/env python3
"""박수(clap) 소리를 기준으로 여러 시점 영상의 싱크를 맞춰 잘라내는 스크립트.

동작 방식
  1) 각 영상에서 모노 오디오를 뽑아 온셋(transient) 엔벨로프를 만든다.
  2) 기준 영상과의 상호상관(cross-correlation)으로 대략적인 시간차를 구한다. (5ms 해상도)
  3) 박수 근처만 고해상도 온셋으로 다시 상관을 취해 0.25ms 단위까지 보정한다.
  4) 박수 위치에 따라 자를 구간을 정한다.
       - 박수가 앞쪽(anchor=head): 박수 이전을 버리고, 가장 먼저 끝나는 영상에 맞춰 뒤를 자른다.
       - 박수가 뒤쪽(anchor=tail): 박수 이후를 버리고, 앞부분이 가장 짧은 영상에 맞춰 앞을 자른다.
  5) 확인용으로 세 시점을 가로로 붙이고 오디오를 전부 겹친 preview 영상을 만든다.

사용 예
  python3 sync_videos.py sample -o synced
  python3 sync_videos.py sample -o synced --fps 30 --anchor tail
  python3 sync_videos.py a.MOV b.MOV c.MOV -o out --ref b.MOV --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SR = 48000                  # 분석용 샘플레이트
COARSE_FRAME = 1024         # 거친 탐색용 STFT 창
COARSE_HOP_MS = 5.0         # 거친 탐색 해상도
FINE_FRAME = 256            # 정밀 보정용 STFT 창
FINE_HOP = 12               # 정밀 보정 해상도 = 0.25ms
FINE_HALF_WIN = 0.30        # 박수 주변 ±0.30초만 정밀 비교
FINE_SEARCH = 0.05          # 거친 추정치 ±50ms 안에서만 보정
CLAP_EDGE_FRAC = 0.35       # 양 끝 이 비율 안에 있는 봉우리만 박수 후보로 본다
CLAP_EDGE_MAX = 12.0        # 다만 끝에서 이 초를 넘어가면 박수로 보지 않는다
CLAP_REL_THRESH = 0.20      # 최대 봉우리의 이 비율 이상이어야 박수로 인정
CAND_LAGS = 8               # 상호상관에서 검토할 후보 시간차 개수
VIDEO_EXTS = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".mts"}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
QUIET = False               # True 면 ffmpeg 진행률(-stats) 출력을 끈다 (배치 처리용)


def ff_base() -> list[str]:
    return ["ffmpeg", "-y", "-v", "error"] + ([] if QUIET else ["-stats"])


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe
# --------------------------------------------------------------------------- #
def check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"[error] {tool} 를 찾을 수 없습니다. ffmpeg 를 설치해 주세요.")


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)

    vstream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if vstream is None:
        raise RuntimeError(f"{path.name}: 비디오 스트림이 없습니다.")
    if not any(s["codec_type"] == "audio" for s in info["streams"]):
        raise RuntimeError(f"{path.name}: 오디오 스트림이 없어 싱크를 맞출 수 없습니다.")

    num, den = (int(v) for v in vstream["r_frame_rate"].split("/"))
    return {
        "duration": float(info["format"]["duration"]),
        "fps": num / den if den else 0.0,
        "width": int(vstream["width"]),
        "height": int(vstream["height"]),
    }


def load_audio(path: Path) -> np.ndarray:
    """첫 번째 오디오 스트림을 48kHz 모노 float32 로 읽어온다."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    x = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
    if x.size == 0:
        raise RuntimeError(f"{path.name}: 오디오를 디코딩하지 못했습니다.")
    peak = float(np.max(np.abs(x)))
    return x / peak if peak > 0 else x


# --------------------------------------------------------------------------- #
# 신호 처리
# --------------------------------------------------------------------------- #
def spectral_flux(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """스펙트럼이 증가한 양만 합산한 온셋 엔벨로프. 박수처럼 순간적인 소리에 크게 반응한다."""
    n_frames = 1 + max(0, (len(x) - frame) // hop)
    if n_frames < 3:
        raise RuntimeError("분석하기에 오디오가 너무 짧습니다.")
    x = np.ascontiguousarray(x)
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, frame), strides=(x.strides[0] * hop, x.strides[0])
    ) * np.hanning(frame).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    return normalize(np.maximum(mag[1:] - mag[:-1], 0.0).sum(axis=1))


def normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32) - v.mean()
    s = v.std()
    return (v / s).astype(np.float32) if s > 0 else v


def xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """순환 상호상관. corr[k] 가 클수록 a 가 b 보다 k 샘플 늦다는 뜻."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    return np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)


def best_lag(corr: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """[lo, hi] 범위(음수 포함) 안의 최대 지점 + 포물선 보간. 봉우리의 뾰족함도 함께 반환."""
    n = len(corr)
    lags = np.arange(lo, hi + 1)
    vals = corr[lags % n]
    i = int(np.argmax(vals))
    peak = float(vals[i])

    lag = float(lags[i])
    if 0 < i < len(vals) - 1:
        y0, y1, y2 = (float(v) for v in vals[i - 1:i + 2])
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            lag += 0.5 * (y0 - y2) / denom

    # 두 번째로 높은 봉우리 대비 비율 = 신뢰도
    guard = max(1, int(0.02 * len(vals)))
    mask = np.abs(lags - lags[i]) > guard
    runner_up = float(np.max(vals[mask])) if mask.any() else 0.0
    ratio = peak / runner_up if runner_up > 0 else float("inf")
    return lag, ratio


def strong_peaks(env: np.ndarray, rate: float, rel: float = 0.25,
                 k: int = 6, suppress: float = 0.8) -> list[tuple[float, float]]:
    """전체 최댓값의 rel 배 이상인 트랜지언트만 (시각, 세기) 로 추린다."""
    top = float(env.max())
    return [(t, v) for t, v in peak_list(env, rate, 0.0, k, suppress) if v >= top * rel]


def corroborate(ref_peaks: list[tuple[float, float]], oth: np.ndarray, rate: float,
                lag: float, tol: float = 0.05, rel: float = 0.25) -> tuple[int, float, float]:
    """ref 의 강한 트랜지언트들이 lag 만큼 밀린 위치에서 other 에도 실제로 있는지 본다.

    박수를 두 번 쳤는데 한 영상이 늦게 시작해 하나만 담긴 경우, 전체 상호상관은
    엉뚱한 박수끼리 붙여도 비슷한 점수를 낸다. 이때 "다른 박수도 맞아떨어지는가"를
    따지면 옳은 정렬만 살아남는다. other 범위 밖으로 나가는 피크는 세지 않는다.

    반환: (맞아떨어진 개수, 범위 안 피크 중 맞은 비율, 맞은 세기의 합)

    비율만 쓰면 "대부분을 범위 밖으로 밀어내고 하나만 맞춘" 엉뚱한 후보가 1.0 을
    받아 이긴다. 그래서 맞은 **개수**를 먼저 보고, 그 다음 비율, 마지막에 세기를 본다.
    """
    thresh = float(oth.max()) * rel
    w = max(1, int(round(tol * rate)))
    n_in = n_hit = 0
    mass = 0.0
    for t, v in ref_peaks:
        j = int(round((t + lag) * rate))
        if not 0 <= j < len(oth):
            continue
        n_in += 1
        m = float(oth[max(0, j - w):j + w + 1].max())
        if m >= thresh:
            n_hit += 1
            mass += min(v, m)
    return n_hit, (n_hit / n_in if n_in else 0.0), mass


def estimate_offset(ref: dict, other: dict, max_offset: float) -> tuple[float, float]:
    """other 가 ref 보다 몇 초 늦은지 추정. (양수 = other 쪽에서 같은 사건이 더 나중에 일어남)"""
    # 1단계: 전체 구간을 5ms 해상도로 거칠게 탐색
    rate = SR / int(round(SR * COARSE_HOP_MS / 1000.0))
    limit = int(round(max_offset * rate))
    corr = xcorr(other["onset"], ref["onset"])
    coarse_lag, ratio = best_lag(corr, -limit, limit)
    coarse = coarse_lag / rate

    # 1.5단계: 상관 봉우리가 여럿이면(= 박수를 여러 번 쳤을 때) 어느 것이 맞는지
    #          "다른 트랜지언트도 함께 맞는가"로 가른다.
    lags = np.arange(-limit, limit + 1)
    vals = corr[lags % len(corr)]
    cands, work = [], vals.copy()
    for _ in range(CAND_LAGS):
        i = int(np.argmax(work))
        if not np.isfinite(work[i]):
            break
        cands.append(lags[i] / rate)
        w = int(rate * 0.8)
        work[max(0, i - w):i + w] = -np.inf

    ref_peaks = strong_peaks(ref["onset"], rate)
    scored = [(corroborate(ref_peaks, other["onset"], rate, L), L) for L in cands]
    # max 는 첫 최댓값을 돌려주므로, 동점이면 원래 상관 1위(cands[0])가 유지된다.
    best_score, best_lag_s = max(scored, key=lambda x: x[0])
    if cands and abs(best_lag_s - cands[0]) > 1e-9:
        # 상관 최댓값이 아닌 후보가 채택됨 = 박수를 여러 번 친 경우. 신뢰도는
        # 채택된 후보가 원래 1위보다 얼마나 더 잘 들어맞는지로 다시 매긴다.
        coarse = best_lag_s
        ratio = (best_score[0] + 1) / (scored[0][0][0] + 1)

    # 2단계: 박수 주변만 0.25ms 해상도로 다시 맞춤
    #        (전체 엔벨로프로 보정하면 잔향·다른 소음 때문에 수십 ms 씩 밀린다)
    #        기준점은 "밀렸을 때 other 안에 실제로 들어오는" 가장 강한 트랜지언트로 잡는다.
    #        ref 의 최대 트랜지언트가 other 에는 안 담겼을 수도 있기 때문이다.
    anchor_t = next((t for t, _ in ref_peaks
                     if 0 <= (t + coarse) * SR < len(other["audio"])), ref["clap_at"])
    pad = FINE_HALF_WIN + FINE_SEARCH
    s_ref, seg_ref = slice_around(ref["audio"], anchor_t, FINE_HALF_WIN)
    s_oth, seg_oth = slice_around(other["audio"], anchor_t + coarse, pad)
    if len(seg_ref) > FINE_FRAME * 2 and len(seg_oth) > FINE_FRAME * 2:
        f_ref = spectral_flux(seg_ref, FINE_FRAME, FINE_HOP)
        f_oth = spectral_flux(seg_oth, FINE_FRAME, FINE_HOP)
        span = int(round(FINE_SEARCH * SR / FINE_HOP))
        base = (s_oth - s_ref) / SR                 # 두 조각의 시작 시각 차이
        center = int(round((coarse - base) * SR / FINE_HOP))
        lag, _ = best_lag(xcorr(f_oth, f_ref), center - span, center + span)
        return base + lag * FINE_HOP / SR, ratio
    return coarse, ratio


def slice_around(x: np.ndarray, center_sec: float, half_sec: float) -> tuple[int, np.ndarray]:
    lo = max(0, int(round((center_sec - half_sec) * SR)))
    hi = min(len(x), int(round((center_sec + half_sec) * SR)))
    return lo, x[lo:hi]


def find_clap(onset: np.ndarray, rate: float) -> float:
    """가장 강한 트랜지언트(= 박수)의 시각(초)."""
    return float(np.argmax(onset)) / rate


# --------------------------------------------------------------------------- #
# 자를 구간 계산
# --------------------------------------------------------------------------- #
def aligned_onset(clips: list[dict], lo: float, hi: float) -> tuple[float, np.ndarray, np.ndarray]:
    """각 뷰의 온셋을 사건 좌표계 [lo, hi] 격자에 정렬해 쌓고 합산한다.

    오프셋이 맞다면 박수는 세 뷰에서 같은 칸에 떨어져 합산본에 아주 뾰족한
    봉우리를 만든다. 뷰별로 "가장 큰 소리"를 따로 고르는 것보다 훨씬 안정적이다.
    """
    rate = SR / int(round(SR * COARSE_HOP_MS / 1000.0))
    n = max(1, int((hi - lo) * rate))
    per = np.zeros((len(clips), n), dtype=np.float32)
    for i, c in enumerate(clips):
        s = max(0, int(round((lo + c["offset"]) * rate)))
        seg = np.clip(c["onset"][s:s + n], 0, None)
        per[i, :len(seg)] = seg
    return rate, per, per.sum(axis=0)


def peak_list(env: np.ndarray, rate: float, base: float,
              k: int = 6, suppress: float = 1.5) -> list[tuple[float, float]]:
    """비최대 억제를 적용한 상위 봉우리 (시각, 세기) 목록."""
    o = env.astype(np.float64).copy()
    out = []
    for _ in range(k):
        i = int(np.argmax(o))
        if not np.isfinite(o[i]) or o[i] <= 0:
            break
        out.append((base + i / rate, float(o[i])))
        w = int(rate * suppress)
        o[max(0, i - w):i + w] = -np.inf
    return out


def plan_window(clips: list[dict], anchor: str, margin: float) -> dict:
    """사건 기준 좌표계 u = t - offset 에서 자를 구간을 정한다.

    lo / hi 는 모든 영상이 존재하는 최대 겹침 구간(박수 포함, 프리뷰에 사용),
    start / end 는 박수 규칙까지 적용한 실제 출력 구간이다.

    박수는 앞에만, 뒤에만, 또는 **양쪽에 다** 있을 수 있다. 정렬 합산 온셋에서
    양 끝 25% 구간에 있으면서 최대 봉우리의 20% 이상인 봉우리를 박수로 본다.
      - 앞쪽 박수가 있으면 그 뒤부터 (여러 개면 가장 늦은 것 뒤부터)
      - 뒤쪽 박수가 있으면 그 앞까지 (여러 개면 가장 이른 것 앞까지)
    """
    lo = max(-c["offset"] for c in clips)
    hi = min(c["duration"] - c["offset"] for c in clips)
    span = hi - lo
    rate, per, agg = aligned_onset(clips, lo, hi)
    peaks = peak_list(agg, rate, lo)

    strong = [t for t, v in peaks if peaks and v >= peaks[0][1] * CLAP_REL_THRESH]
    zone = min(span * CLAP_EDGE_FRAC, CLAP_EDGE_MAX)
    head_hits = [t for t in strong if t <= lo + zone]
    tail_hits = [t for t in strong if t >= hi - zone]
    head_clap = max(head_hits) if head_hits and anchor in ("auto", "head") else None
    tail_clap = min(tail_hits) if tail_hits and anchor in ("auto", "tail") else None

    start = lo if head_clap is None else min(head_clap + margin, hi)
    end = hi if tail_clap is None else max(tail_clap - margin, lo)
    kind = ("both" if head_clap is not None and tail_clap is not None
            else "head" if head_clap is not None
            else "tail" if tail_clap is not None else "none")

    # 검증용: 검출된 박수 위치에서 뷰마다 온셋 봉우리가 얼마나 벌어져 있는지
    spread = 0.0
    for u in (head_clap, tail_clap):
        if u is None:
            continue
        c = int(round((u - lo) * rate))
        w = int(round(0.15 * rate))
        a, b = max(0, c - w), min(per.shape[1], c + w + 1)
        for row in per:
            if b > a:
                spread = max(spread, abs((a + int(np.argmax(row[a:b]))) - c) / rate * 1000.0)

    return {"anchor": kind, "start": start, "end": end, "lo": lo, "hi": hi,
            "head_clap": head_clap, "tail_clap": tail_clap,
            "peak_spread_ms": spread,
            "peaks": [(round(t, 3), round(v, 1)) for t, v in peaks]}


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #
def encode_args(args, crf: int) -> list[str]:
    return ["-c:v", args.vcodec, "-crf", str(crf), "-preset", args.preset,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart"]


def cut(src: Path, dst: Path, start: float, dur: float, args) -> None:
    cmd = ff_base() + ["-ss", f"{start:.6f}", "-i", str(src), "-t", f"{dur:.6f}",
           "-map", "0:v:0", "-map", "0:a:0"]
    if args.fps:
        cmd += ["-vf", f"fps={args.fps}"]
    cmd += encode_args(args, args.crf) + [str(dst)]
    subprocess.run(cmd, check=True)


def make_preview(clips: list[dict], dst: Path, start: float, dur: float, args) -> None:
    """세 시점을 CAM1·CAM2·CAM3 순서로 가로 배치하고, 오디오를 전부 겹쳐서 넣는다.

    박수가 잘려나간 출력본이 아니라 원본에서 박수를 포함한 구간을 뽑아 쓴다.
    싱크가 맞았다면 박수가 여러 번이 아니라 딱 한 번으로 들리고,
    세 화면의 동작도 같은 순간에 움직인다.
    """
    n = len(clips)
    w = args.preview_width // n // 2 * 2
    fps = args.fps or 30            # hstack 은 입력 프레임레이트가 같아야 하므로 통일

    cmd = ff_base()
    parts, vlabels, alabels = [], [], []
    for i, c in enumerate(clips):
        cmd += ["-ss", f"{start + c['offset']:.6f}", "-t", f"{dur:.6f}", "-i", str(c["path"])]
        text = f"CAM {i + 1}  {c['path'].stem}".replace("\\", "").replace(":", "").replace("'", "")
        draw = (f",drawtext=fontfile={FONT}:text='{text}':x=12:y=12:fontsize=24:"
                f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8") if Path(FONT).exists() else ""
        parts.append(f"[{i}:v]fps={fps},scale={w}:-2,setsar=1{draw}[v{i}]")
        parts.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{i}]")
        vlabels.append(f"[v{i}]")
        alabels.append(f"[a{i}]")

    filt = (";".join(parts) + ";" + "".join(vlabels) + f"hstack=inputs={n}[out];"
            + "".join(alabels) + f"amix=inputs={n}:duration=shortest:normalize=0,"
            f"alimiter=limit=0.95[aout]")
    cmd += ["-filter_complex", filt, "-map", "[out]", "-map", "[aout]"]
    cmd += encode_args(args, 23) + [str(dst)]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
def collect_inputs(paths: list[str]) -> list[Path]:
    if len(paths) == 1 and Path(paths[0]).is_dir():
        files = sorted(p for p in Path(paths[0]).iterdir()
                       if p.suffix.lower() in VIDEO_EXTS)
    else:
        files = [Path(p) for p in paths]
    for f in files:
        if not f.is_file():
            sys.exit(f"[error] 파일이 없습니다: {f}")
    if len(files) < 2:
        sys.exit("[error] 최소 2개 이상의 영상이 필요합니다.")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(
        description="박수 소리를 기준으로 다시점 영상의 싱크를 맞춰 자릅니다.")
    ap.add_argument("inputs", nargs="+", help="영상 파일들 또는 영상이 들어있는 폴더")
    ap.add_argument("-o", "--out", default="synced", help="출력 폴더 (기본: synced)")
    ap.add_argument("--ref", help="기준 영상 파일명 (기본: 정렬 순 첫 번째)")
    ap.add_argument("--anchor", choices=["auto", "head", "tail"], default="auto",
                    help="박수가 영상 앞(head)인지 뒤(tail)인지. 기본 auto = 자동 판별")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="박수 소리를 몇 초 잘라낼지. 0.5 면 박수에서 0.5초 더 떨어뜨려 자름")
    ap.add_argument("--max-offset", type=float, default=20.0,
                    help="탐색할 최대 시간차(초). 기본 20")
    ap.add_argument("--fps", type=float, help="출력 fps 통일 (예: 30)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--vcodec", default="libx264")
    ap.add_argument("--preview-width", type=int, default=1920)
    ap.add_argument("--no-preview", action="store_true", help="확인용 합본 영상을 만들지 않음")
    ap.add_argument("--dry-run", action="store_true", help="계산만 하고 인코딩은 건너뜀")
    args = ap.parse_args()

    check_tools()
    files = collect_inputs(args.inputs)

    ref_idx = 0
    if args.ref:
        matches = [i for i, f in enumerate(files) if args.ref in (f.name, str(f))]
        if not matches:
            sys.exit(f"[error] --ref 로 지정한 {args.ref} 를 입력 목록에서 찾을 수 없습니다.")
        ref_idx = matches[0]

    print(f"[1/4] 오디오 분석 ({len(files)}개)")
    coarse_hop = int(round(SR * COARSE_HOP_MS / 1000.0))
    clips = []
    for i, f in enumerate(files):
        meta = probe(f)
        audio = load_audio(f)
        onset = spectral_flux(audio, COARSE_FRAME, coarse_hop)
        clap = find_clap(onset, SR / coarse_hop)
        clips.append({"path": f, "audio": audio, "onset": onset, "clap_at": clap,
                      "duration": meta["duration"], "fps": meta["fps"]})
        print(f"  - {f.name:28s} {meta['duration']:6.2f}s {meta['fps']:>3g}fps  "
              f"박수 추정 @ {clap:6.3f}s (영상의 {clap / meta['duration'] * 100:.0f}% 지점)")

    print(f"[2/4] 기준 = {files[ref_idx].name} / 시간차 계산")
    ref = clips[ref_idx]
    for i, c in enumerate(clips):
        if i == ref_idx:
            c["offset"], c["confidence"] = 0.0, float("inf")
            print(f"  - {c['path'].name:28s} offset {c['offset']:+8.3f}s   기준")
            continue
        c["offset"], c["confidence"] = estimate_offset(ref, c, args.max_offset)
        print(f"  - {c['path'].name:28s} offset {c['offset']:+8.3f}s   "
              f"신뢰도 {c['confidence']:.1f}x")

    plan = plan_window(clips, args.anchor, args.margin)
    u_start, dur = plan["start"], plan["end"] - plan["start"]
    where = {"both": "앞뒤 모두 → 두 박수 사이만 남김",
             "head": "앞쪽 → 박수 이후를 남김",
             "tail": "뒤쪽 → 박수 이전을 남김",
             "none": "박수를 못 찾음 → 겹침 구간 전체를 남김"}[plan["anchor"]]
    print(f"[3/4] 박수 위치: {plan['anchor']} ({where})")
    for name, u in (("앞", plan["head_clap"]), ("뒤", plan["tail_clap"])):
        if u is not None:
            print(f"      {name} 박수 @ 공통 타임라인 {u:.3f}s")
    print(f"      뷰 간 박수 봉우리 편차 {plan['peak_spread_ms']:.1f}ms "
          f"(작을수록 싱크가 정확)")
    if plan["peak_spread_ms"] > 20:
        print("      ! 편차가 큽니다. 뷰마다 다른 소리를 잡았을 수 있으니 결과를 확인하세요.")
    if dur <= 0:
        sys.exit("[error] 남는 공통 구간이 없습니다. 오프셋 추정이 잘못되었을 수 있습니다.")
    print(f"  → 출력 구간 {dur:.3f}s (박수 포함 최대 겹침 구간은 {plan['hi'] - plan['lo']:.3f}s)")

    if args.dry_run:
        return

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for c in clips:
        start = u_start + c["offset"]
        dst = outdir / f"{c['path'].stem}_synced.mp4"
        print(f"  - {dst.name}  (원본 {start:.3f}s 부터 {dur:.3f}s)")
        cut(c["path"], dst, start, dur, args)
        outputs.append(dst)

    (outdir / "sync_report.json").write_text(json.dumps({
        "reference": files[ref_idx].name,
        "anchor": plan["anchor"],
        "head_clap_sec": plan["head_clap"],
        "tail_clap_sec": plan["tail_clap"],
        "peak_spread_ms": round(plan["peak_spread_ms"], 3),
        "duration_sec": round(dur, 6),
        "clips": [{
            "file": c["path"].name,
            "output": o.name,
            "offset_sec": round(c["offset"], 6),
            "cut_start_sec": round(u_start + c["offset"], 6),
            "clap_at_sec": round(c["clap_at"], 4),
            "confidence": None if c["confidence"] == float("inf") else round(c["confidence"], 3),
        } for c, o in zip(clips, outputs)],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[4/4] 확인용 합본 영상")
    if args.no_preview:
        print("  - 건너뜀 (--no-preview)")
    else:
        preview = outdir / "preview_grid.mp4"
        make_preview(clips, preview, plan["lo"], plan["hi"] - plan["lo"], args)
        print(f"  - {preview.name}  (CAM1·CAM2·CAM3 가로 배치 + 오디오 합성, 박수 포함)")

    print(f"완료 → {outdir}/")


if __name__ == "__main__":
    main()
