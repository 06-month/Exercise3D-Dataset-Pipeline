# Phase 8 — SAM 3D Body / SAM-Body4D Runtime Feasibility

## 현재 상태

`WAITING_CHECKPOINT_APPROVAL`. Full 65,595-frame Sapiens2 inference와 SAM body inference는
시작하지 않았다. Local A100은 현재 유휴 상태다.

검증한 upstream은 다음과 같다.

- [SAM-Body4D](https://github.com/gaomingqi/sam-body4d), revision
  `21af1020979ef32ddf6be3597ef59a68bad2f1bf`, MIT code
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), revision
  `b5c765a0d89d789985e186d396315e7590887b94`, SAM License code/checkpoint
- [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas), official implementation

SAM-Body4D official offline script는 initial frame의 모든 human detection을 자동 target으로 삼는다.
이 dataset에서는 그대로 사용하지 않는다. Phase 6 selector의 accepted primary bbox 하나만 SAM 3에
seed하는 adapter를 두고, background candidate는 metadata/ambiguity evidence로만 유지해야 한다.

## 비교할 mode

| Mode | 구성 | 목적 |
|---|---|---|
| A | official SAM 3D Body base + primary target bbox | single-image body prior의 최소 비용 기준 |
| B | SAM-Body4D, `completion.enable=false` | SAM 3 temporal mask + base body, refiner 제외 |
| C | SAM-Body4D, `completion.enable=true` | Diffusion-VAS amodal/refinement 포함 |

Control은 `squat_0001/cam1` 약 42초, severe-occlusion은 `latpulldown_0002/cam2` 약 38초로
정했다. 후자는 1,136 frame 중 959 frame이 conservative `OCCLUSION_RISK`이고 detector candidate는
평균 2.121/frame이지만 visual QA에서 primary identity switch는 0이다.

Mode C의 completion은 upstream config에서 global on/off다. QC-triggered selective ON은 기존 CLI가
아니므로, mode B 전체 결과에서 severe/long occlusion 구간을 clip/subsequence 단위로 mode C에 보내는
scheduler로 구현·검증해야 한다. Amodal output은 image ground truth가 아니라 noisy prior로 표시한다.

## Checkpoint preflight

Local `<CHECKPOINT_ROOT>/sam_body4d`에는 필요한 payload가 없다. 공식 inventory 기준 다운로드는
총 26,803,630,365 bytes(24.963 GiB)이며 weights는 Git에 포함하거나 재배포하지 않는다.

- Mode A 최소: SAM 3D Body package + MoGe-2 + ViTDet, 약 6.422 GiB
- Mode B 누적: Mode A + SAM 3, 약 9.635 GiB
- Mode C 누적: Mode B + Diffusion-VAS 2개 + Depth Anything V2, 24.963 GiB
- SAM 3와 SAM 3D Body는 Hugging Face에서 사전 access acceptance와 인증이 필요

상세 파일/용량/라이선스 출처는
[`sam_body4d_checkpoint_manifest.csv`](../../metadata/results/sam_body4d_checkpoint_manifest.csv)에
기록했다. Credential은 문서, 로그, Git에 기록하지 않는다. 사용자 승인 전 다운로드하지 않는다.

## Upstream 참고 수치와 local gate

SAM-Body4D가 공개한 H800 80GB 측정에서 completion off는 1 target/100 frame에 masklet 15.55초,
4D reconstruction 70.3초였다. Multi-target example에서는 completion on이 total runtime을 약
7.6–9.1배 늘렸다. 이는 A100 local 실측이 아니므로 전체 runtime 확정값으로 쓰지 않는다.

Checkpoint 승인 후 두 pilot clip에서 mode별 다음을 실측한다.

- end-to-end/stage runtime, sec/frame, target track 수
- peak VRAM, mean/p95 GPU utilization, mean/max power
- output/temporary disk, per-frame output size
- body/MHR validity, temporal identity, occlusion failure와 refiner invocation
- full 65,595-frame best/expected/worst projection과 2026-08-15 deadline 판정

Sapiens2 target-only projection만 79.09 GPU-hours다. 2026-08-11 현재 2026-08-15 00:00 UTC까지
남은 시간에서 SAM/body fitting/QC 여유가 작으므로 provisional status는 `DEADLINE_AT_RISK`다.
Local mode A/B/C 실측 전에는 final feasibility를 확정하지 않는다.
