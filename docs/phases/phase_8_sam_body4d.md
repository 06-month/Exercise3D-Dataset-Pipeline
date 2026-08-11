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
이 dataset에서는 그대로 사용하지 않는다. `sam_body_primary_target_runner.py`가 Phase 6 selector의
accepted primary bbox 하나만 SAM 3에 seed하고, background candidate는 metadata/ambiguity evidence로만
유지한다. 이 adapter는 upstream에 존재하지 않는 CLI option을 가장하지 않고 official class/API를
호출한다. Mode A도 official estimator의 `bboxes=` API로 frame당 accepted bbox 0/1개만 전달한다.

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

Local `<CHECKPOINT_ROOT>/sam_body4d`에는 필요한 payload가 없다. Primary-target adapter 기준 필요한
공식 inventory는 총 24,037,682,088 bytes(22.387 GiB)이며 weights는 Git에 포함하거나 재배포하지
않는다.

- Mode A 최소: SAM 3D Body package + MoGe-2, 3.845 GiB
- Mode B 누적: Mode A + SAM 3, 7.059 GiB
- Mode C 누적: Mode B + Diffusion-VAS 2개 + Depth Anything V2, 22.387 GiB
- upstream all-human initialization용 ViTDet 2.576 GiB는 target adapter가 detector를 호출하지 않아 제외
- SAM 3와 SAM 3D Body는 Hugging Face에서 사전 access acceptance와 인증이 필요

상세 파일/용량/라이선스 출처는
[`sam_body4d_checkpoint_manifest.csv`](../../metadata/results/sam_body4d_checkpoint_manifest.csv)에
기록했다. Credential은 문서, 로그, Git에 기록하지 않는다. 사용자 승인 전 다운로드하지 않는다.

## Primary-target preflight 결과

Checkpoint 없이도 input/selection/repository/config 경계까지 검증했다.

| Pilot | Frames | Mode A | Mode B | Mode C | Target seed |
|---|---:|---|---|---|---:|
| `squat_0001/cam1` control | 1,267 | BLOCKED_CHECKPOINT | BLOCKED_CHECKPOINT | BLOCKED_CHECKPOINT | 1 |
| `latpulldown_0002/cam2` severe | 1,136 | BLOCKED_CHECKPOINT | BLOCKED_CHECKPOINT | BLOCKED_CHECKPOINT | 1 |

두 clip 모두 target-valid frame은 전부였고 severe clip의 `OCCLUSION_RISK` 959 frame도 보존했다.
각 mode가 정확히 한 bbox slot/seed만 허용하고 ambiguous first frame을 강제 선택하지 않는 synthetic
test를 포함해 SAM/selector test 11개가 PASS했다. `BLOCKED_CHECKPOINT`는 model execution 실패가 아니라
승인 전 의도된 preflight 종료다.

```bash
python tools/benchmark_sam_body4d.py \
  --mode <A_OR_B_OR_C> \
  --sam-3d-body-root <OFFICIAL_SAM_3D_BODY_REPO> \
  --sam-body4d-root <OFFICIAL_SAM_BODY4D_REPO> \
  --checkpoint-root <CHECKPOINT_ROOT>/sam_body4d \
  --input-frames <PRIVATE_CLIP_FRAME_DIR> \
  --target-selection <PRIVATE_TARGET_SELECTION_NPZ> \
  --frame-count <30_TO_60_SECOND_FRAME_COUNT> \
  --output-dir outputs/runtime/phase8_sam/<CLIP>/<MODE>
```

Mode A에는 `--sam-3d-body-root`, Mode B/C에는 `--sam-body4d-root`만 필요하다. 실제 pilot은 사용자
승인 뒤 `--run`을 추가한다.

`summarize_sam_body_runtime.py`는 여섯 PASS row가 모두 존재해야 Mode C/B ratio와 occlusion runtime
증가를 계산한다. `EXPECTED_CASE`에는 measured severe-frame fraction과 selective-refinement fraction을
명시적으로 요구하며, 이 값이 없으면 낙관적인 숫자를 만들지 않고 중단한다.

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
