# Phase 8 — SAM 3D Body / SAM-Body4D Runtime Feasibility

## 현재 상태와 실행 경계

`PILOT_COMPLETE_REVIEW`. Gated checkpoint access, 22.387 GiB payload integrity, primary-target
adapter, control/severe Mode A/B/C pilot는 완료했다. Full Sapiens2는 실행 중이며 full SAM은
GPU contention을 피하기 위해 그 뒤에 Mode B로 실행한다.

검증한 upstream은 다음과 같다.

- [SAM-Body4D](https://github.com/gaomingqi/sam-body4d), revision
  `21af1020979ef32ddf6be3597ef59a68bad2f1bf`, MIT code
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body), revision
  `b5c765a0d89d789985e186d396315e7590887b94`, SAM License code/checkpoint
- [Diffusion-VAS](https://github.com/Kaihua-Chen/diffusion-vas), official implementation

SAM-Body4D official offline script는 initial frame의 모든 human detection을 자동 target으로 삼는다.
이 dataset에서는 그대로 사용하지 않는다. `sam_body_primary_target_runner.py`가 Phase 6 selector의
accepted primary bbox 하나만 SAM 3에 seed하고, background candidate는 metadata/ambiguity evidence로만
유지한다. Mode A도 official estimator의 `bboxes=` API로 frame당 accepted bbox 0/1개만 전달한다.

## Checkpoint access와 integrity gate

2026-08-11 gated access 승인 후 credential 값을 출력하지 않는 `hf download --dry-run`을 다시
수행했고 SAM 3, SAM 3D Body 및 나머지 네 repository 모두 접근 PASS를 확인했다. 공식 source가
지정한 payload를 받은 뒤 cache를 제외한 required tree를 전수 검증했다.

- required payload: 28 files, 24,037,668,123 bytes(22.387 GiB)
- 누락·예상 밖 파일: 0 / 0
- size mismatch: 0
- SHA-256 mismatch: 0
- checkpoint/config load와 CUDA smoke: PASS
- credential, checkpoint 자체와 cache: Git 비포함

파일별 상대 경로·크기·SHA-256은
[`sam_body4d_checkpoint_integrity.csv`](../../metadata/results/sam_body4d_checkpoint_integrity.csv),
component별 source/terms는
[`sam_body4d_checkpoint_manifest.csv`](../../metadata/results/sam_body4d_checkpoint_manifest.csv)에
기록했다. Primary-target adapter가 detector를 호출하지 않으므로 upstream all-human initialization용
ViTDet 2.576 GiB는 계속 제외한다.

## Pilot 구성

| 조건 | Clip | Frames | 길이 | `OCCLUSION_RISK` | target |
|---|---|---:|---:|---:|---:|
| control | `squat_0001/cam1` | 1,267 | 약 42초 | 0 | 1 |
| severe | `latpulldown_0002/cam2` | 1,136 | 약 38초 | 959(84.42%) | 1 |

| Mode | 구성 | 목적 |
|---|---|---|
| A | official SAM 3D Body base + primary bbox | single-image body 최소 비용 기준 |
| B | SAM-Body4D, `completion.enable=false` | SAM 3 temporal mask + base body |
| C | SAM-Body4D, `completion.enable=true` | amodal/depth/refiner 포함 |

모든 run은 initial target seed와 target track을 각각 1개만 사용했다. Background candidate에는
SAM 3D Body inference를 수행하지 않았다.

## A/B/C 실측

아래 sec/frame은 model initialization을 포함한 end-to-end 값이다. GPU utilization은 0.2초 간격
`nvidia-smi` 표본의 mean/p95, VRAM은 process 구간의 device peak다.

| 조건 | Mode | sec/frame | total | FPS | peak VRAM | GPU mean/p95 | power mean/max |
|---|---|---:|---:|---:|---:|---:|---:|
| control | A | 0.8265 | 1,047.20s | 1.2099 | 7,367 MiB | 36.86/67% | 123.49/361.76 W |
| control | B | 0.9177 | 1,162.70s | 1.0897 | 33,988 MiB | 13.32/95% | 100.00/429.00 W |
| control | C | 1.8202 | 2,306.22s | 0.5494 | 44,175 MiB | 45.48/100% | 203.33/443.74 W |
| severe | A | 0.8319 | 945.05s | 1.2020 | 7,367 MiB | 37.07/68% | 121.72/362.63 W |
| severe | B | 0.9203 | 1,045.43s | 1.0866 | 32,344 MiB | 13.17/95% | 99.89/428.35 W |
| severe | C | 1.8262 | 2,074.61s | 0.5476 | 42,531 MiB | 45.27/100% | 203.07/440.76 W |

Initialization을 제외한 실행 sec/frame은 control A/B/C가 각각 0.8135/0.8968/1.7887,
severe A/B/C가 0.8170/0.8967/1.7901이다. 따라서 severe/control runtime ratio는
A 1.0043, B 0.9999, C 1.0008로, 이 pilot에서는 가림 자체가 throughput을 유의미하게 바꾸지
않았다. Mode C/B 실행시간 ratio는 control 1.9946, severe 1.9964다.

세부 실측은
[`sam_body4d_runtime_pilot.csv`](../../metadata/results/sam_body4d_runtime_pilot.csv)에 있다.

## Refiner ON/OFF와 output sanity

Mode C는 control에서 refiner 1,287회, amodal segmentation 20회, depth 1,267회를 호출했고,
severe에서는 각각 1,154/18/1,136회 호출했다. 두 조건 모두 content completion 호출은 0회였다.
즉 selector의 bbox-overlap 기반 `OCCLUSION_RISK`와 official SAM-Body4D 내부 mask self-IoU
trigger는 같은 정의가 아니다. Severe clip은 외부 가림 조건 runtime은 검증했지만 실제 content
completion 품질 검증 사례로 간주하지 않는다.

Output sanity 결과는 다음과 같다.

- Mode A: 2,403 numeric result 전부 생성, 대표 시작/중간/끝 표본의 모든 numeric array finite
- Mode B/C: 각 조건 frame 수와 동일한 PLY와 render 생성, 누락 0
- 대표 mesh: 18,439 vertices / 36,874 faces, vertex와 bounds finite
- 대표 render: 720×1,280 RGB decode PASS, 한 target body만 존재
- 육안 표본에서 exploding mesh/두 번째 identity/빈 output은 없었지만, 일부 손·팔 형상은 noisy
  prior 특성을 보여 image ground truth나 정답 mesh로 취급하지 않음
- B/C 동일 frame 5개씩 비교한 최대 vertex delta는 control 0.237 mm, severe 0.303 mm였다.
  Mode C가 약 2배 느린 것에 비해 현재 clip에서 material improvement 근거는 부족하다.

따라서 full run 기본 후보는 Mode B다. Mode C를 선택적으로 사용하려면 official content-completion
trigger가 실제 발생하는 별도 짧은 case와 downstream residual 개선 근거를 먼저 확보해야 한다.

## 전체 runtime projection과 8월 15일 판정

Expected prevalence에는 Phase 6 네 sequence/12-camera pilot의 conservative occlusion proxy
2,657/9,732 frame(27.3018%)을 사용했다. 이는 전체 dataset의 확정 prevalence가 아니라 현재 확보한
비편향에 가까운 pilot proxy이며, 그래서 optimistic/expected/pessimistic을 분리한다.

| Scenario | SAM 65,595 frames | Sapiens2 target-only | 한 GPU 순차 합계 |
|---|---:|---:|---:|
| optimistic | 16.35 h | 79.09 h | 95.43 h |
| expected | 20.80 h | 79.09 h | 99.88 h |
| pessimistic | 32.63 h | 79.09 h | 111.71 h |

- optimistic: 모든 frame Mode B, control 실행 rate
- expected: control/severe weighted Mode B + 27.3018%에 Mode C incremental cost
- pessimistic: 모든 frame Mode C, severe 실행 rate

상세 수치는
[`sam_body4d_runtime_projection.csv`](../../metadata/results/sam_body4d_runtime_projection.csv)에 있다.
2026-08-11 08:16 UTC 기준 2026-08-15 00:00 UTC까지 약 87.7시간이므로 optimistic도 deadline을
넘는다. 8월 15일 23:59 UTC를 마감으로 해석하면 expected 계산 자체는 들어오지만 약 12시간만
남아 download 이후 orchestration, Sapiens/SAM validation, triangulation/body fitting, QC, 재시도와
dataset freeze 시간을 보장할 수 없다. Pessimistic은 계산만으로 마감과 사실상 같다.

이 수치는 이후 확정된 2026-08-14 13:00 KST deadline보다 느슨한 과거 가정에서 만든 projection이다.
현재 운영 판정은 다음과 같다.

- 2026-08-14 13:00 KST 전체 Sapiens2+SAM 순차 freeze: `NO_GO`
- SAM full inference mode: Mode B 동결, Mode C selective fallback만 허용
- scheduling: Sapiens2가 GPU를 전용한 뒤 camera 단위 resumable Mode B를 자율 실행

이번 판정은 accuracy-first 정책을 유지하며, 5B teacher·official flip-test·detector·input resolution·
keypoint convention을 변경하지 않는다.

## 재현 명령

```bash
python tools/benchmark_sam_body4d.py \
  --mode <A_OR_B_OR_C> \
  --sam-3d-body-root <OFFICIAL_SAM_3D_BODY_REPO> \
  --sam-body4d-root <OFFICIAL_SAM_BODY4D_REPO> \
  --checkpoint-root <CHECKPOINT_ROOT>/sam_body4d \
  --input-frames <PRIVATE_CLIP_FRAME_DIR> \
  --target-selection <PRIVATE_TARGET_SELECTION_NPZ> \
  --frame-count <30_TO_60_SECOND_FRAME_COUNT> \
  --output-dir outputs/runtime/phase8_sam/<CLIP>/<MODE> \
  --run
```

Mode A에는 `--sam-3d-body-root`, Mode B/C에는 `--sam-body4d-root`만 필요하다. Runtime summary는
여섯 PASS row와 measured prevalence가 모두 있어야 projection을 생성한다.

Full runner는 [`tools/run_sam_body4d_full.py`](../../tools/run_sam_body4d_full.py)다. Mesh만으로
body fitting을 수행하지 않도록 upstream PLY 저장 직전에 MHR pose/shape/scale/joint numeric prior를
별도 compact NPZ로 보존한다. Target source frame index와 selector uncertainty도 함께 남기며,
frame/mesh/numeric/provenance count가 모두 일치해야 camera resume PASS다. 이 payload는 private
output이며 public Git에는 포함하지 않는다.

Compact schema는 source PTS, ambiguity/NO_TARGET/occlusion과 MHR body/hand pose, shape, scale,
expression, 127 joint coordinate/global rotation, 204-d model parameter를 포함한다.
[`tools/consolidate_sam_body_prior.py`](../../tools/consolidate_sam_body_prior.py)는 이 payload를
camera 단위로 통합하되 ambiguous/no-target frame의 model output이 존재하더라도 learned evidence로
보존만 하고 `accepted_prior=false`로 둔다.

Official MHR replay smoke에서는 저장된 204-d parameter와 shape/expression을 JIT model에 다시 넣고
checkpoint의 308-landmark mapping을 적용했을 때 keypoint 최대 차이 `2.68e-7 m`, mesh 최대 차이
`7.15e-7 m`를 확인했다. 따라서 compact parameter는 PLY와 분리된 재현 가능한 body representation으로
사용할 수 있지만, 여전히 monocular learned prior이며 GT가 아니다.

## Mode C selective escalation

전체 기본은 계속 Mode B다. [`configs/sam_mode_c_escalation.json`](../../configs/sam_mode_c_escalation.json)은
Mode C candidate를 occlusion-risk와 Mode B missing/nonfinite 또는 robust temporal/alignment outlier의
교집합으로 제한한다. 후보 clip은 양쪽 15-frame padding, sequence 최대 10%다. Mode C가 identity/PTS
exact match와 schema를 통과하고 content completion이 실제 호출되거나 canonical alignment residual을
10% 이상 낮추며 geometry displacement를 5% 넘게 악화시키지 않을 때만 채택한다. 그렇지 않으면
Mode B를 유지하고 REVIEW로 남긴다.

[`tools/assess_sam_mode_c_escalation.py`](../../tools/assess_sam_mode_c_escalation.py)는 full Mode B와
Phase 9 fit 직후 이 조건을 실행한다. Robust threshold와 geometry-support 조건을 통과한 frame만
bounded review clip으로 저장하며, 후보가 없으면 `PASS_MODE_B_FROZEN`을 기록한다. 이 assessor는
Mode C를 실행하거나 B payload를 교체하지 않으며, 결과 JSON은 final private export의 필수
uncertainty provenance다.
