# Phase 7 — Timestamp-aware Multi-view Triangulation

## 목적과 현재 상태

상태는 `IN_PROGRESS`다. pilot/recovery gate 이후 full pose가 완결된 sequence부터 CPU streaming을
시작했다. 이 단계는 Sapiens2-5B의 view별
2D 출력을 곧바로 3D ground truth로 승격하지 않는다. Phase 2의 PTS offset/drift,
Phase 5의 fixed-camera geometry, view별 confidence와 실제 ray conditioning을 결합해
3D observation proposal과 불확실성을 생성한다.

## 구현

[`tools/triangulate_sapiens2.py`](../../tools/triangulate_sapiens2.py)는 다음을 보존한다.

- cam1 PTS를 reference로 사용하고 cam2/cam3 trajectory만 linear interpolation
- RGB/video frame interpolation 0회
- VGGT model canvas의 fixed K를 working JPEG pixel coordinate로 명시적으로 변환
- OpenCV world-to-camera `Xc = R Xw + t`, cam1 identity gauge, sequence-local arbitrary scale
- confidence-weighted DLT와 Huber IRLS
- joint별 supporting view 수, per-view reprojection, minimum ray angle, DLT conditioning
- source frame bracket/interpolation alpha, pairing error, timing/camera uncertainty provenance
- 308 teacher joint proposal과 별도의 explicit canonical body mapping

Sapiens2→canonical mapping은
[`configs/sapiens2_canonical_joints.json`](../../configs/sapiens2_canonical_joints.json)에
name과 source index를 함께 고정했다. wrist는 teacher body index가 아니라 명시된
`right_wrist=41`, `left_wrist=62`를 사용하며 pelvis/shoulder center는 양쪽 direct joint가
모두 유효할 때만 파생한다. Body-model landmark mapping은 이 파일에서 추측하지 않는다.

## Pilot 측정

기존 4개 target-only pilot의 3,244 reference timestamps를 측정했다. 모든 numeric/schema,
finite/NaN contract는 PASS했지만 pose-camera consistency는 다음과 같았다.

| sequence | canonical reprojection median / p90 | camera gate | Phase 7 gate |
|---|---:|---|---|
| barbellrow_0000 | 7.06 / 30.62 px | PASS | REVIEW |
| squat_0001 | 26.24 / 164.93 px | REVIEW | NO_GO |
| pushup_0001 | 326.93 / 2,004.04 px | REVIEW | NO_GO |
| benchpress_0003 | 7.91 / 97.84 px | PASS | REVIEW |

Gate는 결과에 맞춘 임의 threshold가 아니라 실행 전 선언한 Huber scale 10 px의 배수로
정의한다. PASS는 median ≤1×와 p90 ≤3× 및 camera PASS, REVIEW는 NO_GO 범위 안의 나머지,
NO_GO는 median >2× 또는 p90 >10×다.

## 해석과 안전 조치

`squat_0001/cam2`와 `pushup_0001`의 private overlay에서 target identity와 2D skeleton 자체는
정상이었지만 current refined camera의 human-joint epipolar consistency가 크게 깨졌다.
따라서 낮은 BA background residual만으로 foreground triangulation 사용 가능성을 보장할 수
없다. sparse background support, degeneracy 또는 mirror correspondence 중 무엇이 원인인지는
아직 독립적으로 확정하지 않는다.

NO_GO sequence의 raw triangulation proposal은 진단용으로 보존하지만
`eligible_for_body_fitting=false`이며 pseudo-label export에 사용하지 않는다. REVIEW도 PASS로
승격하지 않고 joint별 residual/conditioning을 계속 전파한다. 다음 gate는 원본 Phase 5 camera를
덮어쓰지 않는 별도 recovery candidate와 held-out-frame 검증이다. Sapiens2 observation으로 camera를
보정하는 경우 그 결과는 독립 calibration이 아니라 observation-conditioned geometry라는
provenance를 명시해야 한다.

## Observation-conditioned recovery gate

[`tools/recover_cameras_from_pose_observations.py`](../../tools/recover_cameras_from_pose_observations.py)는
Phase 5 output을 덮어쓰지 않는 별도 recovery root만 생성한다. canonical direct joints와 Phase 2
timestamp pairing으로 세 가지 essential-pair + tied-scale PnP topology를 fit frame에서만 비교하고,
fit objective가 가장 낮은 topology를 사전 분리한 20% held-out frame에서 검증한다. cam1 identity,
fixed K, sequence-local arbitrary scale은 유지한다.

| sequence | current held-out median / p90 | recovered held-out median / p90 | all-frame canonical median / p90 | gate |
|---|---:|---:|---:|---|
| squat_0001 | 26.21 / 164.86 px | 5.70 / 18.88 px | 5.71 / 18.93 px | REVIEW |
| pushup_0001 | 311.60 / 2,020.79 px | 8.12 / 95.12 px | 8.11 / 96.04 px | REVIEW |

두 sequence 모두 fit/held-out overlap 0이고 held-out degradation/overfit gate를 통과했다. threshold는
변경하지 않았으며 기존 NO_GO proposal은 그대로 남아 있다. recovery 결과는
`SAPIENS2_2D_OBSERVATION_CONDITIONED`이고 Sapiens2와 같은 observation으로 만들었으므로 독립적인
camera 정확도 검증이나 GT가 아니다. 따라서 NO_GO만 해제하되 `REVIEW_OBSERVATION_CONDITIONED`와
원 camera uncertainty를 계속 전파한다. `pushup_0001` p90은 NO_GO 100 px 경계에 가깝기 때문에
fitting/QC에서 특히 보수적으로 취급한다.

## Full streaming contract

[`tools/run_phase7_streaming.py`](../../tools/run_phase7_streaming.py)는 세 camera의 pose metadata와
`poses_2d.npz`가 source frame 수, 308-point shape, finite/schema gate를 모두 통과할 때만 sequence를
시작한다. Phase 5 camera로 initial triangulation을 항상 먼저 수행하며, `NO_GO_TRIANGULATION`일 때만
recovery candidate를 생성하거나 기존 held-out 승인 candidate를 재사용한다. Final metadata에는
`PHASE5_BACKGROUND_BA` 또는 `REVIEW_OBSERVATION_CONDITIONED` source가 남는다. 이 CPU process는
다음 Sapiens2 GPU chunk와 겹치며 원 camera/pose output을 덮어쓰지 않는다.
