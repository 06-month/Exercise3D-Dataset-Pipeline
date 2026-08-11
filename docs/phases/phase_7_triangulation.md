# Phase 7 — Timestamp-aware Multi-view Triangulation

## 목적과 현재 상태

상태는 `PILOT_COMPLETE_CAMERA_RECOVERY_REQUIRED`다. 이 단계는 Sapiens2-5B의 view별
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
