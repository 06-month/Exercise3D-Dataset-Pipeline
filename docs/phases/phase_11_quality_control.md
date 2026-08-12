# Phase 11 — Pseudo-label Quality Control

## 상태와 원칙

상태는 `IN_PROGRESS_STREAMING`이다. Phase 9/Mode C assessment가 완료된 sequence부터 CPU-only로
quality vector를 만들며, deadline export는 누락된 quality output을 동일 함수로 materialize한 뒤
검증한다. 이 단계는 correlated learned evidence를 하나의 “정확도 확률”로 합치지 않는다.

- `not_ground_truth=true`
- `not_calibrated_probability=true`
- `scalar_quality_score_defined=false`
- PASS/REVIEW/FAIL과 source별 residual/reason을 독립적으로 보존
- target ambiguity/NO_TARGET를 background person으로 대체하지 않음
- camera/triangulation/body/Mode C의 기존 status와 threshold를 변경하지 않음

## Frame quality vector

[`tools/build_pseudolabel_quality.py`](../../tools/build_pseudolabel_quality.py)는 Phase 9 reference
timeline에서 다음 evidence를 정렬한다.

- camera별 target mapping/status/confidence, identity risk, duplicate/reflection metadata
- camera별 Sapiens valid-joint fraction
- camera별 SAM accepted/rejected, occlusion, failure reason와 time error
- canonical triangulation valid fraction, quality, reprojection, ray angle
- body valid/confidence와 missing/geometry/prior evidence fraction
- observation/alignment residual과 Mode C review candidate

Frame status는 새 empirical threshold가 아니라 explicit evidence bit로 구성한다.
`TARGET_VIEW_MISSING_OR_ABSTAINED`, `IDENTITY_RISK`, `OCCLUSION_RISK`,
`SAM_PRIOR_REJECTED_OR_INVALID`, `TRIANGULATION_JOINT_MISSING`, `PRIOR_ONLY_JOINT_USED`,
`BODY_JOINT_MISSING`, `MODE_C_REVIEW_CANDIDATE`, `SEQUENCE_CAMERA_REVIEW`를 bitmask로 저장한다.
어떤 bit도 없으면 frame PASS, 하나라도 있으면 REVIEW다. Numeric/schema/finite 계약 실패는 output을
생성하지 않고 stage failure로 처리한다.

## Resume와 export

`quality_vector.npz`와 `metadata.json`의 required field, shape, frame index/PTS, frame status count를
검증한 output만 resume-skip한다. Autonomous supervisor의 새 실행은 Mode C assessment 직후 Phase 11을
호출한다. 현재 살아 있는 기존 supervisor는 재시작하지 않았으며, deadline/final exporter가 complete
body-fit sequence의 누락 quality output을 CPU-only로 생성한 뒤 private build dependency로 포함한다.

## 현재 결과

2026-08-12 10:13 KST 기준 완료된 10 sequence, 6,485 reference frame을 materialize했다.

- sequence REVIEW 10 / FAIL 0
- exporter dependency validation reason 0
- target abstention/unmapped view 8, SAM rejected/unmapped view 8
- prior-only/body-missing/triangulation-missing joint frame 0
- `pushup_0001`의 기존 ambiguity 7은 identity/target/SAM rejection reason으로 그대로 유지
- 기존 body/camera REVIEW를 PASS로 승격하지 않음

Output은 private ignored root에만 있고 public Git에는 code, schema, aggregate만 포함한다.
