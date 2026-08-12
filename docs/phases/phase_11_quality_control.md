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

기존 live supervisor가 새 Phase 11 code를 memory에 load하지 않은 간격은
[`tools/run_quality_control_follower.py`](../../tools/run_quality_control_follower.py)가 보완한다. 이 process는
GPU를 사용하지 않고, complete dependency가 atomic publish된 sequence만 처리한다. 기존 valid
quality는 검증 후 skip하고, concurrent exporter/future supervisor와의 중복 write는 sequence-local
advisory lock으로 방지한다. 실패는 `.runtime/quality_follower_state.json`에 reason을 보존하고
5분 cooldown 후 자동 재시도한다.

Quality validation 후에는 동일 follower가 final exporter의 `validate_sequence()`를 호출해 pose/SAM
run provenance, target/pose/SAM frame/PTS, finite/NaN, triangulation/body/quality gate까지 미리
검증한다. PASS/REVIEW만 `freeze-ready`로 기록하며 INCOMPLETE dependency는 5분 유예
후에도 지속될 때, FAIL은 즉시 dashboard attention으로 올린다. 이로써 deadline
exporter와 같은 validation 계약을 사용하면서 payload copy는 deadline까지 지연한다.
Freeze-ready 후에도 dependency path/size/mtime signature를 lightweight 비교하며, completed payload가
바뀌면 해당 sequence만 validation을 다시 수행한다.

## 현재 결과

2026-08-12 10:46 KST 기준 완료된 11 sequence, 7,147 reference frame을 materialize했다.

- sequence REVIEW 11 / FAIL 0; frame PASS 1,019 / REVIEW 6,128 / FAIL 0
- exporter dependency validation reason 0
- target abstention/unmapped view 8, SAM rejected/unmapped view 8
- prior-only/body-missing/triangulation-missing joint frame 0
- `pushup_0001`의 기존 ambiguity 7은 identity/target/SAM rejection reason으로 그대로 유지
- 기존 body/camera REVIEW를 PASS로 승격하지 않음
- 11/11 completed quality sequence가 exporter preflight `freeze-ready REVIEW`; dependency reason/FAIL 0
- `latpulldown_0003` 662 frame은 occlusion/camera REVIEW를 보존하고 Mode C candidate 79를
  metadata로만 유지; Mode C 실행/채택 0

Output은 private ignored root에만 있고 public Git에는 code, schema, aggregate만 포함한다.
