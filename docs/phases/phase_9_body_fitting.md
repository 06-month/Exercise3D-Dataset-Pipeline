# Phase 9 — Sequence-level Body Fitting

## 상태와 역할

상태는 `IN_PROGRESS_REVIEW`다. 구현과 synthetic test에 더해 첫 full SAM Mode B sequence의
실제 fit을 완료했으며, 나머지는 Mode B input이 생성되는 순서대로 streaming한다.

Phase 9의 목표는 SAM 출력을 복사해 GT로 부르는 것이 아니다. timestamp-aware 3-view geometry를
가장 강한 observation으로 유지하면서 MHR body/pose와 시간 연속성을 약한 prior로 결합한다.
Sapiens2와 SAM 계열은 correlated learned error를 가질 수 있으므로 두 model의 agreement를 독립적인
두 GT의 agreement로 해석하지 않는다.

## Staged fit

[`tools/fit_sequence_body.py`](../../tools/fit_sequence_body.py)는 다음 순서를 고정한다.

1. Phase 7 canonical 3D joint와 quality를 sequence-local arbitrary-scale geometry anchor로 사용
2. 각 camera/frame의 accepted MHR70 canonical prior를 최소 8개 core joint로 robust similarity alignment
3. geometry weight를 MHR view당 weight보다 크게 유지한 weak prior fusion
4. weighted second-difference temporal fit
5. sequence median shape/scale parameter와 scale-invariant `S0` 산출

Triangulated joint가 없는 경우 한 view의 prior만으로 채우지 않는다. timestamp gate 안에서 최소 두
view의 aligned prior가 존재할 때만 `ALIGNED_SAM_PRIOR_ONLY_AT_LEAST_TWO_VIEWS`로 생성하며 confidence
상한을 낮게 둔다. 최종 joint에는 geometry-only, geometry+prior, prior-only, missing evidence code와
observation/prior/temporal residual이 함께 저장된다.

## Representation과 재현성

MHR70→Exercise3D canonical mapping은
[`configs/mhr70_canonical_joints.json`](../../configs/mhr70_canonical_joints.json)에 index/name으로
명시했다. [`tools/verify_mhr_parameter_replay.py`](../../tools/verify_mhr_parameter_replay.py)로 측정한
MHR 204-d model parameter replay의 keypoint/mesh numerical difference는 각각 최대
`2.68e-7 m`, `7.15e-7 m`였다. 이는 parameter serialization의 재현성을 검증할 뿐 model accuracy나
GT 정확도를 검증하는 수치는 아니다.

`S0`의 reference는 sequence median left/right femur length의 평균이다. femur/tibia/torso/shoulder/
hip/arm length를 이 reference로 나누며 MHR beta/shape parameter와 별도 의미로 저장한다.

## Acceptance

- valid point finite, invalid point NaN contract
- frame/PTS와 canonical joint convention exact match
- Phase 7 `eligible_for_body_fitting=true`
- alignment success/normalized residual과 geometry displacement 분포
- prior-only/missing joint 수와 temporal/bone-length consistency
- camera가 REVIEW 또는 observation-conditioned이면 body fit도 REVIEW 유지

Full input이 생기면 sequence 단위로 위 gate를 실행하고 FAIL을 숨기지 않는다.

## First full-input result

`barbellrow_0000`의 590 reference timestamp × 26 canonical joint를 처리했다.

- final joint coverage 1.0, SAM alignment success 1.0, prior-only fraction 0
- median bone-length CV 0.01738
- valid finite/invalid NaN contract PASS
- normalized observation displacement p95 0.05167
- result: `REVIEW_BODY_FIT_QUALITY`, FAIL 0

Displacement p95가 사전 동결 REVIEW 경계 0.05를 소폭 넘고 upstream camera도 REVIEW이므로
자동 PASS로 승격하지 않았다. Mode C assessor가 고른 84 frame은 주로 sequence 경계의 temporal
outlier이며, current Mode B output을 교체할 품질 증거로 사용하지 않는다.

## 사전 동결 quality gate

Full Mode B 결과를 보기 전에 [`configs/phase9_body_fit.json`](../../configs/phase9_body_fit.json)에
threshold를 고정했다. Final joint coverage 95%, SAM alignment success 90%, normalized geometry
displacement p95 0.05, prior-only fraction 2%, median bone-length CV 0.10 밖은 REVIEW다. Coverage 80%
미만, displacement p95 0.20 초과, anthropometric reference invalid 또는 finite/NaN schema 실패는
FAIL이다. Camera status가 PASS가 아니면 다른 지표가 좋아도 REVIEW를 유지한다.
