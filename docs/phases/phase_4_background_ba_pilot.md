# Phase 4 — Fixed-Camera Background BA Pilot

## Physical-camera parameterization

각 camera의 8개 VGGT pose를 robust SO(3)/translation aggregation하고 cam1을 identity gauge로
고정했다. optimization camera 변수는 cam2/cam3 shared extrinsic뿐이며, timestamp pose는 없다.

## Static evidence와 objective

- temporal median/MAD, large difference component, confidence percentile, border mask
- 같은 camera에서 3 timestamps 이상 지속하는 SIFT landmark
- SIFT ratio + USAC_MAGSAC + VGGT-init epipolar/point-map consistency
- fixed intrinsics, Huber robust reprojection, weak pose/point prior, Stage 1/2 sample gate

## Pilot 결과

| sequence | status | 핵심 관찰 |
|---|---|---|
| `barbellrow_0000` | PASS | 충분한 static support와 안정적인 rig |
| `squat_0001` | REVIEW | median 개선, p95 tail과 제한된 support |
| `pushup_0001` | REVIEW | 큰 개선이나 direct 3-camera track 부족 |
| `benchpress_0003` | PASS | occlusion에도 pilot 최대 support |

모든 Stage 1/2가 수렴했다. `squat_0001/cam2`의 6.4 s pose outlier는 수동 hard-code 없이
REJECT됐고, 같은 sample의 유효한 background observation은 유지됐다.

## Gate

PASS 2 / REVIEW 2 / FAIL 0. 동일 default로 Phase 5 전체 확장을 승인했다.
