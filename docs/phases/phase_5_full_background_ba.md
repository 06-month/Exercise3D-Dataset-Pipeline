# Phase 5 — Full Dataset Fixed-Camera Background BA

## Freeze 원칙

Phase 4에서 승인한 matcher, optimizer, threshold, heuristic과 weighting을 변경하지 않았다.
실행 default와 historical hash는 `configs/phase5_background_ba.json`에 기록했다.

## Dataset 결과

- sequences 26, cameras 78
- PASS 11 / REVIEW 14 / FAIL 1
- Stage 1 26/26, Stage 2 25/26 convergence
- point 1,674 → 1,100
- extracted/final observations 16,835 → 11,046
- accepted residual mean 4.113→3.361 px
- median 3.630→2.582 px, p90 8.205→7.425 px, p95 9.850→9.953 px

cam1은 gauge라 변화가 0이다. cam2 rotation change median/p95는 0.457°/2.247°, cam3은
0.376°/3.074°였다. scale은 sequence-local arbitrary이며 initial cam1-cam2 baseline을 보존했다.

## REVIEW / FAIL

REVIEW는 제한된 track support, tail reprojection, direct three-camera track 부재 등의 uncertainty를
유지한다. `pushup_0003`은 Stage 2가 `max_nfev=300`에서 미수렴해 FAIL이다. 알고리즘 freeze를
지키기 위해 silent threshold change나 automatic fallback은 하지 않았다.

## Gate

전체 계산과 파일 validation은 완료됐지만 final camera freeze는 REVIEW다. FAIL camera를
triangulation에 사용하지 않으며 제외/fallback/별도 재최적화 정책을 먼저 결정한다.
