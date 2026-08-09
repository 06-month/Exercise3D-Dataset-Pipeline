# Phase 5 — Full Dataset Fixed-Camera Background BA

## Freeze 원칙

Phase 4에서 승인한 matcher, optimizer, threshold, heuristic과 weighting을 변경하지 않았다.
실행 default와 historical hash는 `configs/phase5_background_ba.json`에 기록했다.

## Dataset 결과

- sequences 26, cameras 78
- Phase 5.1 반영 후 PASS 11 / REVIEW 15 / FAIL 0
- Stage 1 26/26, Stage 2 26/26 convergence
- point 1,674 → 1,100
- extracted/final observations 16,835 → 11,046
- accepted residual mean 4.113→3.361 px
- median 3.630→2.584 px, p90 8.205→7.418 px, p95 9.850→9.953 px

cam1은 gauge라 변화가 0이다. cam2 rotation change median/p95는 0.457°/2.247°, cam3은
0.376°/3.074°였다. scale은 sequence-local arbitrary이며 initial cam1-cam2 baseline을 보존했다.

## REVIEW / FAIL

REVIEW는 제한된 track support, tail reprojection, direct three-camera track 부재 등의 uncertainty를
유지한다. `pushup_0003`은 Phase 5.1에서 동일 objective와 observation으로 Stage 2 budget만
확장해 수렴했으나 sparse support 때문에 REVIEW다. automatic fallback은 사용하지 않았다.

## Gate

26/26 sequence의 계산·파일 validation과 Stage 1/2 수렴을 확인했다. FAIL 0이므로 camera
geometry freeze를 승인한다. REVIEW sequence의 uncertainty/reason은 triangulation에 전달한다.
