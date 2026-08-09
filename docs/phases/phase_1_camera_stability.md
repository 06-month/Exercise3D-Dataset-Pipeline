# Phase 1 — EIS/OIS / Camera Stability Audit

## 목적

고정 tripod camera의 projection을 시간에 대해 고정할 수 있는지 검증한다.

## 방법

temporal foreground mask를 제외한 static background에서 LK track을 만들고 homography/affine
fit의 global motion, spatial residual과 반복성을 평가했다. native-adjacent pair와 long-baseline
pair를 함께 사용했다.

## 결과

- 78/78 `FIXED_CAMERA_OK`
- native-adjacent fit 8,087/8,087
- 반복 global/spatial warp evidence 없음
- foreground false positive 1건 수정 후 전체 재검증

## Gate

최종 geometry에는 physical camera별 pose 하나만 둔다. timestamp별 VGGT pose는 noisy prior일
뿐이며 independent final variable로 사용하지 않는다.
