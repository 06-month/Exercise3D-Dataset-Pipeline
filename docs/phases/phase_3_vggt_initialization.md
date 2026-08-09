# Phase 3 — VGGT-Ω Camera Geometry Initialization

## 범위

VGGT-Ω의 pose, intrinsic, depth, confidence, point map과 feature를 후속 Background BA의
initialization으로 생성한다. Bundle Adjustment, human fitting, SMPL, pseudo-label 생성은 없다.

## 공식 output 해석

- extrinsic: OpenCV world→camera `[R|t]`
- intrinsic: model canvas pixel unit
- depth: positive camera-Z, sequence-local arbitrary scale
- confidence: probability가 아닌 learned ranking score
- point map: depth/K/pose로 camera ray를 world에 역변환한 값

## 실행 및 결과

sequence당 representative PTS 8개와 camera 3대, 총 24 frames를 joint inference했다.

- sequence 26/26 SUCCESS
- camera 78/78 geometry
- sampled camera frames 624
- 실패/필수 payload 누락 0
- PASS 77 / REVIEW 1 (`squat_0001/cam2`)

## Visual Gate

Open3D로 point cloud, confidence, frustum과 world axis를 검사했다. 전역 mirror/180° flip 및
exploding cloud는 없었고 rough scene geometry는 유효했다. thin background sheet와 pose jitter가
있어 최종 camera로 직접 사용하지 않고 Background BA initialization으로만 승인했다.
