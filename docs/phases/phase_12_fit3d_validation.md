# Phase 12 — Fit3D Validation

## 현재 상태

상태는 `METRICS_IMPLEMENTED_WAITING_DATASET`이다. 현재 workspace에는 Fit3D payload나
evidence-backed local mapping이 없으므로 실제 quantitative accuracy 숫자를 만들었다고 주장하지
않는다. Dataset freeze critical path를 방해하는 download/대규모 tolerance experiment도 시작하지
않았다.

## Metric contract

[`tools/evaluate_fit3d_metrics.py`](../../tools/evaluate_fit3d_metrics.py)는 prepared prediction/GT pair에
대해 세 지표를 분리한다.

- MPJPE: prediction/GT pelvis-root를 각 frame에서 뺀 뒤 거리
- N-MPJPE: root alignment 뒤 per-frame scale만 최적화
- PA-MPJPE: per-frame similarity Procrustes

따라서 MPJPE만 높고 N-MPJPE가 낮으면 scale, N-MPJPE가 높고 PA-MPJPE가 낮으면 global
orientation/alignment, PA-MPJPE도 높으면 pose/body error 가능성을 우선 검토한다. Metric 구현은
known scale/rotation synthetic regression으로 검증한다.

## Freeze 이후 확장

Fit3D access와 exact camera/joint convention을 확보하면 50→30 fps, 4→3 camera, 측정된 timing
offset/drift, JPEG/resolution과 camera perturbation을 단계적으로 적용한다. Rotation
0/0.5/1/2/5도와 translation tolerance curve는 private dataset freeze가 완료된 뒤 별도 validation
extension으로 실행한다. Fit3D GT camera에서만 camera error를 직접 측정했다고 표현한다.
