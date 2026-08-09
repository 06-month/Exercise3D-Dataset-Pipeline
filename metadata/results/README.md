# 공개 가능한 실험 결과 요약

이 디렉토리는 private media/geometry payload에서 파생한 비식별 수치 요약만 포함한다.

- `dataset_summary.csv`: dataset/exercise aggregate Background BA 결과
- `camera_summary.csv`: exact pose가 아닌 camera refinement 변화량
- `camera_uncertainty.csv`: residual/support/gating 기반 uncertainty metadata
- `review_sequences.csv`: REVIEW/FAIL 이유와 수치 요약
- `bundle_adjustment_statistics.csv`: sequence별 optimizer와 residual 통계

포함하지 않는 항목: absolute source/output path, exact camera matrices, frame image, depth,
point map, feature, track, checkpoint, 개인 식별 metadata. CSV의 sequence ID는 연구용 logical ID다.

Phase 5 결과는 PASS 11 / REVIEW 14 / FAIL 1이며, FAIL 결과를 downstream camera로 승인하지 않았다.
