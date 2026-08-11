# 공개 가능한 실험 결과 요약

이 디렉토리는 private media/geometry payload에서 파생한 비식별 수치 요약만 포함한다.

- `dataset_summary.csv`: dataset/exercise aggregate Background BA 결과
- `camera_summary.csv`: exact pose가 아닌 camera refinement 변화량
- `camera_uncertainty.csv`: residual/support/gating 기반 uncertainty metadata
- `review_sequences.csv`: REVIEW/FAIL 이유와 수치 요약
- `bundle_adjustment_statistics.csv`: sequence별 optimizer와 residual 통계
- `pushup_0003_recovery.csv`: Phase 5 baseline과 budget-only recovery의 직접 비교
- `target_subject_selection_pilot.csv`: private bbox를 제외한 Phase 6 target identity aggregate
- `sapiens2_target_only_batch_scaling.csv`: target-only batch throughput/resource/equivalence
- `phase6_2_runtime_projection.csv`: stage별 65,595-frame runtime 외삽
- `sam_body4d_checkpoint_manifest.csv`: Phase 8 공식 checkpoint 용량·경로·접근 조건
- `sam_body4d_preflight.csv`: local checkpoint availability와 pilot 실행 가능 상태

포함하지 않는 항목: absolute source/output path, exact camera matrices, frame image, depth,
point map, feature, track, checkpoint, 개인 식별 metadata. CSV의 sequence ID는 연구용 logical ID다.

Phase 5.1 recovery 반영 결과는 PASS 11 / REVIEW 15 / FAIL 0이다. `pushup_0003`은
`RECOVERED_REVIEW`이며 fallback을 사용하지 않았다. REVIEW uncertainty는 downstream에 전달한다.
