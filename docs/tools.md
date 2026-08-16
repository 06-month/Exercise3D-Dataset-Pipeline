# 도구 목록

각 tool의 역할과 source mutation 여부입니다.

| 파일 | 역할 | source mutation |
|---|---|---|
| `scripts/sync_videos.py` | clap/audio 기반 영상 synchronization | 새 derivative 생성 |
| `scripts/build_dataset.py` | sync/frame/manifest orchestration | 지정 output에만 생성 |
| `tools/dataset_inventory.py` | raw/sync/frame provenance inventory | 없음 |
| `tools/eis_background_audit.py` | fixed-camera stability 분석 | 없음 |
| `tools/temporal_sync_audit.py` | PTS/audio/visual offset 및 drift QA | 없음 |
| `tools/vggt_geometry_init.py` | VGGT-Ω initialization | 새 output에만 생성 |
| `tools/visualize_vggt.py` | Open3D geometry QA | optional debug output만 생성 |
| `tools/background_bundle_adjust.py` | shared physical-camera Background BA | 새 output에만 생성 |
| `tools/finalize_background_ba_dataset.py` | dataset-level BA validation/report | BA output metadata 생성 |
| `tools/analyze_background_ba_recovery.py` | Stage 2 budget-only recovery 재현·동일성 검증 | BA output metadata 생성 |
| `tools/sapiens2_pose_smoke.py` | 공식 Sapiens2 5B + DETR single-image smoke/VRAM/latency | optional ignored output만 생성 |
| `tools/sapiens2_pose_pipeline.py` | all-person 5B batch benchmark와 resumable baseline pilot | ignored output만 생성 |
| `tools/detr_person_candidates.py` | explicit sequence allowlist의 official DETR detection-only pass | ignored private output만 생성 |
| `tools/target_subject_selection.py` | all DETR candidate 보존 + bidirectional primary target tracking | ignored private output + aggregate report |
| `tools/sapiens2_target_pipeline.py` | accepted target-only 5B batch benchmark/inference/verification | ignored output만 생성 |
| `tools/validate_target_selection_full.py` | DETR candidate lossless 보존·identity/abstention full gate | ignored aggregate 생성 |
| `tools/summarize_phase6_1.py` | all-person/target-only 비교, ETA와 acceptance gate 집계 | redacted aggregate 생성 |
| `tools/triangulate_sapiens2.py` | PTS-aware weighted triangulation과 pose-camera consistency gate | ignored private output 생성 |
| `tools/recover_cameras_from_pose_observations.py` | NO_GO camera의 별도 observation-conditioned/held-out recovery | ignored private output 생성 |
| `tools/run_phase7_streaming.py` | pose-complete sequence의 triangulation/recovery 자동 streaming | ignored private output 생성 |
| `tools/benchmark_sam_body4d.py` | SAM-Body4D checkpoint preflight와 refiner on/off runtime 측정 | ignored output만 생성 |
| `tools/sam_body_primary_target_runner.py` | primary bbox 1개 adapter와 compact MHR parameter provenance 저장 | ignored private output만 생성 |
| `tools/run_sam_body4d_full.py` | Mode B camera 단위 resume/completeness orchestration | ignored private output만 생성 |
| `tools/run_autonomous_generation.py` | Sapiens resume부터 Phase 7–13까지 장시간 critical path supervision | ignored private output 생성 |
| `tools/monitor_autonomous_generation.py` | 기존 runtime/process/GPU를 읽는 live dashboard와 atomic attention state | ignored `.runtime/dashboard_state.json` 생성 |
| `tools/consolidate_sam_body_prior.py` | frame/PTS/identity-aware MHR numeric prior 통합 | ignored private output 생성 |
| `tools/assess_sam_mode_c_escalation.py` | Mode B failure/outlier 기반 bounded Mode C review clip 선정 | ignored private output 생성 |
| `tools/verify_mhr_parameter_replay.py` | compact 204-d MHR parameter의 official model exact replay 검사 | ignored aggregate 생성 |
| `tools/fit_sequence_body.py` | geometry-dominant staged sequence body fit과 S0 | ignored private output 생성 |
| `tools/build_pseudolabel_quality.py` | target/pose/SAM/geometry/body evidence의 frame/sequence quality vector | ignored private output 생성 |
| `tools/run_quality_control_follower.py` | 완료 body-fit을 감지하는 CPU-only Phase 11 follower | ignored runtime/quality output 갱신 |
| `tools/run_quality_control_follower_watchdog.py` | quality follower exact-identity/absence recovery watchdog | ignored runtime state/log 갱신 |
| `tools/run_predeadline_checkpoint_follower.py` | 증가한 freeze-ready 집합만 immutable checkpoint로 보존하는 CPU-only follower | ignored runtime/private freeze 갱신 |
| `tools/run_predeadline_checkpoint_follower_watchdog.py` | checkpoint follower exact-identity/absence recovery watchdog | ignored runtime state/log 갱신 |
| `tools/export_private_dataset.py` | versioned private dataset export와 byte/SHA/schema 검증 | ignored private output 생성 |
| `tools/evaluate_fit3d_metrics.py` | prepared Fit3D pair의 MPJPE/N-MPJPE/PA-MPJPE 분리 평가 | ignored aggregate 생성 |
| `tools/summarize_sam_body_runtime.py` | A/B/C ratio, occlusion 증가와 best/expected/worst runtime 집계 | redacted aggregate 생성 |
| `tools/check_publication_safety.py` | staged/tracked 공개 안전 검사 | 없음 |
