# 도구 목록

`tools/`의 모든 실행 스크립트를 파이프라인 단계별로 묶었습니다. `source mutation` 열은 그 tool이
raw/synchronized/working source를 건드리는지를 나타냅니다. 모든 tool은 `--help`를 지원하고,
private 경로는 CLI 인자나 환경변수로만 주입합니다.

## 1. Dataset build / inventory

| 파일 | 역할 | source mutation |
|---|---|---|
| `sync_videos.py` | clap(박수) 기반 multi-view 영상 synchronization | 새 derivative 생성 |
| `build_dataset.py` | sync → frame → triplet dataset 배치 빌드 | 지정 output에만 생성 |
| `verify_dataset.py` | 완성된 dataset 검수 | 없음 |
| `check_frame_mapping.py` | 출력 frame이 원본 몇 번째 frame인지 연속 추적 | 없음 |
| `measure_frame_alignment.py` | 출력 frame이 실제로 같은 순간인지 frame 단위 실측 | 없음 |
| `dataset_inventory.py` | raw/sync/frame provenance inventory | 없음 |

## 2. Camera geometry (Phase 1–5)

| 파일 | 역할 | source mutation |
|---|---|---|
| `eis_background_audit.py` | fixed-camera stability(EIS/OIS) 분석 | 없음 |
| `temporal_sync_audit.py` | PTS/audio/visual offset과 drift QA | 없음 |
| `vggt_geometry_init.py` | VGGT-Ω camera/depth initialization | 새 output에만 생성 |
| `visualize_vggt.py` | Open3D geometry QA | optional debug output |
| `background_bundle_adjust.py` | shared physical-camera Background BA | 새 output에만 생성 |
| `finalize_background_ba_dataset.py` | dataset-level BA validation/report | BA output metadata |
| `analyze_background_ba_recovery.py` | Stage 2 budget-only recovery 재현·동일성 검증 | BA output metadata |
| `recover_cameras_from_pose_observations.py` | NO_GO camera의 observation-conditioned/held-out recovery | ignored private output |

## 3. 2D pose / primary target (Phase 6)

| 파일 | 역할 | source mutation |
|---|---|---|
| `sapiens2_pose_smoke.py` | Sapiens2 5B + DETR single-image smoke, VRAM/latency 측정 | optional ignored output |
| `sapiens2_pose_pipeline.py` | all-person 5B batch benchmark와 resumable baseline pilot | ignored output |
| `detr_person_candidates.py` | sequence allowlist 기반 official DETR detection-only pass | ignored private output |
| `target_subject_selection.py` | DETR candidate 전량 보존 + bidirectional primary target tracking | ignored private output + aggregate |
| `sapiens2_target_pipeline.py` | accepted target-only 5B inference/benchmark/verification | ignored output |
| `validate_target_selection_full.py` | candidate lossless 보존·identity/abstention full gate | ignored aggregate |
| `summarize_phase6_1.py` | all-person과 target-only 비교, ETA와 acceptance gate 집계 | redacted aggregate |

## 4. 3D geometry / body (Phase 7–10)

| 파일 | 역할 | source mutation |
|---|---|---|
| `triangulate_sapiens2.py` | PTS-aware weighted triangulation과 pose-camera consistency gate | ignored private output |
| `benchmark_triangulation.py` | timestamp-aware 3-view DLT synthetic throughput benchmark | 없음 |
| `run_phase7_streaming.py` | pose-complete sequence의 triangulation/recovery 자동 streaming | ignored private output |
| `benchmark_sam_body4d.py` | SAM-Body4D checkpoint preflight와 refiner on/off runtime 측정 | ignored output |
| `sam_body_primary_target_runner.py` | primary bbox 1개 adapter와 compact MHR parameter provenance | ignored private output |
| `run_sam_body4d_full.py` | Mode B camera 단위 resume/completeness orchestration | ignored private output |
| `consolidate_sam_body_prior.py` | frame/PTS/identity-aware MHR numeric prior 통합 | ignored private output |
| `assess_sam_mode_c_escalation.py` | Mode B failure/outlier 기반 bounded Mode C clip 선정 | ignored private output |
| `verify_mhr_parameter_replay.py` | compact 204-d MHR parameter의 official model exact replay 검사 | ignored aggregate |
| `fit_sequence_body.py` | geometry-dominant staged sequence body fit과 S0 | ignored private output |
| `summarize_sam_body_runtime.py` | Mode A/B/C ratio와 best/expected/worst runtime 집계 | redacted aggregate |

## 5. Quality / export / validation (Phase 11–13)

| 파일 | 역할 | source mutation |
|---|---|---|
| `build_pseudolabel_quality.py` | target/pose/SAM/geometry/body evidence의 frame·sequence quality vector | ignored private output |
| `run_quality_control_follower.py` | 완료 body-fit을 감지하는 CPU-only Phase 11 follower | ignored runtime/quality output |
| `run_quality_control_follower_watchdog.py` | quality follower exact-identity/absence recovery | ignored runtime state/log |
| `materialize_inference_provenance.py` | 완료 inference의 camera-level config provenance sidecar 생성 | ignored private output |
| `export_private_dataset.py` | versioned private export와 byte/SHA-256/schema 검증 | ignored private output |
| `run_deadline_snapshot.py` | 고정 deadline의 non-destructive private snapshot build | ignored private output |
| `run_deadline_sentinel_watchdog.py` | snapshot sentinel의 중복 없는 recovery | ignored runtime state/log |
| `run_predeadline_checkpoint_follower.py` | 늘어난 freeze-ready 집합만 immutable checkpoint로 보존 | ignored runtime/private freeze |
| `run_predeadline_checkpoint_follower_watchdog.py` | checkpoint follower exact-identity/absence recovery | ignored runtime state/log |
| `evaluate_fit3d_metrics.py` | prepared Fit3D pair의 MPJPE/N-MPJPE/PA-MPJPE 분리 평가 | ignored aggregate |
| `prepare_transfer_snapshot.py` | durable freeze gate 이후 resumable transfer snapshot 준비 | ignored staging 생성 |

## 6. Runtime supervision

운영 규칙은 [operations.md](operations.md)에 있습니다.

| 파일 | 역할 | source mutation |
|---|---|---|
| `run_autonomous_generation.py` | Sapiens resume부터 Phase 7–13까지 critical path supervision | ignored private output |
| `run_autonomous_supervisor_watchdog.py` | live job 중복 없이 supervisor 복구 | ignored runtime state/log |
| `monitor_autonomous_generation.py` | 기존 runtime/process/GPU를 읽는 live dashboard | ignored `.runtime/dashboard_state.json` |
| `run_monitoring_watchdog.py` | dashboard/handoff monitor의 중복 없는 복구 | ignored runtime state/log |
| `checkpoint_handoff_state.py` | live private 운영 상태의 atomic handoff checkpoint | ignored `.runtime` 상태 |

## 7. Rendering / publication

| 파일 | 역할 | source mutation |
|---|---|---|
| `render_public_mesh_showcase.py` | mesh-only render frame으로 공개용 3-view 영상 생성 (source RGB 입력 없음) | 없음 |
| `render_completed_label_overlay.py` | 완료 label의 로컬 전용 3-view overlay 렌더 (비공개) | 없음 |
| `check_publication_safety.py` | staged/tracked 공개 안전 검사 | 없음 |
