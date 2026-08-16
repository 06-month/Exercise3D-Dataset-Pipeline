# Phase 상태 상세

README 요약에서 분리한 phase별 상태표와 실행 노트입니다. 시간순 실행 기록은
[process.md](../process.md), acceptance gate는 [plan.md](../plan.md)를 기준으로 합니다.

## Phase 표

| Phase | 상태 | 핵심 결과 |
|---|---|---|
| 0. Dataset Inventory / Integrity | DONE | 3명, 26 sequences, raw/sync 각 78 videos, working JPEG 65,595장, inventory PASS |
| 1. EIS/OIS / Camera Stability Audit | DONE | 78/78 `FIXED_CAMERA_OK`, native-adjacent fit 8,087/8,087 |
| 2. Temporal Synchronization QA | DONE | PTS offset 546건, median 11.99 ms, p95 25.28 ms, max 31.38 ms |
| 3. VGGT-Ω Initialization | DONE | 26/26 sequence, 78/78 camera, 624 sampled frames, 실패 0 |
| 3G. Open3D Visual Gate | DONE | 전역 mirror/180° flip/exploding cloud 없음, 1 PASS + 3 REVIEW 대표 검사 |
| 4. Fixed-Camera Background BA Pilot | DONE | 4 sequences, PASS 2 / REVIEW 2 / FAIL 0, Stage 1/2 모두 수렴 |
| 5. Full Dataset Background BA | DONE | 26 sequences 실행 완료, Phase 5.1 후 PASS 11 / REVIEW 15 / FAIL 0, Stage 1/2 26/26 |
| 5.1. `pushup_0003` Camera Recovery | DONE | 동일 objective에서 Stage 2 budget만 확장, 322 nfev에서 수렴, `RECOVERED_REVIEW` |
| 6-0. Sapiens2-5B Environment | DONE | A100 80GB smoke PASS, 308 keypoints, peak 19.986 GiB, 4.517 s/image |
| 6-1. Sapiens2 Pose Pilot | DONE | all-person baseline 보존, batch 1/2/4/8/12/16 완료 |
| 6-1A. Primary Target Selection | DONE | 9,732 frame, identity switch 0, ambiguity 7, crop 50.37% 감소 |
| 6-2. Target-only Runtime Gate | PARTIAL COMPLETE | selector `GO_FULL_DATASET`; target 65,430/65,595, pose 75/78 views·61,608 crops 완료 |
| 7. Timestamp-aware Triangulation | PARTIAL COMPLETE | 25/26 sequence, PASS 5 / REVIEW 20 / FAIL 0 |
| 8. SAM Body Runtime Feasibility | PARTIAL COMPLETE/REVIEW | checkpoint integrity PASS; Mode B 72/78 views·58,062 frames 완료 |
| 9. Sequence Body Fitting | PARTIAL COMPLETE/REVIEW | 24/26 sequence, PASS 3 / REVIEW 21 / FAIL 0 |
| 10. Body Shape / Proportion | IMPLEMENTED PARTIAL | sequence-level shape/scale provenance 보존; evidence-backed subject mapping 부재로 cross-sequence fusion 안 함 |
| 11. Pseudo-label Quality Control | PARTIAL COMPLETE | freeze-ready 24/26, REVIEW 24 / FAIL 0; scalar accuracy score 없음 |
| 12. Fit3D Validation | IMPLEMENTED/WAITING DATA | metric regression PASS; local Fit3D payload 부재로 실제 score 미주장 |
| 13. Final Dataset Freeze | DEADLINE SNAPSHOT COMPLETE | immutable deadline build: REVIEW 24 / INCOMPLETE 2 / FAIL 0; best 24-sequence checkpoint integrity PASS |

`pushup_0003`은 Phase 5.1에서 observation, initialization, objective와 gate를 그대로 두고
Stage 2 budget만 300에서 600으로 확장했습니다. 실제 322 evaluations에서 `xtol`로 수렴했고,
제한된 sparse support 때문에 `RECOVERED_REVIEW`로 유지합니다. Dataset-level FAIL은 0이며
camera geometry freeze는 REVIEW uncertainty 전파 조건으로 승인되었습니다.

2026-08-14 13:00 KST deadline에는 24개 sequence가 end-to-end 완료됐고 `deadlift_0002`,
`squat_0003`은 `INCOMPLETE_DEADLINE` provenance로 남았습니다. 이 시점의 immutable build는
요청된 26개 상태를 REVIEW 24 / INCOMPLETE 2 / FAIL 0으로 고정했습니다. 이후 충분한 GPU 자원이
확보되면 5B/flip-test/abstention 설정을 유지하고 completion metadata가 유효한 output은 건너뛰면서
두 sequence의 미완료 stage만 재개합니다.

Phase 7 pilot에서는 2D target이 정상인데도 `squat_0001`과 `pushup_0001`의 current camera와
epipolar consistency가 무너지는 새 evidence가 확인됐습니다. 해당 3D proposal은 fitting/export에서
제외했고, 원본 Phase 5 camera를 덮어쓰지 않는 recovery와 held-out 검증을 요구합니다.
자세한 내용은 [Phase 7 문서](phases/phase_7_triangulation.md)에 있습니다.

Phase 8 primary-target pilot는 control/severe 두 clip의 Mode A/B/C 여섯 run을 완료했습니다.
SAM full-stage projection은 16.35/20.80/32.63시간, Sapiens2 target-only와 한 GPU에서 순차
실행하는 합계는 95.43/99.88/111.71시간입니다. Mode C는 약 2배 느렸지만 이번 severe clip에서
content completion이 호출되지 않아 선택적 refiner 정책은 `REVIEW`로 유지합니다. full 기본은
Mode B이며 Mode C는 evidence가 있는 frame/sequence만 selective escalation합니다. 상세 근거는
[Phase 8 문서](phases/phase_8_sam_body4d.md)에 있습니다.

Full inference가 진행되는 동안 CPU에서는 세 view가 완결된 sequence부터 Phase 7을 자동 실행합니다.
SAM compact prior는 MHR pose/shape/hand/expression/joint/model parameter와 source PTS를 보존하며,
Phase 9는 triangulated geometry를 dominant observation으로 두는 staged fit만 허용합니다. 최종 private
export는 source RGB를 포함하지 않고 stage payload의 byte equality와 SHA-256, PASS/REVIEW/FAIL/
INCOMPLETE 상태를 versioned manifest에 기록합니다.

첫 end-to-end `barbellrow_0000`은 Mode B 3-view 1,770 frame과 body fit 590 timestamp × 26 joint를
완료했습니다. Numeric/mesh/provenance와 finite/NaN contract는 PASS했지만 camera REVIEW와 normalized
geometry displacement p95 0.05167을 그대로 전파해 sequence는 `REVIEW_BODY_FIT_QUALITY`입니다.
Complete sequence 하나만 사용한 private export smoke는 34 files의 size/SHA-256 불일치 없이
freeze-eligible이었으며, REVIEW를 PASS로 승격하지 않았습니다.
강화된 exact-tree exporter smoke는 Phase 11 quality 두 파일을 추가한 36 files/
28,993,394 bytes를 전수 검증했고, clean commit provenance와 immutable read-only reuse까지 PASS했습니다.

두 번째 `squat_0001`도 Mode B 3-view 3,801/3,801 frame과 1,267×26 body fit을 완료했습니다.
Body fit coverage/alignment는 1.0이고 prior-only joint는 0이지만 normalized displacement p95
0.07936과 camera uncertainty를 전파해 REVIEW로 유지합니다. Mode C 후보는 0으로
`PASS_MODE_B_FROZEN`이며 expensive Mode C를 실행하지 않았습니다.

장기 generation job은 deadline snapshot 이후 중단된 상태이며 자동 completion을 주장하지 않습니다.
GPU 환경이 다시 준비되면 singleton/exact-command gate를 확인하고 동일 selection-bound 설정으로
resume한 뒤 Phase 7 → SAM Mode B → prior consolidation → body fit → quality/private export를
sequence별로 이어갑니다. Full SAM 직전에는 8-frame Mode B smoke로 PTS/mesh/MHR numeric schema를
실제 GPU에서 검사합니다. Mode C는 자동 full mode가 아니며
[`configs/sam_mode_c_escalation.json`](../configs/sam_mode_c_escalation.json)의 occlusion+failure/outlier
조건과 B/C 개선 gate를 모두 통과할 때만 선택 후보입니다.
각 full Mode B sequence 뒤에는 이 조건을 실제로 평가해 후보 frame/clip 또는
`PASS_MODE_B_FROZEN`을 private export provenance에 포함합니다.

세부 상태와 acceptance gate는 [plan.md](../plan.md), 시간순 실행 기록은
[process.md](../process.md)를 기준으로 합니다.
