# 파이프라인 상태

Deadline snapshot(2026-08-14 13:00 KST) 기준 상태입니다. acceptance gate는 [plan.md](plan.md),
시간순 실행 기록은 [process.md](process.md)에 있습니다.

## 요약

| | |
|---|---|
| end-to-end 완료 | **24 / 26 sequences** |
| quality 상태 | REVIEW 24 / FAIL 0 |
| 미완료 | `deadlift_0002`, `squat_0003` (`INCOMPLETE_DEADLINE`) |
| immutable deadline build | REVIEW 24 / INCOMPLETE 2 / FAIL 0, checkpoint integrity PASS |

REVIEW를 PASS로 승격하거나 미완료를 완료로 표시하지 않았습니다.

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

## 알려진 이슈와 결정

### Phase 5.1 — `pushup_0003` camera recovery

observation, initialization, objective와 gate는 그대로 두고 Stage 2 budget만 300에서 600으로
확장했습니다. 실제 322 evaluations에서 `xtol`로 수렴했지만 sparse support가 제한적이라
`RECOVERED_REVIEW`로 유지합니다. dataset-level FAIL은 0이고, camera geometry freeze는 REVIEW
uncertainty를 전파하는 조건으로 승인했습니다.

### Phase 7 — epipolar consistency 붕괴

2D target이 정상인데도 `squat_0001`, `pushup_0001`의 current camera와 epipolar consistency가
무너지는 evidence를 확인했습니다. 해당 3D proposal은 fitting/export에서 제외했고, 원본 Phase 5
camera를 덮어쓰지 않는 recovery와 held-out 검증을 요구합니다.
자세한 내용: [Phase 7 문서](phases/phase_7_triangulation.md).

### Phase 8 — Mode C 정책은 REVIEW

control/severe 두 clip의 Mode A/B/C 여섯 run을 완료했습니다. SAM full-stage projection은
16.35 / 20.80 / 32.63시간이고, Sapiens2 target-only와 한 GPU에서 순차 실행하면 합계
95.43 / 99.88 / 111.71시간입니다. Mode C는 약 2배 느렸지만 severe clip에서 content completion이
호출되지 않아 선택적 refiner 정책은 `REVIEW`로 남깁니다. full 기본은 Mode B이며, Mode C는
[`configs/sam_mode_c_escalation.json`](../configs/sam_mode_c_escalation.json)의 occlusion+failure/outlier
조건과 B/C 개선 gate를 모두 통과한 frame/sequence만 selective escalation합니다. full Mode B sequence
뒤에는 이 조건을 실제로 평가해 후보 frame/clip 또는 `PASS_MODE_B_FROZEN`을 export provenance에 넣습니다.

### Phase 10 — cross-sequence fusion 보류

sequence-level shape/scale provenance는 보존하지만, evidence-backed subject mapping이 없어
cross-sequence shape fusion은 수행하지 않았습니다.

## 완료 sequence 사례

**`barbellrow_0000`** (첫 end-to-end) — Mode B 3-view 1,770 frame, body fit 590 timestamp × 26 joint.
numeric/mesh/provenance와 finite/NaN contract는 PASS했지만 camera REVIEW와 normalized geometry
displacement p95 0.05167을 전파해 `REVIEW_BODY_FIT_QUALITY`입니다. 이 sequence 하나로 실행한 private
export smoke는 34 files의 size/SHA-256 불일치가 없었고, 강화된 exact-tree exporter smoke는 Phase 11
quality 두 파일을 더한 36 files / 28,993,394 bytes를 전수 검증했습니다.

**`squat_0001`** — Mode B 3-view 3,801/3,801 frame, body fit 1,267 × 26 joint. coverage/alignment 1.0,
prior-only joint 0이지만 normalized displacement p95 0.07936과 camera uncertainty를 전파해 REVIEW입니다.
Mode C 후보 0으로 `PASS_MODE_B_FROZEN`이며 expensive Mode C를 실행하지 않았습니다.

## Resume 계획

장기 generation job은 deadline snapshot 이후 중단된 상태이며 자동 completion을 주장하지 않습니다.
GPU 환경이 다시 준비되면:

1. singleton/exact-command gate를 확인한다 ([operations.md](operations.md)).
2. 동일 selection-bound 설정으로 resume하고, completion metadata가 유효한 output은 건너뛴다.
3. full SAM 직전에 8-frame Mode B smoke로 PTS/mesh/MHR numeric schema를 실제 GPU에서 검사한다.
4. sequence별로 Phase 7 - SAM Mode B - prior consolidation - body fit - quality/private export를 잇는다.

파이프라인 중간 계약은 유지됩니다. SAM compact prior는 MHR pose/shape/hand/expression/joint/model
parameter와 source PTS를 보존하고, Phase 9는 triangulated geometry를 dominant observation으로 두는
staged fit만 허용하며, 최종 private export는 source RGB 없이 stage payload의 byte equality와 SHA-256,
PASS/REVIEW/FAIL/INCOMPLETE 상태를 versioned manifest에 기록합니다.
