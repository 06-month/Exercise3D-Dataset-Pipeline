# Exercise3D 데이터셋 구축 Canonical Plan

상태 표기: `TODO`, `IN_PROGRESS`, `DONE`, `REVIEW`. 각 Phase는 구현, 정량·시각 검증,
문서 갱신, 공개 안전 검사, commit/push까지 완료되어야 Definition of Done을 만족한다.

## 현재 Gate

- Phase 0–4: `DONE`
- Phase 5 dataset-wide 실행: `DONE`
- Phase 5.1 `pushup_0003` recovery: `DONE` (`RECOVERED_REVIEW`)
- Phase 5 downstream camera freeze gate: `DONE` (REVIEW uncertainty 전파 조건)
- 최종 camera status: PASS 11 / REVIEW 15 / FAIL 0, Stage 1/2 26/26 수렴
- Phase 6: `TODO`, camera quality metadata를 입력으로 pilot 시작 가능

## Phase 0 — Dataset Inventory / Integrity

- 상태: `DONE`
- 목적: raw, synchronized derivative, working frame의 수량·timing·provenance 무결성 확정
- 입력: private raw/synchronized videos, working JPEG, manifest
- 출력: inventory CSV/JSON/summary와 immutable source hash provenance
- 방법: `ffprobe` stream/packet PTS, 파일 수, frame 수, camera mapping 전수 검사
- Acceptance: raw 78, sync 78, working JPEG 65,595, triple-view 26 sequence, 누락·중복 없음
- 결과: PASS. 30/30/60 fps source와 30 fps working derivative의 관계를 기록하고 raw 60 fps 유지
- 다음 gate: source mutation 0건 확인 후 camera stability audit

## Phase 1 — EIS/OIS / Camera Stability Audit

- 상태: `DONE`
- 목적: 시간에 따라 camera projection이 변하는 반복적 warp/electronic stabilization 확인
- 입력: synchronized videos
- 출력: camera별 motion fit, residual, recommendation, QA figure
- 방법: temporal foreground rejection, LK tracks, homography/affine fit, native-adjacent 및 장구간 비교
- Acceptance: 반복 global/spatial warp evidence가 없고 각 physical camera를 fixed pose로 모델링 가능
- 결과: 78/78 `FIXED_CAMERA_OK`, native-adjacent 8,087/8,087 성공, foreground 오탐 1건 수정
- 다음 gate: timestamp별 독립 camera 변수를 금지하고 temporal QA 수행

## Phase 2 — Temporal Synchronization QA

- 상태: `DONE`
- 목적: 시작점뿐 아니라 beginning/middle/end 전체의 offset과 clock drift 측정
- 입력: synchronized videos, raw PTS, working-frame provenance, clap metadata
- 출력: pairwise offset, drift curve, confidence, sequence classification
- 방법: packet/frame PTS 우선, audio cross-correlation, visual motion energy를 상호 검증
- Acceptance: 모든 pair의 evidence와 `TEMPORALLY_STABLE`, `SMALL_CONSTANT_OFFSET`,
  `CLOCK_DRIFT_DETECTED`, `INSUFFICIENT_EVIDENCE` 중 하나의 판정
- 결과: 26 sequences/78 pairs, actual-frame PTS 546건, median 11.99 ms, p95 25.28 ms,
  max 31.38 ms. stable 8, constant 16, drift 2, insufficient 0
- Review: `pushup_0000`, `squat_0001`
- 다음 gate: RGB 재생성 없이 correction metadata만 geometry/matching에 전달

## Phase 3 — VGGT-Ω Camera Geometry Initialization

- 상태: `DONE`
- 목적: 후속 BA용 pose, K, depth, confidence, point map initialization 생성
- 입력: sequence당 representative PTS 8개 × cameras 3대, local official implementation/checkpoint
- 출력: pose/K/depth/confidence/point map/feature, frames.csv, metadata
- 방법: 24 frames joint inference, official preprocessing/output convention 보존
- Acceptance: sequence/camera별 finite value, timestamp provenance, 필수 payload와 status 검증
- 결과: 26/26 sequence, 78/78 camera, 624 sampled frames SUCCESS; PASS 77 / REVIEW 1
- Review: `squat_0001/cam2`
- 다음 gate: initialization을 최종 camera로 간주하지 않고 Open3D visual inspection

## Phase 3G — VGGT Open3D Visual Inspection Gate

- 상태: `DONE`
- 목적: point cloud, camera frustum, confidence와 cross-view coherence를 사람이 재현 가능하게 검사
- 입력: Phase 3 outputs와 source PTS mapping
- 출력: `tools/visualize_vggt.py`, visual checklist/report
- 방법: OpenCV world→camera를 `C=-Rᵀt`로 변환, percentile confidence filter, voxel/random sampling
- Acceptance: mirror/180° flip/exploding cloud 여부와 representative PASS/REVIEW 기록
- 결과: 전역 flip/explosion 없음. `barbellrow_0000` PASS, `squat_0001`, `pushup_0001`,
  `benchpress_0003` REVIEW. thin geometry sheet와 pose jitter 확인
- 다음 gate: 조건부 Background BA initialization으로 사용

## Phase 4 — Fixed-Camera Background BA Pilot

- 상태: `DONE`
- 목적: noisy timestamp-level VGGT pose를 physical camera별 shared pose 1개로 정제
- 입력: Phase 2 timing metadata, Phase 3 geometry, Phase 1 fixed-camera evidence
- 출력: refined K/R/t, sparse static points, track/residual/gating/validation metadata
- 방법: robust SO(3)/translation aggregation, temporal-MAD background, persistent SIFT,
  USAC_MAGSAC, fixed-intrinsic Mode A, Huber two-stage least squares, weak priors
- Hard constraint: cam1 identity gauge, cam2/cam3 shared extrinsic, timestamp별 camera 변수 없음
- Pilot: `barbellrow_0000`, `squat_0001`, `pushup_0001`, `benchpress_0003`
- Acceptance: Stage 1/2 convergence, finite SE(3), pre/post metrics, sample gate, visual coherence
- 결과: PASS 2 / REVIEW 2 / FAIL 0, 모든 Stage 수렴
- 특이사항: `squat_0001/cam2` 6.4 s VGGT pose는 hard-code 없이 REJECT됐고 background observation은 유지
- 다음 gate: 알고리즘/default 동결 후 전체 26 sequence 적용

## Phase 5 — Full Dataset Fixed-Camera Background BA

- 상태: `DONE` (Phase 5.1 recovery 반영)
- 목적: Phase 4 승인 알고리즘을 변경하지 않고 전체 sequence의 최종 camera geometry 후보 생성
- 입력: Phase 2/3/4 산출물과 frozen default
- 출력: 26 sequence의 camera/track/point/residual/metrics/validation 및 dataset summary
- 방법: [configs/phase5_background_ba.json](configs/phase5_background_ba.json)의 default 그대로 실행
- Acceptance: 26/26 파일 완결, Stage 1/2 수렴, SE(3) consistency, source/VGGT payload 불변,
  PASS/REVIEW/FAIL과 uncertainty/gauge/scale provenance 보존
- 결과: Phase 5.1 반영 후 PASS 11 / REVIEW 15 / FAIL 0; Stage 1/2 26/26;
  final points 1,100, observations 11,046; accepted residual median 3.630→2.584 px
- Recovery: `pushup_0003`은 Stage 2 budget만 600으로 확장해 322 nfev에서 수렴,
  sparse support 때문에 `RECOVERED_REVIEW`; fallback 없음
- 다음 gate: camera geometry freeze 승인. REVIEW uncertainty를 Phase 6/7에 전달

## Phase 5.1 — pushup_0003 Camera Recovery

- 상태: `DONE`
- 목적: Phase 5의 유일한 FAIL이 알고리즘 변경 없이 optimization budget만으로 회복 가능한지 검증
- 입력: Phase 5 baseline, 동일 설정의 300-control, Stage 2 `max_nfev=600` recovery run
- 출력: recovered camera/points/residual/validation, optimizer trace, recovery analysis와 갱신된 dataset summary
- 주요 방법: Stage 1 budget 300 유지, Stage 2 budget만 600; objective, loss, observation,
  track/filter/gate, initialization, K, gauge와 scale gauge 동일
- Acceptance: formal convergence, finite SE(3), residual 비악화, visual geometry 정상,
  300-control 재현과 input/init equality
- 결과: 322 nfev에서 `xtol` 수렴, median 4.954→2.559 px, p90 8.037→5.054 px,
  `RECOVERED_REVIEW`, fallback 미사용
- 다음 gate: FAIL 0으로 camera geometry dataset freeze 승인; REVIEW metadata 보존

## Phase 6 — High-Quality 2D Pose Observation

- 상태: `TODO`
- 목적: 모든 camera/frame의 canonical 2D joint와 confidence 생성
- 입력: synchronized frame reference, Phase 5 camera status
- 출력: teacher-native keypoints, canonical mapping, confidence, optional teacher disagreement
- 주요 방법: Sapiens2 1B/5B를 A100 80GB에서 pilot 비교; RTMPose를 primary offline teacher로 강제하지 않음
- Acceptance: mapping/visibility 정의, coverage, left-right sanity, temporal outlier 및 second-teacher uncertainty 검증
- 다음 gate: camera PASS/REVIEW 정책과 2D observation quality 확정 후 triangulation

## Phase 7 — Timestamp-Aware Multi-view Triangulation

- 상태: `TODO`
- 목적: Phase 2 timing correction과 Phase 5 camera를 사용한 robust 3D joints
- 입력: refined camera, temporal metadata, Phase 6 joints
- 출력: 3D joints, reprojection/ray uncertainty, 2-view fallback provenance
- 주요 방법: corrected timestamp pairing, robust triangulation, 3-view 우선, 필요 시 2D trajectory만 interpolation
- Acceptance: cheirality, reprojection, ray angle, cross-view joint consistency와 fallback 이유 저장
- 다음 gate: reliable 3D evidence를 body prior/fitting으로 전달

## Phase 8 — SAM 3D Body / SAM-Body4D Human Prior

- 상태: `TODO`
- 목적: pretrained model로 temporal MHR/body prior 생성
- 입력: RGB reference와 Phase 6/7 evidence
- 출력: temporal body prior, uncertainty, modal/amodal 구분
- 주요 방법: pretrained checkpoint만 사용하고 teacher fine-tuning 금지
- Acceptance: amodal completion을 image GT가 아닌 prior로 표시, model failure/disagreement 기록
- 다음 gate: sequence-level optimization의 weak/noisy prior로만 사용

## Phase 9 — Sequence-Level Body Fitting

- 상태: `TODO`
- 목적: multi-view geometry, time, body constraint를 결합한 최종 body parameter
- 입력: 2D/3D joints, human prior, silhouettes, contacts
- 출력: subject shape, frame pose, global orientation/translation, optional global scale와 residual
- 주요 방법: pretrained prediction 복사 금지; temporal/contact/pose prior와 robust observation objective
- Acceptance: reprojection/3D/body/temporal/contact residual, failure mode, uncertainty 저장
- 다음 gate: subject-level shape consistency

## Phase 10 — Subject-Level Shape / Anthropometric Descriptor

- 상태: `TODO`
- 목적: subject 전체 sequence를 공동 사용한 shape와 scale-invariant descriptor `S0`
- 입력: Phase 9 fits, optional A-pose initialization
- 출력: body shape parameter와 별도 proportion descriptor
- 주요 방법: femur/tibia/torso/shoulder/hip ratio, body beta와 `S0` 의미 분리
- Acceptance: cross-sequence consistency와 scale provenance
- 다음 gate: pseudo-label reliability 통합

## Phase 11 — Pseudo-label Quality Control

- 상태: `TODO`
- 목적: label과 reliability를 함께 저장
- 입력: camera, temporal, teacher, triangulation, fitting diagnostics
- 출력: frame/sequence quality vector와 overall policy
- 주요 방법: uncertainty를 누락하지 않고 source별 provenance 유지
- Acceptance: 모든 label에 quality metadata와 invalid/review reason 존재
- 다음 gate: 외부 ground-truth validation

## Phase 12 — Fit3D Validation

- 상태: `TODO`
- 목적: 전체 pipeline의 정량 accuracy와 calibration tolerance 검증
- 입력: Fit3D의 3-view 구성, clean/degraded perturbation
- 출력: MPJPE, N-MPJPE, PA-MPJPE, PVE, joint-angle MAE, ablation
- 주요 방법: temporal/extrinsic perturbation, tolerance curve, staged loss ablation
- Acceptance: metric 재현성, error attribution, downstream joint-angle 기준 확정
- 다음 gate: final schema와 freeze 승인

## Phase 13 — Final Dataset Freeze

- 상태: `TODO`
- 목적: immutable final schema, split, provenance와 release policy 확정
- 입력: Phase 5–12 승인 결과
- 출력: camera/image references/keypoints/body/quality/metadata schema와 dataset card
- 주요 방법: PTS 기반 record, 비식별 ID, payload checksum, versioned schema
- Acceptance: schema validation, no private payload in Git, documented access/license/citation, reproducible build ID
- 다음 gate: downstream 연구 사용 승인

## Phase별 Git Definition of Done

1. 시작 시 상태를 `IN_PROGRESS`로 변경한다.
2. 구현·실험·validation을 수행한다.
3. `process.md`에 날짜, 명령, 결과, 생성 파일, 실패와 결정을 기록한다.
4. acceptance gate를 적용하고 `DONE` 또는 `REVIEW`로 갱신한다.
5. README 진행표를 갱신한다.
6. source/report에서 private payload와 absolute path를 제거한다.
7. 명시적으로 stage하고 `tools/check_publication_safety.py`와 staged diff를 검토한다.
8. Phase 단위 commit과 push를 수행한다. destructive force push와 history rewrite는 금지한다.
