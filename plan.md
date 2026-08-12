# Exercise3D 데이터셋 구축 Canonical Plan

상태 표기: `TODO`, `IN_PROGRESS`, `DONE`, `REVIEW`. 각 Phase는 구현, 정량·시각 검증,
문서 갱신, 공개 안전 검사, commit/push까지 완료되어야 Definition of Done을 만족한다.

## Deadline critical path — 2026-08-14 13:00 KST

- primary objective: correctness·provenance·identity consistency를 유지하면서 deadline까지
  end-to-end로 완결되고 freeze 가능한 sequence 수를 최대화
- 2026-08-11 22:50 KST 기준 remaining wall-clock 62.17 h
- 최신 target-only Sapiens2 projection 79.09 GPUh와 SAM Mode B 16.35 h는 한 A100에서
  deadline 전 전량 순차 완료가 불가능하므로, sequence-complete streaming으로 변경
- 이미 완료된 4개 pilot sequence output은 검증 후 재사용하고 재추론하지 않음
- GPU scheduling: Sapiens2-5B는 계속 실행하고, pose-complete sequence의 Mode B를 겹쳐
  end-to-end 완결 sequence를 확보한다. 첫 full camera 병렬 peak 61,821 MiB와 completion PASS 확인
- current projection: 2026-08-12 13:09 KST Sapiens current partial 포함 24,135/65,430 crop,
  recent 0.218 crop/s, 전량 ETA 2026-08-14 17:47 KST; 약 -4.79 h deadline margin을
  end-to-end complete sequence 확보와
  명시적 INCOMPLETE provenance로 관리
- SAM policy: Mode B default, Mode C는 실제 failure/occlusion escalation evidence가 있는 경우만 REVIEW
- long-run supervision: current 5B 종료 감시, 불완전 camera selection-bound resume, Phase 7→Mode B→
  consolidation→body fit→versioned private export를 sequence별 자동 진행
- SAM full 직전 8-frame Mode B numeric smoke에서 source PTS/mesh/MHR compact schema exact gate 요구
- deadline에 미완료된 sequence는 `INCOMPLETE_DEADLINE`로 명시하고 PASS로 위장하지 않음
- Fit3D exhaustive tolerance/ablation은 final private dataset critical path를 방해하면 freeze 이후로 이동
- persistent handoff: `HANDOFF.md` + ignored `.runtime/handoff_state.json` 30초 atomic checkpoint;
  completion metadata/schema PASS item만 skip하고 incomplete/corrupt item만 resume
- autonomous monitoring: `.runtime/dashboard_state.json`에 process/GPU/progress/deadline/attention을
  atomic 기록하고 Rich live/`--once`/state-only mode를 제공; selector workload와 measured rate 기반
  overhead-free deadline sequence upper bound 및 completed provenance의 post-SAM p90-adjusted schedule을
  별도 표시; dashboard/handoff monitor는 lifetime lock + exact-identity capped recovery watchdog으로
  유지; verified checkpoint manifest 기반 cumulative immutable build/final snapshot storage를 SAM
  forecast와 합산해 reserve risk를 미리 감지; polling timestamp를 제외한 durable artifact에서
  top-level latest completion event를 산출하며 정상 generation은 AI polling 금지
- remaining order gate: pose-incomplete 14 sequence의 selector target crops/SAM frames pairwise
  dominance inversion 0, measured combined-cost inversion 0; global optimum으로 과장하지 않고
  명백한 shorter-workload command drift만 attention 처리
- expensive camera output별 atomic `run_provenance.json`: checkpoint/config/source/selection/tool/command
  identity를 보존하며 기존 PASS output에는 재추론 없이 sidecar만 materialize
- deadline sentinel: 2026-08-14 13:00 KST에 별도 immutable build ID로 private snapshot/export를
  실행하고 PASS/REVIEW/FAIL/INCOMPLETE를 고정한 뒤 generation은 중단 없이 계속
- deadline build는 hidden resumable staging root에서 만들고 manifest-listed file byte/SHA와 status
  count/privacy flag를 검증한 뒤 directory rename으로 publish; existing final build는 검증 후 reuse만 허용
- first end-to-end gate: `barbellrow_0000` Mode B 1,770/1,770, body fit 590×26,
  export checksum/schema PASS; camera/displacement uncertainty를 숨기지 않고 sequence REVIEW 유지
- second end-to-end gate: `squat_0001` Mode B 3,801/3,801, body fit 1,267×26,
  Mode C candidate 0 `PASS_MODE_B_FROZEN`; displacement/camera uncertainty로 REVIEW 유지

## 현재 Gate

- Phase 0–4: `DONE`
- Phase 5 dataset-wide 실행: `DONE`
- Phase 5.1 `pushup_0003` recovery: `DONE` (`RECOVERED_REVIEW`)
- Phase 5 downstream camera freeze gate: `DONE` (REVIEW uncertainty 전파 조건)
- 최종 camera status: PASS 11 / REVIEW 15 / FAIL 0, Stage 1/2 26/26 수렴
- Phase 6-0 Sapiens2-5B environment/smoke: `DONE`
- Phase 6-1 multi-exercise pose pilot: `DONE`, all-person 결과는 baseline으로 보존
- Phase 6-1A primary target selection: `DONE`, `GO_FULL_DATASET` 조건 충족
- Phase 6-2 target-only runtime gate: `DONE`, autonomous full inference 승인·critical path 진입
- Phase 8 runtime feasibility pilot: `PILOT_COMPLETE_REVIEW`, full Mode B `IN_PROGRESS`, Mode C selective

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

## Phase 6-0 — Sapiens2-5B Pose Environment Preparation

- 상태: `DONE`
- 목적: A100 80GB에서 정확도 우선 offline teacher인 Sapiens2 Pose 5B의 공식 환경과 checkpoint 검증
- 입력: official Meta Sapiens2 repository, `facebook/sapiens2-pose-5b`, representative private frame 1장
- 출력: 별도 Python 3.12/PyTorch 2.7 environment, pose/detector checkpoint, smoke JSON/visualization,
  public dependency/hash manifest와 재현 CLI
- 주요 방법: 공식 1024×768 top-down pipeline, official DETR ResNet-101 DC5 person detector,
  UDP heatmap decode, flip test와 원본 pixel 좌표 복원
- Acceptance: detection/model GPU load, 308 coordinates/confidence, finite/left-right/coordinate sanity,
  peak VRAM·latency 측정, checkpoint/private output 비공개
- 결과: PASS. peak allocated 19.986 GiB, reserved 20.961 GiB, end-to-end median 4.517 s/image;
  5B primary teacher 확정, 1B downgrade/comparison 미수행
- 다음 gate: Phase 6-1에서 소규모 multi-exercise detector/pose/output schema와 throughput 검증

## Phase 6 — High-Quality 2D Pose Observation

- 상태: `IN_PROGRESS` — full DETR/selector gate 완료, resumable 5B target-only 실행 중
- 목적: 모든 camera/frame의 canonical 2D joint와 confidence 생성
- 입력: synchronized frame reference, Phase 5 camera status
- 출력: teacher-native keypoints, canonical mapping, confidence, optional teacher disagreement
- 주요 방법: Sapiens2 Pose 5B primary teacher; 1B는 OOM/비현실적 throughput/instability 또는
  pilot accuracy 동등성 근거가 있을 때만 비교. RTMPose를 primary offline teacher로 강제하지 않음
- Acceptance: mapping/visibility 정의, coverage, left-right sanity, temporal outlier 및 second-teacher uncertainty 검증
- 다음 gate: camera PASS/REVIEW 정책과 2D observation quality 확정 후 triangulation

### Phase 6-1/6-1A — Batch pilot와 primary target identity

- 상태: `DONE`
- pilot: `barbellrow_0000`, `squat_0001`, `pushup_0001`, `benchpress_0003`의 3-view
- baseline: official DETR의 모든 person crop에 5B를 실행한 기존 결과를
  `ALL_DETECTIONS_BASELINE`으로 보존
- target-only: all DETR candidate는 private metadata에 유지하되, forward/backward temporal
  tracking이 합의한 primary target 1명에게만 Sapiens2-5B를 실행
- abstention: no detection은 `NO_TARGET`, 양방향 불일치/낮은 association margin은
  `TARGET_AMBIGUOUS`; 다른 사람을 강제 대체하지 않음
- evidence: multi-frame initialization, bbox IoU/center displacement/scale/aspect, detector score,
  track persistence/duration; appearance model은 bbox-temporal evidence 부족 시에만 검토
- cross-view: PTS와 Phase 5 refined camera geometry interface 및 visibility QA까지만 수행하고,
  이 단계에서 triangulation은 하지 않음
- Acceptance: obvious identity switch 0, ambiguity/background 오류 분석, target-only output 정상,
  batch 1/2/4/8/(12/16) equivalence/throughput, 65,595-frame ETA 재계산
- 최종 gate: `GO_FULL_DATASET`, `REVIEW_TARGET_SELECTION`, `NO_GO`
- 안전 조건: gate 보고 전 full dataset inference를 시작하지 않음
- 결과: 9,732 frame, candidate 19,596, target crop 9,725, ambiguity 7(0.072%),
  no-target 0, obvious identity switch 0, crop reduction 50.37%
- target-only batch: 1/2/4/8/12/16 모두 numerical equivalence PASS. 일반 실행 권장은
  plateau의 최소 batch 4이나, deadline run은 동일한 출력과 안전한 VRAM이 확인된 raw-fastest
  batch 16을 사용
- full gate: 78/78 camera, 65,595 frame, candidate 120,586, target crop 65,430,
  ambiguity 139, `NO_TARGET` 26, identity/F-B/integrity failure 0
- runtime projection: 기존 pilot 9,725 pose를 lossless resume하여 새 inference 55,705 crops.
  cached-detector 보수적 환산 약 66.1 GPU-hours이며 실제 full wall-clock으로 계속 갱신
- first two new cameras steady rate `0.23323 crop/s`; 2026-08-11 19:50 KST projection은
  Sapiens 종료 2026-08-14 12:58 KST로 deadline reserve가 사실상 0
- 판정: target-selection acceptance는 `GO_FULL_DATASET`; autonomous deadline 지침에 따라
  batch 16 full 실행 critical path 진입

## Phase 7 — Timestamp-Aware Multi-view Triangulation

- 상태: `IN_PROGRESS`; full pose 완료 sequence를 자동 감지하는 CPU streaming 실행 중
- 목적: Phase 2 timing correction과 Phase 5 camera를 사용한 robust 3D joints
- 입력: refined camera, temporal metadata, Phase 6 joints
- 출력: 3D joints, reprojection/ray uncertainty, 2-view fallback provenance
- 주요 방법: corrected timestamp pairing, robust triangulation, 3-view 우선, 필요 시 2D trajectory만 interpolation
- Acceptance: cheirality, reprojection, ray angle, cross-view joint consistency와 fallback 이유 저장
- pilot: 4 sequence/3,244 reference timestamps, schema 4/4 PASS. canonical reprojection
  median/p90은 barbellrow 7.06/30.62, squat 26.24/164.93, pushup 326.93/2,004.04,
  benchpress 7.91/97.84 px
- gate: barbellrow/benchpress REVIEW, squat/pushup `NO_GO_TRIANGULATION`; NO_GO proposal은
  보존하되 body fitting/export에서 제외
- recovery: 별도 observation-conditioned root와 20% held-out gate에서 squat/pushup 모두
  NO_GO 해제. all-frame canonical median/p90 5.71/18.93, 8.11/96.04 px
- provenance: 독립 calibration/GT가 아니므로 둘 다 REVIEW 유지, 원 Phase 5 output 보존
- full streaming 현재: 기존 pilot 4/4 final schema PASS, body-fitting eligible 4/4, 나머지 22 대기
- 다음 gate: full pose 완료 sequence의 원 camera consistency 검사 후 NO_GO에만 동일 recovery 적용

## Phase 8 — SAM 3D Body / SAM-Body4D Human Prior

- 상태: `FULL_IN_PROGRESS_REVIEW`; Mode B full policy와 exact-resume/numeric prior schema 동결,
  Sapiens2와 pose-ready sequence 단위 병렬 실행
- 목적: pretrained model로 temporal MHR/body prior 생성
- 입력: RGB reference와 Phase 6/7 evidence
- 출력: temporal body prior, uncertainty, modal/amodal 구분
- 주요 방법: pretrained checkpoint만 사용하고 teacher fine-tuning 금지
- Acceptance: amodal completion을 image GT가 아닌 prior로 표시, model failure/disagreement 기록
- runtime gate: control과 severe-occlusion clip에서 (A) SAM 3D Body base, (B) SAM-Body4D
  completion off, (C) completion on을 비교하고 target selector의 primary bbox 1개만 seed
- adapter: upstream의 모든 initial human 자동 선택 대신 accepted primary bbox 1개만 전달하며,
  사용하지 않는 ViTDet checkpoint는 요구하지 않음
- checkpoint gate: gated access 재확인 후 28 files, 24,037,668,123 bytes(22.387 GiB),
  size/SHA-256/누락 검사 모두 PASS
- pilot 결과: control 1,267 frame + severe 1,136 frame의 A/B/C 6-run PASS, target seed/track 1
- runtime: Mode A 0.827–0.832, Mode B 0.918–0.920, Mode C 1.820–1.826 sec/frame;
  Mode C/B 약 1.98배, severe/control 약 1.00배
- sanity: numeric/mesh/render 누락 0, finite PASS. 단 Mode C content completion 0회이고 B/C mesh
  개선은 표본 최대 0.303 mm라 refiner 효용은 아직 입증되지 않음
- projection: SAM optimistic/expected/pessimistic 16.35/20.80/32.63 h; Sapiens2 target-only와
  한 GPU 순차 합계 95.43/99.88/111.71 h
- deadline gate: 2026-08-14 13:00 KST까지 Sapiens2와 순차 전량은 불가능하므로
  sequence-complete output 최대화 및 미완료 provenance 보존
- full output contract: mesh뿐 아니라 MHR pose/shape/scale/hand/expression/joint coordinate와 rotation,
  204-d model parameter numeric prior, target source-index/PTS/ambiguity/occlusion을 frame별로 저장하고
  camera 단위 exact completeness와 모든 compact NPZ required-field schema를 전수 검사
- 다음 gate: Mode B를 full 기본 후보로 유지하고 실제 completion trigger case에서 Mode C 효용 검증;
  Mode C는 selective escalation evidence가 있을 때만 사용
- Mode C candidate: occlusion-risk와 함께 Mode B missing/nonfinite 또는 sequence median+5 MAD
  temporal/alignment outlier가 있어야 하며, identity/PTS exact match, schema PASS, ≥10% alignment 개선
  또는 content-completion 호출, geometry displacement 증가 ≤5%를 모두 검증한 뒤에만 채택
- full Mode B/body fit 직후 candidate assessor를 자동 실행하되 Mode C 자체는 비교 전 실행/채택하지
  않으며, candidate 또는 `PASS_MODE_B_FROZEN`을 final private manifest에 포함
- first full result: `barbellrow_0000` 3 camera/1,770 frame PASS, 합산 2,960.81초
  (0.59781 frame/s), combined peak 61,821 MiB. mesh/numeric/provenance 수량 exact
- second full result: `squat_0001` 3 camera/3,801 frame PASS, 합산 6,080.57초
  (0.62511 frame/s), combined peak 70,359 MiB. cumulative 5,571 frame rate 0.61617 frame/s

## Phase 9 — Sequence-Level Body Fitting

- 상태: `IN_PROGRESS_REVIEW`; 11 sequence/7,147 reference frame, REVIEW 11/FAIL 0,
  나머지는 Mode B dependency 대기/streaming
- 목적: multi-view geometry, time, body constraint를 결합한 최종 body parameter
- 입력: 2D/3D joints, human prior, silhouettes, contacts
- 출력: subject shape, frame pose, global orientation/translation, optional global scale와 residual
- 주요 방법: triangulated geometry를 dominant anchor로 유지하고, view별 MHR canonical prior를
  sequence-local gauge에 robust similarity alignment한 뒤 weak correlated-prior term과 weighted
  second-difference temporal term을 단계적으로 적용. prior-only joint는 최소 2-view MHR evidence 요구
- Acceptance: reprojection/3D/body/temporal/contact residual, failure mode, uncertainty 저장
- 구현: `tools/fit_sequence_body.py`; MHR parameter→official model replay 최대 keypoint delta
  `2.68e-7 m`, mesh delta `7.15e-7 m`로 numeric contract 검증
- 사전 동결 gate: coverage/alignment/geometry displacement/prior-only/bone CV를 scale-normalized
  PASS/REVIEW/FAIL로 분리하고 camera REVIEW 전파
- first result: `barbellrow_0000` 590×26, coverage/alignment 1.0, prior-only 0,
  median bone CV 0.01738. displacement p95 0.05167와 camera REVIEW 때문에
  `REVIEW_BODY_FIT_QUALITY`; finite/NaN schema FAIL 0
- second result: `squat_0001` 1,267×26, coverage/alignment 1.0, prior-only 0,
  median bone CV 0.02327. displacement p95 0.07936와 camera REVIEW 때문에
  `REVIEW_BODY_FIT_QUALITY`; finite/NaN schema FAIL 0
- 다음 gate: subject-level shape consistency

## Phase 10 — Subject-Level Shape / Anthropometric Descriptor

- 상태: `SEQUENCE_S0_IMPLEMENTED_SUBJECT_MAPPING_UNAVAILABLE`
- 목적: subject 전체 sequence를 공동 사용한 shape와 scale-invariant descriptor `S0`
- 입력: Phase 9 fits, optional A-pose initialization
- 출력: body shape parameter와 별도 proportion descriptor
- 주요 방법: femur/tibia/torso/shoulder/hip ratio, body beta와 `S0` 의미 분리
- Acceptance: cross-sequence consistency와 scale provenance
- 현재 제한: private inventory에 evidence-backed sequence→subject mapping이 없어 `subject_id=null`을
  보존하고 외형/learned shape 기반 추측이나 cross-sequence fusion을 수행하지 않음
- 다음 gate: pseudo-label reliability 통합

## Phase 11 — Pseudo-label Quality Control

- 상태: `IN_PROGRESS_STREAMING`; 완료 body-fit 12 sequence/7,820 frame materialize,
  REVIEW 12/FAIL 0, exporter preflight freeze-ready 12/12
- 목적: label과 reliability를 함께 저장
- 입력: camera, temporal, teacher, triangulation, fitting diagnostics
- 출력: frame/sequence quality vector와 overall policy
- 주요 방법: uncertainty를 누락하지 않고 source별 provenance 유지
- Acceptance: 모든 label에 quality metadata와 invalid/review reason 존재
- 구현: scalar accuracy probability를 만들지 않고 target/pose/SAM/triangulation/body component와
  explicit bitmask reason을 reference frame별로 저장; exact frame/PTS/schema/count validation 후 resume
- streaming: 새 supervisor 실행은 Mode C assessment 직후 생성하며, 현재 live supervisor는 중단하지 않고
  deadline/final exporter가 누락 sequence를 CPU-only로 materialize하는 fallback 유지
- 다음 gate: 외부 ground-truth validation

## Phase 12 — Fit3D Validation

- 상태: `METRICS_IMPLEMENTED_WAITING_DATASET`; exhaustive degradation은 freeze 이후
- 목적: 전체 pipeline의 정량 accuracy와 calibration tolerance 검증
- 입력: Fit3D의 3-view 구성, clean/degraded perturbation
- 출력: MPJPE, N-MPJPE, PA-MPJPE, PVE, joint-angle MAE, ablation
- 주요 방법: temporal/extrinsic perturbation, tolerance curve, staged loss ablation
- Acceptance: metric 재현성, error attribution, downstream joint-angle 기준 확정
- 현재: MPJPE/N-MPJPE/PA-MPJPE metric과 synthetic regression 구현, local Fit3D payload 부재로
  실제 quantitative score는 아직 주장하지 않음
- 다음 gate: final schema와 freeze 승인

## Phase 13 — Final Dataset Freeze

- 상태: `EXPORT_IMPLEMENTED_SMOKE_PASS`; deadline build 입력을 sequence 단위로 누적 중
- 목적: immutable final schema, split, provenance와 release policy 확정
- 입력: Phase 5–12 승인 결과
- 출력: camera/image references/keypoints/body/quality/metadata schema와 dataset card
- 주요 방법: PTS 기반 record, 비식별 ID, payload checksum, versioned schema
- Acceptance: schema validation, no private payload in Git, documented access/license/citation, reproducible build ID
- 구현: versioned private build, byte-exact copy/SHA-256, PASS/REVIEW/FAIL/INCOMPLETE 보존과
  source inventory/frame/PTS/camera/temporal/identity/2D/3D/body provenance 검증
- publication integrity: hidden resumable staging의 symlink/mount traversal 차단, unlisted stale payload prune,
  actual tree↔global/sequence manifest exact-match, Git dirty/diff provenance 보존 후 atomic rename
- streaming preflight: quality 완료 sequence는 exporter와 동일 cross-stage validation으로
  `freeze-ready` PASS/REVIEW를 미리 확정; 5분 지속 dependency 누락/FAIL은 dashboard attention
- predeadline checkpoint follower: `ACTIVE`; 기존 contract-v2 build를 byte/SHA로 전수 검증해 largest durable
  set을 정하고, freeze-ready가 그 집합의 strict superset일 때만 frozen order 기반 deterministic
  immutable build를 CPU-only export; 동일/축소/비-superset set은 재export하지 않음
- checkpoint follower continuity: exact live/resume argv digest pin, 3-cycle absence confirmation,
  final rescan, follower lifetime lock과 capped detached restart를 사용하는 CPU-only watchdog;
  deadline 이후에는 restart하지 않음
- quality follower continuity: follower lifetime singleton lock + exact live/resume argv digest pin,
  3-cycle absence/final rescan/capped detached restart watchdog; quality와 freeze-readiness가 모두
  26/26 validated COMPLETE일 때만 recovery 종료
- monitoring-plane continuity: dashboard/handoff monitor의 shell-safe exact argv persistence + target별
  lifetime singleton lock + 3-cycle absence/final rescan/capped detached recovery watchdog; live target은
  signal하지 않음
- deadline boundary: body-fit NPZ/metadata + Mode-C assessment terminal marker mtime을 UTC cutoff으로
  고정해 post-deadline completion은 INCOMPLETE 유지; cutoff-eligible derived lag/transient export는
  staging checksum-resume 3회 재시도 후 최종 truthful snapshot publish
- freeze contract v2: requested sequence universe/order SHA-256 + status CSV exact match, global 3-file
  provenance와 complete sequence 33-file required set; sentinel expected 26-sequence binding
- supervisor continuity: live/resume argv digest pin + 3-cycle absence confirmation + final rescan +
  capped detached recovery watchdog; recovered supervisor lifetime lock으로 duplicate launch 거부
- deadline continuity: sentinel exact-identity watchdog + sentinel lifetime lock + build-ID-scoped exporter
  lock; concurrent recovery/export caller는 staging mutation 전에 중복 실행 거부
- first smoke: complete `barbellrow_0000`만 사용해 REVIEW 1/FAIL 0/INCOMPLETE 0,
  34 files, payload 28,960,929 bytes, SHA/size mismatch 0, freeze-eligible 확인
- quality/exact-tree smoke: clean commit `250ee73`에서 REVIEW 1/FAIL 0/INCOMPLETE 0,
  36 files/28,993,394 bytes, actual-tree/ownership/hash error 0, immutable reuse PASS
- deadline contract v2 partial smoke: clean commit `7b54214`에서 requested 2-sequence
  universe/order exact bind, REVIEW 1/INCOMPLETE 1, 36 files/28,993,641 bytes,
  verifier error 0, dirty false, freeze eligible false
- single-descriptor source snapshot smoke: clean commit `f1b701e`에서 REVIEW 1,
  36 files/28,993,437 bytes, contract/order/tree/hash error 0, dirty false,
  freeze eligible true, immutable reuse PASS
- predeadline durable checkpoint: current freeze-ready 12 sequence 전체, build
  `exercise3d-predeadline-auto-012-77ac2165e283`, REVIEW 12/FAIL 0/INCOMPLETE 0,
  399 files/377,238,045 bytes, contract/order/tree/hash error 0, freeze eligible true,
  independent verifier file/byte exact PASS
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
