# Exercise3D Chronological Engineering / Research Log

이 문서는 실제 수행 순서와 의사결정을 누적한다. 공개본에는 원본 절대 경로, 피험자
개인정보, media screenshot 및 대용량 numeric payload를 기록하지 않는다. 명령의 private
경로는 `<PRIVATE_DATASET_ROOT>`처럼 치환한다.

## 2026-08-11 — 2026-08-14 13:00 KST autonomous deadline 시작

### Source-of-truth 재검증

- HEAD `ae89fe6`, worktree clean, Draft PR #1과 remote branch 동기화
- A100 80GB idle, private source 65,595 frames와 checkpoint storage 정상, source mutation 0
- deadline 2026-08-14 13:00 KST = 2026-08-14 04:00 UTC
- 2026-08-11 17:31 KST 기준 remaining wall-clock 67.48 h
- 전달된 과거 target 수치 대신 최신 repository result를 채택: 9,732 frames, 9,725 target crops,
  ambiguity 7, identity switch 0, crop reduction 50.3725%
- full target selector/Sapiens2/SAM output은 아직 없고, 4개 pilot sequence Sapiens2 output만 보존됨

### 중간 계획 변경 보고 — 이번 deadline cycle의 유일한 major 변경 보고

- 변경 사유: target-only Sapiens2 실측 projection 79.09 GPUh만으로 remaining 67.48 h를 넘고,
  SAM Mode B 16.35 h 및 downstream을 더하면 한 A100에서 전량 완료가 물리적으로 불가능
- 기존 계획: 전체 Sapiens2 → 전체 triangulation → 전체 SAM → fitting → freeze
- 변경 계획: 기존 4개 pilot output 재사용 + sequence-complete streaming. GPU는 Sapiens2 우선,
  CPU triangulation/QC 병행, SAM Mode B를 dependency 가능한 sequence에만 실행
- 정확도 영향: 5B, official flip-test, detector, target abstention과 accepted threshold는 변경하지 않음
- deadline 영향: 전체 26개 완료 보장은 포기하지 않되, deadline에는 완결 sequence 수를 최대화하고
  나머지는 resumable `INCOMPLETE_DEADLINE` provenance로 동결
- 리스크: Sapiens throughput 저하, Phase 7/9 구현 critical path, SAM output disk 증가
- 즉시 적용: official DETR full 26-sequence resumable pass 시작; full selector 후 Sapiens2 진입

### Phase 6 full 준비와 lossless pilot 재사용

- official DETR full pass는 26 sequence/78 camera에 대해 batch 16, chunk 512, source mutation 없이
  실행 중이다. 완료 camera마다 consolidated bbox/candidate payload와 QA를 원자적으로 기록한다.
- 기존 `ALL_DETECTIONS_BASELINE`에는 모든 candidate의 308-keypoint 결과가 보존되어 있으므로,
  accepted target candidate만 exact gather해 4개 pilot의 target-only output을 만들었다.
- 결과: 12 cameras, 9,732 frames, target poses 9,725, 새 5B inference 0회, elapsed 72.29 s.
- baseline 대비 confident XY/confidence 최대 delta는 12/12 camera 모두 0.0이었다.
- resume chunk는 frame 이름뿐 아니라 현재 selector의 abstention/status/index 및 selected bbox/score가
  일치해야 재사용하도록 강화했다. 기존 45 chunks는 selection-bound 검증 PASS.
- 17개 unit test와 compile PASS.

### Full selector incremental gate

DETR이 먼저 끝난 9 sequences/27 cameras에 full selector와 별도 lossless validator를 적용했다.

- frames 19,224, all candidates 37,966, target crops 19,068
- ambiguity 130, `NO_TARGET` 26, background candidates 18,898
- identity-switch risk 0, forward/backward disagreement 0, integrity failure 0
- candidate offsets/boxes/scores는 DETR consolidated arrays와 exact-match
- gate `GO_FULL_DATASET`

`barbellrow_0003`의 130 ambiguity와 26 NO_TARGET은 촬영 종료 후 target이 화면에서 나가는
구간에 집중됐다. 16-frame private overlay에서 background 사람을 강제 선택하지 않는 올바른
abstention임을 확인했다. 전체 78-camera 완료 후 동일 gate를 다시 실행한다.

### Phase 7 timestamp-aware triangulation pilot

```bash
python tools/triangulate_sapiens2.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --pose-root <PRIVATE_OUTPUT_ROOT>/sapiens2_target_only_full \
  --camera-root <PRIVATE_OUTPUT_ROOT>/background_ba \
  --output-root <PRIVATE_OUTPUT_ROOT>/triangulation \
  --runtime-dir <PRIVATE_OUTPUT_ROOT>/runtime/phase7_triangulation_pilot_gate \
  --sequences barbellrow_0000,squat_0001,pushup_0001,benchpress_0003
```

- schema/finite/NaN contract 4/4 PASS, 3,244 reference timestamps
- canonical source-joint reprojection median/p90 px:
  `barbellrow_0000` 7.06/30.62, `squat_0001` 26.24/164.93,
  `pushup_0001` 326.93/2,004.04, `benchpress_0003` 7.91/97.84
- Huber scale 10 px 배수의 사전 명시 gate 결과: REVIEW 2, NO_GO 2
- private overlay상 squat/pushup target 2D pose는 정상이므로 identity error로 덮지 않았다.
  current refined camera와 human observations의 epipolar inconsistency로 판정했다.
- NO_GO proposal은 진단용으로 보존하지만 `eligible_for_body_fitting=false`이며 export에 사용하지 않는다.
- Phase 5 camera를 덮어쓰지 않고, recovery를 수행한다면 observation-conditioned provenance와
  held-out-frame 검증을 요구한다.

## 2026-08-09 — 초기 synchronization / derivative 구축 기록 이관

### 수행

- clap onset과 waveform cross-correlation으로 triple-view 영상의 공통 구간을 결정했다.
- 서로 다른 native 30/30/60 fps를 보존한 raw에서 synchronized derivative와 30 fps working
  frames를 생성했다.
- output frame이 어떤 source frame에서 왔는지 pixel matching과 PTS로 별도 검사했다.
- source 파일 교체가 필요한 camera filename mapping 오류 1건은 provenance를 보존한 상태로
  수정했고, 이후 inventory에서는 replaced source를 active count에서 제외했다.

### 대표 명령

```bash
python scripts/build_dataset.py \
  --source-root <PRIVATE_DATASET_ROOT>/origin \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --stage all
python scripts/verify_dataset.py --dataset-root <PRIVATE_DATASET_ROOT>
```

### 결정

- sync와 working frame은 derivative이며 raw native frame rate는 유지한다.
- sync residual과 실제 frame-grid offset을 구분한다.
- 실제 data payload는 공개 Git 저장소로 이관하지 않는다.

## 2026-08-09T12:20:46Z — Phase 0 Dataset Inventory / Integrity 완료

### 실행

```bash
python tools/dataset_inventory.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --output-dir <PRIVATE_OUTPUT_ROOT>/inventory
```

### 결과

- 피험자 3명, triple-view 26 sequences
- raw videos 78, synchronized videos 78
- working JPEG 65,595장
- camera source: iPhone 16 / 16 Pro / 17, native 30/30/60 fps
- raw/sync inventory PASS, source/derivative provenance 유지
- source modification 0건

### 결정

모든 후속 report는 frame index만이 아니라 가능한 경우 packet/frame PTS를 함께 저장한다.

## 2026-08-09T12:33:26Z — Phase 1 EIS/OIS / Camera Stability Audit 완료

### 실행

```bash
python tools/eis_background_audit.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --output-dir <PRIVATE_OUTPUT_ROOT>/eis_audit
```

### 방법

temporal background와 foreground component mask를 구성하고, static 영역의 LK feature track에
homography/affine model을 fit했다. native-adjacent와 longer-baseline pair의 global motion,
spatial residual, 반복성을 함께 평가했다.

### 결과

- 78/78 `FIXED_CAMERA_OK`
- native-adjacent fit 8,087/8,087 성공
- 반복 global/spatial warp evidence 없음
- foreground-induced false positive 1건을 mask logic 수정 후 재검증
- source data modification 0건

### 결정

physical camera는 tripod-fixed로 간주한다. downstream final camera에 timestamp별 독립 pose를
두지 않는다.

## 2026-08-09T13:18:24Z — Phase 2 Temporal Synchronization QA 완료

### 실행

```bash
python tools/temporal_sync_audit.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --output-dir <PRIVATE_OUTPUT_ROOT>/temporal_alignment
```

### 방법

- PTS를 frame index보다 우선했다.
- beginning/middle/end를 포함한 여러 window에서 3 camera pair를 검사했다.
- actual-frame mapping, audio waveform/clap, visual motion energy를 함께 사용했다.
- RGB frame은 자르거나 보간하거나 재생성하지 않았다.

### 결과

- 26 sequences, 78 camera pairs
- actual-frame PTS observation 546건
- absolute offset median 11.99 ms, p95 25.28 ms, max 31.38 ms
- 30 fps 1 frame인 33.33 ms 이내 546/546
- `TEMPORALLY_STABLE` 8, `SMALL_CONSTANT_OFFSET` 16,
  `CLOCK_DRIFT_DETECTED` 2, `INSUFFICIENT_EVIDENCE` 0
- drift review: `pushup_0000`, `squat_0001`

### 결정

dataset synchronization은 사용 가능하다. offset/drift는 downstream frame pairing metadata로만
반영하고 video 자체를 보정하지 않는다.

## 2026-08-09T15:52:57Z — Phase 3 VGGT-Ω Geometry Initialization 완료

### 공식 구현 확인

- input preprocessing와 512-class resolution behavior
- joint sequence inference와 output tensor shape
- OpenCV world→camera `[R|t]`
- canvas pixel-unit K, positive camera-Z depth
- arbitrary sequence-local scale/gauge
- probability가 아닌 ranking confidence

### 실행

```bash
python tools/vggt_geometry_init.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-repo <LOCAL_VGGT_REPO> \
  --checkpoint <LOCAL_CHECKPOINT> \
  --output-dir <PRIVATE_OUTPUT_ROOT>/vggt
```

### 결과

- sequence당 8 representative PTS × 3 cameras = 24 images joint inference
- 26/26 sequence SUCCESS, 78/78 camera geometry, 624 sampled camera frames
- 실패·필수 payload 누락 0
- camera quality PASS 77 / REVIEW 1
- REVIEW: `squat_0001/cam2`, rotation dispersion outlier

### 결정

VGGT-Ω 결과는 최종 camera가 아니다. background BA의 initialization/prior로만 사용하며
depth/point-map scale을 metric으로 해석하지 않는다.

## 2026-08-09 — Phase 3 Open3D Visual Inspection Gate 완료

### 구현

`tools/visualize_vggt.py`에 percentile confidence filtering, RGB mapping, world axis, camera
frustum, voxel/max-point sampling, screenshot/PLY debug export와 BA overlay를 구현했다.

### 좌표 검증

world→camera에서 camera center를 `C_world = -R.T @ t`로 계산했다. OpenCV +x right,
+y down, +z forward convention을 raw output에 유지하고 display 변환을 metadata로 분리했다.

### 대표 결과

- `barbellrow_0000`: PASS
- `squat_0001`: REVIEW
- `pushup_0001`: REVIEW
- `benchpress_0003`: REVIEW
- 전역 mirror, 180° flip, exploding point cloud 없음
- distant wall/floor/rack이 thin sheet로 분리되고 일부 camera pose jitter 존재

### `squat_0001/cam2`

특정 sample의 pose가 cluster에서 이탈했지만 전체 scene이 동시에 폭발하지 않아 camera token
pose instability가 중심이고 dynamic foreground/point-map noise가 일부 기여하는 것으로 판단했다.

## 2026-08-09 — Phase 4 Fixed-Camera Background BA Pilot 완료

### 범위

`barbellrow_0000`, `squat_0001`, `pushup_0001`, `benchpress_0003` 네 sequence만 pilot으로 실행했다.

### 방법

- VGGT timestamp pose를 robust SO(3)/translation aggregation해 physical-camera init 생성
- cam1 identity gauge, cam2/cam3 shared extrinsic만 optimization
- temporal median/MAD, confidence, border, persistent SIFT로 static background 추출
- SIFT ratio, USAC_MAGSAC, epipolar/point-map consistency로 cross-view track 구성
- fixed intrinsics Mode A, Huber robust loss, weak pose/point prior, Stage 1/2 gate
- Phase 2 corrected timestamp는 matching pair 선택에만 사용

### 결과

- PASS 2 / REVIEW 2 / FAIL 0
- 모든 sequence Stage 1/2 finite convergence
- `squat_0001/cam2` 6.4 s VGGT pose: aggregation weight 0.001, 자동 REJECT
- 같은 timestamp의 유효 background observation은 shared-pose BA에서 유지
- 수동 sequence/PTS hard-code 없음

### 결정

동일 알고리즘과 default를 변경하지 않는 조건으로 26-sequence 확장을 승인했다.

## 2026-08-09 — Phase 5 Full Dataset Background BA 실행 완료

### Configuration freeze

- historical tool SHA-256: `1f01256e336474fae5c79434323b7c092b618c3b94e171077c80897f95a53feb`
- normalized configuration SHA-256: `df640077fd89f462eec6001f13465e808be47f991d6637175dfbfa24b7d2764a`
- fixed intrinsics, Huber scale 5, max tracks 800, min length 3, max nfev 300
- 새로운 matcher/threshold/heuristic/weighting 추가 없음

### 실행 형태

```bash
python tools/background_bundle_adjust.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-root <PRIVATE_OUTPUT_ROOT>/vggt \
  --output-root <PRIVATE_OUTPUT_ROOT>/background_ba \
  --sequence <SEQUENCE_ID>
python tools/finalize_background_ba_dataset.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-root <PRIVATE_OUTPUT_ROOT>/vggt \
  --output-root <PRIVATE_OUTPUT_ROOT>/background_ba
```

### 결과

- 26/26 sequence output과 78/78 camera summary 생성
- PASS 11 / REVIEW 14 / FAIL 1
- Stage 1 convergence 26/26, Stage 2 convergence 25/26
- point 1,674 initial → 1,100 final
- observation 16,835 extracted → 11,046 final
- 동일 accepted observation의 residual:
  - mean 4.113 → 3.361 px
  - median 3.630 → 2.582 px
  - p90 8.205 → 7.425 px
  - p95 9.850 → 9.953 px
- cam2 rotation change median/p95/max 0.457°/2.247°/2.501°
- cam3 rotation change median/p95/max 0.376°/3.074°/3.207°
- cam1은 exact gauge reference로 변화 0

### REVIEW / FAIL

REVIEW는 low track support, p95 tail, no direct three-camera track 등의 기존 gate 이유를
그대로 보존했다. FAIL은 `pushup_0003` 1건으로 Stage 2가 `max_nfev=300`에 도달했다.
알고리즘 freeze 원칙 때문에 threshold를 바꾸거나 자동 fallback하지 않았다.

### 무결성

- raw/synchronized/working frame 변경 없음
- VGGT numeric payload 변경 없음
- SE(3) inverse/rotation finite sanity 검사 통과
- gauge: robust cam1 physical pose identity
- scale: sequence-local arbitrary, initial cam1-cam2 baseline 보존

### 결정

dataset-level 계산은 완료됐다. 그러나 FAIL refined camera는 triangulation에 사용하지 않는다.
제외, pilot initialization fallback 또는 별도 승인된 재최적화 중 정책을 확정하기 전까지
Phase 6 전체 실행은 보류한다.

## 2026-08-09 — 공개 전용 저장소 migration

### 수행

- 비어 있는 public repository를 clone하고 `main` 최초 bootstrap을 준비했다.
- private workspace는 수정하지 않고 read-only inventory source로만 사용했다.
- dataset-construction 관련 scripts/tools만 이관하고 public-facing legacy project 명칭을 제거했다.
- `--dataset-root`와 `EXERCISE3D_DATASET_ROOT`를 추가하고 output path를 명시할 수 있게 했다.
- 한국어 README/canonical plan/chronological log와 phase/design/QA 문서를 구성했다.
- Phase 5 aggregate numeric CSV만 이관하고 exact K/R/t, media, NPZ와 debug render는 제외했다.
- conservative `.gitignore`, publication safety checker, GitHub Actions check를 추가했다.

### Migration smoke test 범위

- 모든 Python source compile
- 주요 CLI help
- 외부 private dataset을 대상으로 BA `--dry-run`
- staged/tracked file suffix, size, absolute path, credential pattern 검사
- private workspace source tree hash/mtime mutation 없음 확인

### 결정

이후 dataset-construction 변경의 canonical source는 이 public repository다. 각 Phase는
acceptance gate 후 문서화, 안전 검사, commit/push까지 완료해야 한다.

## 2026-08-09 — Phase 5.1 pushup_0003 Camera Recovery 완료

### 원인 분석과 동일성 control

- Phase 5 Stage 1은 cost 10,498.589521에서 정식 수렴했다.
- Stage 2는 cost를 1,672.515861까지 낮췄지만 `max_nfev=300`에서 종료됐다.
- 300-control은 Phase 5의 initial camera, 모든 track/observation array,
  `points_initial`/`points_stage1`, Stage 1 result와 Stage 2 cost를 exact 재현했다.
- 24 sample 모두 GOOD이고 특정 sample reject는 없었다. cam2 residual이 가장 높지만 camera
  explosion 없이 tail step이 작아져, 원인은 발산이 아닌 evaluation budget 부족으로 판단했다.

### 실행

```bash
python tools/background_bundle_adjust.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-root <PRIVATE_VGGT_ROOT> \
  --output-root outputs/local/background_ba/recovery_runs/nfev_600 \
  --sequence pushup_0003 \
  --max-nfev 300 \
  --stage2-max-nfev 600 \
  --optimizer-verbose 2
```

Stage 1은 기존 300 budget을 유지했고 Stage 2 budget만 600으로 확장했다. 새로운 matcher,
threshold, heuristic, loss, weighting 또는 observation 변경은 없다.

### 결과와 Visual QA

- Stage 2 `xtol` 수렴: actual nfev 322, final cost 1,657.953684
- median 4.954229→2.558895 px, p90 8.037446→5.053964 px,
  p95 9.295044→7.055627 px
- final 21 tracks / 183 observations, sample GOOD 24 / DOWNWEIGHT 0 / REJECT 0
- cam2/cam3 robust-init rotation change 2.538° / 1.859°,
  center scene fraction 0.003830 / 0.003494
- Open3D top/side 검사: plausible rig/orientation, mirror·180° flip·explosion 없음
- sparse support 때문에 `RECOVERED_REVIEW`; VGGT fallback 미사용

### Dataset gate와 무결성

- 최종 PASS 11 / REVIEW 15 / FAIL 0
- Stage 1/2 26/26 수렴, per-sequence validation 26/26 PASS
- camera geometry freeze 승인; REVIEW uncertainty는 downstream에 전달
- 외부 private workspace의 raw 78, synchronized 130-file tree, working JPEG 65,595,
  VGGT 689-file numeric tree와 Background BA 1,057-file tree fingerprint가 전/후 동일
- Sapiens2, triangulation, SAM-Body4D, SMPL/human fitting, pseudo-label 수행 없음
- viewer relative debug path가 external dataset root로 해석될 수 있던 경로를 canonical project
  root로 수정했고, 진단 중 생성된 두 debug file은 식별 후 제거하여 external tree를 원상 복구했다.

## 2026-08-09 — Phase 6-0 Sapiens2-5B Pose Environment 완료

### 공식 구현과 detector 결정

- official `facebookresearch/sapiens2` commit
  `7e5bae88456ac418ff0e58e74106c9fe192055d4`를 별도 external source로 clone했다.
- official checkpoint `facebook/sapiens2-pose-5b`의
  `sapiens2_5b_pose.safetensors`만 primary pose weight로 사용했다.
- model card의 RTMDet 문구와 달리 현재 official `docs/POSE.md`, demo shell과
  `vis_pose.py`는 `facebook/detr-resnet-101-dc5`를 사용한다. 실제 실행 code를 우선했다.
- top-down crop 1024×768, Sociopticon 308 points, UDP heatmap decode와 flip-test를 확인했다.

### 환경과 checkpoint

```bash
conda create -y -n sapiens2 python=3.12 pip
conda run -n sapiens2 python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
conda run -n sapiens2 python -m pip install -e <SAPIENS2_REPO>
```

- Python 3.12.13, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118
- transformers 5.14.1, safetensors 0.8.0, OpenCV 5.0.0.93
- A100-SXM4-80GB, driver 535.183.06, compute capability 8.0, BF16 지원 확인
- pose checkpoint 20,480,899,148 bytes; SHA-256
  `b4848da8691c72e14d3ff71319f077363107129bf4128019eb39d072129b2a52`
- detector snapshot revision `96317ca979e231bd960cb3cac31328e0165a3e94`

### Smoke 실행과 결과

```bash
conda run -n sapiens2 python tools/sapiens2_pose_smoke.py \
  --image <PRIVATE_REPRESENTATIVE_FRAME> \
  --sapiens2-root <SAPIENS2_REPO> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --warmup 1 --repeats 3 \
  --output-json outputs/local/sapiens2/smoke.json
```

- representative barbell-row frame에서 person 1명 detection 성공
- pose model GPU load 성공, FP32 model load 약 58.46 s
- 308 keypoint coordinates와 308 confidence 출력, 모두 finite
- confidence ≥0.3 point 100%가 원본 frame 내부, original pixel `(x,y)` 복원 정상
- official 308 flip mapping involution과 body left/right name pair 정상
- end-to-end detector + two-pass flip-test latency median 4.517 s/image
- peak CUDA allocated 19.986 GiB, reserved 20.961 GiB
- visual skeleton의 body/hand/foot 배치와 좌우 ordering plausible

첫 smoke checker는 구형 COCO ankle index pair를 가정해 계산 후 FAIL을 표시했다. 모델 출력 문제가
아니었으며 official 308 metainfo의 name 기반 pair 검사로 수정한 뒤 동일 inference가 PASS했다.
Detector safetensors load의 네 BatchNorm counter warning은 detection이 정상이라 compatibility note로
유지한다.

### 결정

5B는 OOM/instability 없이 동작하므로 primary offline teacher로 확정했다. 1B comparison은 수행하지
않았다. 단일 job 단순 외삽 약 82.3 GPU-hours는 offline 목적에서 허용 가능하며, Phase 6-1에서
official 2 jobs/GPU의 실제 throughput과 multi-exercise robustness를 먼저 측정한다. 전체 26 sequence
inference, triangulation, SAM-Body4D, MHR, SMPL과 pseudo-label generation은 수행하지 않았다.

## 2026-08-11 — Phase 6-1A Primary Target Selection Gate 완료

### 구현과 회귀 검증

- official DETR all-person candidate는 삭제하지 않고 private ragged metadata에 보존
- multi-frame initialization, track duration, IoU, normalized center, scale/aspect, score를 결합
- forward/backward tracking 합의와 cross-view target visibility QA 추가
- detector가 prone target을 상·하체 complementary box로 분할하는 fragmentation을 감지해
  `TARGET_AMBIGUOUS`로 abstain; frame 0이 마지막 frame과 wraparound 연결되던 boundary 수정
- target selector unit test 5개와 Python compile PASS

### 4-sequence pilot와 Visual QA

- 12 camera, 9,732 frame, official DETR person candidate 19,596
- target-only eligible crop 9,725, crop reduction 50.3725%
- `TARGET_AMBIGUOUS` 7, `NO_TARGET` 0, obvious identity switch 0
- ambiguity는 `pushup_0001/cam1` duplicate 1 + fragmentation 6이며 pose crop을 출력하지 않음
- private overlay에서 background crossing/overlap, mirror 후보, lying/prone pose, bbox size reversal,
  candidate order 변화 확인; background person systematic mis-selection 0
- private coordinate/overlay/frame은 ignored `outputs/`에만 보존

### Target-only Sapiens2 benchmark

- batch 1/2/4/8/12/16 모두 PASS 및 batch 1/all-person target baseline equivalence PASS
- raw fastest batch 16: 0.231951 crop/s, reserved 37.426 GiB
- 99% plateau 최소 batch 4 권장: 0.230449 crop/s, reserved 23.801 GiB,
  pose GPU utilization mean 97.309%, mean power 348.408 W
- 65,595 frame target-only stage projection 79.09 GPU-hours, all-person 157.38 GPU-hours 대비
  약 78.30 GPU-hours 감소

### 결정

Target-selection gate는 `GO_FULL_DATASET`이다. 그러나 사용자에게 결과를 보고하고 명시적 승인을
받기 전까지 전체 65,595-frame inference는 시작하지 않는다.

## 2026-08-11 — SAM Body Runtime Feasibility Preflight

### Official interface와 pilot 선정

- SAM-Body4D revision `21af1020979ef32ddf6be3597ef59a68bad2f1bf`
- SAM 3D Body revision `b5c765a0d89d789985e186d396315e7590887b94`
- mode A base, mode B completion off, mode C completion on 비교 계획 동결
- control `squat_0001/cam1`, severe-occlusion `latpulldown_0002/cam2` 선정
- severe clip 1,136 frame detection/selector preflight PASS: 평균 2.121 candidates/frame,
  occlusion risk 959, identity switch/ambiguity 0; private representative overlay 확인

### Checkpoint gate와 결정

Primary-target adapter 기준 필요한 6개 payload set은 local에 없고 총 24,037,668,123 bytes
(22.387 GiB)다. SAM 3와 SAM 3D Body는 gated access가 필요하다. 사용자 조건에 따라
download/model/path/license를 먼저 보고하며 명시적 승인 전 checkpoint 다운로드와 SAM inference를
수행하지 않는다. Provisional deadline verdict는 `DEADLINE_AT_RISK`; local A/B/C 실측 전 final
verdict는 보류한다.

### Primary-target adapter와 6-run preflight

- Mode A는 official SAM 3D Body `bboxes=` API에 frame당 accepted bbox 0/1개를 전달
- Mode B/C는 official SAM-Body4D class를 사용하되 SAM 3 initial object를 accepted bbox 1개로 seed
- upstream all-human initialization용 ViTDet는 호출하지 않아 checkpoint 2.576 GiB도 불필요
- ambiguous first frame, multiple bbox slot, invalid bbox를 강제 실행하지 않는 schema/gate 추가
- control 1,267 frame × A/B/C와 severe 1,136 frame × A/B/C preflight 모두 target seed 1 확인
- control target-valid 1,267/1,267, severe 1,136/1,136; severe occlusion-risk 959 보존
- model 실행은 여섯 경우 모두 승인 전 의도한 `BLOCKED_CHECKPOINT`; download 0 bytes
- SAM adapter/selector synthetic test 11개, Python compile, CLI smoke PASS
- 여섯 mode CSV를 요구하는 runtime summarizer 추가; refiner C/B ratio, control/severe 증가,
  best/expected/worst를 분리하고 expected prevalence 입력이 없으면 숫자 산출 금지
- credential 값을 출력하지 않은 HF auth 확인은 PASS했지만 SAM 3/SAM 3D Body gated access는
  `--dry-run`에서 denied; MoGe-2, Depth Anything V2, 두 official Diffusion-VAS repo는 dry-run PASS
- official setup code 기준 Diffusion-VAS repo ID와 SAM 3D Body `model_config.yaml` requirement를 교정

## 2026-08-11 — SAM Body checkpoint와 A/B/C pilot 완료

### Access, download와 integrity

- 사용자 gated access 승인 후 SAM 3/SAM 3D Body를 포함한 6개 official source의 access dry-run PASS
- required checkpoint tree 28 files, 24,037,668,123 bytes(22.387 GiB) 다운로드 완료
- 모든 payload의 file existence, byte size, SHA-256을 전수 재검증: 누락/불일치/예상 밖 파일 0
- checkpoint/cache/credential은 ignored external storage에만 유지하고 공개 CSV에는 상대 경로,
  크기와 digest만 기록
- 별도 Python 3.12 / PyTorch 2.7.1 CUDA 환경에서 official load, headless EGL과 CUDA smoke PASS
- official loader가 string path를 요구하는 실제 runtime incompatibility를 primary-target runner에서 교정

### Primary-target A/B/C 6-run

- control `squat_0001/cam1`: 1,267 frame, 약 42초, occlusion risk 0
- severe `latpulldown_0002/cam2`: 1,136 frame, 약 38초, occlusion risk 959
- 모든 mode에서 accepted primary target 1명만 처리하고 background detection에는 body inference 미수행
- control A/B/C total 1,047.20/1,162.70/2,306.22초,
  end-to-end 0.8265/0.9177/1.8202 sec/frame
- severe A/B/C total 945.05/1,045.43/2,074.61초,
  end-to-end 0.8319/0.9203/1.8262 sec/frame
- peak VRAM A/B/C 최대 7,367/33,988/44,175 MiB; GPU/power telemetry도 0.2초 간격 보존
- Mode C/B execution ratio control 1.9946, severe 1.9964; severe/control ratio는 모든 mode 약 1.00

### Output sanity와 결정

- Mode A numeric 2,403개, Mode B/C mesh와 render 각각 2,403개 생성, 누락 0
- 시작/중간/끝 numeric finite, PLY 18,439 vertices/36,874 faces finite, JPEG decode와 private visual QA PASS
- Mode C refiner는 control/severe에서 1,287/1,154회 호출됐지만 content completion은 모두 0회
- B/C 대표 mesh 차이는 최대 0.303 mm로 현재 severe clip에서 refiner의 material improvement가
  확인되지 않아 full 기본 후보는 Mode B, Mode C policy는 `REVIEW_SAM_REFINER_POLICY`
- 65,595-frame SAM projection 16.35/20.80/32.63시간, Sapiens2 target-only를 합친 한 GPU 순차
  projection 95.43/99.88/111.71시간
- 2026-08-15 00:00 UTC freeze는 `NO_GO`; end-of-day도 QC/재시도 여유가 작아
  `DEADLINE_AT_RISK`
- 전체 Sapiens2/SAM inference는 실행하지 않았으며 별도 사용자 승인 전 `HOLD`
