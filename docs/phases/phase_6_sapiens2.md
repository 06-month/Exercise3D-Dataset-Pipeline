# Phase 6 — Sapiens2 High-Quality 2D Pose Observation

## Phase 6-0 상태

`DONE`. Offline pseudo-label teacher는 Sapiens2 Pose 5B로 확정했다. 전체 65,595 frame
inference는 수행하지 않았으며, environment preparation과 실제 private frame 1장의 smoke test만
수행했다.

## 공식 구현 분석

- 공식 repository: [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2),
  검증 commit `7e5bae88456ac418ff0e58e74106c9fe192055d4`
- pose checkpoint: [facebook/sapiens2-pose-5b](https://huggingface.co/facebook/sapiens2-pose-5b)
- detector: [facebook/detr-resnet-101-dc5](https://huggingface.co/facebook/detr-resnet-101-dc5)
- model: 5.071B parameters, 15.722 TFLOPs, embed dim 2432, 56 layers, 32 heads
- input: top-down person crop, 1024×768 `(H×W)`
- output: Sociopticon 308 keypoints와 keypoint별 heatmap confidence

Pose model card에는 RTMDet 사용 문구가 있지만, 현재 공식 repository의 `docs/POSE.md`,
`scripts/demo/keypoints308.sh`, `tools/vis/vis_pose.py`는 모두 Hugging Face DETR
`facebook/detr-resnet-101-dc5`를 직접 사용한다. 추측으로 detector를 고르지 않고 실행 가능한
공식 implementation의 DETR 경로를 따랐다.

공식 demo는 DETR의 COCO person label 1을 threshold 0.3으로 선택하고 NMS 0.3을 적용한다.
검출이 없을 때 full-frame bbox로 fallback하는 upstream 동작도 확인했다. Phase 6-1에서는 이
fallback을 confidence/provenance에 반드시 표시해야 하며, 정상 detection과 동일하게 취급하지 않는다.

## Checkpoint와 환경

Checkpoint root 아래 구조는 다음과 같다. 실제 weight는 Git에 포함하지 않는다.

```text
<CHECKPOINT_ROOT>/
└── sapiens2/
    ├── pose/
    │   └── sapiens2_5b_pose.safetensors
    └── detector/
        └── detr-resnet-101-dc5/
            ├── config.json
            ├── preprocessor_config.json
            ├── model.safetensors
            └── pytorch_model.bin
```

- pose weight: 20,480,899,148 bytes, SHA-256
  `b4848da8691c72e14d3ff71319f077363107129bf4128019eb39d072129b2a52`
- Python 3.12.13
- PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118
- transformers 5.14.1, safetensors 0.8.0, OpenCV 5.0.0.93
- GPU: NVIDIA A100-SXM4-80GB, driver 535.183.06, compute capability 8.0

Driver가 보고하는 CUDA 12.2와의 backward-compatible 실행을 위해 PyTorch 공식 CUDA 11.8
wheel을 사용했다. 기본 Phase 0–5 environment는 변경하지 않고 별도 `sapiens2` conda environment를
만들었다. 전체 version/hash manifest는
[`configs/sapiens2_pose_5b_environment.json`](../../configs/sapiens2_pose_5b_environment.json)에 있다.

## 설치와 다운로드

```bash
conda create -y -n sapiens2 python=3.12 pip
conda run -n sapiens2 python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
conda run -n sapiens2 python -m pip install -e <SAPIENS2_REPO>

HF_HOME=<CHECKPOINT_ROOT>/.cache/huggingface \
conda run -n sapiens2 hf download \
  facebook/sapiens2-pose-5b sapiens2_5b_pose.safetensors \
  --local-dir <CHECKPOINT_ROOT>/sapiens2/pose

HF_HOME=<CHECKPOINT_ROOT>/.cache/huggingface \
conda run -n sapiens2 hf download facebook/detr-resnet-101-dc5 \
  --local-dir <CHECKPOINT_ROOT>/sapiens2/detector/detr-resnet-101-dc5
```

## 좌표와 keypoint convention

공식 pipeline은 BGR source를 RGB/ImageNet normalization한 뒤 bbox를 center/scale로 변환하고
UDP affine crop을 1024×768로 만든다. Heatmap decode 후 다음 공식 식으로 원본 image pixel
좌표 `(x, y)`에 복원한다.

```text
xy_original = xy_crop / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
```

Smoke 결과의 308 coordinates와 308 confidence는 모두 finite였고 confidence ≥0.3인 point의
100%가 원본 frame 안에 있었다. 308-entry flip mapping은 involution이며 body의 eye/ear,
shoulder/elbow/wrist, hip/knee/ankle/toe/heel left-right pair가 모두 공식 name mapping과 일치했다.

## Smoke test

```bash
conda run -n sapiens2 python tools/sapiens2_pose_smoke.py \
  --image <PRIVATE_REPRESENTATIVE_FRAME> \
  --sapiens2-root <SAPIENS2_REPO> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --warmup 1 \
  --repeats 3 \
  --output-json outputs/local/sapiens2/smoke.json \
  --save-visualization outputs/local/sapiens2/smoke_pose.jpg
```

- person detection: 1명 성공, bbox가 원본 720×1280 pixel 좌표에서 인물 전체를 포함
- pose model load: 성공, 약 58.46 s
- output: 308 keypoints + 308 confidence, finite PASS
- precision: 공식 demo 기본 FP32
- flip test: enabled, 공식 config 그대로
- end-to-end latency: 4.517 s/image median, warm-up 1회 후 3회
- peak CUDA allocated: 19.986 GiB
- peak CUDA reserved: 20.961 GiB
- visual QA: body/hand/foot 배치와 좌우 ordering이 plausible, bbox/coordinate 복원 정상

Private frame, bbox/keypoint payload와 visualization은 ignored `outputs/`에만 두었고 commit하지
않았다. Detector load 시 transformers가 네 개 BatchNorm `num_batches_tracked` key를 unexpected로
알렸지만 공식 safetensors로 1-person detection이 정상 동작해 non-blocking compatibility note로
기록한다.

## 5B teacher 결정과 Phase 6-1 gate

5B는 A100 80GB에서 OOM이나 pipeline instability 없이 동작하므로 primary offline teacher로
확정한다. 1B comparison/downgrade는 수행하지 않았다. 현재 단일 job 기준 전체 frame 단순 외삽은
약 82.3 GPU-hours이며, 공식 demo의 2 jobs/GPU 전략은 메모리상 pilot 검증 가치가 있다. 실제
throughput, detector fallback rate, crop 수 분포와 output schema를 Phase 6-1 소규모 pilot에서 먼저
검증한 후 26 sequence 전체 실행을 승인한다.

Phase 6-1 진행 가능 상태다. 아직 전체 inference, triangulation, SAM-Body4D, MHR, SMPL 또는
pseudo-label generation은 수행하지 않았다.

## Phase 6-1A — Primary target subject gate

Top-down pose 비용과 multi-view identity contamination을 막기 위해 dataset inference 전에
sequence/camera별 primary subject를 먼저 확정한다. Pipeline 경계는 다음과 같다.

```text
frame
  -> official DETR all-person candidates (private ragged metadata로 전부 보존)
  -> sequence-level forward/backward temporal tracking
  -> agreed primary target 0 또는 1 crop
  -> official Sapiens2 Pose 5B + flip-test
```

Target 초기화/선택은 단일 frame의 largest bbox, 최고 detector score 또는 화면 중심만으로 결정하지
않는다. Sequence 초반의 multi-frame coverage, 전체 track duration/span, detector score, bbox
location/relative area, 연속 frame IoU, normalized center displacement, scale/aspect continuity를 함께
사용한다. Forward와 backward tracker가 다른 candidate를 가리키거나 competing association margin이
작으면 `TARGET_AMBIGUOUS`, real DETR person candidate가 없으면 `NO_TARGET`으로 기록하고 5B pose를
강제로 실행하지 않는다. Appearance embedding은 pilot에서 bbox-temporal evidence가 부족하다고
확인될 때만 추가한다.

Private `target_selection.npz`는 최소 다음을 보존한다.

- `num_person_candidates`, `candidate_offsets`, `all_person_detections_xyxy`, detector scores
- `target_candidate_index`, forward/backward candidate index, selection confidence와 status
- `background_person_count`, ragged background bbox, occlusion risk
- detector duplicate / possible reflection evidence와 identity-switch risk
- PTS timestamp, refined camera geometry interface, cross-view visibility QA

실제 bbox coordinate, overlay와 frame name payload는 ignored `outputs/` 아래에만 둔다. Public
`metadata/results/`에는 sequence/camera aggregate만 기록한다. Full triangulation은 이 gate에서 하지
않으며 PTS와 Phase 5 `cameras_refined.json`을 후속 cross-view identity QA interface로만 연결한다.

현재 pilot에서 모든 person crop을 처리한 결과는 버리지 않고 `ALL_DETECTIONS_BASELINE`으로
분류한다. Selector 적용 뒤 동일 pilot의 `TARGET_ONLY` workload를 별도 output root에서 실행하고,
batch 1/2/4/8/12/16의 numerical equivalence, GPU utilization, VRAM, power, image throughput와
person-crops/s를 다시 측정한다. 전체 65,595-frame runtime은 target-only 결과로만 확정한다.

```bash
conda run -n sapiens2 python tools/detr_person_candidates.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --output-root outputs/detr_person_candidates \
  --runtime-dir outputs/runtime/phase6_1_detr \
  --sequences <EXPLICIT_SEQUENCE_ALLOWLIST>

python tools/target_subject_selection.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --detections-root outputs/detr_person_candidates \
  --output-root outputs/target_selection \
  --runtime-dir outputs/runtime/phase6_1a

conda run -n sapiens2 python tools/sapiens2_target_pipeline.py benchmark \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --selection-root outputs/target_selection \
  --all-detections-root outputs/sapiens2 \
  --batch-sizes 1,2,4,8,12,16 \
  --output-dir outputs/runtime/phase6_1_target_benchmark

conda run -n sapiens2 python tools/sapiens2_target_pipeline.py infer \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --selection-root outputs/target_selection \
  --output-root outputs/sapiens2_target_only \
  --runtime-dir outputs/runtime/phase6_1_target \
  --batch-size <ACCEPTED_BATCH>
```

Full dataset command는 이 pilot command들과 분리하며 acceptance report 전에 자동 호출하지 않는다.
최종 gate는 `GO_FULL_DATASET`, `REVIEW_TARGET_SELECTION`, `NO_GO` 중 하나로 명시한다.
현재 all-person pilot은 같은 official DETR candidate를 이미 lossless ragged array로 저장했으므로
Phase 6-1A에서는 해당 cache를 재사용한다. 이후 dataset run은 `detr_person_candidates.py`로 pose 없는
detection pass를 먼저 수행한다.

## Phase 6-1A 결과와 Visual Gate

4개 pilot sequence의 3-view, 총 9,732 frame을 검사했다. Official DETR candidate 19,596개는
private ragged metadata에 모두 보존했고 target으로 확정한 9,725 crop만 pose-eligible로 만들었다.
따라서 all-person 대비 crop reduction은 50.3725%다.

- obvious identity switch 0, forward/backward disagreement 0
- `NO_TARGET` 0, `TARGET_AMBIGUOUS` 7(0.0719%)
- ambiguity 7건은 모두 `pushup_0001/cam1`: duplicate underdetermination 1건과 상·하체가
  보완적으로 갈라진 DETR fragmentation 6건
- ambiguous frame에서 잘못된 다른 사람을 강제 선택하지 않았고 나머지 두 view는 target 유지
- background crossing, target/background overlap, mirror 후보, 누운 benchpress, prone pushup,
  candidate order 변화, background bbox가 target보다 큰 사례의 private overlay를 확인
- severe-occlusion 후보 `latpulldown_0002/cam2` 1,136 frame도 추가 detection/selector preflight:
  평균 2.121 persons/frame, 최대 대표 장면 5 candidates, identity switch/ambiguity 0

Private overlay, bbox/keypoint coordinate와 source frame은 Git에 포함하지 않는다. 공개 aggregate는
[`target_subject_selection_pilot.csv`](../../metadata/results/target_subject_selection_pilot.csv)에 있다.

## Target-only batch scaling과 전체 runtime projection

Representative target crop 16개에 대해 official DETR timing과 Sapiens2-5B FP32 flip-test를 포함해
batch 1/2/4/8/12/16을 측정했다. 모든 batch가 batch 1 및 보존된 all-person target baseline과
허용오차 내 numerical equivalence를 통과했다. Raw fastest는 batch 16의 0.231951 crop/s지만,
batch 4는 0.230449 crop/s로 0.648%만 느리고 reserved VRAM을 약 13.625 GiB 덜 사용한다.
따라서 fastest의 99% plateau에서 가장 작은 batch 4를 최종 권장한다.

4-sequence crop 분포를 전체 65,595 frame에 외삽하면 약 65,548 target crop이고, measured stage
timing 기반 total은 약 79.09 GPU-hours다. all-person workload 외삽 157.38 GPU-hours보다 약
78.30 GPU-hours 감소한다. 이는 실측 pilot 기반 projection이며 전체 실행 전 추정치다.

Phase 6 target-selection 판정은 `GO_FULL_DATASET`이지만, 전체 inference는 사용자 보고와 명시적
승인 전 `HOLD`한다. Aggregate batch와 stage breakdown은 각각
[`sapiens2_target_only_batch_scaling.csv`](../../metadata/results/sapiens2_target_only_batch_scaling.csv),
[`phase6_2_runtime_projection.csv`](../../metadata/results/phase6_2_runtime_projection.csv)에 있다.
