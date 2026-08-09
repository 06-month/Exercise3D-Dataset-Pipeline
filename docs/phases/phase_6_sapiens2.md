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
