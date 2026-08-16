# Exercise3D mesh-only showcase

이 showcase의 목적은 private 운동 영상을 배포하지 않고도 Exercise3D 파이프라인이 만든
시간적으로 일관된 multi-view body reconstruction을 보여주는 것입니다.

## 공개 범위

공개 영상은 다음 다섯 sequence의 MHR mesh-only render만 사용합니다.

- `benchpress_0004`
- `deadlift_0001`
- `barbellrow_0003`
- `latpulldown_0003`
- `squat_0002`

각 MP4는 세 camera view를 나란히 표시합니다. 원본 RGB, 촬영 배경, 얼굴 pixel, audio,
frame-level numeric label, mesh vertex/face payload와 checkpoint는 포함하지 않습니다.
영상은 원본을 읽지 않는 `tools/render_public_mesh_showcase.py`로 생성했습니다.

## 재생

| Sequence | Frames / FPS | Video |
|---|---:|---|
| `benchpress_0004` | 428 / 15 fps | [MP4](assets/showcase/benchpress_0004_mhr_mesh.mp4) |
| `deadlift_0001` | 518 / 15 fps | [MP4](assets/showcase/deadlift_0001_mhr_mesh.mp4) |
| `barbellrow_0003` | 379 / 15 fps | [MP4](assets/showcase/barbellrow_0003_mhr_mesh.mp4) |
| `latpulldown_0003` | 331 / 15 fps | [MP4](assets/showcase/latpulldown_0003_mhr_mesh.mp4) |
| `squat_0002` | 471 / 15 fps | [MP4](assets/showcase/squat_0002_mhr_mesh.mp4) |

## 재생성

Private SAM-Body4D output을 보유한 환경에서 다음 형태로 생성합니다.

```bash
python tools/render_public_mesh_showcase.py \
  --sequence deadlift_0001 \
  --mesh-render-root outputs/sam_body4d_full \
  --output docs/assets/showcase/deadlift_0001_mhr_mesh.mp4
```

기본값은 source 30 fps에서 매 두 번째 frame을 취해 15 fps로 기록합니다. 시간 길이는 유지되고
GitHub에서 다루기 쉬운 크기로 줄어듭니다. 이 renderer는 source-frame 인자를 제공하지 않으므로
실수로 실제 촬영 pixel을 합성할 수 없습니다.

## 진행 상태

Deadline checkpoint 기준 26개 중 24개 sequence가 end-to-end freeze-ready입니다. 완료된 24개는
모두 REVIEW 상태를 정직하게 유지하며 FAIL은 없습니다. `deadlift_0002`와 `squat_0003`은
INCOMPLETE이며, 충분한 GPU 자원이 확보되면 기존 completion metadata와 checksum을 재검증한 뒤
미완료 stage만 resume할 계획입니다.
