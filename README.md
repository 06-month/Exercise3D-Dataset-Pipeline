<div align="center">

# Exercise3D Dataset Pipeline

**고정된 3대 카메라로 촬영한 운동 영상에서 3D pseudo-label 데이터셋을 만드는 end-to-end 파이프라인**

동기화 → 카메라 보정 → 2D pose → multi-view triangulation → sequence body fitting → 품질 라벨

![status](https://img.shields.io/badge/freeze--ready-24%2F26_sequences-16a34a)
![sequences](https://img.shields.io/badge/sequences-26-2563eb)
![frames](https://img.shields.io/badge/working_frames-65%2C595-2563eb)
![views](https://img.shields.io/badge/camera_views-78-2563eb)
![preview](https://img.shields.io/badge/public_preview-mesh--only-e07b39)

</div>

> **이 저장소는 데이터셋을 배포하지 않습니다.** 파이프라인 코드, 검증 방법, 비식별 집계 결과와
> mesh-only preview만 공개합니다. 원본 영상, 얼굴 pixel, 개인정보, checkpoint, 대용량 geometry
> payload는 포함하지 않습니다.

---

## Preview

아래 미리보기는 실제 파이프라인 출력에서 만든 **3-view MHR mesh-only render**입니다.
왼쪽부터 `cam1` / `cam2` / `cam3`이며, 세 view가 같은 timestamp에서 일관된 자세를 보여줍니다.
촬영 RGB, 배경, 얼굴, audio, numeric label은 들어 있지 않습니다.

<table>
<tr>
<td width="50%" align="center">
<b>Bench press</b> · <code>benchpress_0004</code><br>
<img src="docs/assets/showcase/benchpress_0004_mhr_mesh.gif" width="100%"><br>
428 frames · <a href="docs/assets/showcase/benchpress_0004_mhr_mesh.mp4">전체 MP4</a>
</td>
<td width="50%" align="center">
<b>Deadlift</b> · <code>deadlift_0001</code><br>
<img src="docs/assets/showcase/deadlift_0001_mhr_mesh.gif" width="100%"><br>
518 frames · <a href="docs/assets/showcase/deadlift_0001_mhr_mesh.mp4">전체 MP4</a>
</td>
</tr>
<tr>
<td width="50%" align="center">
<b>Barbell row</b> · <code>barbellrow_0003</code><br>
<img src="docs/assets/showcase/barbellrow_0003_mhr_mesh.gif" width="100%"><br>
379 frames · <a href="docs/assets/showcase/barbellrow_0003_mhr_mesh.mp4">전체 MP4</a>
</td>
<td width="50%" align="center">
<b>Lat pulldown</b> · <code>latpulldown_0003</code><br>
<img src="docs/assets/showcase/latpulldown_0003_mhr_mesh.gif" width="100%"><br>
331 frames · <a href="docs/assets/showcase/latpulldown_0003_mhr_mesh.mp4">전체 MP4</a>
</td>
</tr>
<tr>
<td width="50%" align="center">
<b>Squat</b> · <code>squat_0002</code><br>
<img src="docs/assets/showcase/squat_0002_mhr_mesh.gif" width="100%"><br>
471 frames · <a href="docs/assets/showcase/squat_0002_mhr_mesh.mp4">전체 MP4</a>
</td>
<td width="50%" align="center">
<b>Push-up</b> · <code>pushup_0000</code><br>
<img src="docs/assets/showcase/pushup_0000_mhr_mesh.gif" width="100%"><br>
312 frames · <a href="docs/assets/showcase/pushup_0000_mhr_mesh.mp4">전체 MP4</a>
</td>
</tr>
</table>

GIF는 6초 발췌본이고, MP4는 전체 sequence(15 fps)입니다. `pushup_0000`의 `cam1`은 몸의 장축을
정면에서 바라보는 위치라 plank 자세가 짧게 투영됩니다 — 세 view의 시점 차이를 그대로 보여주는
예시입니다. 생성 방법과 공개 경계는 [showcase 문서](docs/showcase.md)에 있습니다.

---

## 한눈에 보기

| | |
|---|---|
| **피험자** | 3명 (저장소에는 비식별 aggregate만 기록) |
| **Sequence** | 26 synchronized triple-view sequences |
| **Camera view** | 78 (`cam1` iPhone 16, `cam2` iPhone 16 Pro, `cam3` iPhone 17) |
| **Frame** | working JPEG 65,595장, source 30/30/60 fps → working 30 fps |
| **운동 종류** | bench press, deadlift, squat, barbell row, lat pulldown, push-up |
| **Label** | camera pose, 3D joint, MHR body parameter, frame/sequence quality vector |
| **동기화 정확도** | PTS offset median 11.99 ms / p95 25.28 ms / max 31.38 ms |
| **현재 상태** | **24/26 sequence end-to-end 완료** (REVIEW 24 / FAIL 0) |

---

## Pipeline

<div align="center">
  <img src="docs/assets/pipeline.svg" width="100%" alt="Exercise3D pipeline overview">
</div>

| 단계 | 하는 일 | 대표 결과 |
|---|---|---|
| **Input** | audio clap 기반 synchronization과 PTS provenance 확보 | offset median 11.99 ms |
| **Camera geometry** | stability audit → VGGT-Ω initialization → fixed-camera Background BA | 78/78 `FIXED_CAMERA_OK`, BA 26/26 수렴 |
| **Human motion** | Sapiens2 2D pose → PTS-aware triangulation → SAM-Body4D MHR prior → sequence body fit | body fit 24/26 sequence |
| **Labels & freeze** | frame/sequence quality vector, Fit3D metric validation, immutable freeze | freeze-ready 24/26, FAIL 0 |

원본 RGB는 다시 보간하거나 덮어쓰지 않습니다. geometry initialization과 최종 camera calibration을
구분하고, 모든 단계는 불확실성을 숨기지 않고 downstream으로 전달합니다.

---

## 현재 상태 — 24/26 freeze-ready

- 전체 workload: **26 sequences / 78 views / 65,595 frames**
- end-to-end 완료: **24 sequences** (body fitting + quality metadata까지, immutable checkpoint 검증 통과)
- quality 상태: **REVIEW 24 / FAIL 0** — 불확실성을 숨기거나 PASS로 승격하지 않았습니다
- 미완료: `deadlift_0002`, `squat_0003` (`INCOMPLETE_DEADLINE` provenance 유지)
- 다음 단계: A100급 GPU가 확보되면 동일 설정으로 두 sequence의 남은 stage만 resume

`24/26`은 deadline 시점의 정직한 스냅샷입니다. 완료된 output은 checksum과 schema가 일치하면
재계산하지 않습니다. Phase별 상태표와 실행 노트는 [docs/status_notes.md](docs/status_notes.md)에
있습니다.

---

## 빠른 시작

Python 3.10 이상과 `ffmpeg`/`ffprobe`가 필요합니다.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Private dataset 경로는 하드코딩하지 않고 환경변수로 주입합니다
([configs/paths.example.env](configs/paths.example.env)).

```bash
export EXERCISE3D_DATASET_ROOT=/path/to/private/exercise3d
export EXERCISE3D_VGGT_REPO=/path/to/local/vggt-omega
export EXERCISE3D_VGGT_CHECKPOINT=/path/to/checkpoints/vggt_omega_1b_512.pt
```

```bash
# Phase 0 — inventory / integrity
python tools/dataset_inventory.py --dataset-root "$EXERCISE3D_DATASET_ROOT" \
  --output-dir outputs/local/inventory

# Phase 2 — temporal synchronization QA
python tools/temporal_sync_audit.py --dataset-root "$EXERCISE3D_DATASET_ROOT" \
  --output-dir outputs/local/temporal_alignment

# Phase 3 — VGGT-Ω initialization
python tools/vggt_geometry_init.py --dataset-root "$EXERCISE3D_DATASET_ROOT" \
  --vggt-repo "$EXERCISE3D_VGGT_REPO" --checkpoint "$EXERCISE3D_VGGT_CHECKPOINT" \
  --output-dir outputs/local/vggt
```

VGGT-Ω, Sapiens2 Pose 5B, SAM 3D Body checkpoint는 각 upstream에서 별도로 준비합니다. 이 저장소는
checkpoint를 재배포하지 않습니다. Sapiens2는 Python 3.12/PyTorch 2.7 이상 환경이 필요하므로
Phase 0–5 환경과 분리합니다 ([Phase 6 문서](docs/phases/phase_6_sapiens2.md)).

전체 tool 목록은 [docs/tools.md](docs/tools.md)에 있습니다.

---

## 저장소 구조

```text
Exercise3D-Dataset-Pipeline/
├── README.md            # 이 문서
├── plan.md              # phase별 acceptance gate
├── process.md           # 시간순 실행 기록
├── configs/             # freeze된 수치 default와 환경 정의
├── docs/
│   ├── assets/          # 공개 showcase render, 다이어그램
│   ├── design/          # schema, 좌표 규약, privacy policy, 재현성
│   ├── phases/          # phase 0-13 상세 문서
│   └── qa/              # visual QA checklist, troubleshooting
├── scripts/             # synchronization / dataset build
├── tools/               # phase별 실행 tool
├── metadata/results/    # 비식별 집계 결과
└── outputs/example/     # synthetic schema example
```

---

## 문서

| 문서 | 내용 |
|---|---|
| [docs/status_notes.md](docs/status_notes.md) | Phase 0–13 상태표와 실행 노트 |
| [docs/showcase.md](docs/showcase.md) | mesh-only preview 생성 방법과 공개 경계 |
| [docs/design/dataset_schema.md](docs/design/dataset_schema.md) | 최종 dataset schema |
| [docs/design/coordinate_conventions.md](docs/design/coordinate_conventions.md) | 좌표계 규약 |
| [docs/design/privacy_and_data_policy.md](docs/design/privacy_and_data_policy.md) | 개인정보·데이터 정책 |
| [docs/design/reproducibility.md](docs/design/reproducibility.md) | 재현성 요구사항 |
| [docs/tools.md](docs/tools.md) | tool별 역할과 source mutation 여부 |
| [docs/runtime_operations.md](docs/runtime_operations.md) | 장시간 자율 generation 운영 |

---

## QA 원칙

1. raw video, synchronized video, working frame은 immutable source로 취급합니다.
2. frame index보다 PTS provenance를 우선합니다.
3. model prediction은 ground truth가 아니라 noisy observation/prior입니다.
4. fixed physical camera마다 최종 pose 변수는 하나만 둡니다.
5. 사람과 움직이는 장비는 Background BA static track에서 제거합니다.
6. 단일 pixel threshold가 아니라 support, residual, pose plausibility, uncertainty, visual coherence를 함께 평가합니다.
7. `REVIEW`와 `FAIL`을 숨기지 않고 downstream에 전달합니다.
8. Phase acceptance gate를 통과하기 전에는 다음 대규모 단계로 진행하지 않습니다.

---

## 개인정보와 공개 정책

GitHub에 올리지 않는 것: raw/synchronized video, 추출 frame, 얼굴이 포함된 이미지, 개인정보와
GPS metadata, depth/point map/feature/track/checkpoint, 대용량 pseudo-label payload, 제3자 pretrained weight.

공개하는 것: 파이프라인 코드, 방법론, QA 기준, 비식별 수치 요약, synthetic schema example,
그리고 위 5개 sequence의 mesh-only preview.

commit 전에는 다음 검사를 통과해야 합니다.

```bash
git add <명시적 파일 목록>
python tools/check_publication_safety.py
```

`tools/check_publication_safety.py`는 명시적으로 allowlist된 showcase 파일 외의 모든 미디어·payload
확장자와 5 MiB 초과 파일을 차단합니다. 자세한 정책은
[docs/design/privacy_and_data_policy.md](docs/design/privacy_and_data_policy.md)에 있습니다.

---

## Pretrained model dependency

| 모델 | 역할 |
|---|---|
| VGGT-Ω | camera/depth/point-map initialization 전용 (최종 camera로 직접 사용하지 않음) |
| Sapiens2 Pose 5B | primary offline 2D pose teacher, official DETR person detector 사용 |
| SAM 3D Body / SAM-Body4D | MHR body representation과 temporal prior (Mode B 기본, Mode C selective) |
| Fit3D | Phase 12 정량 validation dataset 후보 |

각 모델의 라이선스, 배포 조건, checkpoint 사용 권한은 upstream 정책을 따릅니다.

---

## 라이선스와 인용

정식 오픈소스 라이선스 선택 전 단계이며 기본적으로 모든 권리를 보유합니다. `LICENSE`와 각
third-party upstream license를 확인하십시오. citation 정보는 final dataset freeze 시 추가합니다.
