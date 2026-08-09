# Phase 0 — Dataset Inventory / Integrity

## 목적

원본, synchronization derivative, working frame의 관계를 추측 없이 기록하고 후속 단계가
참조할 immutable provenance를 확정한다.

## 방법

- `ffprobe`로 stream, packet/frame PTS, duration, resolution, frame rate를 수집한다.
- filename과 directory에서 exercise/take/camera mapping을 만들고 triple-view 완결성을 검사한다.
- raw와 derivative의 수량, working-frame count, manifest 경로를 전수 확인한다.
- hash는 provenance가 필요한 private report에만 저장하며 public summary에는 source path를 싣지 않는다.

## 결과

- subjects 3, sequences 26, cameras 3
- raw 78, synchronized 78, working JPEG 65,595
- camera native rate 30/30/60 fps, working derivative 30 fps
- inventory/integrity PASS, source modification 0

## Gate

raw와 60 fps source를 보존하고 camera stability 및 temporal QA로 진행했다.
