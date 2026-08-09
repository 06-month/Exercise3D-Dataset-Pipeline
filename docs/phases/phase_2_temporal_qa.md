# Phase 2 — Temporal Synchronization QA

## 목적

영상 시작뿐 아니라 beginning/middle/end에서 camera pair offset과 clock drift를 측정한다.

## 방법

packet/frame PTS를 우선하고 clap waveform cross-correlation과 visual motion energy를 보조
evidence로 사용했다. 3 camera pair의 offset curve를 sequence 전체에서 비교했다. video cut,
interpolation, resampling, overwrite는 수행하지 않았다.

## 결과

- 26 sequences, 78 camera pairs, actual-frame PTS offsets 546
- absolute offset median 11.99 ms, p95 25.28 ms, max 31.38 ms
- 546/546 observations가 30 fps 1 frame(33.33 ms) 이내
- stable 8, small constant 16, drift detected 2, insufficient evidence 0
- drift metadata: `pushup_0000`, `squat_0001`

## Gate

현재 sync는 dataset 수준에서 사용 가능하다. 보정값은 downstream matching timestamp 선택에만
사용하고 RGB를 다시 만들지 않는다.
