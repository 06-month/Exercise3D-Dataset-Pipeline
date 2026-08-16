# Phase 5.1 — pushup_0003 Camera Recovery

## 목적과 제약

Phase 5의 유일한 FAIL인 `pushup_0003`을 대상으로 optimization budget 부족 여부만 검증했다.
static mask, SIFT/MAGSAC matching, track/filter/sample gate, fixed-K shared-camera model, Huber loss,
VGGT prior, Stage 1/2 objective, cam1 gauge와 sequence-local scale gauge는 변경하지 않았다.

## 재현 명령

```bash
python tools/background_bundle_adjust.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-root <PRIVATE_VGGT_ROOT> \
  --output-root outputs/local/background_ba/recovery_runs/nfev_600 \
  --sequence pushup_0003 \
  --max-nfev 300 \
  --stage2-max-nfev 600 \
  --optimizer-verbose 2

python tools/analyze_background_ba_recovery.py \
  --baseline <PHASE5_OUTPUT>/pushup_0003 \
  --control <CONTROL_300_OUTPUT>/pushup_0003 \
  --recovered <RECOVERED_OUTPUT>/pushup_0003 \
  --trace-log <RECOVERY_LOG> \
  --dataset-validation <FINAL_OUTPUT>/validation.json \
  --dataset-report <FINAL_OUTPUT>/background_ba_dataset_report.md
```

300-control은 Phase 5 baseline의 track/observation arrays, initial cameras,
`points_initial`/`points_stage1`, Stage 1 result와 Stage 2 cost를 재현했다. Recovery run과
baseline의 frozen configuration도 recovery-only budget/verbosity 필드를 제외하면 동일하다.

## 결과

- 기존 Stage 2: nfev 300, cost 1672.515861, max evaluations 종료
- recovery Stage 2: nfev 322/600, cost 1657.953684, `xtol` 정식 수렴
- reprojection median: 4.954229 → 2.558895 px
- reprojection p90: 8.037446 → 5.053964 px
- reprojection p95: 9.295044 → 7.055627 px
- support: final 21 tracks, 183 observations, 3-camera track 1개
- sample gate: GOOD 24 / DOWNWEIGHT 0 / REJECT 0

cam2가 camera별 residual tail이 가장 크지만 모든 sample이 GOOD이고, 300→322 evaluations에서
cam2/cam3 pose 변화는 매우 작았다. 따라서 특정 sample/camera의 폭발보다 sparse shared-camera
문제의 느린 termination settling으로 해석한다.

Open3D 비교에서 camera arrangement와 orientation은 합리적이고 initial/refined frustum이 거의
겹쳤다. mirror, 180° flip, exploding point cloud는 없었다. 다만 sparse support가 제한적이므로
최종 판정은 `RECOVERED_REVIEW`다.

Fallback은 사용하지 않았다. `camera_source=BACKGROUND_BA_RECOVERED`,
`camera_quality=REVIEW`이며 dataset 최종 상태는 PASS 11 / REVIEW 15 / FAIL 0이다.
Camera geometry freeze는 승인하되 REVIEW uncertainty를 downstream에 보존한다.
