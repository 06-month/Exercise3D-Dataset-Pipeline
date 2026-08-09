# Troubleshooting / Decisions

## Open3D import 또는 GUI 실패

viewer는 `EXERCISE3D_OPEN3D_PYTHON`으로 별도 runtime을 지정할 수 있다. display가 없는 서버는
`--headless --save-screenshot`을 사용할 수 있지만 생성 image는 private debug output에만 둔다.

## SciPy/OpenCV runtime 불일치

`EXERCISE3D_BACKGROUND_BA_PYTHON`으로 검증된 runtime을 지정한다. Phase 5 재현 시 dependency를
바꾸기 전에 frozen configuration과 source hash 차이를 기록한다.

## External dataset에 output이 생기는 문제

`--output-dir`/`--output-root`를 public repository의 ignored `outputs/local/...`로 명시한다.
geometry 도구는 raw/synchronized/working-frame subtree와 output overlap을 거부한다.

## Stage 2가 max nfev에 도달

실패를 자동 PASS 처리하거나 threshold를 조용히 바꾸지 않는다. input integrity, Stage 1 상태,
track support와 visual coherence를 먼저 검토하고, 제외/fallback/re-optimization 중 정책을 별도
승인한다.

## Git push 실패

remote, branch, authentication과 remote divergence를 read-only로 확인한다. force push나 history
rewrite 없이 fetch/rebase/PR 정책을 따른다. private payload가 stage됐으면 즉시 unstage하고
추적 여부를 다시 검사한다.
