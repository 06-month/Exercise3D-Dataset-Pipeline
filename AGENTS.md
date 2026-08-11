# Exercise3D Agent Startup Protocol

이 저장소에서 작업을 시작하는 모든 agent는 다음 순서를 지킨다.

1. repository root의 `HANDOFF.md`를 먼저 읽는다.
2. `git status --short --branch`, 현재 branch와 HEAD를 확인한다.
3. `pgrep -af 'sapiens2_target_pipeline.py|run_autonomous_generation.py|run_sam_body4d_full.py|benchmark_sam_body4d.py|run_deadline_snapshot.py'`와 `nvidia-smi`로 실행 중 process/GPU를 확인한다.
4. ignored local state `.runtime/handoff_state.json`과 `outputs/runtime/autonomous_generation/autonomous_generation_state.json`을 확인한다.
5. `HANDOFF.md`에 적힌 마지막 완료 output의 completion metadata/schema를 검증한다.
6. `plan.md`에서는 현재 phase만, `process.md`에서는 최신 기록만 확인한다.
7. 기존 job이 살아 있으면 동일 command를 중복 실행하거나 정상 inference를 중단하지 않는다.
8. job이 죽었을 때만 local handoff state의 exact command와 `HANDOFF.md`의 resume 절차로 재개한다.
9. resume는 completion metadata/checksum/schema가 PASS인 item을 건너뛰고 incomplete/corrupt item만 다시 계산해야 한다.
10. deadline-first autonomous execution을 계속하며 major milestone마다 `HANDOFF.md`, `plan.md`, `process.md`를 갱신한다.

원본·synchronized video·working frame은 immutable이다. Private frame, checkpoint, mesh,
pseudo-label payload, credential 및 절대 private 경로를 public Git에 추가하지 않는다.
