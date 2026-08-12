# Exercise3D Agent Startup Protocol

이 저장소에서 작업을 시작하는 모든 agent는 다음 순서를 지킨다.

1. repository root의 `HANDOFF.md`를 먼저 읽는다.
2. ignored local state `.runtime/handoff_state.json`, `.runtime/dashboard_state.json`,
   `outputs/runtime/autonomous_generation/autonomous_generation_state.json`을 확인한다.
3. `plan.md`에서는 현재 phase만, `process.md`에서는 최신 기록만 확인한다.
4. 관련 phase 문서와 마지막 완료 output의 completion metadata/schema를 확인한다.
5. `git status --short --branch`, 현재 branch, HEAD와 remote를 확인한다.
6. `python tools/monitor_autonomous_generation.py --once`로 process/GPU/progress/attention을
   한 번 확인한다. Monitor가 없거나 stale일 때만 `ps`, `nvidia-smi`, process tree를 직접 확인한다.
7. 기존 job이 살아 있으면 동일 command를 중복 실행하거나 정상 inference를 중단하지 않는다.
8. job이 죽었을 때만 live process absence와 child 부재를 재확인한 뒤 local handoff state의
   exact command와 `HANDOFF.md` resume 절차로 재개한다.
9. resume는 completion metadata/checksum/schema가 PASS인 item을 건너뛰고 incomplete/corrupt item만 다시 계산해야 한다.
10. 정상 상태는 AI가 반복 polling하지 않는다. deadline-first autonomous execution을 계속하며
    실제 attention/recovery/milestone에서만 개입하고 `HANDOFF.md`, `plan.md`, `process.md`를 갱신한다.

원본·synchronized video·working frame은 immutable이다. Private frame, checkpoint, mesh,
pseudo-label payload, credential 및 절대 private 경로를 public Git에 추가하지 않는다.
