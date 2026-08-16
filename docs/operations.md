# Runtime operations

장시간 자율 generation을 운영할 때의 규칙입니다. 실시간 command/PID/progress와 agent 전용
handoff 문서(`HANDOFF.md`, `AGENTS.md`)는 Git에서 제외되어 있으므로, 여기에는 저장소에 남는
운영 계약만 정리합니다. 관련 tool은 [tools.md](tools.md)의 *Runtime supervision* 절에 있습니다.

## 1. 상태 파일

| 파일 | 쓰는 주체 | 내용 |
|---|---|---|
| `.runtime/handoff_state.json` | `checkpoint_handoff_state.py` | live command/PID/progress를 30초 간격으로 atomic 기록 |
| `.runtime/dashboard_state.json` | `monitor_autonomous_generation.py` | 사람용 dashboard와 machine-readable attention state |

두 파일 모두 Git에서 제외됩니다. 새 process를 띄우기 전에 항상 먼저 읽어, 살아 있는 inference를
중복 실행하지 않습니다.

## 2. Dashboard가 보고하는 값

- `last_completed_event` — polling state 갱신 시각이 아니라, atomic camera/sequence output과
  immutable build manifest 중 **가장 최근 durable completion**을 구조화해 기록합니다.
- `OPTIMISTIC_UPPER_BOUND` — frozen selector summary의 exact crop/frame workload와 live measured
  stage rate를 결합한 deadline까지의 terminal 가능 sequence 수. triangulation/body-fit/quality/export
  overhead를 제외한 ceiling이며 **완료 약속이 아닙니다**.
- `EMPIRICAL_P90_POST_SAM_ADJUSTED` — 완료 sequence의 SAM-complete → body-fit/Mode-C terminal
  provenance에서 관측한 post-SAM latency 기반 schedule. 미래 rate 보장이 아니라 deadline risk 범위입니다.
- export state — final deadline build progress와 이미 보존된 best durable checkpoint를 **별도 field**로
  표시합니다. deadline build이 아직 0이어도 유효한 checkpoint sequence를 숨기지 않습니다.

## 3. Follower

두 follower 모두 CPU 전용이며, lifetime singleton lock으로 duplicate writer를 거부합니다.

| Follower | 트리거 | 하는 일 |
|---|---|---|
| Phase 11 quality follower | body-fit 완료 감지 | final exporter과 동일한 validation을 미리 수행하고 `freeze-ready` count와 지속 dependency failure를 dashboard에 보고 |
| Predeadline checkpoint follower | ready 집합이 기존 byte-verified checkpoint의 **strict superset**일 때만 | deterministic immutable build 추가. 동일 집합은 재export하지 않고, final deadline snapshot과 별도 build ID/state 사용 |

## 4. Watchdog

Supervisor, monitor, 두 follower, deadline sentinel에 각각 watchdog이 있습니다. 공통 규칙:

1. persisted resume command의 **exact argv digest**를 pin한다.
2. 연속 process absence(supervisor watchdog은 3회)와 final rescan을 모두 통과한 경우에만 복구한다.
3. 복구는 제한된 detached 실행이며, lifetime advisory lock이 launch race의 중복 stage 실행을 막는다.
4. deadline 도달 이후에는 follower를 재실행하지 않는다.

> tool 경로와 argv 문자열은 watchdog identity의 일부입니다. `tools/` 아래 runtime 스크립트를
> 옮기거나 이름을 바꾸면 persisted state와의 identity 대조가 깨집니다.

## 5. Provenance와 freeze

- 완료된 expensive camera output에는 checkpoint/config/source/selection/tool/command identity를 담은
  `run_provenance.json`을 atomic sidecar로 남깁니다.
- 고정 deadline에는 별도 private snapshot build가 현재 PASS/REVIEW/FAIL/INCOMPLETE 상태를 보존하고,
  장기 generation 자체는 snapshot 이후에도 중단하지 않습니다.
- Snapshot membership은 terminal body-fit/Mode-C marker의 deadline cutoff로 고정합니다. export/retry
  도중 완료된 sequence를 소급 포함하지 않고, transient failure는 hidden staging에서 checksum-resume합니다.
- Cutoff-eligible sequence의 derived sidecar가 순간적으로 누락되면 최대 90초 재시도하되, 최종 시도에도
  불완전하면 INCOMPLETE를 숨기지 않고 immutable snapshot을 publish합니다.
- Freeze contract v2는 26-sequence universe/order와 필수 payload set을 manifest에 bind합니다. INCOMPLETE
  row나 quality/provenance file이 빠진 build은 integrity PASS로 인정하지 않습니다.
- Freeze copy는 source symlink를 거부하고, single open descriptor의 inode/size/time identity를 hash·copy
  전후로 검증하며, file과 final directory rename을 fsync한 뒤에만 publish합니다.
