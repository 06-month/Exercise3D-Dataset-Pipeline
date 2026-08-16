# Runtime operations — 장시간 자율 generation

README에서 분리한 운영 문서입니다. `HANDOFF.md`와 `AGENTS.md`는 agent 전용 로컬 문서로
Git에서 제외되어 있으므로, 이 문서는 저장소에 남는 운영 요약만 담습니다.

장시간 generation을 이어받는 agent는 다른 문서보다 먼저
`HANDOFF.md`를 읽습니다. 실시간 private command/PID/progress는 Git에서 제외된
`.runtime/handoff_state.json`에 30초 간격으로 atomic 저장되며, 살아 있는 inference를 중복
실행하지 않는 startup 순서는 `AGENTS.md`에 고정했습니다.
사람용 live dashboard와 machine-readable attention state는
`tools/monitor_autonomous_generation.py`가 `.runtime/dashboard_state.json`에 atomic 저장합니다.
Top-level `last_completed_event`는 polling state 갱신 시각이 아니라 atomic camera/sequence output과
immutable build manifest 중 가장 최근 durable completion을 구조화해 기록합니다.
Dashboard는 frozen selector summary의 exact crop/frame workload와 live measured stage rate를 결합해
deadline까지 terminal 가능 sequence 수의 `OPTIMISTIC_UPPER_BOUND`도 표시합니다. 이 값은
triangulation/body-fit/quality/export overhead를 제외한 ceiling이며 완료 약속으로 해석하지 않습니다.
완료된 sequence의 SAM-complete→body-fit/Mode-C terminal provenance에서 관측한 post-SAM latency도
별도로 집계해 `EMPIRICAL_P90_POST_SAM_ADJUSTED` schedule을 함께 표시합니다. 이 값 역시
미래 rate/latency 보장이 아니라 deadline risk 범위입니다.
Phase 11 CPU follower는 quality가 완료된 sequence에 대해 final exporter과 동일 validation을
미리 수행하고 dashboard에 `freeze-ready` count와 지속 dependency failure를 보고합니다.
Follower는 lifetime singleton lock으로 duplicate writer를 거부합니다. 별도 exact-identity watchdog은
validated quality/freeze-readiness 26/26 전까지 연속 process absence와 final rescan을 통과한 경우에만
제한된 detached recovery를 수행합니다.
별도 CPU-only predeadline checkpoint follower는 이 ready 집합이 기존 byte-verified
checkpoint의 strict superset이 되었을 때만 deterministic immutable build를 추가합니다.
동일 집합은 재export하지 않으며, final deadline snapshot과 별도 build ID/state를 사용합니다.
그 follower의 watchdog은 live/resume argv digest를 pin하고 연속 absence와 final rescan 후에만
detached recovery합니다. Follower lifetime lock이 recovery race의 duplicate launch를 차단하며,
watchdog은 deadline 도달 이후에는 follower를 재실행하지 않습니다.
별도 CPU-only supervisor watchdog은 살아 있는 supervisor의 exact argv를 persisted resume
command와 digest pin하고, 3회 연속 absence와 final rescan을 통과한 때만 자동
resume합니다. 신규 supervisor는 lifetime advisory lock으로 launch race에서도 중복
stage 실행을 거부합니다.
완료된 expensive camera output에는 checkpoint/config/source/selection/tool/command identity를
담은 `run_provenance.json`을 별도 atomic sidecar로 남깁니다.
고정 deadline에는 별도 private snapshot build가 현재 PASS/REVIEW/FAIL/INCOMPLETE 상태를 보존하며,
장기 generation 자체는 snapshot 이후에도 중단하지 않습니다.
Snapshot membership은 terminal body-fit/Mode-C marker의 deadline cutoff로 고정하여 export/retry 도중
완료된 sequence를 소급 포함하지 않으며, transient failure는 hidden staging에서 checksum-resume합니다.
Cutoff-eligible sequence의 derived sidecar가 순간적으로 누락된 경우 최대 90초를 재시도하되,
최종 시도에도 불완전하면 INCOMPLETE를 숨기지 않고 immutable snapshot을 publish합니다.
Freeze contract v2는 요청한 26-sequence universe/order와 필수 payload set을 manifest에 bind하여
INCOMPLETE row나 quality/provenance file을 누락한 build을 integrity PASS로 인정하지 않습니다.
Deadline sentinel과 build ID별 exporter는 각각 lifetime/advisory lock을 유지하여 recovery
race에서도 동일 snapshot staging/copy/publish를 중복 실행하지 않습니다. Sentinel
watchdog은 exact persisted command identity와 연속 process absence를 확인한 뒤에만 제한된
자동 recovery를 수행합니다.
Freeze copy는 source symlink를 거부하고 single open descriptor의 inode/size/time identity를
hash·copy 전후로 검증하며, file과 final directory rename을 fsync한 뒤만 publish합니다.
Dashboard의 export state는 final deadline build progress와 이미 보존된 best durable checkpoint를
별도 field로 표시하여, deadline build이 아직 0이라도 유효한 checkpoint sequence를 숨기지 않습니다.
