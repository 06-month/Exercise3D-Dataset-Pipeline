# Exercise3D Chronological Engineering / Research Log

이 문서는 실제 수행 순서와 의사결정을 누적한다. 공개본에는 원본 절대 경로, 피험자
개인정보, media screenshot 및 대용량 numeric payload를 기록하지 않는다. 명령의 private
경로는 `<PRIVATE_DATASET_ROOT>`처럼 치환한다.

## 2026-08-11 — 2026-08-14 13:00 KST autonomous deadline 시작

### 2026-08-12 인계, supervisor 복구와 autonomous dashboard

- startup protocol에 따라 handoff/runtime/plan/process/phase/Git을 먼저 읽고 live process를 대조했다.
- Sapiens2 PID 373049, handoff monitor PID 608232, deadline sentinel PID 607755은 정상이고 GPU는
  100%였다. 반면 supervisor state는 23:45 UTC에서 멈췄고 supervisor/SAM process는 없었다.
- duplicate/child 부재를 재확인한 뒤 `.runtime/handoff_state.json`의 frozen exact command로
  supervisor만 복구했다. 기존 10개 PASS/REVIEW row는 재사용했으며 `latpulldown_0003` Phase 7→
  Mode B streaming을 이어갔다. Sapiens/sentinel/handoff monitor는 중단·재실행하지 않았다.
- `tools/monitor_autonomous_generation.py`를 추가했다. 기존 handoff/supervisor/deadline/output state와
  `/proc`, `nvidia-smi`, disk를 읽어 Rich live dashboard 및 atomic
  `.runtime/dashboard_state.json`을 생성하며 별도 progress DB는 만들지 않는다.
- attention은 expected process/supervisor/sentinel 사망, stale/stall, duplicate root process,
  traceback/CUDA OOM/retry exhaustion, validation/NaN-Inf gate FAIL, disk reserve, sustained GPU idle,
  deadline ETA risk/worsening을 탐지한다. 단발 `nvidia-smi` timeout은 마지막 정상 표본을 유지하고
  stale window를 넘겨야 경고한다.
- 첫 실제 dashboard snapshot: Sapiens 33/78 camera, 21,433/65,430 crop; SAM 30/78 camera,
  19,455/65,595 frame; triangulation 11, body fit 10; GPU 100%, combined 62,693 MiB.
  Runtime failure/OOM/retry는 없고 Sapiens ETA가 deadline보다 약 1.1시간 늦어질 위험만 경고했다.
- Mode C assessor의 intentional abstention all-NaN alignment row는 finite row에만 median을 계산하고
  finite value가 없으면 기존대로 NaN을 유지하도록 수정했다. Threshold/candidate/acceptance 변화는
  없으며 warning-free regression을 추가했다. Monitor/state/attention regression을 포함한 전체
  56개 unit test가 PASS했고 30초 state-only dashboard monitor를 시작했다.

### Deadline freeze atomic publication hardening

- Deadline sentinel/export 경로를 요구사항별로 감사했다. 기존 file/manifest write는 atomic이었지만
  build directory가 생성 중에도 final path에 노출되고 direct rerun이 같은 build ID를 갱신할 수 있어,
  immutable snapshot 경계가 충분히 강하지 않았다.
- Export는 hidden `.<build_id>.inprogress` root에서 partial copy를 checksum-resume하고,
  manifest-listed 34-file 기존 smoke를 포함해 byte/SHA-256, safe relative path, duplicate path,
  sequence status/count, freeze eligibility, privacy/source-mutation flag를 검증하도록 보강했다.
- 검증 PASS 뒤 directory rename으로 final build를 한 번에 publish한다. Final manifest가 이미 있으면
  전체 integrity PASS에서 read-only reuse만 하며 corrupt/incomplete final root를 같은 ID로 덮어쓰지 않는다.
- Sentinel도 parseable manifest만으로 COMPLETE 처리하지 않고 integrity verifier를 통과해야 한다.
  현재 살아 있는 sentinel은 deadline에 새 exporter subprocess를 호출하므로 restart하지 않았다.
- 기존 `exercise3d-streaming-smoke-v1`을 read-only 전수 검증해 34 files/28,960,929 bytes,
  mismatch 0을 확인했고 focused regression 7개와 전체 59개 unit test가 PASS했다.
- Dashboard는 sentinel의 `EXPORT_FAILED`, `EXPORT_INTEGRITY_FAILED`, `EXISTING_BUILD_INVALID`와
  구체적 integrity error를 `DEADLINE_SNAPSHOT_FAILED` attention으로 승격하도록 연결했다.

### Phase 11 pseudo-label quality vector streaming 시작

- 기존 Phase 6–9 output의 frame/PTS와 component evidence를 감사하고
  `tools/build_pseudolabel_quality.py`를 구현했다. Correlated learned signal을 calibrated accuracy
  probability나 단일 scalar로 축약하지 않고 source별 vector와 explicit reason bitmask를 저장한다.
- 완료된 10 sequence/6,485 reference frame을 CPU-only materialize했다. Sequence REVIEW 10/FAIL 0,
  target abstention/unmapped 8 view, SAM rejected/unmapped 8 view이며 prior-only/body-missing/
  triangulation-missing joint frame은 0이다. `pushup_0001` ambiguity 7은 강제 target 없이 그대로 남았다.
- Required field/shape/frame index/PTS/status count validation을 통과한 output만 resume-skip한다.
  실제 single-sequence 재호출에서 `resume_skipped=true`와 전역 10-sequence summary 보존을 확인했다.
- New supervisor code는 Mode C assessment 뒤 Phase 11을 호출한다. 현재 살아 있는 supervisor는 restart하지
  않았고, updated deadline/final exporter가 quality 누락 sequence를 CPU-only materialize한 뒤
  `quality/quality_vector.npz`와 metadata를 immutable private build에 포함한다.
- 현재 10 sequence 모두 final exporter validation REVIEW, dependency reason 0이다.
- Phase 11 builder unit/integration regression을 포함한 당시 전체 65개 unit test가 PASS했다.

### Phase 11 CPU-only follower 시작

- Live supervisor는 재시작하지 않았기 때문에 현재 process memory에는 Phase 11 stage가 없다.
  Deadline exporter fallback만으로도 correctness는 보존되지만 새 body-fit의 quality 오류가 deadline에서야
  드러날 수 있어 `tools/run_quality_control_follower.py`를 CPU-only background process로 추가했다.
- Dependency가 모두 있는 sequence만 처리하고, existing valid output은 resume-skip하며 sequence-local
  advisory lock으로 exporter/future supervisor와 concurrent duplicate write를 방지한다. Failure reason은 atomic
  runtime state에 남기고 300초 후 자동 재시도한다.
- 실제 one-shot에서 기존 10/26을 재계산 없이 REVIEW로 검증했고 failure 0,
  나머지 16은 body-fit/Mode-C/SAM prior dependency wait로 정확히 분류했다. 이후 30초 follower를 시작했다.
- Dashboard는 body-fit→quality lag에서 follower 사망과 follower `ATTENTION` failure reason을 자동 승격한다.
- Follower wait/build/retry, dashboard death/stale/failure 계약을 포함한 전체 69개 unit test가 PASS했다.

### Deadline freeze exact-tree hardening

- Resumable hidden staging에서 이전 crash temp나 이전에 complete였다가 현재 incomplete로 분류된
  sequence payload가 남아도 기존 verifier가 manifest-listed file만 검사해 final root에 포함할 수
  있는 공백을 확인했다.
- Exporter는 exact `.<build_id>.inprogress` root에서만 unlisted artifact를 prune하며, nested symlink은
  copy 전에 제거하고 mount point는 거부한다. Source/final build은 삭제하지 않는다.
- Verifier는 actual tree와 manifest tree, sequence status↔file owner, per-sequence↔global file record의
  path/byte/SHA를 exact-match한다. INCOMPLETE/FAIL sequence에 payload가 있으면 publish를 거부한다.
- Git HEAD만 기록하던 provenance를 dirty flag/status hash/tracked diff hash로 보강했다. Hash만
  저장하며 diff text/private path는 manifest에 포함하지 않는다.
- 기존 immutable smoke를 강화 verifier로 read-only 재검증해 34 files/28,960,929 bytes,
  error 0을 확인했다.
- Hardening commit `250ee73`의 clean worktree에서 `barbellrow_0000` quality-inclusive private smoke를
  새 build ID로 publish했다. REVIEW 1/FAIL 0/INCOMPLETE 0, 36 files/28,993,394 bytes,
  `git_worktree_dirty=false`, status/diff empty hash, exact-tree/ownership/SHA error 0이다.
- 같은 build ID를 즉시 재호출했을 때 copy/publish를 반복하지 않고
  `IMMUTABLE_BUILD_REUSED`로 36 files/28,993,394 bytes를 read-only 재검증했다.
- Exact-tree, stale payload, symlink root/target, INCOMPLETE ownership, Git provenance regression을 포함한
  전체 74개 unit test와 publication-safety가 PASS했다.

### Streaming freeze-readiness preflight

- Quality follower에 final exporter의 동일 `validate_sequence()` preflight를 연결했다. Quality만
  존재해도 pose/SAM run provenance, frame/PTS/finite, triangulation/body/quality gate 중 하나가
  깨지면 freeze-ready로 세지 않는다.
- INCOMPLETE dependency는 provenance monitor와 atomic stage publish의 순간적 lag을 허용하려고 300초
  grace를 두고, 이후에도 지속되면 `FREEZE_READINESS_FAILED`로 승격한다. Validation
  FAIL은 즉시 attention이며 valid output을 재계산하지 않는다.
- Freeze-ready dependency의 path/size/mtime signature를 매 cycle 비교해 completed payload가 교체/삭제되면
  해당 sequence만 exporter validation을 다시 수행하도록 했다. Numeric payload을 매번 재로드하지
  않으며 source output을 수정하지 않는다.
- 실제 10개 completed quality sequence를 CPU-only로 검증해 freeze-ready REVIEW 10,
  PASS 0/FAIL 0/dependency reason 0을 확인했다. Quality materialization은 0이어서 기존 payload를
  재계산하지 않았다.
- Readiness grace/failure/dashboard/dependency-change regression을 포함한 전체 77개 unit test와
  publication-safety가 PASS했다.

### `latpulldown_0003` autonomous completion

- Live supervisor가 Mode B cam3 완료 후 compact prior, Phase 9 fit, Mode C assessor까지 자동 관통했다.
  Full Mode B aggregate는 11 sequence/33 camera/21,441 frame이며 OOM/retry는 0이다.
- Body fit은 662×26, coverage/alignment 1.0, prior-only/missing 0, displacement p95 0.07748이다.
  Frozen gate에 따라 displacement + camera uncertainty REVIEW, FAIL 0을 유지했다.
- Mode C candidate 79 frame을 review metadata로 보존했지만 Mode C를 실행/채택하지 않았다.
  Quality follower가 즉시 662-frame vector와 exporter preflight를 완료해 quality/freeze-ready를
  11/26 REVIEW로 증가시켰고 failure/dependency reason은 0이다.

### Deadline point-in-time membership와 export retry

- 기존 sentinel은 deadline 후 exporter를 한 번만 호출했다. Transient copy/source race도
  복구하지 못하고, export 중 새로 완료된 sequence가 순서에 따라 포함될 수 있어
  exact deadline snapshot 경계가 불명확했다.
- Deadline exporter는 body fit NPZ/metadata와 Mode-C assessment의 mtime이 모두 deadline UTC
  이하인 sequence만 eligible로 분류한다. Post-deadline terminal marker는 retry 중에 발견되어도
  `INCOMPLETE`를 유지한다. Derived quality/manifest는 이 terminal payload에서 deadline 후 생성할
  수 있고, marker mtime은 sequence manifest에 보존해 final verifier가 다시 검증한다.
- Exporter가 final verified manifest를 만들지 못한 transient failure에는 기존 hidden staging을
  checksum-resume하며 30초 간격으로 최대 3회 재시도한다. Parseable final integrity error는
  immutable build을 덮어쓰지 않고 즉시 attention으로 종료한다.
- Cutoff-eligible이지만 quality/provenance sidecar lag로 INCOMPLETE인 sequence는 초기 3회에서
  manifest publish를 defer한다. 네 번째 최종 시도에는 defer를 해제해 recovery가 불가해도
  INCOMPLETE를 숨기지 않은 point-in-time snapshot을 생성한다.
- Clean commit `868c0a1`에서 cutoff smoke를 실행했다. `barbellrow_0000`은 REVIEW,
  terminal marker가 없던 `benchpress_0001`은 INCOMPLETE로 고정됐다. Final build는
  2 sequence/36 files/28,993,641 bytes, REVIEW 1/INCOMPLETE 1, exact-tree/marker/SHA error 0,
  `git_worktree_dirty=false`, `freeze_eligible=false`로 정직하게 publish됐다.
- Cutoff/retry/final truthful publish regression을 포함한 전체 81개 unit test와 publication-safety가 PASS했다.
- CPU sleeper sentinel만 PID 1834674로 교체했다. Persistent state는 `WAITING_DEADLINE`,
  point-in-time policy와 `--export-retries 3 --retry-seconds 30`을 보존한다. GPU inference/supervisor는
  중단/재시작하지 않았다.

### Freeze contract v2 sequence-universe binding

- 기존 verifier는 manifest 내부 count/CSV가 서로 맞는지는 검사했지만, requested 26개 중
  INCOMPLETE row 하나를 삭제한 뒤 count를 함께 바꾸는 외부-universe 누락을 독립적으로
  탐지할 수 없었다.
- Contract v2 manifest에 `requested_sequences`와 canonical ordered-list SHA-256을 추가했다.
  Status CSV identity/order가 exact-match해야 하며 sentinel은 자신의 frozen 26-sequence list를
  verifier에 별도 전달한다.
- Global provenance는 source inventory/temporal audit/frame mapping 3 files, complete PASS/REVIEW sequence는
  3-view target/pose/run provenance/SAM prior + geometry/body/quality/sequence manifest 33 files와 exact-match해야
  한다. Internal manifest와 tree에서 둘 다 누락한 payload도 이제 verifier가 거부한다.
- Missing INCOMPLETE row, order mutation, required quality payload omission regression을 포함한 전체
  83개 unit test와 publication-safety가 PASS했다.
- Clean commit `7b54214`에서 deadline partial smoke를 실행했다. Contract v2가 requested
  `barbellrow_0000,benchpress_0001` universe/order를 exact bind했고 REVIEW 1/INCOMPLETE 1,
  36 files/28,993,641 bytes, verifier error 0, dirty false, freeze eligible false로 검증됐다.
- 기존 CPU sleeper PID 1834674의 exact command와 child 0을 확인한 뒤 그 sentinel만
  TERM하고 PID 1846229로 교체했다. 새 process는 `WAITING_DEADLINE`이며 고정된
  26-sequence list를 final verifier에 별도로 전달하는 contract v2 code를 load했다.
  GPU inference와 autonomous supervisor는 중단·재시작하지 않았다.

### Supervisor single-instance recovery watchdog

- 이전에 supervisor 자체가 한 번 종료되어 agent가 수동 복구한 실제 event를 무인
  운영의 single point of failure로 확인했다. Dashboard attention만으로는 자동 복구되지 않았다.
- `run_autonomous_supervisor_watchdog.py`는 현재 live supervisor argv와 handoff의 exact resume
  command SHA-256이 같을 때만 identity를 pin한다. Supervisor가 3회 연속 보이지 않고
  2초 final scan에서도 없을 때만 shell 없이 detached resume하며, 1시간 3회로 제한한다.
- Watchdog singleton lock과 새 supervisor의 lifetime advisory lock을 결합해 동시 launch race에서
  하나만 stage를 진행한다. Duplicate가 보이면 어느 process도 kill하지 않고 attention만 남긴다.
- One-shot live identity gate에서 supervisor PID 1701200과 persisted command digest가 exact-match,
  launch 0, attention false였다. Dashboard는 watchdog death/stale/identity/restart exhaustion을
  machine-readable attention으로 승격한다.
- Code checkpoint `bd943fb`를 push한 뒤 watchdog PID 1864229를 persistent mode로 시작했다.
  Supervisor PID 1701200 observation, child/launch/restart 0, attention false를 확인했다.
- 새 watchdog schema를 load하도록 CPU-only handoff/dashboard monitor만 각각 PID
  1866064/1866198로 교체했다. 첫 handoff monitor 입력의 sequence typo는 즉시
  종료했고 private output을 건드리지 않았다. 수정 process의 frozen order는
  26/26 exact·unique, handoff universe exact, watchdog resume command 보존을 재검증했다.
- Persistent dashboard에서 watchdog RUNNING/restart 0이며 전체 attention은 기존
  `DEADLINE_ETA_AT_RISK` warning 하나뿐이다. GPU inference/supervisor는 중단·재시작하지 않았다.

### Deadline sentinel/export single-instance recovery hardening

- Deadline build의 hidden staging/atomic rename은 이미 강했지만 동일 build ID exporter 두 개가
  동시 시작하면 staging copy/prune/publish를 경쟁할 process-level lock이 없었다.
- Exporter에 build-ID-scoped non-blocking advisory lock을 추가했다. Lock loser는 staging을
  수정하기 전 `BUILD_ALREADY_IN_PROGRESS`/exit 75로 종료하고 sentinel retry path가 재검증한다.
- Deadline sentinel에도 lifetime lock을 추가해 동시 recovery launch를 stage 실행 전
  거부한다. `run_deadline_sentinel_watchdog.py`는 live/persisted exact argv digest를 pin하고,
  3회 연속 absence + 2초 final rescan + 1시간 3회 cap을 통과한 때만 detached recovery한다.
- One-shot identity gate에서 live sentinel PID 1846229와 persisted command digest exact-match,
  launch/restart 0, attention false, snapshot state `WAITING_DEADLINE`를 확인했다.
- 전체 96개 regression과 publication-safety PASS 후 code checkpoint `195d52a`를 push했다.
  Lock file symlink는 `O_NOFOLLOW`로 거부하고 target을 수정하지 않는 regression을 포함한다.
- Exact PID/command/child 0을 확인한 뒤 CPU sleeper sentinel만 PID 1882473으로 교체했다.
  `WAITING_DEADLINE`, single process, child 0이며 cross-process probe가 lifetime lock held를 확인했다.
- Sentinel watchdog PID 1882820을 persistent mode로 시작했다. Live PID 1882473과 persisted
  digest exact-match, launch/restart 0, attention false다. CPU-only handoff/dashboard monitor도 새 schema로
  PID 1883380/1883591에서 각각 단일 인스턴스로 실행 중이다.
- Frozen sequence order/universe는 26/26 exact·unique이고 handoff에 두 watchdog resume command가
  보존됐다. Dashboard attention은 `DEADLINE_ETA_AT_RISK` warning 하나뿐이며
  GPU inference/SAM/autonomous supervisor는 중단·재시작하지 않았다.

### Freeze source byte-snapshot/durability hardening

- 기존 `copy_exact()`은 source hash와 copied hash mismatch를 검출했지만 source path를 hash/copy에서
  두 번 다시 열어 concurrent atomic replacement의 point-in-time semantics가 명시적이지 않았다.
- Source symlink를 거부하고 `O_NOFOLLOW` regular-file descriptor 하나를 통해 hash와 copy를
  수행한다. Device/inode/size/mtime/ctime을 hash·resume-check·copy 전후에 비교하여
  in-place mutation이나 mixed byte stream을 reject한다.
- Copied file/temp metadata의 fsync 후 parent directory를 fsync하고, verified staging을 final build로
  rename한 뒤 output root도 fsync해 crash/power-loss 내구성을 보강했다.
- Source symlink/identity-change/temp cleanup regression을 포함한 전체 98 tests와 publication-safety가
  PASS했고 clean code checkpoint `f1b701e`를 push했다.
- Clean commit에서 `barbellrow_0000` 실제 private smoke를 새 build ID로 publish했다.
  Contract v2 REVIEW 1, 36 files/28,993,437 bytes, requested order/tree/ownership/SHA error 0,
  dirty false, freeze eligible true다. 동일 build ID 재실행은 `IMMUTABLE_BUILD_REUSED`로
  copy/publish 없이 36 files/28,993,437 bytes를 재검증했다.

### 11-sequence predeadline durable checkpoint

- Deadline sentinel을 기다리는 동안 이미 freeze-ready인 11 sequence 전체를 final deadline
  build과 다른 immutable build ID로 미리 보존했다. GPU inference/supervisor는 건드리지 않았다.
- Contract v2 result는 REVIEW 11/FAIL 0/INCOMPLETE 0, 366 files/344,922,733 bytes,
  requested order exact, tree/ownership/SHA error 0, dirty false, freeze eligible true다.
- 동일 build ID를 재호출해 `IMMUTABLE_BUILD_REUSED`와 366-file/344,922,733-byte
  전수 재검증을 확인했다. 남은 sequence generation과 deadline point-in-time snapshot은
  기존 autonomous process에서 계속한다.
- 기존 dashboard는 target deadline build이 아직 없으면 `Export 0 sequences`만 표시해,
  이미 보존된 11-sequence checkpoint를 `latest_materialized_build_id`로만 간접 노출했다.
- Export state에 `durable_checkpoint`를 별도 추가했다. Contract v2 requested/status count/privacy
  consistency와 FAIL/INCOMPLETE 0, freeze eligible를 확인한 build 중 completed sequence 수가 가장
  큰 build를 선택하며, final deadline progress 0과 checkpoint 11을 섞지 않고 모두 표시한다.
- 전체 99 tests와 publication-safety PASS 후 code checkpoint `80f48ab`를 push했다.
  CPU-only quiet dashboard daemon만 exact command/child 0을 확인한 뒤 PID 1900669로 교체했다.
  Persistent state에서 deadline 0, checkpoint 11, REVIEW 11, 366 files/344,922,733 bytes,
  contract consistent/freeze true, 다른 attention 없음을 확인했다.

### Freeze-ready 증가분 autonomous checkpoint follower

- AI polling 없이 이후 완료 sequence도 내구적으로 보존하도록
  `tools/run_predeadline_checkpoint_follower.py`를 추가했다. 이 process는 GPU/inference를 호출하지
  않고 기존 quality follower의 atomic `freeze_readiness`만 소비한다.
- 시작 시 contract-v2 candidate를 큰 순서로 전수 byte/SHA 검증하여 largest valid checkpoint를
  source of truth로 선택한다. Ready 집합이 그 checkpoint의 strict superset이고 deadline 전일 때만
  frozen sequence order와 집합 SHA에 묶인 deterministic build ID로 새 immutable export를 실행한다.
- 기존 집합과 동일하면 아무것도 생성하지 않는다. 변경된 same-size/non-superset 집합, readiness
  validation failure, integrity failure와 disk reserve 부족은 structured attention으로 남긴다.
  Export build lock의 exit 75는 recoverable coordination event로 분류해 자동 재시도한다.
- Follower state는 `.runtime/predeadline_checkpoint_follower_state.json`에 atomic rename+directory fsync로
  저장한다. Lifetime singleton lock으로 duplicate follower를 거부하며 final deadline sentinel/build과
  별도 state/build prefix를 쓴다.
- Dashboard/handoff process discovery에 follower alive/stale/duplicate/structured-attention을 연결했고,
  deadline이 지난 뒤 정상 종료한 follower를 사망으로 오인하지 않도록 predeadline에서만 생존을 요구한다.
- 실제 one-shot은 현재 freeze-ready 11과 기존 checkpoint 11을 전수 검증한 뒤
  `WAITING_FOR_NEW_FREEZE_READY_SEQUENCE`, attempted build/exit code 없음, attention false였다.
  신규 export나 GPU process는 생성하지 않았다. 전체 105개 regression이 PASS했다.
- Publication-safety PASS 후 code commit `711d4fd`를 push하고 persistent follower PID 1916854를
  시작했다. 기존 11-sequence build를 다시 byte-verify한 뒤 child/export 없이 대기한다.
- 새 process marker/state schema를 load하도록 exact argv/cwd/child 0을 확인한 CPU-only dashboard와
  handoff monitor만 각각 PID 1917825/1917827로 교체했다. Sapiens/SAM/supervisor/watchdog/sentinel은
  signal하거나 restart하지 않았다.
- Persistent dashboard는 follower alive/RUNNING, ready 11, best verified checkpoint 11,
  final deadline build 0을 분리해 표시한다. Follower attention은 없고 전체 attention은 기존
  `DEADLINE_ETA_AT_RISK` 하나다.

### Deadline freeze coverage upper-bound forecast

- 기존 dashboard의 Sapiens/SAM 개별 ETA는 전체 26-sequence 종료 시각만 보여 주어, deadline 시점에
  몇 sequence까지 terminal body-fit/freeze 가능할지 직접 드러내지 못했다.
- 78개 selector `summary.json`의 exact `target_only_sapiens_crops`/`frame_count`를 frozen order로
  읽고, 현재 durable crop/SAM camera/partial frame과 measured recent rate를 적용하는 CPU-only
  schedule upper bound를 추가했다. Selector REVIEW의 abstention crop 감소는 그대로 사용하며
  background person이나 ambiguous target을 보충하지 않는다.
- Forecast는 pose-ready dependency와 단일 sequential Mode B stream을 계산하지만
  triangulation/body-fit/quality/export overhead를 의도적으로 0으로 둔다. 따라서
  `OPTIMISTIC_UPPER_BOUND`라는 이름과 assumptions를 machine-readable state에 함께 기록한다.
- 실제 26-sequence inventory는 65,430 target crop/65,595 SAM frame으로 기존 global total과 exact-match했다.
  2026-08-12 12:04 KST rate snapshot에서는 deadline upper bound 25/26, 첫 late sequence
  `squat_0003`, all-sequence optimistic terminal 2026-08-14 15:44 KST였다. 이 ceiling도 전량을
  충족하지 못하므로 `DEADLINE_FREEZE_COVERAGE_AT_RISK` warning을 별도로 기록한다.
- Forecast helper/inventory/partial-SAM scheduling regression을 포함한 전체 107 tests가 PASS했다.
- Code commit `8b55df7`를 push한 뒤 exact argv/cwd/child 0을 확인한 CPU-only quiet dashboard만
  PID 1932669로 교체했다. 이 확인 시점에 기존 supervisor가 `benchpress_0001` pose completion을
  감지해 SAM Mode B PID 1930239를 정상 시작했으며, Sapiens는 36/78 camera 완료 후
  `benchpress_0002/cam1`로 진행했다. GPU inference/supervisor는 signal/restart하지 않았다.

### Empirical post-SAM deadline adjustment

- Overhead-free upper bound가 실제 terminal body-fit 시각을 낙관하는 정도를 정량화하기 위해,
  완료 11 sequence의 세 camera `sam_body_benchmark.csv.created_at_utc`와 body-fit/Mode-C
  `created_at_utc`를 frozen provenance에서 읽었다. Source payload나 output은 수정하지 않았다.
- SAM 최종 camera 완료→body-fit/Mode-C terminal latency는 median 998.73초, p90 1,399.83초,
  최소 847.06초/최대 2,037.30초였다. 음수·6시간 초과·누락 timing은 sample로 사용하지 않고
  machine-readable error로 남긴다.
- 기존 `OPTIMISTIC_UPPER_BOUND`는 그대로 보존하고, 각 future sequence에 관측 p90을 더해
  supervisor의 sequential post-SAM stage를 반영하는 `EMPIRICAL_P90_POST_SAM_ADJUSTED`를 추가했다.
  Pre-SAM/quality/export와 p90 밖 variance는 여전히 제외됨을 assumptions에 명시한다.
- 실제 snapshot에서는 upper/adjusted 모두 deadline 25/26, 첫 late `squat_0003`이었다. Count는 같지만
  adjusted all-sequence terminal은 upper보다 약 23분 늦어져 risk 해석을 더 보수적으로 만든다.
- Provenance timing, p90 schedule, dashboard available-forecast integration을 포함한 전체 113 tests가 PASS했다.
- Publication-safety PASS 후 commit `7ffeb9a`를 push하고 exact argv/cwd/child 0인 CPU-only dashboard만
  PID 1959115로 교체했다. Persistent state는 sample 11, p90 1,399.83초, upper/adjusted 25/26을
  표시한다. SAM concurrency로 Sapiens ETA가 직전보다 57분 악화돼 warning이 추가됐지만 GPU 100%,
  62,823 MiB, OOM/retry/stall 0이므로 frozen concurrent policy를 변경하지 않았다.

### Predeadline checkpoint follower recovery watchdog

- Predeadline checkpoint follower는 lifetime singleton lock과 internal export retry를 갖지만,
  terminal/session 자체가 사라진 경우 dashboard attention 외에는 자동 복구 경로가 없었다.
- `tools/run_predeadline_checkpoint_follower_watchdog.py`를 추가해 repository-local live process와
  handoff에 보존된 exact resume argv SHA를 pin한다. 3회 연속 absence와 2초 final rescan 뒤에만
  detached recovery하며, 최대 3회/시간으로 제한한다. Live process에는 절대 signal하지 않는다.
- Follower 자체 lifetime lock이 watchdog/manual launch race를 닫는다. Watchdog command는 `--once`를
  거부하고 follower와 동일한 timezone-aware deadline만 허용하며, deadline 도달 후에는 restart하지
  않고 `COMPLETE`로 종료한다.
- Dashboard는 deadline 전 아직 quality 또는 unexported ready set이 남은 동안 watchdog의
  dead/stale/duplicate/structured attention을 검사한다. Quality 26/26이더라도 ready count가 durable
  checkpoint보다 크면 follower/watchdog 사망을 숨기지 않는다.
- 실제 live follower PID 1916854에 대한 one-shot은 expected/resume command SHA exact-match,
  missing 0, launch/restart 0, attention false로 identity pin을 완료했다. 전체 111 tests가 PASS했다.
- Publication-safety PASS 후 commit `16fd41f`를 push하고 watchdog PID 1944186을 persistent mode로
  시작했다. Follower PID 1916854 exact-match, restart 0, attention false를 확인했다.
- 새 marker/state를 load하도록 exact argv/cwd/child 0인 CPU-only dashboard/handoff monitor만
  PID 1945200/1945203으로 교체했다. Handoff에는 watchdog exact resume command가 저장됐고,
  dashboard attention은 기존 deadline ETA/coverage warning 두 개뿐이다. GPU job은 건드리지 않았다.

### Phase 11 quality follower lifetime lock과 recovery watchdog

- Quality follower는 sequence-local quality build lock과 internal retry는 갖고 있었지만 process lifetime
  lock이 없어 manual/watchdog recovery race에서 state writer가 중복될 수 있었고, terminal/session
  소실 시 dashboard attention 외 자동 복구가 없었다.
- `run_quality_control_follower.py`에 lifetime singleton advisory lock을 추가했다. 동일 lock owner가
  있으면 output/state를 읽거나 쓰기 전에 exit 3으로 거부한다. Valid existing quality는 기존처럼
  revalidation/resume하고 expensive GPU work는 호출하지 않는다.
- `tools/run_quality_control_follower_watchdog.py`는 live/persisted exact argv SHA를 pin하고 3회 연속
  absence + 2초 final rescan 후에만 최대 3회/시간 detached recovery한다. Live follower에는 signal하지
  않으며 follower lifetime lock이 launch race를 닫는다.
- Watchdog은 state `COMPLETE`, completed 26/26, freeze-ready 26/26, quality/readiness failure 0을 모두
  확인한 뒤에만 recovery를 종료한다. `--once` 또는 sequence set이 없는 resume command는 거부한다.
- Dashboard/handoff에 watchdog dead/stale/duplicate/structured attention과 exact resume command를
  연결했다. 실제 live PID 1819560 one-shot identity pin은 expected/resume SHA exact-match,
  missing/restart 0, attention false였다. 전체 117 tests가 PASS했다.

### Quality follower/watchdog persistent activation

- Publication-safety PASS 후 implementation commit `4600dff`를 push했다. 기존 CPU-only quality
  follower PID 1819560은 exact cwd/argv, child 0, `WAITING_FOR_BODY_FIT`, failure 0을 확인한 뒤에만
  종료했고 GPU inference/supervisor에는 signal하지 않았다.
- 동일 frozen argv를 새 code로 재개한 quality follower PID는 1973073이다. Lifetime lock
  `.runtime/quality_follower.lock`이 held임을 별도 non-mutating probe로 확인했고, 기존 complete 11
  sequence는 materialize/recompute 없이 revalidation했다.
- Persistent quality follower watchdog PID 1973668은 expected/live/resume command SHA exact-match,
  restart 0, attention false다. Exact resume command는 `.runtime/handoff_state.json`에 보존된다.
- 새 watchdog state를 읽도록 exact cwd/argv, child 0인 CPU-only dashboard/handoff monitor만
  PID 1974702/1974706으로 교체했다. 2026-08-12 12:34 KST atomic snapshot은 Sapiens 36/78 camera와
  23,708/65,430 crop, SAM Mode B 34/78 camera와 22,114/65,595 frame, triangulation 12/26,
  body fit/quality/freeze-ready 11/26을 기록했다.
- GPU는 100%, 62,823 MiB, OOM/retry/stall 0이다. Dashboard attention은 Sapiens deadline ETA
  약 2시간 21분 초과와 upper/p90-adjusted freeze coverage 25/26 경고뿐이며 acceptance failure는 없다.

### SAM output storage reserve forecast

- 정적 20 GiB disk threshold만으로는 Mode B 잔여 payload가 reserve를 침범하기 전에 경고할 수 없어,
  dashboard에 완료 PASS camera benchmark의 `output_bytes/frame` nearest-rank p90 forecast를 추가했다.
  현재 partial camera payload는 filesystem free에 이미 포함되므로 produced frame에만 더하고 byte sample은
  atomic complete camera에서만 읽어 중복 차감을 방지한다.
- 2026-08-12 12:45 KST 기준 sample 34 camera/22,114 frame/22,455,935,524 bytes,
  p90 1,038,271.11 bytes/frame이다. Current partial 포함 produced 22,737 frame, remaining 42,858 frame의
  예상 추가 payload는 41.44 GiB다. 현재 free 145.14 GiB에서 SAM 완료 후 103.70 GiB가 예상돼
  20 GiB reserve margin은 83.70 GiB다.
- Forecast는 Sapiens/downstream/checkpoint/unrelated write를 제외한다고 state에 명시한다. 예상 free가
  reserve 아래로 내려가면 `DISK_FORECAST_RESERVE_AT_RISK` warning을 atomic dashboard state에 기록한다.
- 전체 118 tests와 publication safety가 PASS했고 commit `b24f509`를 push했다. Exact argv/cwd/child 0인
  CPU-only dashboard PID 1974702만 종료한 뒤 동일 argv로 PID 1992655를 시작했다. GPU inference,
  SAM, supervisor에는 signal하지 않았다.
- 새 daemon의 first snapshot에서 forecast/schema publish를 확인했다. 같은 시각 Sapiens rate는
  0.223 crop/s, ETA는 deadline 3시간 25분 후이며 upper/p90-adjusted freeze coverage는 24/26,
  첫 projected late sequence는 `deadlift_0002`다. Process alive, GPU saturation, retry/error 0이므로
  frozen shortest-first order와 acceptance policy는 변경하지 않았다.

### Dashboard/handoff monitoring-plane continuity

- Persistent dashboard와 handoff checkpoint monitor에는 lifetime singleton lock이 없고 exact resume
  command도 handoff에 보존되지 않아 terminal/session loss 시 AI polling 없이 복구할 수 없었다.
- 두 persistent mode에 symlink-safe advisory lifetime lock을 추가했다. `--once` 진단은 lock owner의
  state를 수정하지 않고 계속 사용할 수 있으며, duplicate daemon은 output write 전에 exit 3으로 거부한다.
- Handoff process inventory는 basename-exact argv match와 `shlex.join`을 사용하고 dashboard, handoff
  monitor, monitoring watchdog의 last-known exact command를 보존한다. 이 monitoring plane은 workload
  RUNNING 판정에서는 제외해 pipeline completion status를 왜곡하지 않는다.
- `tools/run_monitoring_watchdog.py`는 두 target별 live/resume argv SHA pin, 3-cycle absence confirmation,
  2초 final rescan, 최대 3회/시간 detached recovery를 제공한다. Live target을 signal하지 않고 각
  lifetime lock이 manual/watchdog race를 닫는다. State는 `.runtime/monitoring_watchdog_state.json`이다.
- Dashboard는 watchdog dead/stale/duplicate/structured attention을 표시한다. 관련 22개와 전체 123개
  tests, publication safety가 PASS했다. Implementation `16a8600`과 default handoff output-path validator
  fix `5c93d4e`를 push했다.
- Exact cwd/argv/child 0 확인 후 CPU-only dashboard PID 1992655와 handoff monitor PID 1974706만
  SIGTERM하고 동일 argv로 PID 2006908/2006909를 시작했다. 두 target lock held와 handoff exact command
  persistence를 확인했으며 GPU inference/SAM/supervisor에는 signal하지 않았다.
- Monitoring watchdog PID 2009359은 두 target 모두 live/resume SHA exact, missing/restart 0,
  attention false다. Watchdog command도 handoff에 보존됐고 세 lifetime lock이 모두 held다.
- 2026-08-12 12:55 KST snapshot은 SAM cam2의 새 durable PASS를 반영해 35/78 camera,
  22,787 frame이며 cam3 Mode B 192 frame으로 전진했다. Sapiens 23,964/65,430 crop,
  OOM/retry/stall 0이고 deadline 경고 외 operational attention은 없다.

### Immutable checkpoint/final snapshot storage forecast

- Predeadline follower는 ready set이 1개 늘 때마다 모든 기존 complete sequence를 다시 포함하는 별도
  immutable cumulative build를 만들므로, SAM-only forecast는 freeze write duplication을 제외했다.
- Dashboard가 largest verified checkpoint manifest의 sequence별 payload bytes를 selector frame 수로
  정규화한다. Future sequence에는 관측 최대 16,149.38 bytes/frame을 적용하고, current 11개 이후
  매 sequence checkpoint와 별도 deadline build ID의 final snapshot을 모두 합산한다.
- 2026-08-12 13:01 KST empirical p90 deadline coverage 24/26 기준 새 checkpoint 13개 + final snapshot
  1개의 remaining write는 8.58 GiB다. 모든 26개가 deadline 전에 ready가 되는 observed-max
  시나리오는 checkpoint 15개 + final snapshot 1개, 10.61 GiB다.
- SAM forecast와 합친 projected free는 deadline 24개 기준 95.08 GiB, all-sequence observed-max
  93.05 GiB이며 20 GiB reserve margin은 각각 75.08/73.05 GiB다. Sapiens/compact downstream/
  unrelated write와 manifest filesystem overhead 제외를 state assumptions에 명시한다.
- All-sequence observed-max combined free가 reserve 아래면
  `DISK_COMBINED_FORECAST_RESERVE_AT_RISK`를 기록한다. Helper/real snapshot과 전체 124 tests,
  publication safety가 PASS했고 implementation commit `5bb9c4c`를 push했다.
- Exact argv/cwd/child 0인 CPU-only dashboard PID 2006908만 SIGTERM 후 동일 argv PID 2019834로
  교체했다. Dashboard lock held, monitoring watchdog live/resume SHA exact, restart 0이며 GPU
  inference/SAM/supervisor에는 signal하지 않았다.
- 같은 snapshot에서 Sapiens `benchpress_0002/cam1`이 새 durable PASS가 되어 37/78 camera,
  24,135 crop이며 cam2로 진행했다. SAM은 35/78 camera, cam3 partial 449 frame이고
  OOM/retry/stall 0이다.

### Remaining deadline sequence-order dominance audit

- Deadline projection 24/26에서 process restart/order 변경이 completion count를 개선할 수 있는지
  private frame 없이 selector aggregate workload로 감사했다. 남은 14 sequence는 target crop과 SAM
  frame 수가 모두 같은 오름차순이며, measured-rate combined cost 순서도 동일하다.
- Dashboard에 `REMAINING_TWO_STAGE_WORKLOAD_DOMINANCE`를 추가했다. 완료 pose sequence를 제외한
  모든 earlier/later pair를 비교해 later item이 crops/frames 모두 이하이고 하나 이상 strict-smaller면
  `DEADLINE_SEQUENCE_ORDER_DOMINANCE_INVERSION` warning을 기록한다.
- Live snapshot은 `PARETO_NONDECREASING`, dominance inversion 0, weighted inversion 0이다. 이는
  global two-machine flow-shop optimum 증명이 아니라 명백한 component-wise order 오류가 없다는
  evidence이며 frozen command를 변경하거나 GPU process를 재시작하지 않았다.
- Helper/attention integration과 전체 125 tests, publication safety가 PASS했고 implementation
  commit `f8f603b`를 push했다. Exact CPU-only dashboard PID 2019834만 동일 argv PID 2026797로
  교체했다. Lock held, monitoring watchdog exact identity/restart 0이며 transient observation warning은
  다음 atomic refresh에서 제거돼 deadline warning 두 건만 남았다.
- Activation 중 `benchpress_0001/cam3`가 durable PASS로 완료됐다. SAM 36/78 camera, 23,460 frame,
  body fit 673×26 REVIEW, Mode C candidate 0 `PASS_MODE_B_FROZEN`, quality REVIEW sidecar까지 정상
  materialize됐다. Supervisor는 `WAIT_RUNNING_SAPIENS2`; 다음 `benchpress_0002` 3-view pose 완료를
  기다리며 SAM child 부재는 정상이다. Quality follower/freeze checkpoint의 다음 CPU cycle은 자동으로
  처리하도록 두고 수동 duplicate materialization은 수행하지 않았다.

### Automatic 12-sequence durable checkpoint

- Quality follower가 `benchpress_0001`의 673-frame quality vector와 exporter preflight를 검증해
  quality/freeze-ready를 12/26 REVIEW로 증가시켰다. Failure/dependency reason은 0이며 GPU work나
  기존 sequence recomputation은 수행하지 않았다.
- Predeadline checkpoint follower가 readiness strict superset을 감지해 build
  `exercise3d-predeadline-auto-012-77ac2165e283`를 자동 publish했다. Contract v2 결과는
  REVIEW 12/FAIL 0/INCOMPLETE 0, 399 files/377,238,045 bytes, freeze eligible true다.
- Follower 자체 검증은 manifest file/byte count 399/377,238,045와 verified count가 exact-match했고,
  별도 read-only `verify_frozen_build`도 valid true, errors 0, requested order/tree/ownership/SHA와
  status count 일치를 재확인했다. 이 immutable build는 재-export하지 않는다.
- 2026-08-12 13:09 KST dashboard는 durable checkpoint 12, body fit/quality/freeze-ready 12/26,
  supervisor `WAIT_RUNNING_SAPIENS2`, OOM/retry/stall 0을 기록했다. 기존 11-sequence checkpoint도
  삭제하지 않으며 final deadline sentinel은 별도 build ID로 계속 대기한다.
- 12-sequence 관측치를 반영한 remaining immutable checkpoint + final snapshot forecast는 deadline
  24개 시나리오 8.23 GiB, 모든 26개 관측-max 시나리오 10.25 GiB다. Remaining SAM과 합친 projected
  free는 95.20/93.17 GiB, 20 GiB reserve margin은 75.20/73.17 GiB여서 storage attention은 없다.

### Durable last-completion event monitoring

- 기존 top-level `last_event`는 supervisor의 현재 stage만 반복해 마지막 완료 산출물을 식별하지 못했다.
  Dashboard가 Sapiens/SAM camera terminal metadata, triangulation/body/Mode-C/quality atomic output과
  immutable export manifest의 provenance timestamp를 비교해 `last_completed_event`를 구조화하도록 했다.
- Watchdog/follower의 주기적 state `updated_at`은 후보에서 제외한다. Sequence output은 required payload
  marker가 함께 있을 때만 완료로 인정하며 partial metadata가 더 최신이어도 선택하지 않는다.
  Supervisor stage는 `current_operational_event`로 별도 유지한다.
- 실제 one-shot integration은 최신 event를 12-sequence build
  `exercise3d-predeadline-auto-012-77ac2165e283`의 `DURABLE_CHECKPOINT_PUBLISHED`,
  `FREEZE_ELIGIBLE`, 2026-08-12 13:06:55 KST로 산출했다. GPU/supervisor에는 signal이나 launch를
  수행하지 않았으며 전체 126 regression이 PASS했다.

### Source-of-truth 재검증

- HEAD `ae89fe6`, worktree clean, Draft PR #1과 remote branch 동기화
- A100 80GB idle, private source 65,595 frames와 checkpoint storage 정상, source mutation 0
- deadline 2026-08-14 13:00 KST = 2026-08-14 04:00 UTC
- 2026-08-11 17:31 KST 기준 remaining wall-clock 67.48 h
- 전달된 과거 target 수치 대신 최신 repository result를 채택: 9,732 frames, 9,725 target crops,
  ambiguity 7, identity switch 0, crop reduction 50.3725%
- full target selector/Sapiens2/SAM output은 아직 없고, 4개 pilot sequence Sapiens2 output만 보존됨

### 중간 계획 변경 보고 — 이번 deadline cycle의 유일한 major 변경 보고

- 변경 사유: target-only Sapiens2 실측 projection 79.09 GPUh만으로 remaining 67.48 h를 넘고,
  SAM Mode B 16.35 h 및 downstream을 더하면 한 A100에서 전량 완료가 물리적으로 불가능
- 기존 계획: 전체 Sapiens2 → 전체 triangulation → 전체 SAM → fitting → freeze
- 변경 계획: 기존 4개 pilot output 재사용 + sequence-complete streaming. GPU는 Sapiens2 우선,
  CPU triangulation/QC 병행, SAM Mode B를 dependency 가능한 sequence에만 실행
- 정확도 영향: 5B, official flip-test, detector, target abstention과 accepted threshold는 변경하지 않음
- deadline 영향: 전체 26개 완료 보장은 포기하지 않되, deadline에는 완결 sequence 수를 최대화하고
  나머지는 resumable `INCOMPLETE_DEADLINE` provenance로 동결
- 리스크: Sapiens throughput 저하, Phase 7/9 구현 critical path, SAM output disk 증가
- 즉시 적용: official DETR full 26-sequence resumable pass 시작; full selector 후 Sapiens2 진입

### Phase 6 full 준비와 lossless pilot 재사용

- official DETR full pass는 26 sequence/78 camera에 대해 batch 16, chunk 512, source mutation 없이
  실행 중이다. 완료 camera마다 consolidated bbox/candidate payload와 QA를 원자적으로 기록한다.
- 기존 `ALL_DETECTIONS_BASELINE`에는 모든 candidate의 308-keypoint 결과가 보존되어 있으므로,
  accepted target candidate만 exact gather해 4개 pilot의 target-only output을 만들었다.
- 결과: 12 cameras, 9,732 frames, target poses 9,725, 새 5B inference 0회, elapsed 72.29 s.
- baseline 대비 confident XY/confidence 최대 delta는 12/12 camera 모두 0.0이었다.
- resume chunk는 frame 이름뿐 아니라 현재 selector의 abstention/status/index 및 selected bbox/score가
  일치해야 재사용하도록 강화했다. 기존 45 chunks는 selection-bound 검증 PASS.
- 17개 unit test와 compile PASS.

### Full selector incremental gate

DETR이 먼저 끝난 9 sequences/27 cameras에 full selector와 별도 lossless validator를 적용했다.

- frames 19,224, all candidates 37,966, target crops 19,068
- ambiguity 130, `NO_TARGET` 26, background candidates 18,898
- identity-switch risk 0, forward/backward disagreement 0, integrity failure 0
- candidate offsets/boxes/scores는 DETR consolidated arrays와 exact-match
- gate `GO_FULL_DATASET`

`barbellrow_0003`의 130 ambiguity와 26 NO_TARGET은 촬영 종료 후 target이 화면에서 나가는
구간에 집중됐다. 16-frame private overlay에서 background 사람을 강제 선택하지 않는 올바른
abstention임을 확인했다. 전체 78-camera 완료 후 동일 gate를 다시 실행한다.

### Full DETR / target selector gate와 5B 실행 시작

- official DETR full candidate pass 완료: 26 sequences, 78/78 cameras PASS,
  65,595 frames, person candidates 120,586, pose inference 0회
- full selector/독립 validator 결과: target crops 65,430, ambiguity 139,
  `NO_TARGET` 26, background candidates 55,156, occlusion-risk 19,525
- detector candidate count/offset/bbox/score, source frame/PTS, selected index와 abstention contract를
  전수 검사했고 integrity failure 0, forward/backward disagreement 0, identity-switch risk 0이다.
- 4개 pilot/12 camera의 full selector arrays는 기존 accepted selector와 모든 field가 exact-match했다.
  기존 9,725 target pose의 selection-bound resume가 유효하므로 재추론하지 않는다.
- private representative overlay에서 pushup ambiguity, squat candidate-index 역전,
  benchpress 누운 자세와 10-person scene, mirror/reflection 후보를 다시 확인했다. background를
  target으로 바꾼 사례는 없고, ambiguous pushup frame은 target을 강제 선택하지 않았다.
- 최종 gate: `GO_FULL_DATASET`.

2026-08-11 18:35 KST에 batch 16, chunk 256, official flip-test, primary target only로
resumable full Sapiens2-5B를 시작했다. 새 inference 대상은 55,705 crops이다. cached-detector
benchmark의 보수적 end-to-end 환산 0.234 crops/s에서는 약 66.1 GPUh로, deadline까지 reserve가
매우 작다. GPU는 Sapiens2에 전용하고 CPU triangulation/recovery/QC만 병행한다. 실제 첫 신규
camera wall-clock이 확보되면 ETA를 다시 갱신한다.

### Phase 7 timestamp-aware triangulation pilot

```bash
python tools/triangulate_sapiens2.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --pose-root <PRIVATE_OUTPUT_ROOT>/sapiens2_target_only_full \
  --camera-root <PRIVATE_OUTPUT_ROOT>/background_ba \
  --output-root <PRIVATE_OUTPUT_ROOT>/triangulation \
  --runtime-dir <PRIVATE_OUTPUT_ROOT>/runtime/phase7_triangulation_pilot_gate \
  --sequences barbellrow_0000,squat_0001,pushup_0001,benchpress_0003
```

- schema/finite/NaN contract 4/4 PASS, 3,244 reference timestamps
- canonical source-joint reprojection median/p90 px:
  `barbellrow_0000` 7.06/30.62, `squat_0001` 26.24/164.93,
  `pushup_0001` 326.93/2,004.04, `benchpress_0003` 7.91/97.84
- Huber scale 10 px 배수의 사전 명시 gate 결과: REVIEW 2, NO_GO 2
- private overlay상 squat/pushup target 2D pose는 정상이므로 identity error로 덮지 않았다.
  current refined camera와 human observations의 epipolar inconsistency로 판정했다.
- NO_GO proposal은 진단용으로 보존하지만 `eligible_for_body_fitting=false`이며 export에 사용하지 않는다.
- Phase 5 camera를 덮어쓰지 않고, recovery를 수행한다면 observation-conditioned provenance와
  held-out-frame 검증을 요구한다.

### Phase 7 observation-conditioned camera recovery

원 Phase 5 geometry는 수정하지 않고 별도 private output root에 recovery candidate를 만들었다.
canonical body direct joints와 timestamp-aware pairing을 사용하며, 세 essential-pair/PnP topology 중
fit residual이 가장 작은 것만 선택한 뒤 사전 분리한 20% held-out frame으로 검증했다.

- fit/held-out overlap: 두 sequence 모두 0
- `squat_0001`: held-out median/p90 26.21/164.86 → 5.70/18.88 px,
  all-frame canonical 5.71/18.93 px
- `pushup_0001`: held-out 311.60/2,020.79 → 8.12/95.12 px,
  all-frame canonical 8.11/96.04 px
- 두 sequence schema PASS, NO_GO 0, REVIEW 2
- threshold/Huber scale은 변경하지 않았고 원 NO_GO proposal도 진단용으로 보존

이는 같은 Sapiens2 observation으로 만든 geometry라 독립 calibration evidence가 아니다.
`camera_source=SAPIENS2_2D_OBSERVATION_CONDITIONED`, 최종 상태는 둘 다
`REVIEW_OBSERVATION_CONDITIONED`이며 camera/pose uncertainty를 fitting/export까지 전파한다.
특히 pushup p90 96.04 px는 NO_GO 경계 100 px에 가까워 selective REVIEW QA 대상이다.

### SAM Mode B full-run contract 준비

GPU contention 때문에 5B 실행 중 SAM을 함께 올리지는 않는다. 대신 CPU에서 full runner와
resume validation을 준비했다.

- sequence/camera 단위 Mode B 실행과 PASS output resume
- primary target seed 정확히 1개, source frame index와 selector confidence/ambiguity/occlusion 보존
- upstream PLY 변환 직전 MHR pose/shape/scale/joints를 compact NPZ로 추가 저장
- frame 수와 mesh/MHR numeric/provenance count가 모두 exact일 때만 camera PASS
- partial/old mesh-only output은 complete로 오인하지 않고 deterministic rerun

Mode C는 여전히 selective evidence가 없는 전체 기본값으로 사용하지 않는다.

### First full camera 실측과 Phase 7 streaming 시작

- 첫 신규 대상 `barbellrow_0001/cam1` 481/481 target crop을 약 36분에 완료했다.
- 두 atomic chunk와 consolidated `poses_2d.npz`, bbox/frame/metadata가 모두 생성됐고
  308-keypoint selected payload finite, ambiguity/NO_TARGET 강제 pose 0, camera QA PASS다.
- 첫 camera 기준 serialization 포함 실측은 약 0.22 crop/s이며 A100 utilization 100%, VRAM 약
  36.4 GiB, power 약 300–380 W 범위였다. 2026-08-11 19:12 KST remaining은 65.79 h로
  Sapiens 완료 ETA와 거의 같아 GPU는 계속 5B 전용으로 유지한다.
- `tools/run_phase7_streaming.py`를 별도 CPU process로 시작했다. 각 sequence의 세 camera
  Sapiens schema/수량을 전수 확인한 뒤 Phase 5 geometry를 먼저 triangulate하며,
  `NO_GO_TRIANGULATION`일 때만 disjoint held-out gate를 통과한 observation-conditioned recovery를
  별도 root에서 선택한다. 원 Phase 5 output은 수정하지 않는다.
- 첫 streaming 결과 `barbellrow_0000`은 canonical median/p90 7.06/30.62 px,
  schema PASS, `REVIEW_POSE_CAMERA_CONSISTENCY`, body fitting eligible을 재현했다.
- 기존 pilot 4개를 final root에 모두 materialize했다. schema PASS/body-fitting eligible 4/4,
  원 Phase 5 camera 사용 2, held-out 승인 recovery 사용 2이며 최종 NO_GO는 0이다. Watcher는
  나머지 22 sequence의 세 view pose가 완결될 때까지 30초 간격으로 대기한다.

### SAM numeric provenance와 Phase 9/13 구현 준비

- SAM target provenance에 selector의 `TARGET_AMBIGUOUS`, `NO_TARGET`, source PTS를 추가했다.
- full Mode B compact payload는 bbox/focal/keypoint/camera translation뿐 아니라 MHR global/body/hand
  pose, scale/shape/expression, 127 joint coordinates/global rotations와 204-d model parameter를 보존한다.
- 기존 Mode A private numeric sample의 shape 45, expression 72, model parameter 204를 official MHR
  JIT와 checkpoint의 `308 x 18,566` landmark mapping에 다시 넣었다. 저장 keypoint 최대 차이는
  `2.6822e-7 m`, 평균 `5.1895e-8 m`, mesh 최대 차이는 `7.1526e-7 m`로 exact replay를 확인했다.
- `tools/consolidate_sam_body_prior.py`는 camera별 compact prior를 frame/PTS/identity uncertainty와
  함께 원자적으로 통합하되 ambiguous output은 보존하고 accepted prior로 사용하지 않는다.
- `tools/fit_sequence_body.py`는 triangulated geometry dominant anchor, per-view MHR robust similarity
  alignment, weak correlated-prior fusion, temporal second-difference의 staged fit을 구현했다.
  geometry가 없는 prior-only joint는 최소 두 view가 합의할 때만 low-confidence로 생성한다.
- `tools/export_private_dataset.py`는 source RGB를 복사하지 않고 frame name/index/PTS 및 immutable
  inventory를 보존하며, versioned private payload를 byte-exact copy/SHA-256 검증한다. FAIL과
  INCOMPLETE는 freeze-eligible로 승격하지 않는다.
- 이 단계는 implementation readiness이며 아직 full SAM/body input이 없으므로 실제 Phase 9/13
  acceptance 결과로 간주하지 않는다. 관련 신규/회귀 unit test 18개와 전체 31개 test가 PASS했다.

### 장시간 critical path supervisor 시작

- current Sapiens2 PID가 정상인 동안 30초 간격으로 상태/deadline만 기록하고 GPU에는 개입하지 않는다.
- process 종료 시 78-camera pose completeness를 전수 검사하고, incomplete이면 동일 batch 16/chunk 256,
  selector-bound resume를 최대 2회 수행한다.
- 이후 sequence별 Phase 7 final gate, SAM Mode B, compact prior consolidation, staged body fit을 실행하고
  마지막에 versioned private export/SHA/schema validation을 실행한다.
- stage 실패는 sequence row에 남기고 다른 sequence를 계속 처리하며, export에서 FAIL/INCOMPLETE를
  freeze-eligible로 승격하지 않는다.
- supervisor 시작 시 private storage 여유는 170 GiB다. SAM 예상 약 78 GiB와 compact/export payload를
  수용 가능하지만 sequence마다 확인하고 20 GiB reserve 아래에서는 새 SAM run을 시작하지 않는다.
- Mode C 자동 full 실행은 금지했다. candidate/acceptance는
  `configs/sam_mode_c_escalation.json`에 동결했으며, occlusion 단독으로 escalation하지 않는다.
- supervisor process-alive/storage helper를 포함한 전체 unit test 32개가 PASS했다.
- 5B 종료 직후 full SAM에 들어가기 전에 accepted target 8-frame Mode B smoke를 1회 실행한다.
  source PTS provenance, mesh 8, compact numeric 8과 MHR required field 전체가 exact할 때만 full
  camera run을 허용한다. 이 gate를 포함한 전체 unit test는 33개 PASS다.
- Mode B/body fit 뒤 `assess_sam_mode_c_escalation.py`가 occlusion과 missing/nonfinite,
  median+5 scaled-MAD temporal/alignment outlier의 교집합을 판정한다. 후보 clip은 양쪽 15-frame,
  sequence 10% 상한이며 결과는 export dependency다. 후보를 찾는 것과 C 결과 채택은 분리하고,
  실제 B/C 개선 gate 전에는 Mode B를 덮어쓰지 않는다. 전체 unit test 35개 PASS다.
- private export file manifest에는 sequence payload뿐 아니라 immutable source inventory,
  temporal audit/frame mapping과 각 sequence manifest 자체의 byte count/SHA-256도 포함한다.
- inventory가 피험자 수 3명 aggregate만 보존하고 sequence→subject mapping은 `UNKNOWN`으로 명시하므로,
  외형이나 learned shape로 identity를 추측하지 않는다. Freeze v1은 sequence-level S0를 생성하되
  `subject_id=null`, `SUBJECT_MAPPING_UNAVAILABLE`을 기록하고 cross-sequence shape fusion을 보류한다.

### Sapiens2 full steady-state ETA 갱신

- `barbellrow_0001/cam1`과 cam2는 각각 481/481 target crop, schema PASS로 완료됐다.
- 두 camera completion timestamp 사이 steady-state rate는 `0.23323 crop/s`다. 2026-08-11
  19:50 KST 기준 완료 pose 10,687/65,430, 남은 crop 54,743, complete camera 14/78이다.
- recent-camera rate projection은 Sapiens 종료 `2026-08-14 12:58 KST`로 deadline과 사실상 동일하고
  retry/QC reserve는 없다. Saved-chunk 기준 전체 effective rate는 current incomplete chunk 때문에
  `0.216 crop/s`로 보수적으로 진동한다.
- supervisor state에 current KST 대응 UTC, remaining wall, completed/remaining crop, recent/effective
  throughput, camera/sequence 수, Sapiens ETA, SAM expected 20.8 h와 free storage를 30초마다 기록한다.
- recent-camera ETA monitor 회귀 test를 포함한 전체 unit test 36개가 PASS했다.

### Phase 9 body-fit quality gate 사전 동결

- Full SAM/body 결과 확인 전에 `configs/phase9_body_fit.json`을 추가했다.
- REVIEW: final joint coverage <95%, alignment success <90%, normalized geometry displacement p95 >0.05,
  prior-only >2%, median bone-length CV >0.10 또는 camera status non-PASS.
- FAIL: coverage <80%, normalized displacement p95 >0.20, anthropometric reference invalid,
  valid finite/invalid NaN contract 실패.
- Threshold는 실제 결과를 보고 완화하지 않는다. Gate unit test 포함 전체 37개 PASS다.

### Fit3D metric implementation readiness

- Local workspace에 Fit3D payload가 없어 실제 score나 camera error를 주장하지 않는다.
- Root-aligned MPJPE, per-frame scale-only N-MPJPE, similarity Procrustes PA-MPJPE를 별도로 계산하는
  `tools/evaluate_fit3d_metrics.py`를 구현했다.
- Known scale/rotation synthetic regression 포함 전체 unit test 39개 PASS다.
- Dataset freeze 뒤 Fit3D access/convention을 확보하면 30 fps, 3-camera, timing/JPEG/camera
  perturbation을 단계적으로 적용한다.

### SAM full resume schema 강화

- Camera resume PASS가 mesh/numeric 개수만 검사하던 경로를 강화해 target provenance required field와
  모든 compact NPZ의 MHR required field set을 전수 검사한다.
- 오래된 mesh-only/부분 numeric output은 full runner에서 즉시 `INCOMPLETE`가 되어 retry 대상이며,
  consolidation에서 뒤늦게 반복 실패하지 않는다. Schema mismatch 회귀 test 포함 전체 40개 PASS다.
- Fit3D metric은 root joint 자체가 nonfinite인 frame을 제외하도록 보강했으며 전체 41개 test PASS다.

### Persistent handoff와 sequence-complete GPU streaming

- Sapiens2를 중단하지 않고 8-frame Mode B numeric smoke를 동시에 실행했다. mesh/numeric/PTS와
  MHR required field가 모두 PASS했고 combined peak는 48,525 MiB였다.
- 5B 종료까지 기다리면 deadline에 end-to-end 완결 sequence가 늘지 않으므로, 이미 보고한
  sequence-complete 계획대로 pose-ready sequence를 Phase 7→Mode B→prior→body fit으로 즉시 보내는
  resumable supervisor로 전환했다. 정상 Sapiens PID와 output은 유지했다.
- 첫 full `barbellrow_0000/cam1` Mode B는 590/590 frame, primary target 1명,
  mesh/numeric/provenance 전 completion check PASS. concurrent wall 970.94초,
  0.6077 frame/s, combined peak 61,821 MiB, mean GPU 96.76%, mean power 339.60 W였다.
- public `HANDOFF.md`에는 path-neutral operational checkpoint와 frozen decision/resume gate를,
  `AGENTS.md`에는 10-step startup protocol을 기록했다. exact private command/PID/progress는
  ignored `.runtime/handoff_state.json`에 30초마다 `*.tmp`→atomic rename으로 보존한다.
- checkpoint에는 completed/in-progress/remaining, last completed camera, crop/frame counts,
  config hash, checkpoint identity, source/camera/timing version, Git HEAD/diff hash, active/resume command,
  GPU와 downstream counts가 들어간다. detached monitor PID 575526과 multi-cycle timestamp 전진을 확인했다.
- supervisor resume는 PASS/REVIEW row만 durable complete로 불러오고 incomplete row를 retry한다.
  handoff/resume regression을 포함한 전체 44개 unit test PASS다.
- 완료 camera마다 별도 `run_provenance.json`을 atomic materialize한다. Sapiens 15개와 SAM 1개
  기존 PASS output에 inference 재실행 없이 sidecar를 생성했으며 model/checkpoint identity,
  batch/chunk/mode, source/selection digest, camera/timing version, tool commit/SHA와 exact resume
  command를 기록했다. Live monitor가 새 completion을 30초마다 추가 materialize한다.
- partial chunk까지 포함하는 live throughput/ETA를 handoff state에 추가했다. 20:50 KST 기준
  pose 11,424/65,430, recent-chunk 0.22583 crop/s, Sapiens ETA는 2026-08-14 15:14 KST다.
  단독 projection보다 약 2.23시간 늦지만 Mode B end-to-end sequence를 deadline 전에 확보한다.
- `barbellrow_0000/cam2`도 590/590 completion PASS. cam1+cam2 aggregate는 1,180 frame,
  1,950.22초, 0.6051 frame/s이고 두 camera combined peak는 모두 61,821 MiB다.
- pose/SAM inference provenance sidecar를 consolidated SAM prior와 final private export dependency로
  전파하도록 확장했다. Export는 sidecar 누락 sequence를 complete로 인정하지 않는다.
- deadline 순간의 실제 완료 상태가 이후 generation으로 덮이지 않도록 CPU-only sentinel PID 607755를
  시작했다. 2026-08-14 13:00 KST에 별도 build ID로 export하며 manifest가 이미 있으면 duplicate하지
  않는다. Export exit 2도 expected incomplete snapshot이면 manifest/status를 보존하고 generation은 계속한다.

### First end-to-end sequence와 private export smoke

- `barbellrow_0000` Mode B 3 camera를 모두 완료했다. 각 590 frame이며 elapsed는
  970.94/979.27/1,010.60초, aggregate는 1,770 frame/2,960.81초 = 0.59781 frame/s다.
- camera별 mesh/numeric/PTS/provenance count가 exact이고 nonfinite/missing 0, combined peak
  61,821 MiB, mean GPU utilization 약 96.2%, mean power 약 339 W였다.
- consolidated SAM prior는 output/accepted 1,770/1,770과 inference provenance dependency를 통과했다.
- sequence body fit은 590 timestamp × 26 joint, coverage/alignment 1.0, prior-only 0,
  median bone-length CV 0.01738, finite/NaN contract PASS다.
- 사전 동결 displacement REVIEW 경계 0.05 대비 p95 0.05167이고 camera REVIEW도 전파되어
  `REVIEW_BODY_FIT_QUALITY`로 보존했다. FAIL은 없다.
- Mode C assessor는 84개 reference frame을 `REVIEW_MODE_C_CANDIDATE`로 기록했다. Missing/nonfinite와
  alignment outlier는 0이고 주로 sequence boundary temporal outlier라, Sapiens2+Mode B critical path를
  중단하지 않고 Mode C 실행/채택 0으로 Mode B를 유지했다.
- complete sequence 하나만 사용한 private export smoke는 REVIEW 1/FAIL 0/INCOMPLETE 0,
  34 files, payload 28,960,929 bytes, 누락·size·SHA-256 mismatch 0, `freeze_eligible=true`였다.
- 2026-08-11 21:10 KST에는 Sapiens2 16/78 camera, 11,677/65,430 crop까지 완료했다.
  recent 0.22073 crop/s projection은 2026-08-14 16:49 KST로 deadline보다 약 3.82시간 늦다.
  `squat_0001/cam1` Mode B를 다음 streaming input으로 시작했고 기존 두 job은 중단하지 않았다.

### Second end-to-end sequence — squat_0001

- `squat_0001` Mode B cam1/cam2/cam3 각 1,267 frame을 모두 first attempt에서 완료했다.
  Elapsed는 2,023.09/2,033.27/2,024.20초, aggregate 3,801 frame/6,080.57초 =
  0.62511 frame/s다. 세 camera combined peak는 70,359 MiB이고 OOM은 없었다.
- Mesh/numeric/PTS/provenance count와 required schema가 3,801/3,801 exact, nonfinite/missing/temp 0이다.
  Cam2/cam3 occlusion-risk 545/689를 frame provenance로 보존했다.
- Consolidated prior accepted/output 3,801/3,801, camera PASS 3/3이다.
- Body fit은 1,267×26, coverage/alignment 1.0, prior-only 0, median bone CV 0.02327,
  finite/NaN PASS다. Displacement p95 0.07936과 camera uncertainty로
  `REVIEW_BODY_FIT_QUALITY`; FAIL은 없다.
- Mode C candidate는 0이고 `PASS_MODE_B_FROZEN`; Mode C 실행/채택 0이다.
- 두 full sequence cumulative SAM은 5,571 frame/9,041.38초 = 0.61617 frame/s다.
- 같은 구간 Sapiens2는 `pushup_0004` 3-view를 PASS로 완료해 durable 18/78 camera가 됐고,
  `pushup_0002/cam1` first chunk까지 current partial 포함 12,951/65,430 crop이다.
  Recent 0.20622 crop/s projection은 Sapiens 종료 2026-08-14 21:27 KST다.
- Supervisor는 `pushup_0001`의 Phase 7 recovery provenance를 재확인한 뒤 Mode B `cam1`을
  즉시 시작했다. 정상 Sapiens와 deadline/handoff monitor는 중단하지 않았다.

## 2026-08-09 — 초기 synchronization / derivative 구축 기록 이관

### 수행

- clap onset과 waveform cross-correlation으로 triple-view 영상의 공통 구간을 결정했다.
- 서로 다른 native 30/30/60 fps를 보존한 raw에서 synchronized derivative와 30 fps working
  frames를 생성했다.
- output frame이 어떤 source frame에서 왔는지 pixel matching과 PTS로 별도 검사했다.
- source 파일 교체가 필요한 camera filename mapping 오류 1건은 provenance를 보존한 상태로
  수정했고, 이후 inventory에서는 replaced source를 active count에서 제외했다.

### 대표 명령

```bash
python scripts/build_dataset.py \
  --source-root <PRIVATE_DATASET_ROOT>/origin \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --stage all
python scripts/verify_dataset.py --dataset-root <PRIVATE_DATASET_ROOT>
```

### 결정

- sync와 working frame은 derivative이며 raw native frame rate는 유지한다.
- sync residual과 실제 frame-grid offset을 구분한다.
- 실제 data payload는 공개 Git 저장소로 이관하지 않는다.

## 2026-08-09T12:20:46Z — Phase 0 Dataset Inventory / Integrity 완료

### 실행

```bash
python tools/dataset_inventory.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --output-dir <PRIVATE_OUTPUT_ROOT>/inventory
```

### 결과

- 피험자 3명, triple-view 26 sequences
- raw videos 78, synchronized videos 78
- working JPEG 65,595장
- camera source: iPhone 16 / 16 Pro / 17, native 30/30/60 fps
- raw/sync inventory PASS, source/derivative provenance 유지
- source modification 0건

### 결정

모든 후속 report는 frame index만이 아니라 가능한 경우 packet/frame PTS를 함께 저장한다.

## 2026-08-09T12:33:26Z — Phase 1 EIS/OIS / Camera Stability Audit 완료

### 실행

```bash
python tools/eis_background_audit.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --output-dir <PRIVATE_OUTPUT_ROOT>/eis_audit
```

### 방법

temporal background와 foreground component mask를 구성하고, static 영역의 LK feature track에
homography/affine model을 fit했다. native-adjacent와 longer-baseline pair의 global motion,
spatial residual, 반복성을 함께 평가했다.

### 결과

- 78/78 `FIXED_CAMERA_OK`
- native-adjacent fit 8,087/8,087 성공
- 반복 global/spatial warp evidence 없음
- foreground-induced false positive 1건을 mask logic 수정 후 재검증
- source data modification 0건

### 결정

physical camera는 tripod-fixed로 간주한다. downstream final camera에 timestamp별 독립 pose를
두지 않는다.

## 2026-08-09T13:18:24Z — Phase 2 Temporal Synchronization QA 완료

### 실행

```bash
python tools/temporal_sync_audit.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --output-dir <PRIVATE_OUTPUT_ROOT>/temporal_alignment
```

### 방법

- PTS를 frame index보다 우선했다.
- beginning/middle/end를 포함한 여러 window에서 3 camera pair를 검사했다.
- actual-frame mapping, audio waveform/clap, visual motion energy를 함께 사용했다.
- RGB frame은 자르거나 보간하거나 재생성하지 않았다.

### 결과

- 26 sequences, 78 camera pairs
- actual-frame PTS observation 546건
- absolute offset median 11.99 ms, p95 25.28 ms, max 31.38 ms
- 30 fps 1 frame인 33.33 ms 이내 546/546
- `TEMPORALLY_STABLE` 8, `SMALL_CONSTANT_OFFSET` 16,
  `CLOCK_DRIFT_DETECTED` 2, `INSUFFICIENT_EVIDENCE` 0
- drift review: `pushup_0000`, `squat_0001`

### 결정

dataset synchronization은 사용 가능하다. offset/drift는 downstream frame pairing metadata로만
반영하고 video 자체를 보정하지 않는다.

## 2026-08-09T15:52:57Z — Phase 3 VGGT-Ω Geometry Initialization 완료

### 공식 구현 확인

- input preprocessing와 512-class resolution behavior
- joint sequence inference와 output tensor shape
- OpenCV world→camera `[R|t]`
- canvas pixel-unit K, positive camera-Z depth
- arbitrary sequence-local scale/gauge
- probability가 아닌 ranking confidence

### 실행

```bash
python tools/vggt_geometry_init.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-repo <LOCAL_VGGT_REPO> \
  --checkpoint <LOCAL_CHECKPOINT> \
  --output-dir <PRIVATE_OUTPUT_ROOT>/vggt
```

### 결과

- sequence당 8 representative PTS × 3 cameras = 24 images joint inference
- 26/26 sequence SUCCESS, 78/78 camera geometry, 624 sampled camera frames
- 실패·필수 payload 누락 0
- camera quality PASS 77 / REVIEW 1
- REVIEW: `squat_0001/cam2`, rotation dispersion outlier

### 결정

VGGT-Ω 결과는 최종 camera가 아니다. background BA의 initialization/prior로만 사용하며
depth/point-map scale을 metric으로 해석하지 않는다.

## 2026-08-09 — Phase 3 Open3D Visual Inspection Gate 완료

### 구현

`tools/visualize_vggt.py`에 percentile confidence filtering, RGB mapping, world axis, camera
frustum, voxel/max-point sampling, screenshot/PLY debug export와 BA overlay를 구현했다.

### 좌표 검증

world→camera에서 camera center를 `C_world = -R.T @ t`로 계산했다. OpenCV +x right,
+y down, +z forward convention을 raw output에 유지하고 display 변환을 metadata로 분리했다.

### 대표 결과

- `barbellrow_0000`: PASS
- `squat_0001`: REVIEW
- `pushup_0001`: REVIEW
- `benchpress_0003`: REVIEW
- 전역 mirror, 180° flip, exploding point cloud 없음
- distant wall/floor/rack이 thin sheet로 분리되고 일부 camera pose jitter 존재

### `squat_0001/cam2`

특정 sample의 pose가 cluster에서 이탈했지만 전체 scene이 동시에 폭발하지 않아 camera token
pose instability가 중심이고 dynamic foreground/point-map noise가 일부 기여하는 것으로 판단했다.

## 2026-08-09 — Phase 4 Fixed-Camera Background BA Pilot 완료

### 범위

`barbellrow_0000`, `squat_0001`, `pushup_0001`, `benchpress_0003` 네 sequence만 pilot으로 실행했다.

### 방법

- VGGT timestamp pose를 robust SO(3)/translation aggregation해 physical-camera init 생성
- cam1 identity gauge, cam2/cam3 shared extrinsic만 optimization
- temporal median/MAD, confidence, border, persistent SIFT로 static background 추출
- SIFT ratio, USAC_MAGSAC, epipolar/point-map consistency로 cross-view track 구성
- fixed intrinsics Mode A, Huber robust loss, weak pose/point prior, Stage 1/2 gate
- Phase 2 corrected timestamp는 matching pair 선택에만 사용

### 결과

- PASS 2 / REVIEW 2 / FAIL 0
- 모든 sequence Stage 1/2 finite convergence
- `squat_0001/cam2` 6.4 s VGGT pose: aggregation weight 0.001, 자동 REJECT
- 같은 timestamp의 유효 background observation은 shared-pose BA에서 유지
- 수동 sequence/PTS hard-code 없음

### 결정

동일 알고리즘과 default를 변경하지 않는 조건으로 26-sequence 확장을 승인했다.

## 2026-08-09 — Phase 5 Full Dataset Background BA 실행 완료

### Configuration freeze

- historical tool SHA-256: `1f01256e336474fae5c79434323b7c092b618c3b94e171077c80897f95a53feb`
- normalized configuration SHA-256: `df640077fd89f462eec6001f13465e808be47f991d6637175dfbfa24b7d2764a`
- fixed intrinsics, Huber scale 5, max tracks 800, min length 3, max nfev 300
- 새로운 matcher/threshold/heuristic/weighting 추가 없음

### 실행 형태

```bash
python tools/background_bundle_adjust.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-root <PRIVATE_OUTPUT_ROOT>/vggt \
  --output-root <PRIVATE_OUTPUT_ROOT>/background_ba \
  --sequence <SEQUENCE_ID>
python tools/finalize_background_ba_dataset.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-root <PRIVATE_OUTPUT_ROOT>/vggt \
  --output-root <PRIVATE_OUTPUT_ROOT>/background_ba
```

### 결과

- 26/26 sequence output과 78/78 camera summary 생성
- PASS 11 / REVIEW 14 / FAIL 1
- Stage 1 convergence 26/26, Stage 2 convergence 25/26
- point 1,674 initial → 1,100 final
- observation 16,835 extracted → 11,046 final
- 동일 accepted observation의 residual:
  - mean 4.113 → 3.361 px
  - median 3.630 → 2.582 px
  - p90 8.205 → 7.425 px
  - p95 9.850 → 9.953 px
- cam2 rotation change median/p95/max 0.457°/2.247°/2.501°
- cam3 rotation change median/p95/max 0.376°/3.074°/3.207°
- cam1은 exact gauge reference로 변화 0

### REVIEW / FAIL

REVIEW는 low track support, p95 tail, no direct three-camera track 등의 기존 gate 이유를
그대로 보존했다. FAIL은 `pushup_0003` 1건으로 Stage 2가 `max_nfev=300`에 도달했다.
알고리즘 freeze 원칙 때문에 threshold를 바꾸거나 자동 fallback하지 않았다.

### 무결성

- raw/synchronized/working frame 변경 없음
- VGGT numeric payload 변경 없음
- SE(3) inverse/rotation finite sanity 검사 통과
- gauge: robust cam1 physical pose identity
- scale: sequence-local arbitrary, initial cam1-cam2 baseline 보존

### 결정

dataset-level 계산은 완료됐다. 그러나 FAIL refined camera는 triangulation에 사용하지 않는다.
제외, pilot initialization fallback 또는 별도 승인된 재최적화 중 정책을 확정하기 전까지
Phase 6 전체 실행은 보류한다.

## 2026-08-09 — 공개 전용 저장소 migration

### 수행

- 비어 있는 public repository를 clone하고 `main` 최초 bootstrap을 준비했다.
- private workspace는 수정하지 않고 read-only inventory source로만 사용했다.
- dataset-construction 관련 scripts/tools만 이관하고 public-facing legacy project 명칭을 제거했다.
- `--dataset-root`와 `EXERCISE3D_DATASET_ROOT`를 추가하고 output path를 명시할 수 있게 했다.
- 한국어 README/canonical plan/chronological log와 phase/design/QA 문서를 구성했다.
- Phase 5 aggregate numeric CSV만 이관하고 exact K/R/t, media, NPZ와 debug render는 제외했다.
- conservative `.gitignore`, publication safety checker, GitHub Actions check를 추가했다.

### Migration smoke test 범위

- 모든 Python source compile
- 주요 CLI help
- 외부 private dataset을 대상으로 BA `--dry-run`
- staged/tracked file suffix, size, absolute path, credential pattern 검사
- private workspace source tree hash/mtime mutation 없음 확인

### 결정

이후 dataset-construction 변경의 canonical source는 이 public repository다. 각 Phase는
acceptance gate 후 문서화, 안전 검사, commit/push까지 완료해야 한다.

## 2026-08-09 — Phase 5.1 pushup_0003 Camera Recovery 완료

### 원인 분석과 동일성 control

- Phase 5 Stage 1은 cost 10,498.589521에서 정식 수렴했다.
- Stage 2는 cost를 1,672.515861까지 낮췄지만 `max_nfev=300`에서 종료됐다.
- 300-control은 Phase 5의 initial camera, 모든 track/observation array,
  `points_initial`/`points_stage1`, Stage 1 result와 Stage 2 cost를 exact 재현했다.
- 24 sample 모두 GOOD이고 특정 sample reject는 없었다. cam2 residual이 가장 높지만 camera
  explosion 없이 tail step이 작아져, 원인은 발산이 아닌 evaluation budget 부족으로 판단했다.

### 실행

```bash
python tools/background_bundle_adjust.py \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --vggt-root <PRIVATE_VGGT_ROOT> \
  --output-root outputs/local/background_ba/recovery_runs/nfev_600 \
  --sequence pushup_0003 \
  --max-nfev 300 \
  --stage2-max-nfev 600 \
  --optimizer-verbose 2
```

Stage 1은 기존 300 budget을 유지했고 Stage 2 budget만 600으로 확장했다. 새로운 matcher,
threshold, heuristic, loss, weighting 또는 observation 변경은 없다.

### 결과와 Visual QA

- Stage 2 `xtol` 수렴: actual nfev 322, final cost 1,657.953684
- median 4.954229→2.558895 px, p90 8.037446→5.053964 px,
  p95 9.295044→7.055627 px
- final 21 tracks / 183 observations, sample GOOD 24 / DOWNWEIGHT 0 / REJECT 0
- cam2/cam3 robust-init rotation change 2.538° / 1.859°,
  center scene fraction 0.003830 / 0.003494
- Open3D top/side 검사: plausible rig/orientation, mirror·180° flip·explosion 없음
- sparse support 때문에 `RECOVERED_REVIEW`; VGGT fallback 미사용

### Dataset gate와 무결성

- 최종 PASS 11 / REVIEW 15 / FAIL 0
- Stage 1/2 26/26 수렴, per-sequence validation 26/26 PASS
- camera geometry freeze 승인; REVIEW uncertainty는 downstream에 전달
- 외부 private workspace의 raw 78, synchronized 130-file tree, working JPEG 65,595,
  VGGT 689-file numeric tree와 Background BA 1,057-file tree fingerprint가 전/후 동일
- Sapiens2, triangulation, SAM-Body4D, SMPL/human fitting, pseudo-label 수행 없음
- viewer relative debug path가 external dataset root로 해석될 수 있던 경로를 canonical project
  root로 수정했고, 진단 중 생성된 두 debug file은 식별 후 제거하여 external tree를 원상 복구했다.

## 2026-08-09 — Phase 6-0 Sapiens2-5B Pose Environment 완료

### 공식 구현과 detector 결정

- official `facebookresearch/sapiens2` commit
  `7e5bae88456ac418ff0e58e74106c9fe192055d4`를 별도 external source로 clone했다.
- official checkpoint `facebook/sapiens2-pose-5b`의
  `sapiens2_5b_pose.safetensors`만 primary pose weight로 사용했다.
- model card의 RTMDet 문구와 달리 현재 official `docs/POSE.md`, demo shell과
  `vis_pose.py`는 `facebook/detr-resnet-101-dc5`를 사용한다. 실제 실행 code를 우선했다.
- top-down crop 1024×768, Sociopticon 308 points, UDP heatmap decode와 flip-test를 확인했다.

### 환경과 checkpoint

```bash
conda create -y -n sapiens2 python=3.12 pip
conda run -n sapiens2 python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
conda run -n sapiens2 python -m pip install -e <SAPIENS2_REPO>
```

- Python 3.12.13, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118
- transformers 5.14.1, safetensors 0.8.0, OpenCV 5.0.0.93
- A100-SXM4-80GB, driver 535.183.06, compute capability 8.0, BF16 지원 확인
- pose checkpoint 20,480,899,148 bytes; SHA-256
  `b4848da8691c72e14d3ff71319f077363107129bf4128019eb39d072129b2a52`
- detector snapshot revision `96317ca979e231bd960cb3cac31328e0165a3e94`

### Smoke 실행과 결과

```bash
conda run -n sapiens2 python tools/sapiens2_pose_smoke.py \
  --image <PRIVATE_REPRESENTATIVE_FRAME> \
  --sapiens2-root <SAPIENS2_REPO> \
  --checkpoint-root <CHECKPOINT_ROOT> \
  --warmup 1 --repeats 3 \
  --output-json outputs/local/sapiens2/smoke.json
```

- representative barbell-row frame에서 person 1명 detection 성공
- pose model GPU load 성공, FP32 model load 약 58.46 s
- 308 keypoint coordinates와 308 confidence 출력, 모두 finite
- confidence ≥0.3 point 100%가 원본 frame 내부, original pixel `(x,y)` 복원 정상
- official 308 flip mapping involution과 body left/right name pair 정상
- end-to-end detector + two-pass flip-test latency median 4.517 s/image
- peak CUDA allocated 19.986 GiB, reserved 20.961 GiB
- visual skeleton의 body/hand/foot 배치와 좌우 ordering plausible

첫 smoke checker는 구형 COCO ankle index pair를 가정해 계산 후 FAIL을 표시했다. 모델 출력 문제가
아니었으며 official 308 metainfo의 name 기반 pair 검사로 수정한 뒤 동일 inference가 PASS했다.
Detector safetensors load의 네 BatchNorm counter warning은 detection이 정상이라 compatibility note로
유지한다.

### 결정

5B는 OOM/instability 없이 동작하므로 primary offline teacher로 확정했다. 1B comparison은 수행하지
않았다. 단일 job 단순 외삽 약 82.3 GPU-hours는 offline 목적에서 허용 가능하며, Phase 6-1에서
official 2 jobs/GPU의 실제 throughput과 multi-exercise robustness를 먼저 측정한다. 전체 26 sequence
inference, triangulation, SAM-Body4D, MHR, SMPL과 pseudo-label generation은 수행하지 않았다.

## 2026-08-11 — Phase 6-1A Primary Target Selection Gate 완료

### 구현과 회귀 검증

- official DETR all-person candidate는 삭제하지 않고 private ragged metadata에 보존
- multi-frame initialization, track duration, IoU, normalized center, scale/aspect, score를 결합
- forward/backward tracking 합의와 cross-view target visibility QA 추가
- detector가 prone target을 상·하체 complementary box로 분할하는 fragmentation을 감지해
  `TARGET_AMBIGUOUS`로 abstain; frame 0이 마지막 frame과 wraparound 연결되던 boundary 수정
- target selector unit test 5개와 Python compile PASS

### 4-sequence pilot와 Visual QA

- 12 camera, 9,732 frame, official DETR person candidate 19,596
- target-only eligible crop 9,725, crop reduction 50.3725%
- `TARGET_AMBIGUOUS` 7, `NO_TARGET` 0, obvious identity switch 0
- ambiguity는 `pushup_0001/cam1` duplicate 1 + fragmentation 6이며 pose crop을 출력하지 않음
- private overlay에서 background crossing/overlap, mirror 후보, lying/prone pose, bbox size reversal,
  candidate order 변화 확인; background person systematic mis-selection 0
- private coordinate/overlay/frame은 ignored `outputs/`에만 보존

### Target-only Sapiens2 benchmark

- batch 1/2/4/8/12/16 모두 PASS 및 batch 1/all-person target baseline equivalence PASS
- raw fastest batch 16: 0.231951 crop/s, reserved 37.426 GiB
- 99% plateau 최소 batch 4 권장: 0.230449 crop/s, reserved 23.801 GiB,
  pose GPU utilization mean 97.309%, mean power 348.408 W
- 65,595 frame target-only stage projection 79.09 GPU-hours, all-person 157.38 GPU-hours 대비
  약 78.30 GPU-hours 감소

### 결정

Target-selection gate는 `GO_FULL_DATASET`이다. 그러나 사용자에게 결과를 보고하고 명시적 승인을
받기 전까지 전체 65,595-frame inference는 시작하지 않는다.

## 2026-08-11 — SAM Body Runtime Feasibility Preflight

### Official interface와 pilot 선정

- SAM-Body4D revision `21af1020979ef32ddf6be3597ef59a68bad2f1bf`
- SAM 3D Body revision `b5c765a0d89d789985e186d396315e7590887b94`
- mode A base, mode B completion off, mode C completion on 비교 계획 동결
- control `squat_0001/cam1`, severe-occlusion `latpulldown_0002/cam2` 선정
- severe clip 1,136 frame detection/selector preflight PASS: 평균 2.121 candidates/frame,
  occlusion risk 959, identity switch/ambiguity 0; private representative overlay 확인

### Checkpoint gate와 결정

Primary-target adapter 기준 필요한 6개 payload set은 local에 없고 총 24,037,668,123 bytes
(22.387 GiB)다. SAM 3와 SAM 3D Body는 gated access가 필요하다. 사용자 조건에 따라
download/model/path/license를 먼저 보고하며 명시적 승인 전 checkpoint 다운로드와 SAM inference를
수행하지 않는다. Provisional deadline verdict는 `DEADLINE_AT_RISK`; local A/B/C 실측 전 final
verdict는 보류한다.

### Primary-target adapter와 6-run preflight

- Mode A는 official SAM 3D Body `bboxes=` API에 frame당 accepted bbox 0/1개를 전달
- Mode B/C는 official SAM-Body4D class를 사용하되 SAM 3 initial object를 accepted bbox 1개로 seed
- upstream all-human initialization용 ViTDet는 호출하지 않아 checkpoint 2.576 GiB도 불필요
- ambiguous first frame, multiple bbox slot, invalid bbox를 강제 실행하지 않는 schema/gate 추가
- control 1,267 frame × A/B/C와 severe 1,136 frame × A/B/C preflight 모두 target seed 1 확인
- control target-valid 1,267/1,267, severe 1,136/1,136; severe occlusion-risk 959 보존
- model 실행은 여섯 경우 모두 승인 전 의도한 `BLOCKED_CHECKPOINT`; download 0 bytes
- SAM adapter/selector synthetic test 11개, Python compile, CLI smoke PASS
- 여섯 mode CSV를 요구하는 runtime summarizer 추가; refiner C/B ratio, control/severe 증가,
  best/expected/worst를 분리하고 expected prevalence 입력이 없으면 숫자 산출 금지
- credential 값을 출력하지 않은 HF auth 확인은 PASS했지만 SAM 3/SAM 3D Body gated access는
  `--dry-run`에서 denied; MoGe-2, Depth Anything V2, 두 official Diffusion-VAS repo는 dry-run PASS
- official setup code 기준 Diffusion-VAS repo ID와 SAM 3D Body `model_config.yaml` requirement를 교정

## 2026-08-11 — SAM Body checkpoint와 A/B/C pilot 완료

### Access, download와 integrity

- 사용자 gated access 승인 후 SAM 3/SAM 3D Body를 포함한 6개 official source의 access dry-run PASS
- required checkpoint tree 28 files, 24,037,668,123 bytes(22.387 GiB) 다운로드 완료
- 모든 payload의 file existence, byte size, SHA-256을 전수 재검증: 누락/불일치/예상 밖 파일 0
- checkpoint/cache/credential은 ignored external storage에만 유지하고 공개 CSV에는 상대 경로,
  크기와 digest만 기록
- 별도 Python 3.12 / PyTorch 2.7.1 CUDA 환경에서 official load, headless EGL과 CUDA smoke PASS
- official loader가 string path를 요구하는 실제 runtime incompatibility를 primary-target runner에서 교정

### Primary-target A/B/C 6-run

- control `squat_0001/cam1`: 1,267 frame, 약 42초, occlusion risk 0
- severe `latpulldown_0002/cam2`: 1,136 frame, 약 38초, occlusion risk 959
- 모든 mode에서 accepted primary target 1명만 처리하고 background detection에는 body inference 미수행
- control A/B/C total 1,047.20/1,162.70/2,306.22초,
  end-to-end 0.8265/0.9177/1.8202 sec/frame
- severe A/B/C total 945.05/1,045.43/2,074.61초,
  end-to-end 0.8319/0.9203/1.8262 sec/frame
- peak VRAM A/B/C 최대 7,367/33,988/44,175 MiB; GPU/power telemetry도 0.2초 간격 보존
- Mode C/B execution ratio control 1.9946, severe 1.9964; severe/control ratio는 모든 mode 약 1.00

### Output sanity와 결정

- Mode A numeric 2,403개, Mode B/C mesh와 render 각각 2,403개 생성, 누락 0
- 시작/중간/끝 numeric finite, PLY 18,439 vertices/36,874 faces finite, JPEG decode와 private visual QA PASS
- Mode C refiner는 control/severe에서 1,287/1,154회 호출됐지만 content completion은 모두 0회
- B/C 대표 mesh 차이는 최대 0.303 mm로 현재 severe clip에서 refiner의 material improvement가
  확인되지 않아 full 기본 후보는 Mode B, Mode C policy는 `REVIEW_SAM_REFINER_POLICY`
- 65,595-frame SAM projection 16.35/20.80/32.63시간, Sapiens2 target-only를 합친 한 GPU 순차
  projection 95.43/99.88/111.71시간
- 2026-08-15 00:00 UTC freeze는 `NO_GO`; end-of-day도 QC/재시도 여유가 작아
  `DEADLINE_AT_RISK`
- 전체 Sapiens2/SAM inference는 실행하지 않았으며 별도 사용자 승인 전 `HOLD`
