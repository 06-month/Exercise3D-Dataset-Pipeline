# Exercise3D Pipeline Handoff

> 이 파일은 현재 operational checkpoint다. 정확한 private 경로·전체 command·실시간 PID는
> Git에서 제외된 `.runtime/handoff_state.json`이 source of truth다.

## Current objective

- 최종 deadline: 2026-08-14 13:00 KST
- 현재 목표: correctness/provenance를 유지하며 end-to-end freeze 가능한 sequence 수 최대화
- 현재 phase: Phase 6 target-only Sapiens2-5B + Phase 7/8 sequence streaming
- acceptance gate: target selector `GO_FULL_DATASET`; SAM Mode B full 10-sequence와 private export smoke `PASS`

## Current pipeline state

- DONE: Phase 0–5, Phase 6 pilot/target selector, Phase 8 A/B/C pilot와 checkpoint integrity
- RUNNING: Phase 6 full 5B inference, pose-complete sequence의 Phase 7→SAM Mode B→Phase 9 streaming
- RUNNING: Phase 11 quality 21/26 sequence; CPU follower/watchdog + exporter fallback 연결
- TODO: remaining triangulation, SAM prior consolidation, body fitting/QC, deadline private export/freeze
- BLOCKED: 없음. Dashboard attention은 deadline ETA/coverage WARNING 두 개뿐이다.
- REVIEW/FAIL: quality REVIEW 21/FAIL 0; body fit PASS 2/REVIEW 19/FAIL 0

## Active job

- 2026-08-13 01:25 KST 이전: parent session (PID 1694093) 종료로 전 GPU workload와 supervisor/watchdog
  process group 사망. Monitoring plane (handoff/dashboard/monitoring watchdog)과 deadline sentinel만 생존.
  OOM/reboot 아님, session kill. 2026-08-13 01:27 KST 전체 복구 완료.
- Sapiens2 PID 2979192, `setsid` 독립 session으로 시작, PPID=1. Output `outputs/sapiens2_target_only_full`.
  완료 camera/chunk를 검증 후 skip하며 `squat_0002/cam2`부터 resume. Singleton lock 사용 중.
- autonomous supervisor PID 2980339, `setsid` 독립 session, PPID=1. `--wait-sapiens-pid 2979192`.
  기존 15 completed sequence row를 인식하고 resume. 현재 exact PID/stage는 dashboard/handoff state가 source of truth.
- supervisor watchdog은 복구 supervisor와 old pinned SHA가 달라 ATTENTION이었으나, live supervisor 1개,
  persisted resume argv exact match, child 0, 두 lifetime lock held, restart 0/3을 확인하고 old CPU-only
  watchdog PID 2981054만 종료했다. `--adopt-live-command --once`가 exactly-one live/resume match를 검증해
  SHA를 명시적으로 repin했고, normal argv로 detached 재기동한 PID 197832가 PPID=1, RUNNING,
  attention false, restart 0/3이다. Supervisor PID 2980339와 GPU inference는 signal/restart하지 않았다.
- 2026-08-13 22:09 KST authoritative dashboard snapshot: Sapiens 64/78 camera,
  49,633/65,430 crop, current `squat_0000/cam2`; triangulation/body/quality 21/21/21 sequence,
  quality REVIEW 21/FAIL 0. Last verified checkpoint는
  `exercise3d-predeadline-auto-021-32a51bf7c071`, 21 sequence, `freeze_eligible=true`다.
- Sapiens recent throughput 0.220 crop/s; projected ETA는 deadline 약 5시간 5분 후 risk.
  OOM/retry/stall은 없음
- Phase 7 initial/final triangulation reuse는 pose NPZ/metadata, selected camera refinement/validation,
  first-frame shape source, temporal report, VGGT canvas metadata, canonical config와 triangulation tool의
  privacy-safe size/mtime/ctime signature를 atomic `.phase7_source_identity.json`에 저장한다. Exact signature,
  camera source와 `COMPLETE`가 모두 일치할 때만 skip한다. 실행 직전 marker는 `IN_PROGRESS`로 바꾸고
  실행 전후 dependency가 같을 때만 `COMPLETE`로 승격하므로 interrupted/changed-source output은 재사용하지
  않는다. 기존 12 completed sequence는 supervisor successful-row gate로 호출하지 않아 재삼각화하지 않으며,
  다음 새 sequence부터 적용된다.
- SAM durable 63/78 camera, 47,940/65,595 source frame, 21/26 full sequence; aggregate 약 0.567 frame/s.
  현재 active SAM PID는 없고 supervisor는 `WAIT_RUNNING_SAPIENS2`다. `barbellrow_0003`은 기존 payload만
  completion 재검증하여 세 camera 모두 `resume_skipped=true` PASS; GPU child/recomputation 0이다.
  Future `run_sam_body4d_full.py` invocation은 GPU child 생성 전
  `.runtime/sam_body4d_full.lock`을 lifetime 동안 보유하고, exact coordinator 또는 Mode B
  benchmark/primary child의 resolved output이 같은 SAM root 아래인지 `/proc`에서 검사한다.
  Legacy coordinator/orphan child가 남았거나 process table을 검사할 수 없으면 fail-closed로 거부한다.
  13:51 KST read-only probe는 dashboard와 동일하게 matching child 0을 확인했으며, supervisor는
  restart하지 않아 다음 normal subprocess부터 current guarded entrypoint가 자동 적용된다.
  Camera PASS gate는 benchmark/profile/mesh/numeric schema뿐 아니라 target provenance의 exact frame
  lengths/source indices, first accepted seed, forced ambiguous/no-target 금지, valid finite bbox와 abstention
  NaN bbox, finite bounded confidence, strictly increasing finite PTS를 확인한다. 마지막 완료
  `benchpress_0001/cam3` 673-frame output은 신규 checks 전부 PASS했다. Mesh/numeric required object root와
  존재하는 focal/render per-object root는 real directory `1` 하나만 허용하고 recursive extra payload 및
  각 numeric `object_id != 1`을 거부한다. 같은 실출력은 exact-object checks도 전부 PASS했다.
  모든 accepted target frame의 mesh/numeric은 필수지만 selector abstention 뒤 tracker propagation이 멈춘
  tail output은 optional이다. Extra abstention output도 downstream `accepted_prior = output_valid & target_valid`
  때문에 절대 채택되지 않는다. `barbellrow_0003`은 target-valid/output-valid가 cam1 731/735,
  cam2 700/702, cam3 684/688이며 accepted prior 총 2,115, abstention view 156을 그대로 보존한다.
  SAM compact-prior consolidation은 current provenance/numeric inventory의 privacy-safe
  size/mtime/ctime signature를 metadata에 저장한다. Retry 시 output의 frame/PTS/target flags/bbox,
  accepted-prior/finite payload/QA와 signature가 모두 일치할 때만 skip하고 source/output drift는 atomic
  rebuild한다. 기존 12 completed sequence는 supervisor successful-row gate로 호출 자체를 skip하며,
  다음 새 sequence부터 signature가 생성된다.
  Sequence body-fit은 canonical triangulation/metadata, 3-view compact prior/metadata, gate config와 모든
  fitting parameter의 privacy-safe size/mtime/ctime signature를 저장한다. Existing PASS/REVIEW output은
  source identity, frame/PTS/joint convention, finite/NaN/evidence/confidence schema, QA count와 current
  acceptance gate 재평가가 모두 exact할 때만 skip한다. 기존 `benchpress_0001` 673-frame REVIEW output은
  signature를 요구하지 않는 read-only schema/gate audit에서 PASS했으며 기존 12 sequence는 재계산하지 않았다.
  Mode C assessment marker도 body-fit NPZ/metadata, triangulation support, 3-view compact priors와 policy/
  canonical config의 size/mtime/ctime signature를 저장한다. Existing marker는 camera/count/signal/source-index/
  clip/threshold 및 selected-total↔PASS/REVIEW 결정을 검증한 뒤에만 timestamp를 보존해 skip한다. 기존
  `benchpress_0001` marker는 read-only audit에서 candidate 0 `PASS_MODE_B_FROZEN`으로 contract PASS했다.
  Phase 11 quality output은 selection/pose/SAM prior, triangulation/body/Mode C artifacts와 quality builder의
  privacy-safe size/mtime/ctime signature를 metadata에 저장한다. Exact signature와 full quality schema가
  일치할 때만 skip하며 build 실행 전후 source가 바뀌면 publish하지 않고 retry한다. Source-bound 정책 전
  완료되어 persisted completion에 기록된 기존 12개 unsigned quality는 grandfather하여 재계산하지 않는다.
  새 signed completion은 follower fast path에서도 source drift를 확인하며 corrupt NPZ는 follower 종료 대신
  bounded retry state로 전환한다. Live quality follower는 재시작하지 않았고 다음 supervisor quality
  subprocess부터 current builder가 자동 적용된다.
- GPU: A100 80GB, 22:09 KST Sapiens-only snapshot 36,375/81,920 MiB, utilization 100%,
  305.31 W, 54°C. observed OOM/retry 없음
- exact live command/PID/progress/ETA: `.runtime/handoff_state.json`
- long-running handoff monitor PID 2006909는 target-complete SAM semantics 변경 전 code를 load하고 있어,
  exact argv/cwd, child 0, lifetime lock, monitoring-watchdog identity, restart 0/3을 확인한 뒤 그 PID만
  SIGTERM했다. 수동 launch하지 않았으며 monitoring watchdog의 3-cycle/final-rescan이 current code로
  자동 복구한다. Exact current PID/state는 `.runtime/monitoring_watchdog_state.json`이 source of truth다.
- deadline snapshot sentinel PID 2171153: 2026-08-14 13:00 KST에 completed sequence와
  `INCOMPLETE` 목록을 별도 versioned private build로 export; local state는
  `.runtime/deadline_snapshot_state.json`, 현재 `WAITING_DEADLINE`
  Exporter는 hidden `.<build_id>.inprogress`에서 checksum-resume한 뒤 전수 integrity PASS 시 final
  directory로 atomic rename한다. Staging stale/unlisted file과 symlink는 정확한 hidden root에서만
  제거하고 actual tree↔manifest/sequence ownership exact-match를 검증한다. Existing final manifest는
  검증 후 reuse하며 같은 ID를 덮어쓰지 않는다.
  Deadline membership은 body fit NPZ/metadata + Mode-C assessment marker mtime이 cutoff 이하인
  sequence로 고정하며 post-deadline completion은 INCOMPLETE로 유지한다. Cutoff-eligible
  INCOMPLETE은 derived sidecar lag을 위해 최대 3회/30초 간격으로 staging checksum-resume하고,
  네 번째 최종 시도에는 truthful INCOMPLETE snapshot을 반드시 publish한다.
  Freeze contract v2는 requested 26-sequence list/order hash와 status CSV를 exact-match하고,
  global provenance 3 files + complete sequence당 required 33-file set을 강제한다. Sentinel은
  고정된 26-sequence list와 exact cutoff를 verifier에 별도로 전달한다. Exporter는 cutoff 전에
  output root/lock/staging을 만들지 않으며 manifest `created_at_utc >= cutoff`도 검증한다. 현재 live
  sentinel을 재시작하지 않아도 deadline export subprocess가 이 code를 load한다. Cutoff eligibility에서
  terminal marker 3개의 dev/inode/size/mtime/ctime identity를 고정해 validation 이후 copy descriptor와
  exact-match하지 않으면 publish를 중단하고 retry하므로 post-cutoff replacement가 섞이지 않는다.
  Source ctime 자체도 cutoff 이하이어야 하고 privacy-safe timestamp를 sequence manifest에 남겨 verifier가
  재검사하므로 post-cutoff replacement의 mtime backdating도 `INCOMPLETE`로 보존한다.
  Exporter는 valid unsigned legacy quality를 schema 검증 후 그대로 재사용하여 기존 12개를 다시 쓰지 않고,
  signed quality는 current Phase 11 source signature까지 exact해야 재사용한다. 각 complete sequence의
  32개 copied source dependency는 validation 전후 dev/inode/size/mtime/ctime identity가 동일해야 하며 deadline terminal
  marker identity와도 교차검증한다. 실제 copy descriptor가 동일 identity를 다시 요구하므로 validation,
  cutoff eligibility, copy 사이의 replacement는 `INCOMPLETE` 또는 sentinel retry로 남는다. 다음 checkpoint/
  deadline exporter subprocess가 current code를 자연스럽게 load하며 live process restart는 없다.
  GPU inference/supervisor는 건드리지 않았다.
- deadline sentinel watchdog PID 1882820: live/persisted command digest exact-match.
  Exporter `2cfd6b7` 반영 뒤 dashboard가 loaded-code drift를 탐지해, old CPU-only sentinel PID 2076548의
  exact argv/cwd, child 0, `WAITING_DEADLINE`, lifetime lock, restart budget 0/3을 확인한 후 그 PID에만
  SIGTERM을 보냈다. 수동 launch 없이 watchdog의 3-cycle/final-rescan이 PID 2171153을 복구했고 current
  exporter SHA exact, `WAITING_DEADLINE`, restart 1/3, attention false다. State는
  `.runtime/deadline_sentinel_watchdog_state.json`.
  Sentinel lifetime lock은 별도 process probe에서 held로 확인했고 exporter는 build ID별
  lock을 staging mutation 전에 취득한다.
  Sentinel runtime state는 process 시작 시 loaded sentinel/exporter tool SHA를 고정하고 dashboard가
  current on-disk SHA와 비교한다. Mismatch/missing은 `DEADLINE_SENTINEL_CODE_DRIFT`이며 exact argv만으로
  loaded Python implementation이 최신이라고 간주하지 않는다.
- dashboard monitor: `tools/monitor_autonomous_generation.py`; atomic state는
  `.runtime/dashboard_state.json`. Quiet daemon PID 208462이며 `--once`는 snapshot,
  기본은 Rich live, `--quiet`는 state-only daemon이다. Export section은 final deadline
  build progress와 contract-v2 best durable checkpoint progress를 별도로 보존한다. Selector exact workload와
  measured rate를 사용한 22:09 KST overhead-free deadline upper bound와 empirical p90-adjusted forecast는
  모두 24/26이며 첫 late sequence는 `deadlift_0002`다. Disk free 115.026 GiB, SAM-final 예상 free
  97.950 GiB, combined deadline/all 예상 free 94.639/92.624 GiB로 reserve attention은 없다.
  남은 14 sequence의 exact selector workload audit은 target crops와 SAM frames 양쪽 모두
  `PARETO_NONDECREASING`, dominance/combined-cost inversion 0이다. 이는 global optimum 증명이 아니라
  뒤 sequence가 두 GPU workload 모두 더 작은 명백한 order 오류가 없다는 지속 gate다.
  Top-level `last_completed_event`는 polling state timestamp를 제외하고 atomic camera/sequence payload와
  immutable manifest 중 최신 durable completion의 stage/sequence/camera/build/status/UTC·KST 시각을
  구조화하며 `current_operational_event`와 분리한다.
- Historical supervisor failure row는 더 강한 terminal quality PASS/REVIEW가 있으면 dashboard ERROR로
  재보고하지 않는다 (`b8d7a01`). Current code activation 전 dashboard PID 2065337의 exact argv/cwd,
  child 0, singleton lock, monitoring-watchdog identity와 restart 0/3을 확인한 뒤 그 CPU PID만 controlled
  replace했다. New quiet dashboard PID 208462는 PPID=1, exact persisted argv, watchdog-observed singleton,
  restart 0/3이며 inference/supervisor는 건드리지 않았다.
- Monitoring-plane watchdog PID 2009359: dashboard/handoff monitor의 live/resume exact argv SHA를
  각각 pin한다. Latest-completion code activation 중 exec-scoped manual daemon이 첫 state 뒤 종료되자
  3-cycle/final-rescan 경로로 dashboard를 복구한 이력이 있다. 현재 dashboard PID 208462와 recovered
  handoff monitor는 각각 exact live/resume identity, missing 0, attention false다. 두 monitor와
  watchdog의 lifetime lock은 모두 held다.
  3회 연속 absence + 2초 final rescan 후 target별 최대 3회/시간 detached recovery하고 live process는
  signal하지 않는다. Exact target/watchdog commands는 `.runtime/handoff_state.json`, state는
  `.runtime/monitoring_watchdog_state.json`에 atomic 보존된다.
- Phase 11 CPU follower PID 2981075: `setsid` 독립 session, PPID=1. quality 21/26이며
  `barbellrow_0003` SAM camera provenance 3개와 prior copy 3개를 atomic materialize했다. Retry cycle이
  freeze-readiness 21/26을 검증했고 failures 0이다. State는
  `.runtime/quality_follower_state.json`.
- Quality follower watchdog PID 2981113: RUNNING, attention false.
- CPU-only predeadline checkpoint follower PID 2981156: `setsid` 독립 session, PPID=1.
  last verified checkpoint build `exercise3d-predeadline-auto-021-32a51bf7c071`, 21 sequence,
  696 files/766,963,670 bytes, freeze_eligible/integrity verified=true. State attention false.
  State는 `.runtime/predeadline_checkpoint_follower_state.json`.
- Checkpoint follower watchdog PID 2981216: RUNNING, attention false.
- Deadline sentinel watchdog PID 2981238: RUNNING, deadline sentinel PID 2171153 관찰 중.

Public-safe Sapiens command 형태:

```bash
<SAPIENS_PYTHON> tools/sapiens2_target_pipeline.py infer \
  --dataset-root <PRIVATE_DATASET_ROOT> \
  --selection-root outputs/target_selection_full \
  --output-root outputs/sapiens2_target_only_full \
  --runtime-dir outputs/runtime/phase6_full_target_inference \
  --sequences <FROZEN_SHORTEST_FIRST_SEQUENCE_LIST> \
  --cameras cam1,cam2,cam3 --batch-size 16 --chunk-size 256 \
  --loader-workers 8 --prefetch-batches 4 --retry-failures 1 --save-overlays 0
```

## Completed work

- full selector: 65,595 frame, target 65,430, ambiguity 139, `NO_TARGET` 26, identity/integrity failure 0
- Sapiens2 pose: complete 39 camera와 current partial 합계 25,501 accepted target crops;
  `benchpress_0002`까지 13 sequence 3-view schema/finite PASS
- Phase 7 final: 13 sequence schema PASS/body-fit eligible, NO_GO 0. 첫 신규 source-bound
  `benchpress_0002` initial/final marker는 각각 15개 dependency, `COMPLETE`, current signature exact,
  `PHASE5_BACKGROUND_BA`, schema/finite PASS이며 absolute private path 0이다.
- concurrent Mode B 8-frame smoke: mesh/numeric/PTS schema PASS, combined peak 48,525 MiB
- full Mode B `barbellrow_0000`: 3 camera/1,770 frame, 전 completion check PASS,
  합산 2,960.81초(0.59781 frame/s), combined peak 61,821 MiB
- full Mode B `squat_0001`: 3 camera/3,801 frame, 전 completion check PASS,
  합산 6,080.57초(0.62511 frame/s), combined peak 70,359 MiB
- first body fit: 590 timestamp × 26 canonical joint, finite/NaN contract와 coverage/alignment PASS;
  displacement p95 0.05167 및 camera REVIEW 전파로 `REVIEW_BODY_FIT_QUALITY`
- Mode C assessor: boundary temporal outlier 중심 84-frame `REVIEW_MODE_C_CANDIDATE`;
  `squat_0001`은 후보 0 `PASS_MODE_B_FROZEN`; Mode C 실행/채택 0, Mode B payload 유지
- private export smoke: REVIEW 1/FAIL 0/INCOMPLETE 0, 34 files, SHA/size mismatch 0,
  `freeze_eligible=true`
- quality/exact-tree private smoke: commit `250ee73`, REVIEW 1/FAIL 0/INCOMPLETE 0,
  36 files/28,993,394 bytes, `git_worktree_dirty=false`, exact-tree error 0;
  same build ID rerun은 `IMMUTABLE_BUILD_REUSED`
- deadline contract v2 partial smoke: clean commit `7b54214`, requested order
  `barbellrow_0000,benchpress_0001` exact bind, REVIEW 1/INCOMPLETE 1,
  36 files/28,993,641 bytes, verifier error 0, `git_worktree_dirty=false`,
  `freeze_eligible=false` (의도된 truthful partial snapshot)
- single-descriptor source snapshot smoke: clean commit `f1b701e`, REVIEW 1,
  36 files/28,993,437 bytes, contract v2 requested order exact, verifier error 0,
  `git_worktree_dirty=false`, `freeze_eligible=true`; same build ID rerun은
  `IMMUTABLE_BUILD_REUSED`로 copy 없이 전수 검증
- current valid predeadline checkpoint: build `exercise3d-predeadline-auto-021-32a51bf7c071`,
  completed/freeze-ready 21 sequence를 모두 보존하고 `freeze_eligible=true`; independent verifier가
  requested order/tree/ownership/SHA와 file/byte identity를 재확인했다.
  이 build는 final deadline build ID와 별도이며 남은 generation은 계속한다.
- completed Sapiens 38 camera와 SAM 36 camera의 `run_provenance.json` materialize PASS;
  model/checkpoint/config/source/selection/tool/exact-resume identity 포함
- `latpulldown_0003`: 662×26, coverage/alignment 1.0, prior-only/missing 0,
  displacement p95 0.07748 + camera uncertainty로 REVIEW; Mode C candidate 79, 실행/채택 0
- Body fit/quality complete 21 sequence/15,980 reference frame, quality REVIEW 21/FAIL 0.
- `benchpress_0001`: SAM 3-view 2,019/2,019 frame PASS; body fit 673×26, coverage/alignment 1.0,
  prior-only/missing 0, displacement p95 0.10771 + camera uncertainty로 REVIEW; Mode C candidate 0
  `PASS_MODE_B_FROZEN`; quality REVIEW sidecar와 freeze-readiness 검증을 완료했다. Checkpoint follower가
  새 12-sequence immutable build를 자동 publish·검증했으며 기존 output을 재계산하지 않았다.
- `barbellrow_0003`: 3-view SAM source 2,271 frame 중 target-valid 2,115, payload 2,125.
  Intentional abstention tail 156 view는 accepted=false다. Prior consolidation PASS, body fit 757×26,
  final coverage 0.92262, alignment 0.91590, prior-only 7 joint, displacement p95 0.04835,
  FAIL 0/CAMERA_UNCERTAINTY 포함 REVIEW. Mode C는 실행하지 않고 candidate 150만 보존했으며 quality REVIEW다.

## Remaining work

- Sapiens2: 14/78 camera, 15,797 target crop remaining at the 21:59 KST snapshot
- SAM full: 15/78 camera, 17,655 source frame remaining; full-complete sequence 21/26
- Quality/freeze-ready/checkpoint: 21/26. `barbellrow_0003` 포함 integrity-verified durable snapshot 보존
- critical path: pose-complete sequence → Phase 7 gate → Mode B → compact prior → body fit → Mode C candidate QA → export
- Phase 11은 body fit/Mode C 뒤 CPU-only로 생성하며 deadline exporter가 누락 output을 자동 materialize한다.
- supervisor CSV의 `barbellrow_0003,INCOMPLETE,SAM_MODE_B`는 old completion gate에서 두 번 소진된
  historical row다. Live supervisor의 in-memory row/CSV를 수동 수정하지 않는다. Sapiens 종료 후 final
  sequence pass가 current subprocess code를 load해 resume-skip으로 정식 REVIEW upsert한다.

## Resume instructions

1. `HANDOFF.md` → `.runtime/handoff_state.json` → `.runtime/dashboard_state.json` → plan/process/phase
   문서 → Git state 순서로 읽는다.
2. `python tools/monitor_autonomous_generation.py --once`를 실행한다. State가 없거나 stale일 때만
   startup `ps`/`nvidia-smi`/process-tree 명령으로 직접 대조한다. 살아 있는 동일 job은 절대 중복 실행하지 않는다.
3. `attention_required=false`이고 progress가 정상이라면 AI가 `wait`/`ps`/`nvidia-smi`/`tail` polling을
   반복하지 않고 autonomous process에 맡긴다.
4. Sapiens camera는 `metadata.json` PASS, frame 수, `(N,308,2)/(N,308)` shape, finite/abstention을 검사한다.
   SAM camera는 benchmark/profile/provenance, exact target-valid coverage, timeline-bounded mesh/numeric frame
   identity와 required field를 검사한다. Abstention frame payload 수가 source frame 수보다 적다는 이유만으로
   incomplete 처리하거나 background person을 채우지 않는다.
5. 죽어 있으면 live process absence와 child 부재를 다시 확인한 뒤 `.runtime/handoff_state.json`의
   exact frozen resume command를 사용한다. Sapiens와 SAM runner는 PASS camera/chunk를 검증 후 skip하며,
   두 runner의 lifetime lock과 output-bound legacy/orphan process guard를 우회하지 않는다. SAM prior
   Phase 7 triangulation, SAM prior consolidation, body-fit, Mode C assessment, Phase 11 quality도 current
   source signature와 output/acceptance contract가 exact할 때만 materialization을 skip한다. Phase 7 marker가 없거나
   `IN_PROGRESS`/signature mismatch이면 해당 sequence가 실제 재호출될 때만 재삼각화한다.
6. supervisor가 죽었으면 local state의 exact supervisor command로 재실행한다. `--overwrite`는 사용하지 않는다.
   단, `.runtime/supervisor_watchdog_state.json`의 watchdog이 정상이면 수동 launch하지 말고
   자동 recovery 결과를 사용한다. `ATTENTION`/재시도 소진일 때만 수동 개입한다.
   Healthy live supervisor와 persisted resume command가 exact match하지만 pinned SHA만 stale이면 generic
   auto-repin하지 않는다. Exactly-one live/resume identity, child/locks/restart budget을 확인하고 old CPU
   watchdog만 종료한 뒤 `run_autonomous_supervisor_watchdog.py --once --adopt-live-command` one-shot을 사용하고,
   persistent watchdog은 flag 없는 기존 exact argv로 시작한다.
7. resume 후 `python -m unittest discover -s tests -p 'test_*.py'`와 마지막 completed camera completion gate를 확인한다.
8. dashboard 또는 handoff monitor가 없으면 먼저 `.runtime/monitoring_watchdog_state.json`을 확인한다.
   Watchdog이 RUNNING이면 수동 launch하지 않고 3-cycle automatic recovery를 사용한다. Watchdog
   ATTENTION/restart exhaustion일 때만 exact absence를 재확인하고 handoff의 target resume command로
   복구한다. 두 target lifetime lock을 우회하지 않는다.
9. `.runtime/deadline_snapshot_state.json`의 sentinel이 없으면 `HANDOFF.md`와 local resume
   command로 복구한다. 단 deadline sentinel watchdog이 RUNNING이면 수동 launch하지
   말고 exact-identity recovery에 맡긴다. 기존 deadline build manifest가 있으면 duplicate
   export하지 않는다. Watchdog attention/restart exhaustion일 때만 수동 개입한다.
10. body-fit count가 quality count보다 큰데 quality follower가 없으면 먼저
    `.runtime/quality_follower_watchdog_state.json`을 확인한다. Watchdog이 RUNNING이면 수동 launch하지
    말고 3-cycle automatic recovery를 사용한다. Watchdog ATTENTION/restart exhaustion일 때만 exact
    absence와 child 부재를 재확인해 handoff resume command로 복구한다. Follower lifetime lock을
    우회하지 않으며 valid sequence는 재계산하지 않는다.
11. `freeze_readiness.failures`는 completed quality의 export validation failure이다. 5분 유예 후
    누락 sidecar가 지속되거나 FAIL이면 dashboard `FREEZE_READINESS_FAILED`를 보고한다.
    Quality/acceptance threshold를 낮추지 말고 reason의 source stage만 recovery한다.
12. deadline 전 checkpoint follower가 없으면 먼저
    `.runtime/predeadline_checkpoint_follower_watchdog_state.json`을 확인한다. Watchdog이 RUNNING이면
    수동 launch하지 말고 3-cycle automatic recovery를 사용한다. Watchdog ATTENTION/restart exhaustion일
    때만 exact absence와 exporter child 부재를 재확인해 handoff resume command로 복구한다.
    Ready 집합이 기존 checkpoint와 같을 때 `attempted_build_id=null`이어야 하며, deadline 후
    follower/watchdog 종료는 정상이다.

Dashboard 사용:

```bash
python tools/monitor_autonomous_generation.py --once
python tools/monitor_autonomous_generation.py --refresh-seconds 10
python tools/monitor_autonomous_generation.py --quiet --refresh-seconds 30
tmux new-window -n exercise3d-dashboard \
  'cd <REPOSITORY_ROOT> && python tools/monitor_autonomous_generation.py --refresh-seconds 10'
```

기존 inference를 dashboard/tmux로 옮기기 위해 restart하지 않는다.

## Frozen decisions

- Sapiens2-5B + official flip-test가 primary 2D teacher다.
- primary target 한 명만 추론하며 background candidate는 metadata만 보존한다.
- `TARGET_AMBIGUOUS`/`NO_TARGET`은 강제 선택하지 않는다.
- SAM full 기본은 Mode B, Mode C는 evidence 기반 selective escalation만 허용한다.
- camera는 Background BA를 사용하고 REVIEW uncertainty를 downstream에 전파한다.
- raw/synchronized video/working frame은 immutable이며 temporal correction은 metadata/pairing 수준이다.
- private payload/checkpoint/frame/얼굴/credential은 public Git에 올리지 않는다.
- deadline은 2026-08-14 13:00 KST다.

## Current exceptions / uncertainty

- Phase 5 camera REVIEW 15개와 observation-conditioned Phase 7 recovery provenance를 유지한다.
- target ambiguity 139, `NO_TARGET` 26은 abstention으로 보존한다.
- sequence→subject mapping은 `UNKNOWN`; cross-sequence shape fusion은 하지 않는다.
- Fit3D payload 부재로 exhaustive validation은 freeze 이후다.
- 한 A100에서 전량 순차 completion은 deadline 전 불가능하며 incomplete는 `INCOMPLETE_DEADLINE`로 남긴다.
- 첫 Mode C 후보는 missing/nonfinite/alignment failure가 아니라 주로 sequence boundary temporal
  outlier다. Sapiens2+Mode B 동시 GPU critical path를 방해하지 않고 REVIEW evidence로 보존한다.

## Runtime estimates

- 2026-08-13 21:59 KST snapshot: Sapiens recent rate 0.220 crop/s,
  ETA는 deadline 약 4시간 55분 후. Downstream overhead-free와 empirical p90-adjusted schedule은
  deadline까지 모두 24/26이며 first late는 `deadlift_0002`다.
- deadline margin: Sapiens 전량 기준 약 -4.92 h; 대신 Mode B complete sequence와
  deadline snapshot을 내구적으로 확보
- concurrent SAM Mode B aggregate 47,940 durable source frame, measured 약 0.567 frame/s;
  standalone expected 20.80 h projection은
  historical baseline이며 Mode C는 약 1.99배라 full default 금지
- live 병렬 ETA와 deadline margin은 `.runtime/handoff_state.json`에서 확인한다.

## Git state

- branch: `agent/phase-5-1-pushup-0003-recovery`
- HEAD `b8d7a01`: recovered historical supervisor row dashboard suppression. Explicit exactly-one supervisor
  watchdog repinning은 `15701f9`. Recent target-complete fixes는
  runner `6e08988`, provenance/handoff consumers `0667c70`, prior consolidation `fdee353`, finite body
  consensus warning `80d9d80`. Earlier validation-bound freeze copy `2cfd6b7`; source-bound Phase 11 quality
  resume `533959e`; source-bound Phase 7
  triangulation reuse `fadd5c7` + in-flight source-race
  guard `c55bc5c`; source-bound Mode C assessment resume `1c8046e`; source-bound body-fit
  resume `6cacff1`; source-bound SAM prior resume
  `e641bfa`; SAM single-target exact-tree
  gate `d3fc911`; SAM provenance completion
  gate `11a91ca`; streaming transient retry
  `0412590`; SAM duplicate-resume/orphan guard
  `46cdced`; Sapiens duplicate-resume guard `30a051d`; marker ctime cutoff attestation `6e802b1`;
  sentinel loaded-code identity `60eadb8`; deadline marker identity binding `e31098c`; premature
  deadline publication gate `ddd3461`; durable latest-completion event `a0ad72c`; remaining deadline
  order audit `f8f603b`; immutable freeze storage forecast `5bb9c4c`; monitoring-plane recovery
  watchdog `16a8600` + default-path validation
  fix `5c93d4e`; SAM output storage forecast `b24f509`; quality follower recovery
  watchdog `4600dff`; empirical downstream
  deadline forecast `7ffeb9a`; checkpoint follower
  recovery watchdog `16fd41f`; deadline freeze
  coverage forecast `8b55df7`; autonomous
  predeadline checkpoint follower `711d4fd`; durable
  checkpoint dashboard `80f48ab`; predeadline checkpoint manifest source commit `54a8d2c`; exact `HEAD`는
  `git rev-parse HEAD`와 local state의 `git_commit`으로 확인
- Draft PR: #1 (`https://github.com/06-month/Exercise3D-Dataset-Pipeline/pull/1`)
- pushed: `b8d7a01`까지 origin branch와 동기화 완료
- dirty: 이 handoff 문서 commit 직전 기준 tracked code/docs 외 private runtime/output은 Git 제외;
  exact clean/dirty 상태는 `git status --short`로 확인

## Last updated

- 2026-08-13 22:10 KST
