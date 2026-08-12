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
- RUNNING: Phase 11 quality vector 10/26 sequence; live supervisor는 유지하고 exporter fallback 연결
- TODO: remaining triangulation, SAM prior consolidation, body fitting/QC, deadline private export/freeze
- BLOCKED: 없음
- REVIEW/FAIL: camera PASS 11/REVIEW 15/FAIL 0; body fit REVIEW 10/FAIL 0

## Active job

- Sapiens2 PID 373049, 시작 2026-08-11 18:35 KST, output `outputs/sapiens2_target_only_full`
- autonomous supervisor는 2026-08-12 08:45 KST 이후 사라진 것을 live process와
  stale state로 확인한 뒤, 중복/child 부재를 재확인하고 exact resumable command로 09:44 KST 복구했다.
  현재 exact PID/stage는 dashboard/handoff state가 source of truth다.
- 2026-08-12 10:30 KST dashboard snapshot: current streaming sequence `latpulldown_0003`,
  SAM Mode B child PID 1705755 (`cam3`) active
- Sapiens durable 34/78 camera, current partial 포함 22,106/65,430 crop; PID 373049 alive,
  current `benchpress_0001/cam2`
- Sapiens recent-chunk throughput 0.222 crop/s; 병렬 effective 0.216 crop/s;
  projected ETA 2026-08-14 16:42 KST로 deadline 약 3시간 42분 risk
- SAM durable 32/78 camera, 20,779/65,595 frame, 10/26 full sequence; aggregate 0.586 frame/s
- GPU: A100 80GB, combined snapshot 62,693 MiB/100%; observed OOM/retry 없음
- exact live command/PID/progress/ETA: `.runtime/handoff_state.json`
- handoff monitor PID 608232: 30초마다 `.runtime/handoff_state.json`을 atomic rename으로 갱신;
  `updated_at_utc` 증가와 exact active/resume command/stage count 보존 확인 완료
- deadline snapshot sentinel PID 607755: 2026-08-14 13:00 KST에 completed sequence와
  `INCOMPLETE` 목록을 별도 versioned private build로 export; local state는
  `.runtime/deadline_snapshot_state.json`, 현재 `WAITING_DEADLINE`
  Exporter는 hidden `.<build_id>.inprogress`에서 checksum-resume한 뒤 전수 integrity PASS 시 final
  directory로 atomic rename한다. Existing final manifest는 검증 후 reuse하며 같은 ID를 덮어쓰지 않는다.
- dashboard monitor: `tools/monitor_autonomous_generation.py`; atomic state는
  `.runtime/dashboard_state.json`. `--once`는 snapshot, 기본은 Rich live, `--quiet`는 state-only daemon이다.
- Phase 11 CPU follower PID 1786236: complete body-fit/Mode-C dependency만 감지해 quality를
  atomic materialize/validate한다. State는 `.runtime/quality_follower_state.json`; 2026-08-12 10:27 KST
  10/26 REVIEW, failure 0, remaining 16은 dependency wait. Exact resume command/cwd도 같은 state에
  보존하며 GPU work는 하지 않는다.

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
- Sapiens2 pose: complete 33 camera와 current partial 합계 21,433 accepted target crops;
  `latpulldown_0003`까지 11 sequence 3-view schema/finite PASS
- Phase 7 final: 11 sequence schema PASS/body-fit eligible, NO_GO 0
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
- completed Sapiens 34 camera와 SAM 32 camera의 `run_provenance.json` materialize PASS;
  model/checkpoint/config/source/selection/tool/exact-resume identity 포함
- Phase 11: body-fit complete 10 sequence/6,485 reference frame, REVIEW 10/FAIL 0;
  target abstention/SAM rejection view 8/8, missing/prior-only joint frame 0

## Remaining work

- Sapiens2: 44/78 camera, current partial 포함 43,324 target crops
- Phase 7 이후: `latpulldown_0003` 및 이후 pose-complete sequence
- SAM full: 32/78 camera PASS, `latpulldown_0003/cam3` RUNNING, full-complete sequence 10/26
- critical path: pose-complete sequence → Phase 7 gate → Mode B → compact prior → body fit → Mode C candidate QA → export
- Phase 11은 body fit/Mode C 뒤 CPU-only로 생성하며 deadline exporter가 누락 output을 자동 materialize한다.

## Resume instructions

1. `HANDOFF.md` → `.runtime/handoff_state.json` → `.runtime/dashboard_state.json` → plan/process/phase
   문서 → Git state 순서로 읽는다.
2. `python tools/monitor_autonomous_generation.py --once`를 실행한다. State가 없거나 stale일 때만
   startup `ps`/`nvidia-smi`/process-tree 명령으로 직접 대조한다. 살아 있는 동일 job은 절대 중복 실행하지 않는다.
3. `attention_required=false`이고 progress가 정상이라면 AI가 `wait`/`ps`/`nvidia-smi`/`tail` polling을
   반복하지 않고 autonomous process에 맡긴다.
4. Sapiens camera는 `metadata.json` PASS, frame 수, `(N,308,2)/(N,308)` shape, finite/abstention을 검사한다. SAM camera는 benchmark/profile/provenance, mesh/numeric 수량과 required field를 모두 검사한다.
5. 죽어 있으면 live process absence와 child 부재를 다시 확인한 뒤 `.runtime/handoff_state.json`의
   exact frozen resume command를 사용한다. Sapiens와 SAM runner는 PASS camera/chunk를 검증 후 skip한다.
6. supervisor가 죽었으면 local state의 exact supervisor command로 재실행한다. `--overwrite`는 사용하지 않는다.
7. resume 후 `python -m unittest discover -s tests -p 'test_*.py'`와 마지막 completed camera completion gate를 확인한다.
8. handoff monitor가 없으면 `python tools/checkpoint_handoff_state.py ... --poll-seconds 30`을
   동일 root/sequence 설정으로 재실행하고 `updated_at_utc`가 전진하는지 확인한다.
9. `.runtime/deadline_snapshot_state.json`의 sentinel이 없으면 `HANDOFF.md`와 local resume
   command로 복구한다. 기존 deadline build manifest가 있으면 duplicate export하지 않는다.
10. body-fit count가 quality count보다 큰데 quality follower가 없으면
    `.runtime/quality_follower_state.json`과 dashboard의 exact command/process evidence를 확인한 뒤
    `tools/run_quality_control_follower.py`를 resume한다. Valid sequence는 재계산하지 않는다.

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

- Sapiens current recent-chunk rate 0.23401 crop/s, streaming ETA 2026-08-14 14:06 KST
- deadline margin: Sapiens 전량 기준 약 -1.1 h; 대신 Mode B complete sequence와 deadline snapshot 확보
- concurrent SAM Mode B aggregate 19,455 frame/33,159.28초 = 0.58671 frame/s;
  standalone expected 20.80 h projection은
  historical baseline이며 Mode C는 약 1.99배라 full default 금지
- live 병렬 ETA와 deadline margin은 `.runtime/handoff_state.json`에서 확인한다.

## Git state

- branch: `agent/phase-5-1-pushup-0003-recovery`
- latest commit: 현재 `HEAD` (handoff/streaming milestone `9684cd2` 이상);
  exact hash는 `git rev-parse HEAD`와 local state의 `git_commit`으로 확인
- Draft PR: #1 (`https://github.com/06-month/Exercise3D-Dataset-Pipeline/pull/1`)
- pushed: handoff/streaming supervisor와 문서 milestone remote 동기화 완료
- dirty: 정상 milestone 직후 없음; 이후 실행 중 checkpoint 문서 갱신 여부는 `git status`로 확인

## Last updated

- 2026-08-12 09:53 KST
