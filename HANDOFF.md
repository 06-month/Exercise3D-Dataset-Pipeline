# Exercise3D Pipeline Handoff

> 이 파일은 현재 operational checkpoint다. 정확한 private 경로·전체 command·실시간 PID는
> Git에서 제외된 `.runtime/handoff_state.json`이 source of truth다.

## Current objective

- 최종 deadline: 2026-08-14 13:00 KST
- 현재 목표: correctness/provenance를 유지하며 end-to-end freeze 가능한 sequence 수 최대화
- 현재 phase: Phase 6 target-only Sapiens2-5B + Phase 7/8 sequence streaming
- acceptance gate: target selector `GO_FULL_DATASET`; SAM Mode B numeric smoke `PASS`

## Current pipeline state

- DONE: Phase 0–5, Phase 6 pilot/target selector, Phase 8 A/B/C pilot와 checkpoint integrity
- RUNNING: Phase 6 full 5B inference, pose-complete sequence의 Phase 7→SAM Mode B→Phase 9 streaming
- TODO: remaining triangulation, SAM prior consolidation, body fitting/QC, private export/freeze
- BLOCKED: 없음
- REVIEW/FAIL: camera PASS 11/REVIEW 15/FAIL 0; pilot triangulation final REVIEW 4/FAIL 0

## Active job

- Sapiens2 PID 373049, 시작 2026-08-11 18:35 KST, output `outputs/sapiens2_target_only_full`
- autonomous supervisor PID 537033, 시작 2026-08-11 20:17 KST, output `outputs/runtime/autonomous_generation`
- 현재 SAM child: `barbellrow_0000/cam2` Mode B full 준비/실행 중
- Sapiens 완료: pilot 4 sequence + `barbellrow_0001`, 15/78 camera, target crop 11,168/65,430
- pre-concurrency steady throughput: 0.23323 crop/s; 병렬 steady throughput은 첫 full SAM camera 후 재계산
- GPU: A100 80GB, cam1 합산 peak 61,821 MiB/100%; OOM 없음
- exact live command/PID/progress/ETA: `.runtime/handoff_state.json`
- handoff monitor PID 575526: 30초마다 `.runtime/handoff_state.json`을 atomic rename으로 갱신;
  `updated_at_utc` 증가와 exact active/resume command/stage count 보존 확인 완료

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
- Sapiens2 pose: 11,168 accepted target crops; `barbellrow_0001` 1,443/1,443, schema/finite PASS
- Phase 7 final: 4 pilot sequence schema PASS/body-fit eligible, NO_GO 0
- concurrent Mode B 8-frame smoke: mesh/numeric/PTS schema PASS, combined peak 48,525 MiB
- full Mode B `barbellrow_0000/cam1`: 590/590, 전 completion check PASS,
  970.94초(0.6077 frame/s), combined peak 61,821 MiB

## Remaining work

- Sapiens2: 63/78 camera, 54,262 target crops
- Phase 7 이후: `barbellrow_0001` 및 이후 pose-complete sequence
- SAM full: 1/78 camera PASS, `barbellrow_0000/cam2` RUNNING, full-complete sequence 0/26
- critical path: pose-complete sequence → Phase 7 gate → Mode B → compact prior → body fit → Mode C candidate QA → export

## Resume instructions

1. `git status --short --branch`, `HANDOFF.md`, `.runtime/handoff_state.json`을 확인한다.
2. 위 startup `pgrep` 명령과 `nvidia-smi`를 실행한다. 살아 있는 동일 job은 절대 중복 실행하지 않는다.
3. Sapiens camera는 `metadata.json` PASS, frame 수, `(N,308,2)/(N,308)` shape, finite/abstention을 검사한다. SAM camera는 benchmark/profile/provenance, mesh/numeric 수량과 required field를 모두 검사한다.
4. 죽어 있으면 `.runtime/handoff_state.json`의 `active_processes[].command`에서 동일 frozen command를 사용한다. Sapiens와 SAM runner는 PASS camera/chunk를 검증 후 skip한다.
5. supervisor가 죽었으면 local state의 exact supervisor command로 재실행한다. `--overwrite`는 사용하지 않는다.
6. resume 후 `python -m unittest discover -s tests -p 'test_*.py'`와 마지막 completed camera completion gate를 확인한다.
7. handoff monitor가 없으면 `python tools/checkpoint_handoff_state.py ... --poll-seconds 30`을
   동일 root/sequence 설정으로 재실행하고 `updated_at_utc`가 전진하는지 확인한다.

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

## Runtime estimates

- Sapiens pre-concurrency recent rate 0.23323 crop/s, 기존 단독 ETA 2026-08-14 12:58 KST
- SAM Mode B standalone expected 20.80 h; Mode C는 약 1.99배라 full default 금지
- live 병렬 ETA와 deadline margin은 `.runtime/handoff_state.json`에서 확인한다.

## Git state

- branch: `agent/phase-5-1-pushup-0003-recovery`
- latest pushed commit: `c50f72609092d54b008d9daf34e8a674daf89618`
- Draft PR: #1 (`https://github.com/06-month/Exercise3D-Dataset-Pipeline/pull/1`)
- unpushed: handoff/streaming supervisor 변경(현재 milestone commit 전)
- dirty: `tools/run_autonomous_generation.py`, test, `HANDOFF.md`, `AGENTS.md`, `.gitignore`

## Last updated

- 2026-08-11 20:35 KST
