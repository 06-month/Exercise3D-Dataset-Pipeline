# 재현성과 Git Workflow

## Path injection

private data와 public source를 분리한다. `EXERCISE3D_DATASET_ROOT` 또는 `--dataset-root`로
입력을 지정하고, read-only source를 사용할 때 output은 ignored `outputs/`에 명시한다.

## Provenance

- source packet/frame PTS와 logical sequence/camera mapping
- 실행 tool commit/hash와 normalized configuration hash
- model upstream revision/checkpoint hash(파일 자체는 미배포)
- coordinate, gauge, scale convention
- input/output file integrity와 mutation count

## Phase workflow

Phase 시작 시 plan 상태를 `IN_PROGRESS`로 바꾸고, acceptance 이후 process/README를 갱신한다.
private scan과 staged diff를 거쳐 Phase 단위 commit/push한다. force push, history rewrite,
원본 overwrite는 금지한다. bootstrap 이후 default branch에서 직접 작업하지 않고 branch와
review workflow를 사용한다.
