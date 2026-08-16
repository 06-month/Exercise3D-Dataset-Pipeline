# 기여 및 Phase 운영 규칙

1. 작업 전 `docs/plan.md`에서 해당 Phase를 `IN_PROGRESS`로 표시한다.
2. private dataset path는 CLI/env로 주입하고 source를 overwrite하지 않는다.
3. 알고리즘 변경과 dataset batch 실행을 같은 Phase에서 섞지 않는다.
4. 결과와 실패를 `docs/process.md`에 시간순으로 남긴다.
5. acceptance gate 이후 README/plan/report를 갱신한다.
6. 파일을 명시적으로 stage하고 publication safety 및 staged diff를 검사한다.
7. bootstrap 이후 feature branch에서 Phase 단위 commit과 review를 사용한다.
8. force push, history rewrite, raw media/checkpoint commit은 금지한다.

Commit message 예:

```text
phase 6: add high-quality 2D pose observations
```
