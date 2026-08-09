# Privacy / Data Publication Policy

## 공개하지 않는 항목

- raw/synchronized video 및 전체 extracted frame
- 얼굴, 신체 식별 정보 또는 private 촬영 환경이 보이는 image/screenshot
- 피험자 개인정보, 원래 식별자, GPS/geolocation, EXIF payload
- checkpoint, pretrained weights, depth, point map, feature, track, PLY
- exact per-sequence final camera K/R/t payload와 비공개 pseudo-label dataset
- private storage absolute path, access token, credential 및 signed URL

## 공개 가능한 항목

- pipeline source, CLI와 configuration template
- 한국어 방법론과 acceptance criteria
- aggregate 또는 sequence-id 수준의 비식별 QA 수치
- exact geometry를 포함하지 않는 refinement/uncertainty statistics
- synthetic/redacted schema와 directory example

## Commit 절차

1. 파일을 명시적으로 stage한다.
2. `python tools/check_publication_safety.py`를 실행한다.
3. staged filename, type, size와 symlink를 확인한다.
4. `git diff --cached --stat`와 `git diff --cached`를 검토한다.
5. media/weight/numeric payload와 absolute path가 0건일 때만 commit한다.

`.gitignore`는 방어선이지 승인 목록이 아니다. ignored 파일이라도 `git add -f`로 강제
추가하지 않는다. 사람 얼굴이 포함될 가능성이 있는 debug image는 redaction 여부와 무관하게
기본 공개 금지다.
