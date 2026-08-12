# Final Private Dataset Schema v1

Phase 13 freeze candidate는 versioned directory와 SHA-256 manifest로 구성한다. 원본 RGB/video를
복제하지 않고 source frame name/index/PTS만 logical reference로 저장한다.

Build는 `<output>/.<build_id>.inprogress`에서 camera/sequence payload checksum을 재사용하며
구축한다. `sequence_status.csv`, 모든 manifest-listed file의 byte/SHA-256, status count와 privacy
flag를 전수 검증한 뒤에만 directory rename으로 `<output>/<build_id>`를 한 번에 publish한다.
동일 build ID의 exporter는 `<output>/.locks/<build_id>.lock`을 process lifetime 동안
non-blocking advisory lock하며, lock을 얻지 못한 caller는 hidden staging을 건드리기 전에
exit code 75로 종료한다. Deadline sentinel 자체도 별도 lifetime lock으로 single instance를
강제하고 watchdog recovery launch race를 차단한다.
각 source payload는 symlink를 거부하고 `O_NOFOLLOW`로 한 regular-file descriptor를 연 뒤,
device/inode/size/mtime/ctime identity를 hash·copy 전후로 재검증한다. 즉 path가 atomic
replacement되거나 file이 in-place mutation되는 동안 섞인 byte stream은 publish하지 않는다.
Copied file, metadata file, staging→final directory rename 후에는 해당 directory를 fsync한다.
최종 manifest가 존재하는 build ID는 immutable하며, 재실행은 전수 integrity PASS일 때 read-only
reuse만 허용한다. Corrupt/불일치 final build를 같은 ID로 덮어쓰지 않는다.
Resume staging에 이전 시도의 temp/unlisted payload가 남아 있으면 final manifest tree에 없는
파일만 정확한 hidden staging root 안에서 제거한다. Nested symlink/mount traversal은 허용하지
않으며, publish verifier는 actual regular-file tree가 manifest + 두 root metadata file과 exact-match인지
검증한다. PASS/REVIEW sequence의 local manifest file set과 global manifest ownership도 서로 일치해야 한다.
Build provenance는 Git commit 뿐 아니라 worktree dirty flag, status hash, tracked diff hash를 저장하며
diff 내용이나 private path는 manifest에 노출하지 않는다.

Deadline build의 sequence membership은 export 완료 시각이 아니라 고정 cutoff으로 결정한다.
`body_fit.npz`, body metadata, Mode-C assessment 세 terminal marker가 모두 존재하고 mtime이
cutoff 이하인 sequence만 validation/export 후보다. Deadline 후 완료된 sequence는 exporter
retry 중에 새로 보이더라도 `INCOMPLETE`로 유지한다. Quality와 manifest는 pre-deadline
terminal payload에서 deadline 후 파생할 수 있지만, sequence manifest에 세 terminal marker의
mtime provenance를 남기고 verifier가 cutoff을 다시 확인한다.
Cutoff-eligible sequence가 quality/provenance sidecar lag로 INCOMPLETE이면 sentinel은 initial attempt +
3회 retry 동안 publish를 defer한다. 최종 시도에는 defer flag를 제거해 sidecar가 여전히
누락됐더라도 해당 sequence를 INCOMPLETE로 보존한 immutable manifest를 반드시 생성한다.

`freeze_contract_version=2`는 caller가 요청한 sequence universe/order 전체를
`requested_sequences`와 canonical JSON SHA-256에 bind한다. `sequence_status.csv`는 이 list와
순서/identity가 exact-match해야 하며, 따라서 INCOMPLETE row를 삭제하고 count를 다시 맞춰도
verifier를 통과할 수 없다. Global provenance는 고정 3-file set, PASS/REVIEW sequence는
camera 3개의 target/pose/provenance/SAM prior와 geometry/body/quality/sequence manifest 33-file set을
정확히 만족해야 한다. Legacy smoke contract v1은 historical read-only verification만 유지하고,
deadline sentinel은 expected 26-sequence list를 verifier에 별도 전달한다.

```text
<build_id>/
├── dataset_manifest.json
├── sequence_status.csv
├── provenance/
│   ├── source_inventory.json
│   ├── temporal_audit.json
│   └── temporal_camera_frame_mapping.csv
└── sequences/<sequence_id>/
    ├── sequence_manifest.json
    ├── view/
    │   ├── cam{1,2,3}_target_selection.npz
    │   ├── cam{1,2,3}_target_metadata.json
    │   ├── cam{1,2,3}_pose_2d.npz
    │   └── cam{1,2,3}_pose_metadata.json
    ├── geometry/
    │   ├── triangulated_3d.npz
    │   ├── canonical_3d.npz
    │   └── metadata.json
    ├── body/
    │   ├── cam{1,2,3}_sam_body_prior.npz
    │   ├── cam{1,2,3}_sam_body_metadata.json
    │   ├── body_fit.npz
    │   ├── metadata.json
    │   └── mode_c_escalation.json
    └── quality/
        ├── quality_vector.npz
        └── metadata.json
```

## 핵심 array semantics

- view 2D: source frame index/name/PTS, 308 `(x,y)`, confidence/valid, primary target presence
- target identity: all person candidate ragged metadata, selected index/confidence, ambiguity/NO_TARGET,
  background/mirror/duplicate/occlusion evidence
- geometry: 308-point proposal와 canonical 26 joints, supporting views, reprojection, ray angle,
  DLT conditioning, source frame brackets, timing/camera uncertainty
- SAM prior: target provenance, MHR70 2D/3D, body/hand pose, shape/scale/expression,
  127 joint coordinate/global rotation, 204-d replayable parameter
- sequence fit: canonical 3D/confidence, evidence code, geometry/prior residual, temporal fit,
  sequence shape/scale consensus와 별도 scale-invariant `S0`
- quality: frame별 target/pose/SAM/triangulation/body component vector, categorical flag bitmask,
  PASS/REVIEW/FAIL. Calibrated accuracy probability나 합성 scalar score로 해석하지 않음

모든 valid numeric payload는 finite여야 하고 invalid point는 NaN이다. `PASS`, `REVIEW`, `FAIL`,
`INCOMPLETE`를 분리하며 REVIEW를 PASS로 승격하지 않는다. Camera geometry가 observation-conditioned면
그 provenance와 original Phase 5 uncertainty를 함께 보존한다.

## Subject와 scale

현재 private inventory에는 전체 피험자 수 3명이라는 aggregate는 있지만 evidence-backed
sequence→subject mapping은 없다. 따라서 v1 freeze는 `subject_id=null`과
`subject_mapping_status=SUBJECT_MAPPING_UNAVAILABLE`을 사용한다. 얼굴/외형이나 learned shape로
sequence identity를 임의 추론하지 않고 cross-sequence shape fusion도 수행하지 않는다. Mapping이
별도 provenance와 함께 제공될 때만 pseudonymous subject ID와 subject-level consensus를 새 version에
추가할 수 있다.

각 sequence world는 arbitrary scale/cam1 gauge이므로 서로 직접 합치지 않는다. `S0` reference는
sequence median left/right femur length의 평균이며 body shape parameter와 의미를 혼동하지 않는다.

Logical per-timestamp machine-readable contract는
[`examples/final_record.schema.json`](../../examples/final_record.schema.json)에 있다.
