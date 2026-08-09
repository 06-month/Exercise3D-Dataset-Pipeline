# Final Dataset Schema 초안

Phase 13에서 동결할 record는 PTS 기반이며 label과 quality를 함께 저장한다.

```text
sequence_id
subject_id                 # 비식별 ID
exercise_id
timestamp                  # source PTS provenance
camera/
    K, R, t
    camera_quality
image_references           # private storage logical reference
keypoints_2d/
keypoints_3d/
body/
    pose
    shape
    global_orient
    translation
quality/
    reprojection
    triangulation
    temporal
    teacher
    camera
    overall
metadata/
```

원본 RGB를 schema payload에 복제하지 않는다. `subject_id`는 공개 가능한 random/pseudonymous
identifier여야 하며 원본 filename/개인정보와 직접 연결하지 않는다. body beta와 downstream
scale-invariant anthropometric descriptor `S0`는 서로 다른 필드와 의미를 가진다.

machine-readable 초안은 `examples/final_record.schema.json`에 있다.
