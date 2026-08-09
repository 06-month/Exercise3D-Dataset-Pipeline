# Visual QA Checklist

camera geometry representative sequence마다 다음을 `PASS`, `REVIEW`, `FAIL`로 기록한다.

1. camera arrangement plausible
2. camera orientation plausible
3. static background coherent
4. ground plane roughly coherent
5. wall/rig structure plausible
6. cross-view point-cloud overlap plausible
7. no obvious mirrored geometry
8. no severe duplicated scene structure
9. no isolated exploding point cloud
10. per-camera temporal pose stability

visual 판정은 정량 residual을 대체하지 않는다. 두 evidence를 함께 보며 `REVIEW`를 임의로
PASS로 승격하지 않는다. 공개 Git에는 사람을 식별할 수 있는 render/screenshot을 올리지 않는다.
