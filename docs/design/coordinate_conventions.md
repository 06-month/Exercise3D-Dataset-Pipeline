# Camera / World Coordinate Convention

## VGGT-Ω extrinsic

저장 pose는 OpenCV world→camera `camera-from-world` 행렬이다.

```text
X_camera = R_world_to_camera @ X_world + t_world_to_camera
C_world = -R_world_to_camera.T @ t_world_to_camera
```

camera axis는 +x image right, +y image down, +z camera forward다. OpenGL/Blender display를 위한
axis flip은 raw numeric output에 적용하지 않고 visualization transform으로만 관리한다.

## Intrinsic

```text
K = [[fx, 0, cx],
     [0, fy, cy],
     [0,  0,  1]]
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

VGGT K는 model canvas pixel 좌표다. raw/working image 좌표로 옮길 때 `frames.csv`의
crop/resize/padding provenance를 적용해야 한다. distortion coefficient는 VGGT output에 없다.

## Depth / point map / confidence

- depth는 positive camera-Z지만 metric scale이 보장되지 않는다.
- point map은 depth, K와 pose에서 파생한 동일 gauge의 world point다.
- confidence는 `[1,∞)` ranking score이며 probability나 표준편차가 아니다.
- confidence filtering은 absolute threshold가 아니라 frame/sequence percentile을 쓴다.

## Background BA gauge와 scale

- cam1 robust physical pose를 exact identity로 고정한다.
- cam2/cam3은 cam1 기준 relative transform이다.
- global scale은 sequence-local arbitrary다.
- Phase 5는 initial cam1-cam2 baseline을 scale anchor로 보존하지만 metric anchor는 아니다.
- 서로 다른 sequence의 coordinates를 similarity alignment 없이 직접 합치지 않는다.
