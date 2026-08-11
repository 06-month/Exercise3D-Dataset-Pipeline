import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.consolidate_sam_body_prior import (
    REQUIRED_PRIOR_FIELDS,
    consolidate_camera,
    load_mapping,
)


class ConsolidateSamBodyPriorTest(unittest.TestCase):
    def test_consolidation_keeps_ambiguous_output_but_does_not_accept_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            camera = root / "sam" / "sequence" / "cam1" / "mode_b_private_output"
            numeric = camera / "mhr_numeric" / "1"
            numeric.mkdir(parents=True)
            np.savez_compressed(
                camera / "target_provenance.npz",
                frame_names=np.asarray(["00000000.jpg", "00000001.jpg"]),
                source_frame_names=np.asarray(["000000.jpg", "000001.jpg"]),
                source_frame_indices=np.asarray([0, 1], dtype=np.int32),
                target_bboxes_xyxy=np.asarray(
                    [[10, 20, 110, 220], [np.nan, np.nan, np.nan, np.nan]],
                    dtype=np.float32,
                ),
                target_valid=np.asarray([True, False]),
                target_selection_confidence=np.asarray([0.9, 0.4], dtype=np.float32),
                target_ambiguous=np.asarray([False, True]),
                no_target=np.asarray([False, False]),
                occlusion_risk=np.asarray([False, True]),
                timestamp_pts_seconds=np.asarray([0.0, 1.0 / 30.0]),
            )
            shapes = {
                "focal_length": (),
                "bbox": (4,),
                "pred_keypoints_3d": (70, 3),
                "pred_keypoints_2d": (70, 2),
                "pred_cam_t": (3,),
                "pred_pose_raw": (266,),
                "global_rot": (3,),
                "body_pose_params": (133,),
                "hand_pose_params": (108,),
                "scale_params": (28,),
                "shape_params": (45,),
                "expr_params": (72,),
                "pred_joint_coords": (127, 3),
                "pred_global_rots": (127, 3, 3),
                "mhr_model_params": (204,),
            }
            self.assertEqual(set(shapes), set(REQUIRED_PRIOR_FIELDS))
            for frame in range(2):
                np.savez_compressed(
                    numeric / f"{frame:08d}.npz",
                    **{
                        key: np.full(shape, frame + 1, dtype=np.float32)
                        for key, shape in shapes.items()
                    },
                )
            mapping = load_mapping(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "mhr70_canonical_joints.json"
            )
            output = root / "consolidated"

            qa = consolidate_camera(
                root / "sam" / "sequence" / "cam1",
                output,
                "sequence",
                "cam1",
                mapping,
            )

            self.assertEqual(qa["status"], "PASS")
            self.assertEqual(qa["output_valid_count"], 2)
            self.assertEqual(qa["accepted_prior_count"], 1)
            with np.load(output / "sam_body_prior.npz", allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["output_valid"], [True, True])
                np.testing.assert_array_equal(payload["accepted_prior"], [True, False])
                self.assertEqual(payload["canonical_local_3d"].shape, (2, 26, 3))
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(metadata["not_ground_truth"])


if __name__ == "__main__":
    unittest.main()
