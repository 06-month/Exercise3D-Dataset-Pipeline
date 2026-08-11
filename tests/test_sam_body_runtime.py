import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tools.benchmark_sam_body4d import (
    MODE_A,
    MODE_B,
    MODE_C,
    prepare_body4d_config_text,
    prepare_target_input,
    required_checkpoint_components,
)
from tools.sam_body_primary_target_runner import (
    load_target_input,
    run_mode_body4d,
    save_body_prior_numeric,
)
from tools.run_sam_body4d_full import completion_status
from tools.summarize_sam_body_runtime import compute_pass_summary
from tools.summarize_sam_body_runtime import read_rows


class SamBodyRuntimeTest(unittest.TestCase):
    @staticmethod
    def benchmark_row(
        candidate: str,
        camera: str,
        mode: str,
        elapsed: float,
        initialization: float,
    ) -> dict[str, str]:
        return {
            "candidate": candidate,
            "camera": camera,
            "mode": mode,
            "status": "PASS",
            "persons_targeted": "1",
            "frames_processed": "100",
            "elapsed_wall_seconds": str(elapsed),
            "model_initialization_seconds": str(initialization),
            "peak_nvidia_vram_mib": "20000",
            "refinement_model_seconds": "0",
        }

    def test_mode_requirements_exclude_bypassed_vitdet(self) -> None:
        files_a, dirs_a = required_checkpoint_components(MODE_A)
        files_b, dirs_b = required_checkpoint_components(MODE_B)
        files_c, dirs_c = required_checkpoint_components(MODE_C)

        self.assertNotIn("sam3/sam3.pt", files_a)
        self.assertIn("sam-3d-body-dinov3/model_config.yaml", files_a)
        self.assertIn("sam3/sam3.pt", files_b)
        self.assertIn("sam3/sam3.pt", files_c)
        self.assertFalse(dirs_a)
        self.assertFalse(dirs_b)
        self.assertEqual(len(dirs_c), 2)
        self.assertFalse(any("vitdet" in item.lower() for item in files_c))

    def test_body4d_config_only_changes_checkpoint_and_completion(self) -> None:
        source = """paths:
  ckpt_root: \"path to global checkpoint root\"
completion:
  enable: true
  max_occ_len: 25
"""
        prepared = prepare_body4d_config_text(
            source, Path("/private/checkpoints"), completion_enabled=False
        )

        self.assertIn('ckpt_root: "/private/checkpoints"', prepared)
        self.assertIn("enable: false", prepared)
        self.assertIn("max_occ_len: 25", prepared)

    def test_target_adapter_emits_one_bbox_slot_per_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            for frame in range(3):
                (frames / f"{frame:06d}.jpg").touch()
            selection = root / "target_selection.npz"
            np.savez_compressed(
                selection,
                frame_name=np.asarray([f"{frame:06d}.jpg" for frame in range(3)]),
                candidate_offsets=np.asarray([0, 2, 4, 6], dtype=np.int64),
                all_person_detections_xyxy=np.asarray(
                    [
                        [10, 20, 110, 220],
                        [300, 20, 360, 180],
                        [305, 18, 365, 178],
                        [11, 21, 111, 221],
                        [12, 22, 112, 222],
                        [310, 20, 370, 180],
                    ],
                    dtype=np.float32,
                ),
                target_candidate_index=np.asarray([0, 1, -1], dtype=np.int32),
                target_selection_confidence=np.asarray([0.99, 0.98, 0.4], dtype=np.float32),
                target_ambiguous=np.asarray([False, False, True]),
                no_target=np.asarray([False, False, False]),
                occlusion_risk=np.asarray([False, True, True]),
                timestamp_pts_seconds=np.asarray([0.0, 1.0 / 30.0, 2.0 / 30.0]),
            )
            clip = root / "clip"
            target_input = root / "target_input.npz"

            summary = prepare_target_input(
                frames, selection, 0, 3, clip, target_input
            )
            loaded = load_target_input(target_input)

            self.assertEqual(summary["target_seed_count"], 1)
            self.assertEqual(summary["target_valid_frame_count"], 2)
            self.assertEqual(loaded["target_bboxes_xyxy"].shape, (3, 4))
            np.testing.assert_array_equal(
                loaded["target_bboxes_xyxy"][:2],
                np.asarray([[10, 20, 110, 220], [11, 21, 111, 221]], dtype=np.float32),
            )
            np.testing.assert_array_equal(loaded["target_valid"], [True, True, False])
            np.testing.assert_array_equal(loaded["target_ambiguous"], [False, False, True])
            np.testing.assert_array_equal(loaded["no_target"], [False, False, False])
            np.testing.assert_allclose(
                loaded["timestamp_pts_seconds"], [0.0, 1.0 / 30.0, 2.0 / 30.0]
            )
            self.assertTrue(np.isnan(loaded["target_bboxes_xyxy"][2]).all())
            self.assertEqual(len(list(clip.glob("*.jpg"))), 3)

    def test_body4d_clip_cannot_force_ambiguous_first_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "frames"
            frames.mkdir()
            (frames / "000000.jpg").touch()
            selection = root / "target_selection.npz"
            np.savez_compressed(
                selection,
                frame_name=np.asarray(["000000.jpg"]),
                candidate_offsets=np.asarray([0, 1], dtype=np.int64),
                all_person_detections_xyxy=np.asarray(
                    [[10, 20, 110, 220]], dtype=np.float32
                ),
                target_candidate_index=np.asarray([-1], dtype=np.int32),
                target_selection_confidence=np.asarray([0.4], dtype=np.float32),
                target_ambiguous=np.asarray([True]),
                no_target=np.asarray([False]),
                occlusion_risk=np.asarray([True]),
                timestamp_pts_seconds=np.asarray([0.0]),
            )

            with self.assertRaisesRegex(RuntimeError, "accepted primary target"):
                prepare_target_input(
                    frames,
                    selection,
                    0,
                    1,
                    root / "clip",
                    root / "target_input.npz",
                )

    def test_body4d_adapter_seeds_exactly_one_object(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def synchronize() -> None:
                return None

        class FakeTorch:
            cuda = FakeCuda()

            @staticmethod
            def autocast(*args, **kwargs):
                return nullcontext()

        class FakePredictor:
            def __init__(self) -> None:
                self.boxes = []

            def init_state(self, video_path):
                return {"images": video_path}

            def clear_all_points_in_video(self, state) -> None:
                return None

            def add_new_points_or_box(self, **kwargs):
                self.boxes.append(kwargs["box"].copy())
                return None, [1], None, None

        class FakeOfflineApp:
            last_instance = None

            def __init__(self, config_path: str) -> None:
                FakeOfflineApp.last_instance = self
                self.pipeline_mask = None
                self.pipeline_rgb = None
                self.depth_model = None
                self.predictor = FakePredictor()
                self.RUNTIME = {}
                self.OUTPUT_DIR = ""

            def on_mask_generation(self, **kwargs) -> None:
                return None

            def on_4d_generation(self) -> None:
                return None

        fake_module = SimpleNamespace(
            torch=FakeTorch(),
            device="cpu",
            OfflineApp=FakeOfflineApp,
            load_sam_3d_body=lambda *args, **kwargs: (None, None),
            SAM3DBodyEstimator=object,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = root / "clip"
            frames.mkdir()
            (frames / "00000000.jpg").touch()
            arguments = SimpleNamespace(
                mode=MODE_B,
                sam_body4d_root=root,
                body4d_config=root / "body4d.yaml",
                input_frames=frames,
                output_dir=root / "output",
            )
            target = {
                "frame_names": np.asarray(["00000000.jpg"]),
                "target_valid": np.asarray([True]),
                "target_bboxes_xyxy": np.asarray(
                    [[20, 30, 180, 270]], dtype=np.float32
                ),
            }

            fake_pil = SimpleNamespace(
                Image=SimpleNamespace(
                    open=lambda path: SimpleNamespace(size=(200, 300))
                )
            )
            with mock.patch(
                "tools.sam_body_primary_target_runner.load_offline_module",
                return_value=fake_module,
            ), mock.patch.dict("sys.modules", {"PIL": fake_pil}):
                profile = run_mode_body4d(arguments, target)

            self.assertEqual(profile["target_seed_count"], 1)
            self.assertEqual(profile["persons_processed"], 1)
            self.assertIsNotNone(FakeOfflineApp.last_instance)
            seeded = FakeOfflineApp.last_instance.predictor.boxes
            self.assertEqual(len(seeded), 1)
            np.testing.assert_allclose(
                seeded[0], np.asarray([[0.1, 0.1, 0.9, 0.9]], dtype=np.float32)
            )

    def test_compact_mhr_prior_excludes_mesh_and_is_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = [{
                "body_pose_params": np.arange(6, dtype=np.float32),
                "shape_params": np.ones(4, dtype=np.float32),
                "pred_keypoints_3d": np.zeros((3, 3), dtype=np.float32),
                "pred_vertices": np.zeros((18439, 3), dtype=np.float32),
            }]

            count, keys = save_body_prior_numeric(
                outputs, "00000007.jpg", [1], root
            )

            self.assertEqual(count, 1)
            self.assertIn("body_pose_params", keys)
            self.assertNotIn("pred_vertices", keys)
            with np.load(root / "1" / "00000007.npz", allow_pickle=False) as payload:
                self.assertEqual(int(payload["object_id"]), 1)
                self.assertIn("shape_params", payload.files)
                self.assertNotIn("pred_vertices", payload.files)

    def test_full_resume_requires_mesh_numeric_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "mode_b_private_output"
            mesh = private / "mesh_4d_individual" / "1"
            numeric = private / "mhr_numeric" / "1"
            mesh.mkdir(parents=True)
            numeric.mkdir(parents=True)
            (root / "sam_body_benchmark.csv").write_text(
                "status,frames_processed,elapsed_wall_seconds,peak_nvidia_vram_mib,gpu_utilization_mean_pct,power_mean_w\n"
                "PASS,2,10,30000,20,100\n",
                encoding="utf-8",
            )
            (root / "mode_b_profile.json").write_text(
                '{"frames_processed":2,"input_frames":2,"target_seed_count":1,"persons_processed":1}',
                encoding="utf-8",
            )
            np.savez_compressed(
                private / "target_provenance.npz",
                frame_names=np.asarray(["0.jpg", "1.jpg"]),
                target_valid=np.asarray([True, False]),
            )
            for index in range(2):
                (mesh / f"{index}.ply").touch()
                np.savez_compressed(numeric / f"{index}.npz", value=np.asarray(index))

            self.assertEqual(completion_status(root, 2)["status"], "PASS")
            (numeric / "1.npz").unlink()
            result = completion_status(root, 2)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertIn("numeric_prior_complete", result["reason"])

    def test_runtime_summary_separates_best_expected_worst(self) -> None:
        rows = [
            self.benchmark_row("control", "cam1", MODE_A, 80, 10),
            self.benchmark_row("control", "cam1", MODE_B, 110, 10),
            self.benchmark_row("control", "cam1", MODE_C, 320, 20),
            self.benchmark_row("severe", "cam2", MODE_A, 100, 10),
            self.benchmark_row("severe", "cam2", MODE_B, 160, 10),
            self.benchmark_row("severe", "cam2", MODE_C, 500, 20),
        ]

        summary = compute_pass_summary(
            rows,
            ("control", "cam1"),
            ("severe", "cam2"),
            full_frame_count=1000,
            severe_frame_fraction=0.25,
            selective_refinement_fraction=0.20,
        )

        self.assertAlmostEqual(summary["projections"]["BEST_CASE"]["seconds"], 1010)
        self.assertAlmostEqual(
            summary["projections"]["EXPECTED_CASE"]["seconds"], 1620
        )
        self.assertAlmostEqual(summary["projections"]["WORST_CASE"]["seconds"], 4820)
        self.assertGreater(
            summary["comparisons"]["severe"]["refiner_on_off_execution_ratio"],
            summary["comparisons"]["control"]["refiner_on_off_execution_ratio"],
        )

    def test_runtime_reader_enriches_profile_call_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "control" / "mode_c"
            result.mkdir(parents=True)
            (result / "sam_body_benchmark.csv").write_text(
                "mode,candidate,camera,status\nC,control,cam1,PASS\n",
                encoding="utf-8",
            )
            (result / "mode_c_profile.json").write_text(
                '{"refinement_model_calls": 12, "content_completion_calls": 3}',
                encoding="utf-8",
            )

            rows = read_rows(root)

            self.assertEqual(rows[0]["refinement_model_calls"], "12")
            self.assertEqual(rows[0]["content_completion_calls"], "3")


if __name__ == "__main__":
    unittest.main()
