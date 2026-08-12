import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from sapiens2_target_pipeline import (  # noqa: E402
    acquire_inference_lock,
    command_matches_inference_job,
    guarded_infer_command,
    target_chunk_matches_selection,
)


class FakeSelection(SimpleNamespace):
    def detection(self, frame: int):
        start = int(self.candidate_offsets[frame])
        candidate = int(self.target_candidate_index[frame])
        absolute = start + candidate
        return (
            self.all_person_detections_xyxy[absolute : absolute + 1],
            self.all_person_detection_scores[absolute : absolute + 1],
            False,
        )


class TargetChunkResumeTest(unittest.TestCase):
    def test_inference_lifetime_lock_refuses_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inference.lock"
            first = acquire_inference_lock(path)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_inference_lock(path))
            assert first is not None
            first.close()
            recovered = acquire_inference_lock(path)
            self.assertIsNotNone(recovered)
            assert recovered is not None
            recovered.close()

    def test_inference_lock_refuses_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "keep.txt"
            target.write_text("keep", encoding="utf-8")
            lock = root / "inference.lock"
            lock.symlink_to(target)
            with self.assertRaises(OSError):
                acquire_inference_lock(lock)
            self.assertEqual(target.read_text(encoding="utf-8"), "keep")

    def test_command_match_binds_script_command_and_output_root(self) -> None:
        cwd = Path(__file__).resolve().parents[1]
        output = cwd / "outputs" / "pose"
        argv = [
            "python",
            "tools/sapiens2_target_pipeline.py",
            "infer",
            "--output-root",
            "outputs/pose",
        ]
        self.assertTrue(command_matches_inference_job(argv, cwd, output))
        self.assertFalse(
            command_matches_inference_job(
                [*argv[:2], "verify", *argv[3:]], cwd, output
            )
        )
        self.assertFalse(
            command_matches_inference_job(
                argv, cwd, cwd / "outputs" / "other"
            )
        )

    def test_guard_refuses_legacy_same_output_job_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = Namespace(
                instance_lock=root / "inference.lock",
                output_root=root / "pose",
            )
            with (
                patch(
                    "sapiens2_target_pipeline.matching_inference_processes",
                    return_value=[1234],
                ),
                patch("sapiens2_target_pipeline.infer_command") as infer,
                patch("builtins.print"),
            ):
                self.assertEqual(guarded_infer_command(args), 3)
            infer.assert_not_called()

    def test_guard_holds_lock_for_full_inference_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = root / "inference.lock"
            args = Namespace(instance_lock=lock, output_root=root / "pose")

            def observe_lock(_args):
                self.assertIsNone(acquire_inference_lock(lock))
                return 0

            with (
                patch(
                    "sapiens2_target_pipeline.matching_inference_processes",
                    return_value=[],
                ),
                patch(
                    "sapiens2_target_pipeline.infer_command",
                    side_effect=observe_lock,
                ),
            ):
                self.assertEqual(guarded_infer_command(args), 0)
            released = acquire_inference_lock(lock)
            self.assertIsNotNone(released)
            assert released is not None
            released.close()

    def test_process_discovery_fails_closed_without_proc(self) -> None:
        from sapiens2_target_pipeline import matching_inference_processes

        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing-proc"
            with self.assertRaisesRegex(RuntimeError, "process table is unavailable"):
                matching_inference_processes(
                    Path(temporary) / "pose", proc_root=missing
                )

    def make_selection(self) -> FakeSelection:
        return FakeSelection(
            num_person_candidates=np.asarray([1, 1], dtype=np.int16),
            target_candidate_index=np.asarray([0, -1], dtype=np.int16),
            target_selection_confidence=np.asarray([0.9, 0.1], dtype=np.float32),
            target_ambiguous=np.asarray([False, True]),
            no_target=np.asarray([False, False]),
            target_status=np.asarray(["TARGET", "TARGET_AMBIGUOUS"]),
            occlusion_risk=np.asarray([False, True]),
            candidate_offsets=np.asarray([0, 1, 2], dtype=np.int64),
            all_person_detections_xyxy=np.asarray(
                [[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]],
                dtype=np.float32,
            ),
            all_person_detection_scores=np.asarray([0.95, 0.8], dtype=np.float32),
        )

    def write_chunk(self, path: Path, selection: FakeSelection) -> None:
        np.savez_compressed(
            path,
            frame_index=np.asarray([0, 1], dtype=np.int32),
            frame_name=np.asarray(["000000.jpg", "000001.jpg"]),
            target_present=np.asarray([True, False]),
            num_person_candidates=selection.num_person_candidates,
            target_candidate_index=selection.target_candidate_index,
            target_selection_confidence=selection.target_selection_confidence,
            target_ambiguous=selection.target_ambiguous,
            no_target=selection.no_target,
            target_status=selection.target_status,
            occlusion_risk=selection.occlusion_risk,
            bbox_xyxy=np.asarray(
                [[10.0, 20.0, 30.0, 40.0], [np.nan] * 4], dtype=np.float32
            ),
            bbox_score=np.asarray([0.95, np.nan], dtype=np.float32),
        )

    def test_resume_requires_current_selection_and_detection(self) -> None:
        selection = self.make_selection()
        frames = [Path("000000.jpg"), Path("000001.jpg")]
        with tempfile.TemporaryDirectory() as directory:
            chunk = Path(directory) / "chunk.npz"
            self.write_chunk(chunk, selection)
            with np.load(chunk, allow_pickle=False) as payload:
                self.assertTrue(
                    target_chunk_matches_selection(payload, frames, 0, selection)
                )

            changed = self.make_selection()
            changed.all_person_detections_xyxy[0, 0] += 1.0
            with np.load(chunk, allow_pickle=False) as payload:
                self.assertFalse(
                    target_chunk_matches_selection(payload, frames, 0, changed)
                )


if __name__ == "__main__":
    unittest.main()
