import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from sapiens2_target_pipeline import target_chunk_matches_selection  # noqa: E402


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
