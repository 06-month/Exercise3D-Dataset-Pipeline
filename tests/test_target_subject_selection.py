import unittest

import numpy as np

from tools.target_subject_selection import (
    minimum_cost_assignment,
    rank_tracks,
    track_candidates,
)


class TargetSubjectSelectionTest(unittest.TestCase):
    def test_rectangular_hungarian_assignment(self) -> None:
        cost = np.asarray([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0]])
        self.assertEqual(set(minimum_cost_assignment(cost)), {(0, 1), (1, 0)})

    def test_persistent_target_beats_larger_high_score_passer(self) -> None:
        frame_count = 100
        boxes = []
        scores = []
        for frame in range(frame_count):
            target = np.asarray(
                [0.25 + 0.001 * frame, 0.20, 0.65 + 0.001 * frame, 0.90],
                dtype=np.float32,
            )
            frame_boxes = [target]
            frame_scores = [0.93]
            if 35 <= frame < 55:
                frame_boxes.append(
                    np.asarray(
                        [
                            0.05 + 0.02 * (frame - 35),
                            0.05,
                            0.75 + 0.02 * (frame - 35),
                            0.98,
                        ],
                        dtype=np.float32,
                    )
                )
                frame_scores.append(0.99)
            boxes.append(np.stack(frame_boxes))
            scores.append(np.asarray(frame_scores, dtype=np.float32))

        for reverse in (False, True):
            tracks, _ = track_candidates(
                boxes, scores, max_gap=4, threshold=0.30, reverse=reverse
            )
            ranked = rank_tracks(tracks, boxes, frame_count)
            self.assertEqual(len(ranked[0][0].observations), frame_count)
            self.assertTrue(
                all(
                    observation.candidate == 0
                    for observation in ranked[0][0].observations.values()
                )
            )

    def test_candidate_order_changes_do_not_change_identity(self) -> None:
        random = np.random.default_rng(7)
        frame_count = 120
        boxes = []
        scores = []
        expected = []
        for frame in range(frame_count):
            target = np.asarray(
                [
                    0.25 + 0.0005 * frame,
                    0.18 + 0.02 * np.sin(frame / 10),
                    0.67 + 0.0005 * frame,
                    0.92 + 0.02 * np.sin(frame / 10),
                ],
                dtype=np.float32,
            )
            background = np.asarray(
                [0.78 - 0.002 * frame, 0.30, 0.94 - 0.002 * frame, 0.78],
                dtype=np.float32,
            )
            order = random.permutation(2)
            candidates = [target, background]
            candidate_scores = [0.91, 0.995]
            boxes.append(np.stack([candidates[index] for index in order]))
            scores.append(
                np.asarray([candidate_scores[index] for index in order], dtype=np.float32)
            )
            expected.append(int(np.flatnonzero(order == 0)[0]))

        choices = []
        for reverse in (False, True):
            tracks, _ = track_candidates(
                boxes, scores, max_gap=4, threshold=0.30, reverse=reverse
            )
            primary = rank_tracks(tracks, boxes, frame_count)[0][0]
            choices.append(
                [primary.observations[frame].candidate for frame in range(frame_count)]
            )
        self.assertEqual(choices[0], expected)
        self.assertEqual(choices[1], expected)


if __name__ == "__main__":
    unittest.main()
