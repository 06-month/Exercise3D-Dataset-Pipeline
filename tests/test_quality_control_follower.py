import tempfile
import unittest
from argparse import ArgumentTypeError, Namespace
from pathlib import Path
from unittest.mock import patch

from tools.run_quality_control_follower import (
    dependency_paths,
    parse_list,
    run_cycle,
)


class QualityControlFollowerTest(unittest.TestCase):
    def args(self, root: Path) -> Namespace:
        return Namespace(
            selection_root=root / "selection",
            pose_root=root / "pose",
            triangulation_root=root / "triangulation",
            sam_prior_root=root / "sam_prior",
            sam_mode_c_review_root=root / "mode_c",
            body_fit_root=root / "body",
            output_root=root / "quality",
            runtime_state=root / "state.json",
            sequences=["sequence"],
            poll_seconds=1.0,
            retry_seconds=30.0,
            once=True,
        )

    def materialize_dependencies(self, args: Namespace) -> None:
        for path in dependency_paths(args, "sequence").values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def test_parse_list_rejects_duplicates(self) -> None:
        self.assertEqual(parse_list("one,two"), ["one", "two"])
        with self.assertRaises(ArgumentTypeError):
            parse_list("one,one")

    def test_waits_without_building_until_all_dependencies_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            with patch(
                "tools.run_quality_control_follower.build_sequence_quality"
            ) as build:
                state = run_cycle(args, {}, {}, monotonic_now=10.0)
            build.assert_not_called()
            self.assertEqual(state["status"], "RUNNING")
            self.assertEqual(state["completed_sequence_count"], 0)
            self.assertEqual(state["waiting"][0]["sequence"], "sequence")

    def test_builds_ready_sequence_and_records_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            self.materialize_dependencies(args)
            completed: dict[str, str] = {}
            with (
                patch(
                    "tools.run_quality_control_follower.validate_existing",
                    side_effect=[(False, "missing"), (True, "REVIEW")],
                ),
                patch(
                    "tools.run_quality_control_follower.build_sequence_quality",
                    return_value={"qa": {"sequence_status": "REVIEW"}},
                ) as build,
            ):
                state = run_cycle(args, completed, {}, monotonic_now=10.0)
            build.assert_called_once()
            self.assertEqual(completed, {"sequence": "REVIEW"})
            self.assertEqual(state["status"], "COMPLETE")
            self.assertEqual(state["newly_validated"], ["sequence"])
            self.assertEqual(state["materialized"], ["sequence"])

    def test_failure_reason_survives_retry_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            self.materialize_dependencies(args)
            retry_state: dict[str, dict] = {}
            with (
                patch(
                    "tools.run_quality_control_follower.validate_existing",
                    return_value=(False, "missing"),
                ),
                patch(
                    "tools.run_quality_control_follower.build_sequence_quality",
                    side_effect=RuntimeError("synthetic failure"),
                ),
            ):
                first = run_cycle(args, {}, retry_state, monotonic_now=10.0)
                second = run_cycle(args, {}, retry_state, monotonic_now=20.0)
            self.assertEqual(first["status"], "ATTENTION")
            self.assertIn("synthetic failure", first["failures"][0]["reason"])
            self.assertIn("synthetic failure", second["failures"][0]["reason"])
            self.assertEqual(second["failures"][0]["retry_in_seconds"], 20.0)


if __name__ == "__main__":
    unittest.main()
