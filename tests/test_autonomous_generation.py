import os
import subprocess
import unittest
from pathlib import Path

from tools.run_autonomous_generation import free_gib, process_alive


class AutonomousGenerationTest(unittest.TestCase):
    def test_process_alive_for_current_and_finished_process(self) -> None:
        self.assertTrue(process_alive(os.getpid()))
        process = subprocess.Popen(["true"])
        process.wait()
        self.assertFalse(process_alive(process.pid))
        self.assertGreater(free_gib(Path(__file__)), 0)


if __name__ == "__main__":
    unittest.main()
