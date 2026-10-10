"""Small fake simulator/validator fixture for mobile collection resume."""

import importlib.util
import json
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch


RUNNER_PATH = Path(__file__).resolve().parents[1] / "run-mobile-traffic.py"
SPEC = importlib.util.spec_from_file_location("mobile_traffic", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.source = self.root / "kernel-1.traceg"
        self.source.write_text("-grid dim = (1,1,1)\n#BEGIN_TB\n")
        self.config = self.root / "config"
        self.config.write_text("")
        self.args = SimpleNamespace(
            output=self.output, rejected_inputs={}, revalidate=False,
            validator_sha256=RUNNER.sha256(RUNNER.VALIDATOR), binary=self.root / "simulator",
            library_dir=None, max_cycles=0, timeout=10,
            validation_semaphore=threading.Semaphore(1))
        self.calls = []

    def existing(self, kernel, status=None, log=""):
        run = self.output / f"kernel-{kernel}"
        run.mkdir()
        (run / "simulator.log").write_text(log)
        if status is not None:
            RUNNER.write_json(run / "status.json", {"kernel_id": kernel, **status})
        return run

    def fake_process(self, command, **kwargs):
        self.calls.append(command)
        if command[0] == str(self.args.binary):
            kwargs["stdout"].write("gpu_tot_issued_cta = 1\nGPGPU-Sim: *** exit detected ***\n")
        else:
            report = Path(command[command.index("--json-out") + 1])
            RUNNER.write_json(report, {"passed": True, "source_counts": {}, "counts": {}})
        return SimpleNamespace(returncode=0)

    def run_kernel(self, kernel):
        return RUNNER.run_kernel(self.args, kernel, self.source, self.config, None, 4)

    def test_pin_all_validator_modules_and_detect_mutation(self):
        self.args.binary.write_text("fake simulator")
        target = RUNNER.pin_tools(self.output, self.args.binary, None)
        RUNNER.check_pinned_tools(target)
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertIn("validator/l2_traffic_schema.py", manifest)
        self.assertIn("validator/validate_l2_traffic.py", manifest)
        self.assertIn("run-mobile-traffic.py", manifest)
        original = RUNNER.manifest_fingerprint(RUNNER.validator_manifest(target / "validator"))
        schema = target / "validator/l2_traffic_schema.py"
        schema.write_text(schema.read_text() + "\n# changed dependency\n")
        self.assertNotEqual(original, RUNNER.manifest_fingerprint(
            RUNNER.validator_manifest(target / "validator")))
        with self.assertRaisesRegex(ValueError, "Pinned collection tool changed"):
            RUNNER.check_pinned_tools(target)

    def test_two_passed_one_interrupted_one_terminal_failure(self):
        passed = {"kernel_completed": True, "passed": True,
                  "validator_sha256": self.args.validator_sha256}
        for kernel in (1, 2):
            self.existing(kernel, passed)
        partial = self.existing(3, log="simulation was running\n")
        (partial / "partial-records.txt").write_text("preserve me\n")
        failed = {"kernel_completed": False, "passed": False, "returncode": 1,
                  "failure": ["deadlock detected"]}
        self.existing(4, failed, "deadlock detected\n")
        with patch.object(RUNNER.subprocess, "run", side_effect=self.fake_process):
            results = [self.run_kernel(kernel) for kernel in (1, 2, 3, 4)]
        self.assertEqual([result["passed"] for result in results], [True, True, True, False])
        self.assertEqual(len(self.calls), 2)  # Only interrupted kernel simulates and validates.
        archives = list((self.output / "interrupted-attempts").iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "partial-records.txt").read_text(), "preserve me\n")

    def test_completed_unvalidated_does_not_simulate(self):
        self.existing(1, {"kernel_completed": True, "passed": False, "returncode": 0})
        with patch.object(RUNNER.subprocess, "run", side_effect=self.fake_process):
            self.assertTrue(self.run_kernel(1)["passed"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][1], str(RUNNER.VALIDATOR))

    def test_changed_validator_or_explicit_revalidate(self):
        for kernel, fingerprint in ((1, "old"), (2, self.args.validator_sha256)):
            self.existing(kernel, {"kernel_completed": True, "passed": True,
                                   "validator_sha256": fingerprint})
        with patch.object(RUNNER.subprocess, "run", side_effect=self.fake_process):
            self.assertTrue(self.run_kernel(1)["passed"])
            self.args.revalidate = True
            self.assertTrue(self.run_kernel(2)["passed"])
        self.assertEqual(len(self.calls), 2)

    def test_completed_log_without_status_is_recovered(self):
        self.existing(1, log="gpu_tot_issued_cta = 1\nGPGPU-Sim: *** exit detected ***\n")
        with patch.object(RUNNER.subprocess, "run", side_effect=self.fake_process):
            result = self.run_kernel(1)
        self.assertTrue(result["passed"])
        self.assertTrue(result["recovered_from_simulator_log"])
        self.assertEqual(len(self.calls), 1)

    def test_killed_status_replays_but_assertion_without_status_is_preserved(self):
        self.existing(1, {"kernel_completed": False, "passed": False,
                          "returncode": -signal.SIGTERM, "failure": ["Issued CTAs=None, expected=1"]})
        self.existing(2, log="Assertion barrier_count failed\n")
        with patch.object(RUNNER.subprocess, "run", side_effect=self.fake_process):
            self.assertTrue(self.run_kernel(1)["passed"])
            self.assertFalse(self.run_kernel(2)["passed"])
        self.assertEqual(len(self.calls), 2)


if __name__ == "__main__":
    unittest.main()
