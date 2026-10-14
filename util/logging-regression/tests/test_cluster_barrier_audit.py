#!/usr/bin/env python3
"""Independent failure checks for the cluster-barrier regression audit."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location("cluster_gate", Path(__file__).with_name("run-cluster-barriers.py"))
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


class ClusterAuditTest(unittest.TestCase):
    def event(self, event, cta, cycle, generation=0):
        return {"event": event, "kernel_uid": 1, "cluster": [0, 0, 0],
                "generation": generation, "cta": [cta, 0, 0], "warp": 0,
                "cycle": cycle, "mask": 0xffffffff, "pc": 32 + cta * 16}

    def events(self):
        return [self.event("arrive", 0, 2**32 + 10), self.event("wait", 0, 2**32 + 11),
                self.event("arrive", 1, 2**32 + 20), self.event("release", 0, 2**32 + 20),
                self.event("wait", 1, 2**32 + 21), self.event("release", 1, 2**32 + 21)]

    def log(self, events):
        return "\n".join("CLUSTER_BARRIER " + json.dumps(value) for value in events)

    def test_valid_different_pc_and_64bit_cycles(self):
        self.assertTrue(gate.synchronization_audit(self.log(self.events()), {"arrive": 2, "wait": 2})["passed"])

    def test_release_before_peer_arrival(self):
        events = self.events()
        events[3]["cycle"] = 2**32 + 19
        with self.assertRaisesRegex(ValueError, "released before"):
            gate.synchronization_audit(self.log(events))

    def test_missing_wait_or_release(self):
        for remove in (1, 3):
            with self.subTest(remove=remove), self.assertRaisesRegex(ValueError, "participant mismatch"):
                events = self.events()
                del events[remove]
                gate.synchronization_audit(self.log(events))

    def test_duplicate_arrival(self):
        with self.assertRaisesRegex(ValueError, "Duplicate arrival"):
            gate.synchronization_audit(self.log(self.events() + [self.events()[0]]))

    def test_duplicate_release(self):
        with self.assertRaisesRegex(ValueError, "Duplicate wait/release"):
            gate.synchronization_audit(self.log(self.events() + [self.events()[3]]))

    def test_stas_before_or_without_initialization(self):
        for initialized in (False, True):
            with self.subTest(initialized=initialized):
                stas = {"event": "stas_complete", "kernel_uid": 1, "cluster": [0, 0, 0],
                        "cta": [0, 0, 0], "bar_addr": 0x1000400, "cycle": 100}
                memory = [stas]
                if initialized:
                    memory.append(dict(stas, event="init", cycle=101))
                log = self.log(self.events()) + "\n" + "\n".join("CLUSTER_MEMORY " + json.dumps(value) for value in memory)
                with self.assertRaisesRegex(ValueError, "before destination initialization"):
                    gate.synchronization_audit(log)

    def test_arrived_participant_can_exit_before_wait(self):
        events = self.events()
        events[4] = self.event("exit", 1, 2**32 + 21)
        del events[5]
        self.assertTrue(gate.synchronization_audit(self.log(events), {"arrive": 2, "wait": 1})["passed"])

    def test_partial_exit_cannot_excuse_missing_wait(self):
        events = self.events()
        events[4] = dict(self.event("exit", 1, 2**32 + 21), mask=0xffff)
        del events[5]
        with self.assertRaisesRegex(ValueError, "participant mismatch"):
            gate.synchronization_audit(self.log(events))

    def test_text_and_tracez_fixture_generation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fixture = gate.make_fixture("partial-final-warp", 980004)
            gate.write_text_fixture(root / "kernel.traceg", fixture)
            gate.write_tracez_fixture(root / "kernel.tracez", fixture)
            text = (root / "kernel.traceg").read_text()
            self.assertIn("0000ffff 0 UCGABAR_ARV", text)
            self.assertIn("-block dim = (48,1,1)", text)
            self.assertGreater((root / "kernel.tracez").stat().st_size, 100)


if __name__ == "__main__":
    unittest.main()
