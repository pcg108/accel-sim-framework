import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "validate_traces.py"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def add_kernel(root: Path, kernel_id: int, *, candidate: bool, bad_elapsed: bool = False) -> None:
    kernel = f"kernel_{kernel_id}_test_kernel"
    l1_root = root / "l2_trace"
    common = Path(kernel) / "shader_0" / "scheduler_0"
    write(
        l1_root / common / "l1_to_l2_requests.txt",
        "request_uid=10 PC=0x00000020 dynamic_warp=0 inst_ordinal=2 "
        "addr=0x1000 subpartition=0 set_index=0 tag=0x1000 sector_mask=0x1 "
        "size=32 type=READ l1_to_l2_cycle=100\n",
    )
    elapsed = 51 if bad_elapsed else 50
    write(
        l1_root / common / "l2_to_icnt_timing.txt",
        f"request_uid=10 start_cycle=100 end_cycle=150 elapsed_cycle={elapsed}\n",
    )
    write(
        root / "icache_status" / common / "icache_access.txt",
        "PC=0x00000020 warp=0 status=HIT\n",
    )
    write(
        root / "dcache_status" / common / "dcache_access.txt",
        "request_uid=10 PC=0x00000020 warp=0 addr=0x1000 size=32 status=MISS\n",
    )
    write(
        root / "warp_assignments" / kernel / "shader_0.txt",
        "CTA 0 (ctaid=0) assigned to shader core 0:\n"
        "  Warp 0 (warp_in_cta=0 dynamic_warp_id=0) -> Scheduler 0\n\n",
    )
    write(
        root / "traces" / "fixture" / f"kernel-{kernel_id}.traceg",
        f"-kernel name = test_kernel\n-kernel id = {kernel_id}\n"
        "-grid dim = (1,1,1)\n-block dim = (1,1,1)\n",
    )


def l1_request(uid: int, addr: int = 0x1000, sector_mask: int = 1) -> str:
    return (
        f"request_uid={uid} PC=0x00000020 dynamic_warp=0 inst_ordinal=2 "
        f"addr=0x{addr:x} subpartition=0 set_index=0 tag=0x1000 "
        f"sector_mask=0x{sector_mask:x} size=32 type=READ l1_to_l2_cycle=100\n"
    )


def extended_timing(
    request_uid: int,
    transport_uid: int,
    service_uid: int,
    *,
    addr: int = 0x1000,
    sector_mask: int = 1,
    start: int = 100,
    end: int = 150,
) -> str:
    return (
        f"request_uid={request_uid} transport_uid={transport_uid} "
        f"service_uid={service_uid} addr=0x{addr:x} sector_mask=0x{sector_mask:x} "
        f"size=32 start_cycle={start} end_cycle={end} elapsed_cycle={end - start}\n"
    )


class ValidatorTests(unittest.TestCase):
    def invoke(self, candidate: Path, golden: Path, *extra: str):
        json_out = candidate / "report.json"
        markdown_out = candidate / "report.md"
        command = [
            sys.executable,
            str(SCRIPT),
            "--candidate",
            str(candidate),
            "--golden",
            str(golden),
            "--join-policy",
            "strict",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            *extra,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return completed, json.loads(json_out.read_text(encoding="utf-8")), markdown_out

    def test_l2_family_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, golden = root / "candidate", root / "golden"
            add_kernel(candidate, 1, candidate=True)
            add_kernel(golden, 1, candidate=False)
            completed, report, markdown = self.invoke(candidate, golden)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(report["passed"])
            self.assertEqual(report["issues"]["error_count"], 0)
            self.assertTrue(report["comparisons"]["l2_trace"]["files"]["passed"])
            self.assertEqual(
                report["comparisons"]["l2_trace"]["semantics"]["1"]["metrics"]
                ["request_records"]["delta"],
                0,
            )
            self.assertEqual(
                report["comparisons"]["dcache_status"]["semantics"]["1"]
                ["histograms"]["status_hist"]["MISS"]["delta"],
                0,
            )
            self.assertIn("**Result:** PASS", markdown.read_text(encoding="utf-8"))

    def test_legacy_directory_and_option_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, golden = root / "candidate", root / "golden"
            add_kernel(candidate, 1, candidate=True)
            add_kernel(golden, 1, candidate=False)
            (candidate / "l2_trace").rename(candidate / "l1_trace")
            completed, report, _ = self.invoke(candidate, golden)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report["comparisons"]["l2_trace"]["files"]["passed"])
            for option in ("--candidate-l2-dir", "--candidate-l1-dir"):
                completed, report, _ = self.invoke(candidate, golden, option, "l1_trace")
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_bad_elapsed_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, golden = root / "candidate", root / "golden"
            add_kernel(candidate, 1, candidate=True, bad_elapsed=True)
            add_kernel(golden, 1, candidate=False)
            completed, report, _ = self.invoke(candidate, golden)
            self.assertEqual(completed.returncode, 1)
            self.assertFalse(report["passed"])
            self.assertTrue(
                any("elapsed_cycle" in message for message in report["issues"]["error_samples"])
            )

    def test_explicit_subset_ignores_other_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, golden = root / "candidate", root / "golden"
            add_kernel(candidate, 1, candidate=True)
            add_kernel(golden, 1, candidate=False)
            add_kernel(candidate, 2, candidate=True, bad_elapsed=True)
            add_kernel(golden, 2, candidate=False)
            completed, report, _ = self.invoke(candidate, golden, "--kernel-ids", "1")
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(report["inputs"]["kernel_ids"], [1])

    def test_extended_ordinary_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, golden = root / "candidate", root / "golden"
            add_kernel(candidate, 1, candidate=True)
            add_kernel(golden, 1, candidate=False)
            timing = (
                candidate
                / "l2_trace"
                / "kernel_1_test_kernel"
                / "shader_0"
                / "scheduler_0"
                / "l2_to_icnt_timing.txt"
            )
            write(timing, extended_timing(10, 10, 10))
            completed, report, _ = self.invoke(candidate, golden)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            stats = report["datasets"]["candidate"]["l2_trace"]["kernels"]["1"]
            self.assertEqual(stats["timing_schema_hist"], {"extended": 1})
            self.assertEqual(stats["timing_identity_hist"], {"ordinary": 1})
            self.assertEqual(stats["join"]["matched_requests"], 1)

    def test_extended_sector_split_joins_on_parent_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, golden = root / "candidate", root / "golden"
            add_kernel(candidate, 1, candidate=True)
            add_kernel(golden, 1, candidate=False)
            timing = (
                candidate
                / "l2_trace"
                / "kernel_1_test_kernel"
                / "shader_0"
                / "scheduler_0"
                / "l2_to_icnt_timing.txt"
            )
            write(
                timing,
                extended_timing(10, 11, 11, addr=0x1000, sector_mask=1)
                + extended_timing(10, 12, 12, addr=0x1020, sector_mask=2, end=155),
            )
            completed, report, _ = self.invoke(candidate, golden)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            stats = report["datasets"]["candidate"]["l2_trace"]["kernels"]["1"]
            self.assertEqual(stats["timing_identity_hist"], {"sector_split": 2})
            self.assertEqual(stats["join"]["unique_requests"], 1)
            self.assertEqual(stats["join"]["matched_requests"], 1)
            self.assertEqual(stats["join"]["timing_records"], 2)
            self.assertEqual(stats["join"]["orphan_timing_records"], 0)

    def test_extended_lrc_shared_service_keeps_request_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate, golden = root / "candidate", root / "golden"
            add_kernel(candidate, 1, candidate=True)
            add_kernel(golden, 1, candidate=False)
            common = (
                candidate
                / "l2_trace"
                / "kernel_1_test_kernel"
                / "shader_0"
                / "scheduler_0"
            )
            write(
                common / "l1_to_l2_requests.txt",
                l1_request(10, 0x1000, 1) + l1_request(20, 0x1020, 2),
            )
            write(
                common / "l2_to_icnt_timing.txt",
                extended_timing(10, 11, 30, addr=0x1000, sector_mask=1)
                + extended_timing(20, 21, 30, addr=0x1020, sector_mask=2, end=160),
            )
            completed, report, _ = self.invoke(candidate, golden)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            stats = report["datasets"]["candidate"]["l2_trace"]["kernels"]["1"]
            self.assertEqual(stats["timing_identity_hist"], {"shared_service": 2})
            self.assertEqual(stats["join"]["unique_requests"], 2)
            self.assertEqual(stats["join"]["matched_requests"], 2)
            self.assertEqual(stats["join"]["missing_requests"], 0)
            self.assertEqual(stats["join"]["orphan_timing_records"], 0)


if __name__ == "__main__":
    unittest.main()
