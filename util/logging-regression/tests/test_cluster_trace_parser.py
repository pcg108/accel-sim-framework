#!/usr/bin/env python3
"""Compile the production parser and exercise cluster preflight in both formats."""
import ctypes
import lzma
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
ARV = "0010 ffffffff 0 UCGABAR_ARV 0 0 0 0 NoVal"
WAIT = "0020 ffffffff 0 UCGABAR_WAIT 0 0 0 0 NoVal"
EXIT = "0030 ffffffff 0 EXIT 0 0 0 0 NoVal"


def header(ctas, threads=32):
    return f"""-kernel name = cluster_parser_fixture
-kernel id = 700001
-grid dim = ({ctas},1,1)
-block dim = ({threads},1,1)
-shmem = 0
-nregs = 16
-binary version = 90
-cuda stream id = 0
-accelsim tracer version = 6
-enable lineinfo = 0
#traces format = PC mask dest_num reg_dests opcode src_num reg_srcs memory immediates values
"""


def compress_zstd(data):
    lib = ctypes.CDLL("libzstd.so")
    lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    lib.ZSTD_compressBound.restype = ctypes.c_size_t
    lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.ZSTD_compress.restype = ctypes.c_size_t
    output = ctypes.create_string_buffer(lib.ZSTD_compressBound(len(data)))
    size = lib.ZSTD_compress(output, len(output), data, len(data), 1)
    return output.raw[:size]


def write_fixture(path, blocks, instructions=None, threads=32, omit_rank=False):
    """blocks contains (cluster_id, local_rank); original CTA ID is its index."""
    instructions = instructions or [[ARV, WAIT, EXIT] for _ in blocks]
    prefix = header(len(blocks), threads)
    if path.suffix == ".tracez":
        blob = bytearray(prefix.encode() + b"\0" + struct.pack("<II", 0, 100))
        offsets = []
        for lines in instructions:
            offsets.append(len(blob))
            compressed = compress_zstd(("\n".join(lines) + "\n").encode())
            blob += struct.pack("<III", len(lines), 1, len(compressed)) + compressed
        index_offset = len(blob)
        blob += struct.pack("<I", len(blocks))
        for i, ((cluster, rank), offset, lines) in enumerate(zip(blocks, offsets, instructions)):
            blob += struct.pack("<11I", i, 0, 0, cluster, 0, 0, rank, 0, 0, rank, 1)
            blob += struct.pack("<IIQI", len(lines), 1, offset, len(lines))
        blob += struct.pack("<Q", index_offset)
        path.write_bytes(blob)
    else:
        text = prefix
        for i, ((cluster, rank), lines) in enumerate(zip(blocks, instructions)):
            text += f"\n#BEGIN_TB\nthread block = {i},0,0\ncluster id = {cluster},0,0\ncluster cta = {rank},0,0\n"
            if not omit_rank:
                text += f"cluster rank = {rank}\n"
            text += f"warp = 0\ninsts = {len(lines)}\n" + "\n".join(lines) + "\n#END_TB\n"
        path.write_bytes(lzma.compress(text.encode()) if path.suffix == ".xz" else text.encode())


class ClusterParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="cluster-parser-")
        cls.directory = Path(cls.tmp.name)
        driver = cls.directory / "driver.cc"
        driver.write_text('''#include "trace_parser.h"
#include <exception>
#include <iostream>
int main(int argc, char **argv) {
  try {
    trace_parser parser;
    kernel_trace_t *k = parser.parse_kernel_info(argv[1]);
    std::cout << "CLUSTER_CTAS=" << k->cluster_barrier_ctas.size() << "\\n";
    delete k;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\\n";
    return 2;
  }
}
''')
        cls.binary = cls.directory / "parser-test"
        subprocess.run(["g++", "-std=c++17", "-O0", "-pthread", "-I/usr/local/cuda/include",
                        "-I" + str(ROOT / "gpu-simulator/trace-parser"),
                        "-I" + str(ROOT / "util/tracer_nvbit/tracer_tool"),
                        str(driver), str(ROOT / "gpu-simulator/trace-parser/trace_parser.cc"),
                        "-lzstd", "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_case(self, blocks, instructions=None, expected=None, threads=32, omit_rank=False):
        formats = ("traceg", "traceg.xz") if omit_rank else ("traceg", "traceg.xz", "tracez")
        for fmt in formats:
            with self.subTest(format=fmt):
                path = self.directory / ("case." + fmt)
                write_fixture(path, blocks, instructions, threads, omit_rank)
                result = subprocess.run([str(self.binary), str(path)], text=True, capture_output=True, timeout=10)
                if isinstance(expected, str):
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn(expected, result.stderr)
                else:
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(f"CLUSTER_CTAS={len(blocks) if expected is None else expected}", result.stdout)

    def test_two_contiguous_clusters(self):
        self.check_case([(0, 0), (0, 1), (1, 0), (1, 1)])

    def test_repeated_generations(self):
        self.check_case([(0, 0)], [[ARV, WAIT, ARV, WAIT, EXIT]])

    def test_inactive_variants_do_not_enable_preflight(self):
        self.check_case([(0, 0)], [[ARV.replace("ffffffff", "00000000").replace("ARV", "ARV.RELAXED"), EXIT]], expected=0)

    def test_no_cluster_instructions(self):
        self.check_case([(0, 0), (0, 0)], [[EXIT], [EXIT]], expected=0)

    def test_early_exit_warp_allowed(self):
        self.check_case([(0, 0), (0, 1)], [[ARV, WAIT, EXIT], [EXIT]])

    def test_empty_warp_in_active_cluster_rejected(self):
        self.check_case([(0, 0), (0, 1)], [[ARV, WAIT, EXIT], []], expected="zero-instruction warp")

    def test_empty_warp_without_cluster_preserves_behavior(self):
        self.check_case([(0, 0)], [[]], expected=0)

    def test_partial_exit_before_arrival(self):
        partial_exit = EXIT.replace("ffffffff", "0000ffff")
        remaining = [s.replace("ffffffff", "ffff0000") for s in [ARV, WAIT, EXIT]]
        self.check_case([(0, 0)], [[partial_exit, *remaining]])

    def test_partial_exit_between_arrival_and_wait(self):
        partial_exit = EXIT.replace("ffffffff", "0000ffff")
        remaining = [s.replace("ffffffff", "ffff0000") for s in [WAIT, EXIT]]
        self.check_case([(0, 0)], [[ARV, partial_exit, *remaining]])

    def test_full_exit_after_arrival_needs_no_wait(self):
        self.check_case([(0, 0)], [[ARV, EXIT]])

    def test_invalid_exit_mask_rejected_only_for_cluster_kernel(self):
        self.check_case([(0, 0)], [[ARV, WAIT, EXIT, EXIT]], expected="not live")
        self.check_case([(0, 0)], [[EXIT, EXIT]], expected=0)

    def test_partial_final_warp(self):
        self.check_case([(0, 0)], [[s.replace("ffffffff", "0000ffff") for s in [ARV, WAIT, EXIT]]], threads=16)

    def test_divergent_mask(self):
        self.check_case([(0, 0)], [[ARV.replace("ffffffff", "00000001"), WAIT, EXIT]], expected="divergent")

    def test_active_variant(self):
        self.check_case([(0, 0)], [[ARV.replace("ARV", "ARV.RELAXED"), WAIT, EXIT]], expected="unsupported active")

    def test_wait_without_arrive(self):
        self.check_case([(0, 0)], [[WAIT, EXIT]], expected="without matching arrival")

    def test_double_arrive(self):
        self.check_case([(0, 0)], [[ARV, ARV, WAIT, EXIT]], expected="second cluster arrival")

    def test_unmatched_arrive(self):
        self.check_case([(0, 0)], [[ARV]], expected="unmatched cluster arrival")

    def test_duplicate_rank(self):
        self.check_case([(0, 0), (0, 0)], expected="duplicate ranks")

    def test_noncontiguous_cluster(self):
        self.check_case([(0, 0), (1, 0), (0, 1), (1, 1)], expected="noncontiguous")

    def test_incomplete_cluster(self):
        self.check_case([(0, 1)], expected="exceeds cluster size")

    def test_missing_metadata(self):
        self.check_case([(0, 0)], expected="missing CTA, cluster", omit_rank=True)


if __name__ == "__main__":
    unittest.main()
