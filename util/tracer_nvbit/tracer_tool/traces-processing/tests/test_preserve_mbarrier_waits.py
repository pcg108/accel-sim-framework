#!/usr/bin/env python3
"""Verify the real preprocessor preserves synchronization in text and tracez."""
import ctypes
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'post-traces-processing.cpp'


def read_tracez(path):
    data = path.read_bytes()
    header_end = data.index(b'\0') + 1
    dictionary_size, _ = struct.unpack_from('<II', data, header_end)
    dictionary = data[header_end + 8:header_end + 8 + dictionary_size]
    index = struct.unpack_from('<Q', data, len(data) - 8)[0]
    assert struct.unpack_from('<I', data, index)[0] == 1
    tb = struct.unpack_from('<11I', data, index + 4)
    assert tb[-1] == 1
    count, chunks, offset = struct.unpack_from('<IIQ', data, index + 48)
    stored_count, stored_chunks = struct.unpack_from('<II', data, offset)
    assert (count, chunks) == (stored_count, stored_chunks)
    lib = ctypes.CDLL('libzstd.so')
    lib.ZSTD_createDCtx.restype = ctypes.c_void_p
    lib.ZSTD_freeDCtx.argtypes = [ctypes.c_void_p]
    lib.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    lib.ZSTD_decompress_usingDict.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                             ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.ZSTD_decompress_usingDict.restype = ctypes.c_size_t
    context = lib.ZSTD_createDCtx()
    result = b''
    offset += 8
    try:
        for _ in range(chunks):
            size = struct.unpack_from('<I', data, offset)[0]
            payload = data[offset + 4:offset + 4 + size]
            length = lib.ZSTD_getFrameContentSize(payload, size)
            assert length < 1000000
            decoded = ctypes.create_string_buffer(length)
            actual = lib.ZSTD_decompress_usingDict(context, decoded, length, payload, size,
                                                   dictionary, len(dictionary))
            assert actual == length
            result += decoded.raw[:actual]
            offset += 4 + size
    finally:
        lib.ZSTD_freeDCtx(context)
    lines = result.decode().splitlines()
    assert len(lines) == count
    return lines


class PreserveWaits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='preserve-waits-')
        cls.root = Path(cls.temporary.name)
        cls.binary = Path(os.environ.get('TRACE_PREPROCESSOR_BINARY', cls.root / 'preprocessor'))
        if 'TRACE_PREPROCESSOR_BINARY' not in os.environ:
            subprocess.run(['g++', '-std=c++17', '-O2', '-pthread', str(SOURCE), '-lzstd',
                            '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def check_instructions(self, lines):
        for text in (True, False):
            with self.subTest(format='text' if text else 'tracez'):
                with tempfile.TemporaryDirectory(dir=self.root) as tmp:
                    directory = Path(tmp)
                    trace = directory / 'kernel-1.trace'
                    trace.write_text('-kernel name = preserve_waits\n-kernel id = 1\n'
                                     '-grid dim = (1,1,1)\n-block dim = (32,1,1)\n'
                                     '-enable lineinfo = 0\n' + ''.join(
                                         '0 0 0 0 0 0 0 0 0 0 0 ' + line + '\n' for line in lines))
                    listing = directory / 'kernelslist'
                    listing.write_text(trace.name + '\n')
                    command = [str(self.binary), str(listing), '-j', '1'] + (['--text'] if text else [])
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    if text:
                        output = (directory / 'kernel-1.traceg').read_text().splitlines()
                        first = next(i for i, line in enumerate(output) if line.startswith('insts = '))
                        count = int(output[first].split('=')[1])
                        actual = output[first + 1:first + 1 + count]
                    else:
                        actual = read_tracez(directory / 'kernel-1.tracez')
                    self.assertEqual(actual, lines)

    def test_sparse_lanes_arrive_at_remote_rank(self):
        self.check_instructions([
            '6a00 01000001 1 R255 SYNCS.ARRIVE.TRANS64.RED.A1T0 2 R0 R255 4 2 0x430 117440512 0 0',
            '4590 ffffffff 0 SYNCS.PHASECHK.TRANS64.TRYWAIT 2 R0 R4 4 1 0x7000430 0 0 0',
        ])

    def test_all_address_encodings_and_negative_delta(self):
        for addresses in ['0 0x1000430 0x7000430', '1 0x1000430 100663296',
                          '2 0x7000430 -100663296']:
            self.check_instructions([
                '6a00 00000003 1 R255 SYNCS.ARRIVE.TRANS64.RED.A1T0 2 R0 R255 4 ' + addresses + ' 0 0',
                '4590 ffffffff 0 SYNCS.PHASECHK.TRANS64.TRYWAIT 2 R0 R4 4 1 0x1000430 0 0 0',
                '4590 ffffffff 0 SYNCS.PHASECHK.TRANS64.TRYWAIT 2 R0 R4 4 1 0x7000430 0 0 0',
            ])

    def test_waits_without_visible_arrivals_and_replay_order(self):
        wait = '4590 ffffffff 0 SYNCS.PHASECHK.TRANS64.TRYWAIT 2 R0 R4 4 1 0x490 0 0 0'
        self.check_instructions([
            '0000 ffffffff 0 REPLAY_START 0 0 0 0', wait,
            '45a0 ffffffff 0 SYNCS.PHASECHK.TRANS64 2 R0 R4 4 1 0x490 0 0 0',
            '0000 ffffffff 0 REPLAY_END 0 0 0 0', wait,
            wait.replace('ffffffff', '00000000'),
        ])

    def test_multicast_target_wait(self):
        self.check_instructions([
            '4600 ffffffff 0 UTMALDG.4D.MULTICAST 0 2 0x400 1 0x00ff 2 0 4 0x780000 1 0 0',
            '4590 ffffffff 0 SYNCS.PHASECHK.TRANS64.TRYWAIT 2 R0 R4 4 1 0x7000400 0 0 0',
        ])


if __name__ == '__main__':
    unittest.main()
