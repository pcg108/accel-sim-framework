#!/usr/bin/env python3
"""Compile/run focused standalone TraceFileManager tests (no CUDA required)."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--build-dir', type=Path)
parser.add_argument('--cxx', default='g++')
args = parser.parse_args()
include = args.source_root / 'gpu-simulator/gpgpu-sim/src/gpgpu-sim'
source = Path(__file__).with_name('trace-file-manager-runtime.cc')
with tempfile.TemporaryDirectory(prefix='trace-file-manager-build-') as tmp:
    build = args.build_dir or Path(tmp)
    build.mkdir(parents=True, exist_ok=True)
    flags = [args.cxx, '-std=c++17', '-O2', '-fno-builtin', '-Wall', '-Wextra', '-Werror', '-Wno-missing-field-initializers', '-I', str(include)]
    subprocess.run(flags + ['-DTRACE_FILE_MANAGER_SECOND_TU', '-c', str(source), '-o', str(build / 'second.o')], check=True)
    wrappers = ['fopen', 'fputs', 'vfprintf', '__vfprintf_chk', 'fflush', 'fclose', 'setvbuf', 'getrlimit']
    subprocess.run(flags + [str(source), str(build / 'second.o'), '-o', str(build / 'trace-file-manager-test')]
                   + ['-Wl,--wrap=' + name for name in wrappers], check=True)
    subprocess.run([str(build / 'trace-file-manager-test')], check=True, timeout=60)
