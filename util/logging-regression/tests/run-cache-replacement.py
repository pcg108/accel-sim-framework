#!/usr/bin/env python3
"""Compile production cache replacement tests against an isolated CMake build."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('runtime', Path(__file__).with_name('run-l2-traffic-runtime.py'))
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.SOURCE = Path(__file__).with_name('cache-replacement-runtime.cc')
    binary = runtime.build(args.build_dir.resolve(), args.output)
    process = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.output / 'test.log').write_text(process.stdout)
    (args.output / 'result.json').write_text(json.dumps({'passed': process.returncode == 0, 'returncode': process.returncode}, indent=2) + '\n')
    print(process.stdout, end='')
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
