#!/usr/bin/env python3
"""Supervise versioned traffic collections, including a verified capture handoff.

The JSON manifest contains collections with name, output, argv, and optionally
capture_status. A capture-dependent collection starts only after state=complete
and passed=true. Each runner and its output tools are independently pinned.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def write_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    root = manifest.parent
    config = json.loads(manifest.read_text())
    lock = (root / '.supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    running, streams, states = {}, {}, {}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process in running.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.time()
    while not stopping:
        for spec in config['collections']:
            name = spec['name']
            output = Path(spec['output'])
            state = states.setdefault(name, {'state': 'pending', 'output': str(output)})
            if state['state'] in ('complete', 'failed', 'capture_failed'):
                continue
            if name not in running:
                capture_path = spec.get('capture_status')
                if capture_path:
                    try:
                        capture = json.loads(Path(capture_path).read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        state.update(state='waiting_for_capture')
                        continue
                    state['capture_stage'] = capture.get('state')
                    if capture.get('state') == 'failed':
                        state.update(state='capture_failed', failure=capture.get('error'))
                        continue
                    if capture.get('state') != 'complete' or not capture.get('passed'):
                        state.update(state='waiting_for_capture')
                        continue
                    state['captured_kernels'] = capture.get('kernel_count')
                command = list(spec['argv'])
                if (output / 'provenance.json').is_file():
                    command.append('--resume')
                stream = (root / f'{name}-runner.log').open('a', buffering=1)
                process = subprocess.Popen(command, cwd=config['cwd'], stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                running[name], streams[name] = process, stream
                state.update(state='running', pid=process.pid, argv=command, started_unix=time.time())
            summary_path = output / 'summary.json'
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                state.update({key: summary.get(key) for key in (
                    'selected_kernels', 'finished_kernels', 'passed_kernels', 'failed_kernels',
                    'completed_simulations', 'validated_source_counts', 'validated_record_counts')})
                state['summary'] = str(summary_path)
            process = running[name]
            if process.poll() is not None:
                summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
                state.update(state='complete' if summary.get('complete') else 'failed',
                             passed=summary.get('passed', False), returncode=process.returncode,
                             finished_unix=time.time())
                streams.pop(name).close()
                del running[name]
        terminal = all(s['state'] in ('complete', 'failed', 'capture_failed') for s in states.values())
        write_json(root / 'summary.json', {
            'state': 'finished' if terminal else 'running', 'pid': os.getpid(),
            'started_unix': started, 'updated_unix': time.time(), 'complete': terminal,
            'all_requested_kernels_validated': terminal and all(s.get('passed') for s in states.values()),
            'collections': states})
        if terminal:
            return 0 if all(s.get('passed') for s in states.values()) else 1
        time.sleep(15)
    for process in running.values():
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    write_json(root / 'supervisor-interrupted.json', {'time': time.time(), 'collections': states})
    return 130


if __name__ == '__main__':
    raise SystemExit(main())
