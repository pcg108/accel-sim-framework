#!/usr/bin/env python3
"""Validate large logs without retaining every parsed traffic record in memory.

All UID references are joined before partitioning. A parent, every split child,
all LRC contributors and service/timing rows therefore stay together, including
incorrect cross-references. Each partition uses the existing strict checker;
its counts are combined before checking the real final drain/LRC counters.
The simulator and all source/timing log formats are unchanged.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import validate_l2_traffic as core
from l2_traffic_schema import SOURCE_FILES, TRACKED_SOURCES, key_values, number

UID_PATTERN = re.compile(r'(?:\b(?:request_uid|transport_uid|service_uid)=|"(?:request_uid|transport_uid|service_uid|source_uid)"\s*:\s*"?)(0[xX][0-9a-fA-F]+|[0-9]+)')


def identities(line, filename):
    """Extract references without relaxing validation of the original row."""
    # UID-only scanning avoids allocating full JSON/dict records twice before
    # the actual validation pass. Untracked contributors need special handling
    # because their sentinel parent IDs do not identify a shared request.
    if 'other_read' not in line:
        return [int(x, 16 if x.lower().startswith('0x') else 10)
                for x in UID_PATTERN.findall(line)]
    try:
        if filename == 'l2_traffic_audit.jsonl':
            row = json.loads(line)
            keys = ['transport_uid'] if row.get('source') == 'other_read' else ['source_uid', 'transport_uid']
            return [number(row[k], k) for k in keys if k in row]
        row = key_values(line)
        keys = ['request_uid', 'transport_uid', 'service_uid']
        ids = [number(row[k], k) for k in keys if k in row]
        if filename == 'lrc_to_l2_requests.txt':
            for member in json.loads(row['contributors']):
                ids.append(number(member['transport_uid'], 'transport_uid'))
                if member.get('source') != 'other_read':
                    ids.append(number(member['request_uid'], 'request_uid'))
        return ids
    except (ValueError, TypeError, KeyError, AttributeError):
        # Preserve malformed rows for the strict checker. Any recognizable IDs
        # still join their component, so malformed rows cannot hide duplicates.
        return [int(x, 16 if x.lower().startswith('0x') else 10)
                for x in UID_PATTERN.findall(line)]


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, uid):
        p = self.parent
        p.setdefault(uid, uid)
        while p[uid] != uid:
            p[uid] = p[p[uid]]
            uid = p[uid]
        return uid

    def join(self, ids):
        if not ids:
            return
        root = self.find(ids[0])
        for uid in ids[1:]:
            other = self.find(uid)
            if root != other:
                low, high = sorted((root, other))
                self.parent[high] = low
                root = low


def bucket(uid, count):
    # Mix high bits too: monotonic transport UIDs often have regular strides.
    uid = (uid ^ (uid >> 30)) * 0xbf58476d1ce4e5b9 & ((1 << 64) - 1)
    uid = (uid ^ (uid >> 27)) * 0x94d049bb133111eb & ((1 << 64) - 1)
    return (uid ^ (uid >> 31)) % count


def inputs(run):
    paths = []
    for name in ['l2_traffic_audit.jsonl', 'l2_accesses.txt']:
        p = run / name
        paths.append(p if p.exists() else p.with_name(p.name + '.gz'))
    names = set(SOURCE_FILES) | {'lrc_to_l2_requests.txt', 'l2_to_icnt_timing.txt'}
    paths.extend(p for p in sorted((run / 'l2_trace').rglob('*'))
                 if p.is_file() and p.name.removesuffix('.gz') in names)
    return paths


def lines(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        yield from handle


class Writers:
    def __init__(self, limit=96):
        self.handles = OrderedDict()
        self.limit = limit

    def put(self, path, line):
        handle = self.handles.pop(path, None)
        if handle is None:
            if len(self.handles) >= self.limit:
                _, oldest = self.handles.popitem(last=False)
                oldest.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = gzip.open(path, 'at', encoding='utf-8', compresslevel=1)
        self.handles[path] = handle
        handle.write(line)

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def add_numbers(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            add_numbers(target.setdefault(key, {}), value)
        elif isinstance(value, int) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value
        elif key not in target:
            target[key] = value
        elif target[key] != value:
            raise ValueError(f'inconsistent partition metadata {key}')


def merge_issues(issues, report, seen_warnings):
    issues.error_count += report['error_count']
    issues.errors.extend(report['error_samples'][:max(0, issues.max_samples - len(issues.errors))])
    # The only warning in the strict traffic validator describes shared warp
    # metadata. Report that once, just as an unpartitioned run would.
    repeated = 0
    for message in report['warning_samples']:
        if message == 'warp_assignments missing: input warp membership was not checked':
            if message in seen_warnings:
                repeated += 1
                continue
            seen_warnings.add(message)
        if len(issues.warnings) < issues.max_samples:
            issues.warnings.append(message)
    issues.warning_count += report['warning_count'] - repeated


def validate(run, max_merged=4, *, partitions=128, work=None, max_partition_bytes=2 * 1024**3,
             subprocess_partitions=True, keep_partitions=False, max_index_uids=64000000):
    run = Path(run).resolve()
    if max_merged < 1 or partitions < 1 or max_partition_bytes < 1 or max_index_uids < 1:
        raise ValueError('merge limit, partition count, byte limit and UID index limit must be positive')
    work = Path(work) if work is not None else Path(tempfile.mkdtemp(prefix='l2-validation-'))
    work.mkdir(parents=True, exist_ok=True)
    if any(work.iterdir()):
        raise ValueError('validation work directory must be empty')
    start = time.monotonic()
    issues = core.TrafficIssues()
    graph = Components()
    inventory = {}
    files = inputs(run)
    if not (run / 'l2_trace').is_dir():
        issues.error(f'missing source log directory: {run / "l2_trace"}')
    progress = {'stage': 'indexing', 'files': 0, 'partitions': partitions, 'rows': 0}
    def checkpoint(**values):
        progress.update(values)
        progress['wall_seconds'] = time.monotonic() - start
        (work / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    checkpoint()
    for path in files:
        digest = hashlib.sha256()
        try:
            for line in lines(path):
                digest.update(line.encode('utf-8'))
                if line.strip():
                    graph.join(identities(line, path.name.removesuffix('.gz')))
                    if len(graph.parent) > max_index_uids:
                        raise ValueError('UID index exceeds --max-index-uids; validation incomplete')
                    progress['rows'] += 1
                    if progress['rows'] % 100000 == 0:
                        checkpoint(indexed_uids=len(graph.parent), current_file=str(path))
            inventory[str(path)] = digest.hexdigest()
        except (OSError, EOFError, UnicodeError) as error:
            issues.error(f'cannot read {path}: {error}')
        checkpoint(files=progress['files'] + 1, indexed_uids=len(graph.parent))
    checkpoint(stage='partitioning', files=0, rows=0)
    sizes = [0] * partitions
    rows = [0] * partitions
    writers = Writers()
    try:
        for path in files:
            digest = hashlib.sha256()
            relative = path.relative_to(run)
            if relative.suffix != '.gz':
                relative = relative.with_name(relative.name + '.gz')
            try:
                for line in lines(path):
                    digest.update(line.encode('utf-8'))
                    if not line.strip():
                        continue
                    ids = identities(line, path.name.removesuffix('.gz'))
                    shard = bucket(graph.find(ids[0]), partitions) if ids else 0
                    sizes[shard] += len(line.encode('utf-8'))
                    rows[shard] += 1
                    if sizes[shard] > max_partition_bytes:
                        raise ValueError('a UID component partition exceeds the byte limit; increase --partitions, or inspect a giant connected component')
                    writers.put(work / f'partition-{shard:04d}' / relative, line)
                    progress['rows'] += 1
                    if progress['rows'] % 100000 == 0:
                        checkpoint(current_file=str(path))
                if digest.hexdigest() != inventory.get(str(path)):
                    issues.error(f'input changed during validation: {path}')
            except (OSError, EOFError, UnicodeError) as error:
                issues.error(f'cannot read {path}: {error}')
            checkpoint(files=progress['files'] + 1)
    finally:
        writers.close()
    uid_count = len(graph.parent)
    del graph
    gc.collect()
    (work / 'input-manifest.json').write_text(json.dumps(inventory, indent=2) + '\n')
    checkpoint(stage='validating', finished_partitions=0, largest_partition_bytes=max(sizes))
    result = {'run': str(run), 'counts': {}, 'source_counts': {}, 'consistency': {},
              'merge_size_distribution': {}, 'mixed_source_groups': 0, 'timing_coverage': {}}
    incoming = forwarded = excluded = 0
    enabled = set()
    warp_checked = True
    seen_warnings = set()
    selected = [i for i, count in enumerate(rows) if count] or [0]
    for shard in selected:
        directory = work / f'partition-{shard:04d}'
        directory.mkdir(exist_ok=True)
        (directory / 'l2_trace').mkdir(exist_ok=True)
        for name in ['l2_traffic_audit.jsonl', 'l2_accesses.txt']:
            p = directory / (name + '.gz')
            if not p.exists():
                with gzip.open(p, 'wt'):
                    pass
        for name in ['simulator.log', 'warp_assignments']:
            source = run / name
            if source.exists():
                (directory / name).symlink_to(source, target_is_directory=source.is_dir())
        report_path = work / f'partition-{shard:04d}.json'
        if subprocess_partitions:
            code = ('import json,sys; from pathlib import Path; from validate_l2_traffic import validate; '
                    'r=validate(Path(sys.argv[1]), int(sys.argv[3]), _defer_global_counters=True); '
                    'Path(sys.argv[2]).write_text(json.dumps(r))')
            with (work / f'partition-{shard:04d}.log').open('w') as log:
                child = subprocess.run([sys.executable, '-c', code, str(directory), str(report_path), str(max_merged)],
                                       cwd=Path(__file__).parent, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode or not report_path.exists():
                raise RuntimeError(f'partition {shard} validator exited {child.returncode}; see {work}')
            report = json.loads(report_path.read_text())
        else:
            report = core.validate(directory, max_merged, _defer_global_counters=True)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
        merge_issues(issues, report['issues'], seen_warnings)
        report['merge_size_distribution'] = {int(k): v for k, v in report['merge_size_distribution'].items()}
        for name in ['counts', 'source_counts', 'consistency', 'merge_size_distribution', 'timing_coverage']:
            add_numbers(result[name], report[name])
        result['mixed_source_groups'] += report['mixed_source_groups']
        counters = report['counter_reconciliation']
        incoming += counters.get('incoming_audit_weighted', 0)
        forwarded += counters.get('forwarded_results_and_excluded', 0)
        excluded += counters.get('excluded_read_services', 0)
        enabled.update(report['coverage']['lrc_enabled_observed'])
        warp_checked &= report['coverage']['warp_assignments_checked']
        # Successful temporary partitions can be discarded; original input logs
        # and all reports/hashes remain. Keep failed partitions for diagnostics.
        if report['passed'] and not keep_partitions:
            shutil.rmtree(directory)
        checkpoint(finished_partitions=progress['finished_partitions'] + 1)
    counters, drain = core.check_stats(run / 'simulator.log', {}, {}, set(), issues,
                                      _counter_totals=(incoming, forwarded, excluded))
    result.update(passed=not issues.error_count, issues=issues.as_dict(), counter_reconciliation=counters,
                  final_drain=drain, coverage={
                      'warp_assignments_checked': warp_checked,
                      'absent_source_categories': sorted(TRACKED_SOURCES - result['source_counts'].keys()),
                      'lrc_enabled_observed': sorted(enabled),
                      'input_instruction_trace_check': 'performed by the smoke runner; this validator checks immutable audit attribution',
                      'cache_set_tag_check': 'schema only; cache indexing geometry is not independently recomputed'},
                  validation_resources={'mode': 'uid-connected-components', 'indexed_uids': uid_count,
                      'partitions': len(selected), 'largest_partition_bytes': max(sizes), 'work': str(work),
                      'parent_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'child_peak_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                      'wall_seconds': time.monotonic() - start})
    checkpoint(stage='complete', passed=result['passed'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--json-out', required=True, type=Path)
    parser.add_argument('--lrc-max-merged', type=int, default=4)
    parser.add_argument('--partitions', type=int, default=128)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--max-partition-bytes', type=int, default=2 * 1024**3)
    parser.add_argument('--max-index-uids', type=int, default=64000000)
    args = parser.parse_args()
    work = args.work_dir or args.json_out.with_name(args.json_out.stem + '-work')
    if work.exists() and any(work.iterdir()):
        parser.error('work directory already contains a validation attempt')
    report = validate(args.run, args.lrc_max_merged, partitions=args.partitions, work=work,
                      max_partition_bytes=args.max_partition_bytes, max_index_uids=args.max_index_uids)
    args.json_out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['issues']['error_count']} errors; {args.json_out}", flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
