#!/usr/bin/env python3
"""Exercise TMA destination masks and barrier delivery in the real simulator.

Generated traces are semantic regressions, not hardware measurements. A remote
only transfer intentionally has no issuing-CTA barrier: an implicit local
completion must fail. Empty and inactive transfers likewise have no barrier.
"""
import argparse
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('cluster', Path(__file__).with_name('run-cluster-barriers.py'))
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


def fixture(name, kernel):
    destinations = {'empty': 0, 'empty-oob': 0, 'inactive': 3,
                    'remote': 2, 'remote-oob': 2, 'self': 1, 'both': 3}[name]
    inactive = name == 'inactive'
    oob = name.endswith('oob')
    entries = []
    for rank in range(2):
        address = (rank << 24) | 0x400
        receives = bool(destinations & (1 << rank)) and not inactive
        records = []
        if receives:
            records.append(cluster.initialize(0x10, address))
            records.append(f'0020 00000001 1 R0 SYNCS.ARRIVE.TRANS64 2 R1 R2 8 1 0x{address:x} 0 0 0 Val 1 0 1 20 1 0')
        records += [cluster.instruction(0x30, 'UCGABAR_ARV'), cluster.instruction(0x40, 'UCGABAR_WAIT')]
        if rank == 0:
            addresses = ' '.join(hex(0x100000 + 4 * i) for i in range(8)) if not oob else ''
            records.append(f'0050 {0 if inactive else 0xffffffff:08x} 0 UTMALDG.1D.MULTICAST 0 4 0x400 1 0x{destinations:x} {0 if oob else 32} {32 if oob else 0} 3 {0 if oob else 8} {addresses} 0 0 NoVal')
        if receives:
            records.append(f'0060 00000001 1 P0 SYNCS.PHASECHK.TRANS64.TRYWAIT 2 R1 R2 8 1 0x{address:x} 0 0 0 Val 1 0 1 0 1 0')
        # Wait for TMA destinations before any CTA exits/unregisters its barrier.
        records += [cluster.instruction(0x70, 'UCGABAR_ARV'), cluster.instruction(0x80, 'UCGABAR_WAIT'), cluster.instruction(0x90, 'EXIT')]
        entries.append(dict(cta=rank, cluster=0, rank=rank, warps=[records]))
    return dict(header=cluster.header(kernel, 2), entries=entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--library-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--disable-lrc', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.binary = args.binary.resolve(); args.library_dir = args.library_dir.resolve()
    args.timeout = 60
    reports = []
    for index, name in enumerate(['empty', 'empty-oob', 'inactive', 'remote', 'remote-oob', 'self', 'both']):
        data = fixture(name, 90000 + index)
        for form in ['text', 'tracez']:
            path = args.output / (f'kernel-{90000 + index}-{name}' + ('.traceg' if form == 'text' else '.tracez'))
            (cluster.write_text_fixture if form == 'text' else cluster.write_tracez_fixture)(path, data)
            run = args.output / (name + '-' + form)
            status, log = cluster.simulator(args, run, args.config.resolve(), path, True,
                                           ['-gpgpu_max_cycle', '10000', *(['-gpgpu_lrc_enabled', '0'] if args.disable_lrc else [])])
            audit_path = run / 'l2_traffic_audit.jsonl'
            rows = [json.loads(line) for line in audit_path.read_text().splitlines()] if audit_path.exists() else []
            incoming = [r for r in rows if r['event'] == 'ingress']
            expected = 1 if name in ['remote', 'self', 'both'] else 0
            drains = [json.loads(line.split(' ', 1)[1]) for line in log.splitlines() if line.startswith('L2_TRAFFIC_FINAL_DRAIN ')]
            masks = [r['multicast_destinations'] for r in incoming]
            timing_rows = [dict(word.split('=', 1) for word in line.split())
                           for timing in (run / 'l2_trace').rglob('l2_to_icnt_timing.txt')
                           for line in timing.read_text().splitlines()]
            timing_uids = [int(row['transport_uid']) for row in timing_rows]
            passed = (status['completed'] and 'gpu_tot_issued_cta = 2' in log and len(drains) == 1 and drains[0]['complete'] and
                      len(incoming) == expected and len(timing_rows) == expected and
                      set(timing_uids) == {row['transport_uid'] for row in incoming} and masks == ([2 if name == 'both' else 1] if expected else []))
            reports.append(dict(name=name, format=form, passed=passed, status=status, ingress=len(incoming), destinations=masks))
    report = dict(passed=all(r['passed'] for r in reports), cases=reports)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
