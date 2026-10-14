#!/usr/bin/env python3
"""Compile and exercise real sector splitting using an existing CMake build.

This focused harness does not run a kernel or alter simulator source/config.
Byte-mask findings are reported separately from logging provenance, so existing
splitter defects cannot be mistaken for missing contributor metadata.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


FRAMEWORK = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("l2-traffic-runtime.cc")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validator"))
from l2_traffic_schema import parse_lrc_line


def build(build_dir, output):
    commands = json.loads((build_dir / "compile_commands.json").read_text())
    main = next(row for row in commands if row["file"].endswith("/main.cc"))
    compile_argv = shlex.split(main["command"])
    compile_argv[compile_argv.index("-o") + 1] = str(output / "runtime.o")
    compile_argv[compile_argv.index("-c") + 1] = str(SOURCE)
    subprocess.run(compile_argv, cwd=main["directory"], check=True)
    link_file = build_dir / "CMakeFiles/accel-sim.out.dir/link.txt"
    link = shlex.split(link_file.read_text())
    link = [str(output / "runtime.o") if arg.endswith("/main.cc.o") else arg
            for arg in link if not arg.endswith("/accel-sim.cc.o")]
    # The harness reaches initialization through a different object than the
    # normal frontend, so resolve cyclic simulator archives as a link group.
    first_archive = next(i for i, arg in enumerate(link) if arg.endswith(".a"))
    link.insert(first_archive, "-Wl,--start-group")
    link.append("-Wl,--end-group")
    executable = output / "l2-traffic-runtime"
    link[link.index("-o") + 1] = str(executable)
    subprocess.run(link, cwd=build_dir, check=True)
    (output / "build-commands.json").write_text(json.dumps({
        "compile": compile_argv, "compile_cwd": main["directory"],
        "link": link, "link_cwd": str(build_dir)}, indent=2) + "\n")
    return executable


def verify(output, completed):
    rows = [json.loads(line) for line in (output / "audit.jsonl").read_text().splitlines()]
    sources = {r["source_uid"]: r for r in rows if r["event"] == "departure"}
    ingress = [r for r in rows if r["event"] == "ingress"]
    groups = []
    for path in (output / "l2_trace").rglob("lrc_to_l2_requests.txt"):
        groups.extend(parse_lrc_line(line) for line in path.read_text().splitlines())
    errors, byte_errors = [], []
    timings = []
    for path in (output / "l2_trace").rglob("l2_to_icnt_timing.txt"):
        for line in path.read_text().splitlines():
            fields = dict(word.split("=", 1) for word in line.split())
            if set(fields) != {"request_uid", "transport_uid", "service_uid", "addr",
                               "sector_mask", "size", "start_cycle", "end_cycle", "elapsed_cycle"}:
                errors.append("timing wire fields changed")
            timings.append({key: int(value, 0) for key, value in fields.items()})
    by_transport = {row["transport_uid"]: row for row in timings}
    tracked_ingress = {row["transport_uid"]: row for row in ingress
                       if row["source"] != "other_read"}
    if len(by_transport) != len(timings):
        errors.append("duplicate completion after reply retry/multicast/disposal")
    if set(by_transport) != set(tracked_ingress):
        errors.append("completion identities differ from tracked ingress")
    services = {member["transport_uid"]: group["service_uid"]
                for group in groups for member in group["contributors"]}
    group_ends = defaultdict(set)
    for uid, timing in by_transport.items():
        child = tracked_ingress.get(uid)
        if child is None:
            continue
        expected_service = services.get(uid, uid)
        parent = sources[child["source_uid"]]
        expected = {"request_uid": child["source_uid"], "service_uid": expected_service,
                    "start_cycle": parent["source_cycle"], "addr": int(child["addr"], 16),
                    "sector_mask": int(child["sector_mask"], 16), "size": child["size"]}
        for key, value in expected.items():
            if timing[key] != value:
                errors.append(f"transport {uid}: wrong completion {key}")
        if timing["end_cycle"] < child["cycle"] or timing["elapsed_cycle"] != timing["end_cycle"] - timing["start_cycle"]:
            errors.append(f"transport {uid}: invalid timing interval")
        group_ends[expected_service].add(timing["end_cycle"])
    if any(len(ends) != 1 for ends in group_ends.values()):
        errors.append("LRC contributors did not share service completion")
    if not any(row["start_cycle"] > 2 ** 32 for row in timings):
        errors.append("64-bit timing case was not exercised")
    if not any(row["multicast_destinations"] == 4 for row in tracked_ingress.values()):
        errors.append("multicast timing case was not exercised")
    children = defaultdict(list)
    for row in ingress:
        children[row["source_uid"]].append(row)
    all_members = [member for group in groups for member in group["contributors"]]
    member_uids = [m["transport_uid"] for m in all_members]
    if len(set(member_uids)) != len(member_uids):
        errors.append("duplicate finalized contributor (possible retry emission)")
    if set(member_uids) != {r["transport_uid"] for r in ingress if not r["is_write"]}:
        errors.append("missing or extra finalized contributor identities")
    # Legacy special64B children share an address, and the actual address-based
    # LRC merges each pair despite their distinct sector masks. Preserve that
    # behavior: this fixture checks bytes/provenance, not cache semantics.
    if len(groups) != 16 or sorted(len(g["contributors"]) for g in groups) != [1] * 12 + [2, 2, 3, 4]:
        errors.append("wrong singleton/split/mixed/untracked-representative group inventory")
    fields = ("source", "source_cycle", "kernel_uid", "sm", "scheduler",
              "dynamic_warp", "inst_ordinal", "PC")
    for member in all_members:
        if member["source"] == "other_read":
            continue
        parent = sources.get(member["request_uid"])
        if parent is None:
            errors.append(f"orphan contributor {member['transport_uid']}")
            continue
        for field in fields:
            expected = int(parent[field], 16) if field == "PC" else parent[field]
            if member[field] != expected:
                errors.append(f"contributor {member['transport_uid']}: lost {field}")
    for uid, parent in sources.items():
        sectors = children[uid]
        expected = 1 if parent["data_size"] == 32 else parent["data_size"] // 32
        if len(sectors) != expected:
            errors.append(f"source {uid}: expected {expected} children, found {len(sectors)}")
        for child in sectors:
            sector = int(child["sector_mask"], 16).bit_length() - 1
            sector_bytes = ((1 << 32) - 1) << (32 * sector)
            expected_bytes = int(parent["byte_mask"], 16) & sector_bytes
            if int(child["byte_mask"], 16) != expected_bytes:
                byte_errors.append({"source_uid": uid, "transport_uid": child["transport_uid"],
                                    "expected": hex(expected_bytes), "actual": child["byte_mask"],
                                    "data_size": parent["data_size"]})
    report = {"provenance_passed": completed and not errors, "provenance_errors": errors,
              "sector_bytes_passed": not byte_errors, "sector_byte_findings": byte_errors,
              "counts": {"sources": len(sources), "ingress": len(ingress),
                         "lrc_results": len(groups), "contributors": len(all_members),
                         "timings": len(timings), "tracked_ingress": len(tracked_ingress)},
              "note": "Production splitter, completion hooks and reply retry paths with controlled service times. Full cache/DRAM completion is covered by synthetic runs."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "audit.jsonl").exists():
        parser.error("output already contains an audit; use a fresh output directory")
    executable = build(args.build_dir, args.output)
    if args.compile_only:
        print(f"Compiled: {executable}")
        return 0
    config = FRAMEWORK / "gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100"
    shutil.copy2(config / "config_hopper_islip.icnt", args.output)
    env = dict(os.environ)
    env.update(L2_TRACE_DIR=str(args.output / "l2_trace"),
               L2_TRAFFIC_AUDIT_FILE=str(args.output / "audit.jsonl"))
    argv = [str(executable), "-config", str(config / "gpgpusim.config")]
    (args.output / "run-command.json").write_text(json.dumps({"argv": argv,
        "cwd": str(args.output), "environment": {k: env[k] for k in
            ("L2_TRACE_DIR", "L2_TRAFFIC_AUDIT_FILE")}}, indent=2) + "\n")
    with (args.output / "runtime.log").open("w") as log:
        process = subprocess.run(argv, cwd=args.output, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=120)
    if process.returncode:
        print(f"Runtime harness failed ({process.returncode}); {args.output / 'runtime.log'}")
        return 1
    report = verify(args.output, "RUNTIME_L2_PROVENANCE_PASS" in (args.output / "runtime.log").read_text())
    print(json.dumps(report, indent=2))
    return 0 if report["provenance_passed"] and report["sector_bytes_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
