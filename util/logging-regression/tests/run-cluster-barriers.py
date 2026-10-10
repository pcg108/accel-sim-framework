#!/usr/bin/env python3
"""Isolated Hopper cluster-barrier behavior and captured-workload regression gate.

Generated fixtures are simulator tests, not captured performance measurements.
Every invocation requires a fresh output directory and pins its binary, configs,
runner and validator modules. Existing collection directories are only read.
"""

import argparse
import collections
import ctypes
import ctypes.util
import importlib.util
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

FRAMEWORK = Path(__file__).resolve().parents[3]
HARNESS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("l2_smoke", HARNESS / "run-l2-traffic-smoke.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)
DEBUG_ENV = ("MOBILE_MBAR_DEBUG", "MOBILE_MBAR_DEFER_STAS", "CLUSTER_BARRIER_TRACE")


def instruction(pc, opcode, mask=0xffffffff):
    return f"{pc:04x} {mask:08x} 0 {opcode} 0 0 0 0 NoVal"


def delay(count, mask=0xffffffff):
    # A true register dependency makes this a deterministic delayed participant.
    return [f"{0x8000 + i * 16:04x} {mask:08x} 1 R10 IADD3 3 R10 R255 R255 0 0 0 NoVal"
            for i in range(count)]


def memory(pc, opcode, address, mask=0xffffffff, destination=False):
    registers = "1 R2" if destination else "0"
    return f"{pc:04x} {mask:08x} {registers} {opcode} 1 R1 4 1 0x{address:x} 4 0 0 NoVal"


def initialize(pc, address):
    # The traced packed count represents one expected arrival, initial phase 0.
    return (f"{pc:04x} 00000001 1 UR63 SYNCS.EXCH.64 2 R0 UR10 8 1 0x{address:x} 0 "
            "0 0 Val 1 00000000 1 00000000 1 7ffff800001ffffe")


def header(kernel, ctas, threads=32):
    return f"""-kernel name = cluster_barrier_fixture_{kernel}
-kernel id = {kernel}
-grid dim = ({ctas},1,1)
-block dim = ({threads},1,1)
-shmem = 2048
-nregs = 16
-binary version = 90
-cuda stream id = 0
-shmem base_addr = 0x0000780800000000
-local mem base_addr = 0x00007807e5000000
-nvbit version = 1.8
-accelsim tracer version = 6
-enable lineinfo = 0

#traces format = PC mask dest_num reg_dests opcode src_num reg_srcs memory immediates values
"""


def make_fixture(name, kernel):
    """Return trace topology, instructions and independent expected audit counts."""
    ctas = 4 if name in ("skewed-init-remote-stas", "noncontiguous-clusters") else 2
    threads = 48 if name == "partial-final-warp" else 32
    generations = 2 if name == "repeated-generations-different-pcs" else 1
    entries = []
    expected = collections.Counter()
    for cta in range(ctas):
        rank = cta % 2
        warps = []
        for warp in range((threads + 31) // 32):
            mask = (1 << min(32, threads - 32 * warp)) - 1
            records = []
            if name == "partial-exit-before-arrive":
                records.append(instruction(0x8, "EXIT", 0xffff0000))
                mask = 0x0000ffff
            if name == "exited-participant" and rank == 1:
                records.append(instruction(0x10, "EXIT", mask))
                warps.append(records)
                continue
            if name == "skewed-init-remote-stas":
                if rank == 1:
                    records.extend(delay(150))
                records.append(initialize(0x10, (rank << 24) | 0x400))
            if name in ("sync-load-no-destination", "sync-store-ack"):
                opcode = "LDG.E.BYPASS" if "load" in name else "STG.E.BYPASS"
                records.append(memory(0x20, opcode, 0x100000 + cta * 4096, mask))
            if name in ("sync-l1-miss-no-destination", "sync-l1-hit-no-destination", "sync-shared-load-no-destination"):
                address = 0x400 if "shared" in name else 0x100000 + cta * 4096
                if "hit" in name:
                    records.append(memory(0x10, "LDG.E", address, mask, destination=True))
                    records.append(f"0018 {mask:08x} 1 R3 IADD3 3 R2 R255 R255 0 0 0 NoVal")
                records.append(memory(0x20, "LDS" if "shared" in name else "LDG.E", address, mask))
            if name == "async-ldgsts-not-ordered":
                records.append(memory(0x20, "LDGSTS.E.BYPASS", 0x100000 + cta * 4096, mask))
            if name.startswith("async-tma"):
                addresses = " ".join(f"0x{0x100000 + cta * 4096 + 4 * lane:x}" for lane in range(8))
                if "load" in name:
                    records.append(initialize(0x10, (rank << 24) | 0x400))
                    records.extend(delay(100))
                    records.append(f"0020 {mask:08x} 0 UTMALDG.1D 0 4 0x{((rank << 24) | 0x400):x} 0 32 0 3 8 {addresses} 0 0 NoVal")
                else:
                    records.append(f"0020 {mask:08x} 0 UTMAREDG.1D.ADD.F32 0 4 0x0 0 32 0 3 8 {addresses} 0 0 NoVal")
            if name == "inactive-barrier":
                records.extend([instruction(0x30, "UCGABAR_ARV", 0),
                                instruction(0x40, "UCGABAR_WAIT", 0)])
            if name == "arrive-nonblocking" and rank == 1:
                records.extend(delay(300))
            for generation in range(generations):
                pc = 0x100 + generation * 0x100 + cta * 0x20 + warp * 0x10
                if name != "wait-without-arrival":
                    opname = "UCGABAR_ARV.UNSUPPORTED" if name == "unsupported-modifier" else "UCGABAR_ARV"
                    active = mask >> 1 if name == "divergent-participation" else mask
                    records.append(instruction(pc, opname, active))
                    expected["arrive"] += 1
                if name == "duplicate-arrival":
                    records.append(instruction(pc + 8, "UCGABAR_ARV", mask))
                    expected["arrive"] += 1
                if name == "arrive-nonblocking" and rank == 0:
                    # This initialization after ARRV must execute while the
                    # peer still delays its arrival; ARRV cannot block a warp.
                    records.append(initialize(0x50, 0x400))
                if name != "exit-after-arrive-before-wait" or rank == 0:
                    records.append(instruction(pc + 16, "UCGABAR_WAIT", mask))
                    expected["wait"] += 1
            if name == "skewed-init-remote-stas" and rank == 0:
                records.append(memory(0x400, "STAS", (0x01000400 << 32) | 0x01000500, 1))
                expected["stas_complete"] += 1
            if name == "skewed-init-remote-stas" and rank == 1:
                records.extend(delay(250))
            if name == "async-tma-store-not-ordered":
                records.extend([instruction(0x500, "UTMACMDFLUSH", mask),
                                instruction(0x510, "DEPBAR.LE", mask)])
            if name != "wait-at-eof":
                records.append(instruction(0x7000, "EXIT", mask))
            if name == "empty-warp" and cta == 1:
                records.clear()
            warps.append(records)
        entries.append({"cta": cta, "cluster": cta // 2, "rank": rank, "warps": warps})
    if name == "noncontiguous-clusters":
        entries = [entries[i] for i in (0, 2, 1, 3)]
    if name == "duplicate-cta":
        entries[1]["cta"] = entries[0]["cta"]
    if name == "nondense-ranks":
        entries[1]["rank"] = 3
    negative = {"duplicate-arrival", "wait-without-arrival", "unsupported-modifier",
                "divergent-participation", "noncontiguous-clusters", "duplicate-cta",
                "nondense-ranks", "insufficient-capacity", "concurrent-clusters", "empty-warp"}
    return {"name": name, "header": header(kernel, ctas, threads), "entries": entries,
            "expected": dict(expected), "negative": name in negative,
            "threads": threads, "ctas": ctas, "generations": generations}


def write_text_fixture(path, fixture):
    lines = [fixture["header"]]
    for item in fixture["entries"]:
        lines.extend(["#BEGIN_TB", f"thread block = {item['cta']},0,0",
                      f"cluster id = {item['cluster']},0,0", f"cluster cta = {item['rank']},0,0",
                      f"cluster rank = {item['rank']}"])
        for warp, records in enumerate(item["warps"]):
            lines.extend([f"warp = {warp}", f"insts = {len(records)}", *records])
        lines.append("#END_TB")
    path.write_text("\n".join(lines) + "\n")


def write_tracez_fixture(path, fixture):
    """Use the actual tracez format with no dictionary and one subchunk/warp."""
    zstd = ctypes.CDLL(ctypes.util.find_library("zstd"))
    zstd.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
    zstd.ZSTD_compressBound.restype = ctypes.c_size_t
    zstd.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                   ctypes.c_size_t, ctypes.c_int]
    zstd.ZSTD_compress.restype = ctypes.c_size_t
    zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
    zstd.ZSTD_isError.restype = ctypes.c_uint
    indexes = []
    with path.open("wb") as output:
        output.write(fixture["header"].encode() + b"\0" + struct.pack("<II", 0, 512))
        for item in fixture["entries"]:
            warp_indexes = []
            for records in item["warps"]:
                raw = ("\n".join(records) + "\n").encode()
                buffer = ctypes.create_string_buffer(zstd.ZSTD_compressBound(len(raw)))
                size = zstd.ZSTD_compress(buffer, len(buffer), raw, len(raw), 3)
                if zstd.ZSTD_isError(size):
                    raise RuntimeError("Failed to compress synthetic tracez fixture")
                warp_indexes.append((len(records), 1, output.tell(), len(records)))
                output.write(struct.pack("<III", len(records), 1, size) + buffer.raw[:size])
            indexes.append((item, warp_indexes))
        index_offset = output.tell()
        output.write(struct.pack("<I", len(indexes)))
        for item, warps in indexes:
            output.write(struct.pack("<11I", item["cta"], 0, 0, item["cluster"], 0, 0,
                                     item["rank"], 0, 0, item["rank"], len(warps)))
            for record in warps:
                output.write(struct.pack("<IIQI", *record))
        output.write(struct.pack("<Q", index_offset))


def events(log, prefix):
    return [json.loads(line[len(prefix) + 1:]) for line in log.splitlines()
            if line.startswith(prefix + " ")]


def synchronization_audit(log, expected=None):
    barriers = events(log, "CLUSTER_BARRIER")
    memory_events = events(log, "CLUSTER_MEMORY")
    counts = collections.Counter(event["event"] for event in barriers)
    if expected:
        for kind in ("arrive", "wait"):
            if counts[kind] != expected.get(kind, 0):
                raise ValueError(f"Expected {expected.get(kind, 0)} {kind} events, found {counts[kind]}")
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for event in barriers:
        key = (event.get("kernel_uid"), tuple(event["cluster"]), event["generation"])
        grouped[key][event["event"]].append(event)
    for key, group in grouped.items():
        arrivals = group.get("arrive", [])
        releases = group.get("release", [])
        if not arrivals:
            continue
        participants = {(tuple(event["cta"]), event["warp"]) for event in arrivals}
        if len(participants) != len(arrivals):
            raise ValueError(f"Duplicate arrival audit events for {key}")
        waits = group.get("wait", [])
        waited = {(tuple(event["cta"]), event["warp"]): event for event in waits}
        released = {(tuple(event["cta"]), event["warp"]): event for event in releases}
        exited = collections.defaultdict(int)
        for event in group.get("exit", []):
            exited[(tuple(event["cta"]), event["warp"])] |= event.get("mask", 0)
        arrived_masks = {(tuple(event["cta"]), event["warp"]): event.get("mask", 0xffffffff)
                         for event in arrivals}
        omitted = participants - set(waited)
        if (set(waited) - participants or set(released) != set(waited)
                or any((exited[who] & arrived_masks[who]) != arrived_masks[who] for who in omitted)):
            raise ValueError(f"Arrival/wait/release participant mismatch for {key}")
        if len(waits) != len(waited) or len(releases) != len(released):
            raise ValueError(f"Duplicate wait/release audit events for {key}")
        if any(released[who]["cycle"] < waited[who]["cycle"] for who in waited):
            raise ValueError(f"Release preceded matching wait for {key}")
        if releases and min(event["cycle"] for event in releases) < max(event["cycle"] for event in arrivals):
            raise ValueError(f"Cluster {key} released before all recorded arrivals")
    initialized = {}
    stas = []
    for event in sorted(memory_events, key=lambda value: value["cycle"]):
        if event["event"] not in ("init", "stas_complete"):
            continue
        key = (event.get("kernel_uid"), tuple(event["cluster"]), event["bar_addr"])
        if event["event"] == "init":
            initialized[key] = event["cycle"]
        else:
            if key not in initialized or initialized[key] > event["cycle"]:
                raise ValueError(f"Remote STAS completed before destination initialization: {event}")
            stas.append(event["cycle"] - initialized[key])
    if expected and len(stas) != expected.get("stas_complete", 0):
        raise ValueError(f"Expected {expected.get('stas_complete', 0)} STAS completions, found {len(stas)}")
    return {"passed": True, "barrier_events": dict(counts), "cluster_generations": sum(bool(group.get("arrive")) for group in grouped.values()),
            "mbarrier_initializations": len(initialized), "remote_stas_completions": len(stas),
            "minimum_init_to_stas_cycles": min(stas) if stas else None}


def prepare_config(args, model):
    config = args.output / "configs" / model
    config.mkdir(parents=True)
    gpu = args.framework / "gpu-simulator"
    gpgpu = gpu / "gpgpu-sim/configs"
    if model == "mobile-ampere":
        main = gpgpu / "model-cfgs/gpgpusim_8sm.config"
        timing = args.resnet_baseline.parent / "config/trace.config"
    elif model == "mobile-hopper":
        main = gpgpu / "model-cfgs/gpgpusim_8sm_hopper.config"
        timing = gpu / "configs/tested-cfgs/SM90_H100/trace.config"
    else:
        main = gpgpu / "tested-cfgs" / model / "gpgpusim.config"
        timing = gpu / "configs/tested-cfgs" / model / "trace.config"
        shutil.copy2(main.parent / "config_hopper_islip.icnt", config)
    shutil.copy2(main, config / "gpgpusim.config")
    shutil.copy2(timing, config / "trace.config")
    return config


def simulator(args, run, config, trace, enabled, extra=()):
    run.mkdir(parents=True)
    (run / trace.name).symlink_to(trace.resolve())
    (run / "kernelslist.g").write_text(trace.name + "\n")
    for auxiliary in config.glob("*.icnt"):
        shutil.copy2(auxiliary, run)
    environment = os.environ.copy()
    for name in (*smoke.LOG_ENV, *DEBUG_ENV):
        environment.pop(name, None)
    settings = {"LD_LIBRARY_PATH": str(args.library_dir) + ":/usr/local/cuda/lib64",
                "CLUSTER_BARRIER_TRACE": "1"}
    if enabled:
        settings.update(LOG_CACHE_TRACES="1", L2_TRACE_DIR=str(run / "l2_trace"),
                        L2_ACCESS_TRACE_FILE=str(run / "l2_accesses.txt"),
                        L2_TRAFFIC_AUDIT_FILE=str(run / "l2_traffic_audit.jsonl"))
    environment.update(settings)
    argv = [str(args.binary), "-config", str(config / "gpgpusim.config"), "-config",
            str(config / "trace.config"), "-trace", str(run / "kernelslist.g"),
            "-gpgpu_concurrent_kernel_sm", "0", "-gpgpu_max_concurrent_kernel", "1",
            "-gpgpu_perf_sim_memcpy", "0", "-trace_enabled", "0", *extra]
    smoke.write_json(run / "command.json", {"argv": argv, "cwd": str(run), "environment": settings,
                     "unset_inherited_environment": [*smoke.LOG_ENV, *DEBUG_ENV],
                     "trace": str(trace.resolve()), "trace_sha256": smoke.sha256(trace)})
    print(f"Running {run.name} logging={'on' if enabled else 'off'}", flush=True)
    start = time.monotonic()
    timed_out = False
    with (run / "simulator.log").open("w") as output:
        try:
            result = subprocess.run(argv, cwd=run, env=environment, stdout=output,
                                    stderr=subprocess.STDOUT, timeout=args.timeout)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
    log = (run / "simulator.log").read_text(errors="replace")
    completed = (returncode == 0 and "GPGPU-Sim: *** exit detected ***" in log
                 and "deadlock detected" not in log and "break due to reaching" not in log)
    status = {"completed": completed, "returncode": returncode, "timed_out": timed_out,
              "wall_seconds": time.monotonic() - start, "logging_enabled": enabled}
    smoke.write_json(run / "status.json", status)
    return status, log


def validate_traffic(args, run):
    validator = args.output / "tools/validator/validate_l2_traffic.py"
    with (run / "validator.log").open("w") as output:
        result = subprocess.run([sys.executable, str(validator), "--run", str(run),
                                 "--json-out", str(run / "traffic-validation.json")],
                                stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout)
    report = json.loads((run / "traffic-validation.json").read_text())
    if result.returncode or not report.get("passed"):
        raise ValueError(f"Strict completion validation failed: {run}")
    return report


def synthetic_cases(args):
    names = ["skewed-init-remote-stas", "arrive-nonblocking", "repeated-generations-different-pcs",
             "wait-at-eof", "partial-final-warp", "inactive-barrier", "exited-participant",
             "partial-exit-before-arrive", "exit-after-arrive-before-wait",
             "sync-load-no-destination", "sync-store-ack", "sync-l1-miss-no-destination",
             "sync-l1-hit-no-destination", "sync-shared-load-no-destination", "async-ldgsts-not-ordered",
             "async-tma-load-not-ordered", "async-tma-store-not-ordered", "empty-warp",
             "duplicate-arrival", "wait-without-arrival", "unsupported-modifier",
             "divergent-participation", "noncontiguous-clusters", "duplicate-cta", "nondense-ranks",
             "insufficient-capacity", "concurrent-clusters"]
    selected = set(args.cases.split(",")) if args.cases else set(names)
    unknown = selected - set(names)
    if unknown:
        raise ValueError(f"Unknown synthetic cases: {sorted(unknown)}")
    config = prepare_config(args, "mobile-hopper")
    reports = {}
    for index, name in enumerate(names):
        if name not in selected:
            continue
        fixture = make_fixture(name, 980000 + index)
        parent = args.output / "synthetic" / name
        parent.mkdir(parents=True)
        smoke.write_json(parent / "fixture.json", fixture)
        text_trace = parent / f"kernel-{980000 + index}.traceg"
        compressed_trace = text_trace.with_suffix(".tracez")
        write_text_fixture(text_trace, fixture)
        write_tracez_fixture(compressed_trace, fixture)
        format_reports = {}
        for trace_format, trace in (("text", text_trace), ("tracez", compressed_trace)):
            run = parent / trace_format
            overrides = ["-gpgpu_n_clusters", "1" if name == "insufficient-capacity" else "2",
                         "-gpgpu_shader_cta", "1"]
            if name == "concurrent-clusters":
                overrides.extend(["-gpgpu_concurrent_kernel_sm", "1", "-gpgpu_max_concurrent_kernel", "2"])
            status, log = simulator(args, run, config, trace, name.startswith("async-"), overrides)
            try:
                if fixture["negative"]:
                    if status["completed"] or status["timed_out"] or not re.search(r"(?:cluster barrier:|Cluster trace preflight|Cluster barrier|cluster barrier kernel=|Unsupported active cluster barrier)", log):
                        raise ValueError("Malformed/unsupported cluster replay was not diagnosed promptly")
                    format_reports[trace_format] = {"passed": True, "expected_rejection": True, "status": status}
                    continue
                if not status["completed"]:
                    raise ValueError(f"Synthetic replay failed: {run / 'simulator.log'}")
                statistics = smoke.architectural_statistics(log)
                smoke.write_json(run / "architectural-statistics.json", statistics)
                audit = synchronization_audit(log, fixture["expected"])
                smoke.write_json(run / "synchronization-validation.json", audit)
                barrier_events = events(log, "CLUSTER_BARRIER")
                mem_events = events(log, "CLUSTER_MEMORY")
                if name == "arrive-nonblocking":
                    init = next(event for event in mem_events if event["event"] == "init")
                    peer_arrival = next(event for event in barrier_events
                                        if event["event"] == "arrive" and event["cta"][0] == 1)
                    if init["cycle"] >= peer_arrival["cycle"]:
                        raise ValueError("ARRV blocked progress before peer arrival")
                if name.startswith("sync-"):
                    event_name = "store_ack" if name == "sync-store-ack" else "load_complete"
                    completed = [event for event in mem_events if event["event"] == event_name]
                    if len(completed) < fixture["ctas"]:
                        raise ValueError(f"Missing {event_name} audit for synchronous memory fixture")
                    for arrival in (event for event in barrier_events if event["event"] == "arrive"):
                        preceding = [event for event in completed if event["cta"] == arrival["cta"]]
                        if not preceding or max(event["cycle"] for event in preceding) > arrival["cycle"]:
                            raise ValueError(f"Arrival did not wait for synchronous {event_name}")
                if name.startswith("async-"):
                    end_cycles = []
                    for timing in (run / "l2_trace").rglob("l2_to_icnt_timing.txt"):
                        for line in timing.read_text().splitlines():
                            fields = dict(field.split("=", 1) for field in line.split())
                            end_cycles.append(int(fields["end_cycle"]))
                    last_arrival = max(event["cycle"] for event in barrier_events if event["event"] == "arrive")
                    if not end_cycles or last_arrival >= min(end_cycles):
                        raise ValueError(f"Fixture did not demonstrate ARRV before asynchronous completion: {name}")
                    validate_traffic(args, run)
                format_reports[trace_format] = {"passed": True, "status": status, "audit": audit,
                                                 "architectural_statistics": statistics}
            except Exception as error:
                format_reports[trace_format] = {"passed": False, "failure": str(error), "status": status}
        if not fixture["negative"] and all(value["passed"] for value in format_reports.values()):
            equal = format_reports["text"]["architectural_statistics"] == format_reports["tracez"]["architectural_statistics"]
            if not equal:
                format_reports["format_equality"] = {"passed": False, "failure": "Text/tracez statistics differ"}
        reports[name] = {"passed": all(value["passed"] for value in format_reports.values()), "formats": format_reports}
        smoke.write_json(parent / "summary.json", reports[name])
        smoke.write_json(args.output / "synthetic-summary.json", reports)
    return reports


def captured_cases(args):
    cases = [("mobile-hopper", 8039), ("SM90_H100", 8039), ("SM90_H200", 8039),
             ("mobile-hopper", 6542), ("mobile-hopper", 9870), ("mobile-hopper", 13076),
             ("mobile-ampere", 26)]
    selected = set(args.cases.split(",")) if args.cases else {f"{model}:{kernel}" for model, kernel in cases}
    unknown = selected - {f"{model}:{kernel}" for model, kernel in cases}
    if unknown:
        raise ValueError(f"Unknown captured cases: {sorted(unknown)}")
    configs = {}
    reports = {}
    for model, kernel in cases:
        case = f"{model}:{kernel}"
        if case not in selected:
            continue
        if model not in configs:
            configs[model] = prepare_config(args, model)
        if kernel == 26:
            trace = (args.resnet_baseline / "kernel-26.traceg").resolve(strict=True)
        else:
            candidates = list(args.hopper_traces.glob(f"kernel-{kernel}-*.traceg.xz"))
            if len(candidates) != 1:
                raise ValueError(f"Cannot select unique trace for kernel {kernel}: {candidates}")
            trace = candidates[0]
        parent = args.output / "captured" / f"{model}-kernel-{kernel}"
        parent.mkdir(parents=True)
        report = {"passed": False, "model": model, "kernel": kernel}
        try:
            on, log = simulator(args, parent / "logging-on", configs[model], trace, True)
            report["logging_on"] = on
            if not on["completed"]:
                if "deadlock detected" in log:
                    report["classification"] = "simulation_deadlock"
                elif "Accessing non-existent mbarrier" in log:
                    report["classification"] = "missing_mbarrier"
                elif on["timed_out"]:
                    report["classification"] = "simulation_timeout"
                else:
                    report["classification"] = "simulation_failure"
                raise ValueError(f"Captured replay failed: {parent / 'logging-on/simulator.log'}")
            stats_on = smoke.architectural_statistics(log)
            smoke.write_json(parent / "logging-on/architectural-statistics.json", stats_on)
            traffic = validate_traffic(args, parent / "logging-on")
            report["traffic_validation"] = {key: traffic[key] for key in ("passed", "counts", "source_counts")}
            report["final_drain"] = json.loads(next(line.split(" ", 1)[1] for line in stats_on
                                                   if line.startswith("L2_TRAFFIC_FINAL_DRAIN ")))
            if kernel == 8039:
                report["synchronization"] = synchronization_audit(log, {"arrive": 576, "wait": 576, "stas_complete": 36})
                inventory = smoke.trace_inventory(trace)
                report["instructions"] = smoke.validate_source_instructions(parent / "logging-on", trace, inventory)
                if report["instructions"]["cta_assignments"] != 48:
                    raise ValueError("Kernel 8039 did not launch all 48 CTAs")
                insns = int(next(line.split("=", 1)[1] for line in stats_on if line.startswith("gpu_tot_sim_insn ")))
                if insns != 19840968:
                    raise ValueError(f"Kernel 8039 instruction mismatch: {insns}")
            elif kernel == 26:
                counts = traffic["source_counts"]
                if counts.get("l1_bypass_load") != 131584 or counts.get("l1_write") != 100352:
                    raise ValueError(f"ResNet 26 source count mismatch: {counts}")
                baseline = smoke.architectural_statistics((args.resnet_baseline / "simulator.log").read_text(errors="replace"))
                report["baseline_statistics_equal"] = baseline == stats_on
                if baseline != stats_on:
                    raise ValueError("ResNet 26 architectural statistics differ from completion-v2 baseline")
            else:
                report["synchronization"] = synchronization_audit(log)
            off, off_log = simulator(args, parent / "logging-off", configs[model], trace, False)
            report["logging_off"] = off
            if not off["completed"]:
                raise ValueError("Logging-disabled captured replay failed")
            stats_off = smoke.architectural_statistics(off_log)
            smoke.write_json(parent / "logging-off/architectural-statistics.json", stats_off)
            report["logging_on_off_equal"] = stats_on == stats_off
            if stats_on != stats_off:
                report["statistics_difference"] = {"on_only": list((collections.Counter(stats_on) - collections.Counter(stats_off)).elements()),
                                                   "off_only": list((collections.Counter(stats_off) - collections.Counter(stats_on)).elements())}
                raise ValueError("Logging on/off architectural statistics differ")
            report["passed"] = True
        except Exception as error:
            report["failure"] = str(error)
        smoke.write_json(parent / "summary.json", report)
        reports[case] = report
        smoke.write_json(args.output / "captured-summary.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--library-dir", required=True, type=Path)
    parser.add_argument("--framework", type=Path, default=FRAMEWORK)
    parser.add_argument("--mode", choices=("synthetic", "captured"), default="synthetic")
    parser.add_argument("--cases", help="Comma-separated fixture names, or model:kernel selections")
    parser.add_argument("--hopper-traces", type=Path, default=Path("/accel-sim/accel-sim-framework/data/gemma4-h100/traces"))
    parser.add_argument("--resnet-baseline", type=Path, default=Path("/results/mobile-traffic-20261007/completion-timing-v2/ampere-resnet/kernel-26"))
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    for name in ("binary", "library_dir", "framework"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "INCOMPLETE").write_text("Cluster barrier verification in progress\n")
    tools = args.output / "tools"
    tools.mkdir()
    shutil.copy2(args.binary, tools / "accel-sim.out")
    (tools / "tests").mkdir()
    shutil.copy2(Path(__file__), tools / "tests")
    shutil.copy2(HARNESS / "run-l2-traffic-smoke.py", tools)
    (tools / "lib").mkdir()
    shutil.copy2(args.library_dir / "libcudart.so", tools / "lib/libcudart.so")
    shutil.copytree(HARNESS / "validator", tools / "validator", ignore=shutil.ignore_patterns("__pycache__", "tests"))
    provenance = {"invocation": [sys.executable, *sys.argv], "cwd": os.getcwd(),
                  "binary_input": str(args.binary), "binary_sha256": smoke.sha256(args.binary),
                  "runner_sha256": smoke.sha256(Path(__file__)), "mode": args.mode,
                  "library": str(args.library_dir), "library_sha256": smoke.sha256(args.library_dir / "libcudart.so")}
    for label, repo in (("framework", args.framework), ("gpgpusim", args.framework / "gpu-simulator/gpgpu-sim")):
        provenance[label + "_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        provenance[label + "_status"] = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
        (args.output / f"{label}.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo, text=True))
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True).splitlines()
        provenance[label + "_untracked_sources"] = {}
        for name in untracked:
            source = repo / name
            if source.is_file() and source.suffix in (".h", ".cc", ".cpp", ".py", ".md", ".config"):
                destination = args.output / "source-untracked" / label / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                provenance[label + "_untracked_sources"][name] = smoke.sha256(source)
    args.binary = tools / "accel-sim.out"
    args.library_dir = tools / "lib"
    smoke.write_json(args.output / "provenance.json", provenance)
    summary = {"passed": False, "mode": args.mode}
    try:
        reports = synthetic_cases(args) if args.mode == "synthetic" else captured_cases(args)
        summary["cases"] = reports
        summary["passed"] = bool(reports) and all(value["passed"] for value in reports.values())
    except Exception as error:
        summary["failure"] = str(error)
    smoke.write_json(args.output / "summary.json", summary)
    config_hashes = {str(path.relative_to(args.output)): smoke.sha256(path)
                     for path in (args.output / "configs").rglob("*") if path.is_file()}
    smoke.write_json(args.output / "config-sha256.json", config_hashes)
    (args.output / "INCOMPLETE").rename(args.output / ("SUCCESS" if summary["passed"] else "FAILED"))
    print(json.dumps({"passed": summary["passed"], "output": str(args.output),
                      "cases": {key: value["passed"] for key, value in summary.get("cases", {}).items()},
                      "failure": summary.get("failure")}, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
