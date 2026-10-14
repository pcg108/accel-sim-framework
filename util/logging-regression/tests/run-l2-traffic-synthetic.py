#!/usr/bin/env python3
"""Exercise missing L2 source classes after the complete H100 kernel gate passes.

These deliberately small, generated SASS traces test simulator logging paths;
they are not captured workloads or hardware performance measurements.
"""

import argparse
import collections
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

HARNESS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("l2_smoke", HARNESS / "run-l2-traffic-smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def ordinary(pc, opcode, address, destination=True):
    registers = "1 R2" if destination else "0"
    return f"{pc:04x} ffffffff {registers} {opcode} 1 R1 4 1 0x{address:x} 4 0 0 NoVal"


def reduction(pc, opcode, address):
    addresses = " ".join(f"0x{address + 4 * index:x}" for index in range(8))
    return f"{pc:04x} ffffffff 0 {opcode} 0 4 0x0 0 32 0 3 8 {addresses} 0 0 NoVal"


def trace_text(kernel, stream=0, writeback=False, write_allocate=False):
    header = f"""-kernel name = synthetic_l2_sources_{kernel}
-kernel id = {kernel}
-grid dim = (1,1,1)
-block dim = (128,1,1)
-shmem = 0
-nregs = 16
-binary version = 90
-cuda stream id = {stream}
-shmem base_addr = 0x0000780800000000
-local mem base_addr = 0x00007807e5000000
-nvbit version = 1.8
-accelsim tracer version = 6
-enable lineinfo = 0

#traces format = PC mask dest_num reg_dests opcode src_num reg_srcs memory immediates values

#BEGIN_TB
thread block = 0,0,0
cluster id = 0,0,0
cluster cta = 0,0,0
cluster rank = 0
"""
    warps = [
        [ordinary(0x10, "LDG.E", 0x100000), ordinary(0x20, "STG.E", 0x300000, False)],
        [ordinary(0x10, "LDG.E.BYPASS", 0x100000), ordinary(0x20, "STG.E.BYPASS", 0x400000, False)],
        [ordinary(0x10, "LDG.E.BYPASS", 0x100000),
         ordinary(0x20, "ATOMG.E.ADD.STRONG.GPU", 0x500000),
         ordinary(0x30, "RED.E.ADD.STRONG.GPU", 0x600000, False)],
        [ordinary(0x10, "LDG.E.BYPASS", 0x100000),
         reduction(0x20, "UTMAREDG.1D.ADD.F32", 0x700000),
         reduction(0x30, "UBLKRED.ADD.F32.G.S", 0x800000),
         "0040 ffffffff 0 UTMACMDFLUSH 0 0 0 0 NoVal",
         "0050 ffffffff 0 DEPBAR.LE 0 0 0 0 NoVal"],
    ]
    if writeback:
        warps = [[ordinary(0x10 * (index + 1), "STG.E", 0x100000 + 0x400 * index, False)
                  for index in range(4)], [], [], []]
        # Deliberately stores-only: eviction traffic must complete after the
        # final CTA exits, without a trailing load extending the kernel.
    if write_allocate:
        # One active lane writes only 4 bytes of a cold 32-byte sector. With
        # FETCH_ON_WRITE this necessarily creates an auxiliary allocation read.
        partial_store = ordinary(0x10, "STG.E.BYPASS", 0x900000, False)
        warps = [[partial_store.replace("ffffffff", "00000001", 1)], [], [], []]
    for warp, instructions in enumerate(warps):
        instructions.append("00f0 ffffffff 0 EXIT 0 0 0 0 NoVal")
        header += f"\nwarp = {warp}\ninsts = {len(instructions)}\n" + "\n".join(instructions) + "\n"
    return header + "\n#END_TB\n"


def check_drain_controls(run, argv, environment, source_log, drain):
    """A stores-only fixture must drain identically off, and fail when cut short."""
    off = run / "logging-off"
    off.mkdir()
    shutil.copy2(run / "config_hopper_islip.icnt", off)
    off_environment = environment.copy()
    for log_name in smoke.LOG_ENV:
        off_environment.pop(log_name, None)
    with (off / "simulator.log").open("w") as output:
        process = subprocess.run(argv, cwd=off, env=off_environment, stdout=output,
                                 stderr=subprocess.STDOUT, timeout=120)
    off_log = (off / "simulator.log").read_text(errors="replace")
    if process.returncode or smoke.architectural_statistics(source_log) != smoke.architectural_statistics(off_log):
        raise RuntimeError(f"Stores-only drain changed with logging disabled: {off}")

    interrupted = run / "interrupted-drain"
    interrupted.mkdir()
    shutil.copy2(run / "config_hopper_islip.icnt", interrupted)
    interrupted_environment = environment.copy()
    interrupted_environment.update({
        "L2_TRACE_DIR": str(interrupted / "l2_trace"),
        "L2_ACCESS_TRACE_FILE": str(interrupted / "l2_accesses.txt"),
        "L2_TRAFFIC_AUDIT_FILE": str(interrupted / "l2_traffic_audit.jsonl"),
    })
    limited_argv = argv + ["-gpgpu_max_cycle", str(drain["finish_cycle"] + 1)]
    with (interrupted / "simulator.log").open("w") as output:
        subprocess.run(limited_argv, cwd=interrupted, env=interrupted_environment,
                       stdout=output, stderr=subprocess.STDOUT, timeout=120)
    limited_log = (interrupted / "simulator.log").read_text(errors="replace")
    reports = [json.loads(line.split(" ", 1)[1]) for line in limited_log.splitlines()
               if line.startswith("L2_TRAFFIC_FINAL_DRAIN ")]
    if len(reports) != 1 or reports[0]["complete"] or reports[0]["drain_cycles"] != 1:
        raise RuntimeError(f"Simulation limit did not interrupt final draining: {interrupted}")
    with (interrupted / "validator.log").open("w") as output:
        process = subprocess.run([
            sys.executable, str(HARNESS / "validator/validate_l2_traffic.py"),
            "--run", str(interrupted), "--json-out", str(interrupted / "traffic-validation.json")],
            stdout=output, stderr=subprocess.STDOUT)
    report = json.loads((interrupted / "traffic-validation.json").read_text())
    if process.returncode == 0 or report["passed"]:
        raise RuntimeError(f"Strict validation accepted incomplete final draining: {interrupted}")
    cta_limited = run / "completed-cta-limit"
    cta_limited.mkdir()
    shutil.copy2(run / "config_hopper_islip.icnt", cta_limited)
    first_trace = Path(argv[argv.index("-trace") + 1]).read_text().splitlines()[0]
    first_name = Path(first_trace).name
    shutil.copy2(run / first_trace, cta_limited / first_name)
    second_trace = cta_limited / "kernel-990010.traceg"
    second_trace.write_text(trace_text(990010, writeback=True))
    kernel_list = cta_limited / "kernelslist.g"
    kernel_list.write_text(first_name + "\n" + second_trace.name + "\n")
    cta_argv = argv.copy()
    cta_argv[cta_argv.index("-trace") + 1] = str(kernel_list)
    cta_argv.extend(["-gpgpu_max_completed_cta", "1"])
    with (cta_limited / "simulator.log").open("w") as output:
        subprocess.run(cta_argv, cwd=cta_limited, env=off_environment,
                       stdout=output, stderr=subprocess.STDOUT, timeout=120)
    cta_log = (cta_limited / "simulator.log").read_text(errors="replace")
    cta_reports = [json.loads(line.split(" ", 1)[1]) for line in cta_log.splitlines()
                   if line.startswith("L2_TRAFFIC_FINAL_DRAIN ")]
    if (len(cta_reports) != 1 or cta_reports[0]["complete"] or
            cta_log.count("launching kernel name:") != 1 or
            "break due to reaching the maximum cycles" not in cta_log):
        raise RuntimeError(f"Completed-CTA limit was lost across stats reset: {cta_limited}")
    return {"logging_on_off_equal": True, "cycle_limit_rejected": True,
            "completed_cta_limit_stops_later_launches": True,
            "interrupted_drain": reports[0]}


def run_case(args, config, name, kernels, lrc_enabled, concurrent, writeback=False):
    run = args.output / name
    run.mkdir()
    traces = []
    write_allocate = name.startswith("write-allocate")
    for kernel, stream in kernels:
        trace = run / f"kernel-{kernel}.traceg"
        trace.write_text(trace_text(kernel, stream, writeback, write_allocate))
        traces.append(trace)
    (run / "kernelslist.g").write_text("".join(path.name + "\n" for path in traces))
    shutil.copy2(config / "config_hopper_islip.icnt", run)
    environment = os.environ.copy()
    for log_name in smoke.LOG_ENV:
        environment.pop(log_name, None)
    settings = {"LOG_CACHE_TRACES": "1", "L2_TRACE_DIR": str(run / "l2_trace"),
                "L2_ACCESS_TRACE_FILE": str(run / "l2_accesses.txt"),
                "L2_TRAFFIC_AUDIT_FILE": str(run / "l2_traffic_audit.jsonl")}
    environment.update(settings)
    argv = [str(args.binary), "-config", str(config / "gpgpusim.config"),
            "-config", str(config / "trace.config"), "-trace", str(run / "kernelslist.g"),
            "-gpgpu_concurrent_kernel_sm", str(int(concurrent)),
            "-gpgpu_max_concurrent_kernel", "2" if concurrent else "1",
            "-gpgpu_perf_sim_memcpy", "0", "-trace_enabled", "0",
            "-gpgpu_lrc_enabled", str(int(lrc_enabled))]
    if writeback:
        # Test-only conflict cache: eight sets, one way, write-back, lazy fetch.
        # One KiB avoids get_max_cache_multiplier's integer-KiB denominator=0.
        # The first gate always uses the unchanged H100 write-through cache.
        argv.extend(["-gpgpu_adaptive_cache_config", "0", "-gpgpu_cache:dl1",
                     "S:8:128:1,L:B:m:L:L,A:512:64,16:0,32",
                     "-gpgpu_l1_cache_write_ratio", "0"])
    if name.startswith("dirty-writebacks-dram"):
        argv.extend(["-gpgpu_cache:dl2", "S:256:128:20,L:B:m:N:X,A:384:4,32:0,32",
                     "-gpgpu_simple_dram_model", "0" if name.endswith("detailed") else "1"])
    if write_allocate:
        argv.extend(["-gpgpu_cache:dl2", "S:256:128:20,L:B:m:F:X,A:384:4,32:0,32",
                     "-gpgpu_simple_dram_model", "0" if name.endswith("detailed") else "1"])
    smoke.write_json(run / "command.json", {"argv": argv, "cwd": str(run), "environment": settings,
                                            "trace_sha256": {path.name: smoke.sha256(path) for path in traces}})
    print(f"Running synthetic case {run.name}", flush=True)
    start = time.monotonic()
    with (run / "simulator.log").open("w") as log:
        process = subprocess.run(argv, cwd=run, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=120)
    log = (run / "simulator.log").read_text(errors="replace")
    completed = process.returncode == 0 and "GPGPU-Sim: *** exit detected ***" in log
    completed = completed and "deadlock detected" not in log and "break due to reaching" not in log
    status = {"completed": completed, "returncode": process.returncode,
              "wall_seconds": time.monotonic() - start}
    smoke.write_json(run / "status.json", status)
    if not completed:
        raise RuntimeError(f"Synthetic simulator failed: {run / 'simulator.log'}")
    with (run / "validator.log").open("w") as validator_log:
        result = subprocess.run([sys.executable, str(HARNESS / "validator/validate_l2_traffic.py"),
                                 "--run", str(run), "--json-out", str(run / "traffic-validation.json")],
                                stdout=validator_log, stderr=subprocess.STDOUT)
    report = json.loads((run / "traffic-validation.json").read_text())
    if result.returncode or not report["passed"]:
        raise RuntimeError(f"Synthetic log consistency failed: {run / 'traffic-validation.json'}")
    counts = report["source_counts"]
    drain_lines = [line.split(" ", 1)[1] for line in log.splitlines()
                   if line.startswith("L2_TRAFFIC_FINAL_DRAIN ")]
    if len(drain_lines) != 1:
        raise RuntimeError(f"Missing or repeated final drain report: {run}")
    drain = json.loads(drain_lines[0])
    if not drain["complete"]:
        raise RuntimeError(f"Synthetic final drain did not complete: {run}")
    if write_allocate:
        if counts.get("l1_bypass_store") != 1:
            raise RuntimeError(f"Partial FETCH_ON_WRITE store was not exercised: {run}")
        return {"passed": True, "test_only_fetch_on_write_config": True,
                "dram_model": "detailed" if name.endswith("detailed") else "simple",
                "source_counts": counts, "consistency_counts": report["counts"],
                "final_drain": drain}
    if writeback:
        writebacks = 0
        for path in (run / "l2_trace").rglob("l1_to_l2_requests.txt"):
            with path.open() as source_log:
                for line in source_log:
                    fields = dict(word.split("=", 1) for word in line.split())
                    writebacks += fields.get("type") == "WRITEBACK"
        if not writebacks:
            raise RuntimeError(f"Synthetic conflict stores did not produce dirty writebacks: {run}")
        if drain["drain_cycles"] <= 0:
            raise RuntimeError(f"Stores-only eviction fixture did not exercise final draining: {run}")
        completion_after_finish = False
        for path in (run / "l2_trace").rglob("l2_to_icnt_timing.txt"):
            for line in path.read_text().splitlines():
                fields = dict(word.split("=", 1) for word in line.split())
                completion_after_finish |= int(fields["end_cycle"]) > drain["finish_cycle"]
        if not completion_after_finish:
            raise RuntimeError(f"No source completion depended on final draining: {run}")
        controls = check_drain_controls(run, argv, environment, log, drain) if name == "dirty-writebacks" else {}
        return {"passed": True, "test_only_l1_writeback_config": True,
                "dirty_writeback_records": writebacks, "source_counts": counts,
                "consistency_counts": report["counts"], "final_drain": drain,
                "drain_controls": controls}
    for kind in ("l1_miss", "l1_write", "l1_bypass_load", "l1_bypass_store", "tma_store"):
        if counts.get(kind, 0) <= 0:
            raise RuntimeError(f"Missing exercised synthetic source {kind}: {run}")
    if bool(report["counts"]["lrc_results"]) != lrc_enabled:
        raise RuntimeError(f"LRC output presence does not match setting: {run}")
    by_kernel = collections.Counter()
    observed_opcodes = collections.Counter()
    first_last = {}
    ordinal_opcodes = {
        0: ["LDG", "STG"],
        1: ["LDG.BYPASS", "STG.BYPASS"],
        2: ["LDG.BYPASS", "ATOMG", "RED"],
        3: ["LDG.BYPASS", "UTMAREDG", "UBLKRED"],
    }
    # Every case has one CTA per kernel; assigned dynamic warp IDs need not
    # equal warp-in-CTA IDs when two kernels share an SM.
    mapping = {}
    for path in (run / "warp_assignments").glob("kernel_*/shader_*.txt"):
        kernel = int(path.parent.name.split("_")[1])
        sm = int(path.stem.split("_")[-1])
        for match in re.finditer(r"warp_in_cta=(\d+) dynamic_warp_id=(\d+)", path.read_text()):
            mapping[(kernel, sm, int(match[2]))] = int(match[1])
    with (run / "l2_traffic_audit.jsonl").open() as audit:
        for line in audit:
            event = json.loads(line)
            if event["event"] != "departure":
                continue
            kernel = event["kernel_uid"]
            by_kernel[kernel] += 1
            warp = mapping[(kernel, event["sm"], event["dynamic_warp"])]
            opcode = ordinal_opcodes[warp][event["inst_ordinal"]]
            if int(event["PC"], 16) != 0x10 * (event["inst_ordinal"] + 1):
                raise RuntimeError(f"Wrong source PC for synthetic opcode {opcode}: {run}")
            expected_source = {
                "LDG": "l1_miss", "STG": "l1_write", "LDG.BYPASS": "l1_bypass_load",
                "STG.BYPASS": "l1_bypass_store", "ATOMG": "l1_bypass_load",
                "RED": "l1_bypass_load", "UTMAREDG": "tma_store", "UBLKRED": "tma_store",
            }[opcode]
            if event["source"] != expected_source:
                raise RuntimeError(f"Wrong source classification for synthetic opcode {opcode}: {run}")
            observed_opcodes[opcode] += 1
            if kernel not in first_last:
                first_last[kernel] = [event["cycle"], event["cycle"]]
            first_last[kernel][1] = event["cycle"]
    if set(by_kernel) != {kernel for kernel, _ in kernels}:
        raise RuntimeError(f"Missing concurrent-kernel attribution: {run}")
    if not all(observed_opcodes[opcode] for opcode in ("LDG", "STG", "LDG.BYPASS", "STG.BYPASS",
                                                     "ATOMG", "RED", "UTMAREDG", "UBLKRED")):
        raise RuntimeError(f"A targeted synthetic opcode emitted no source requests: {run}")
    if concurrent:
        intervals = list(first_last.values())
        if max(interval[0] for interval in intervals) > min(interval[1] for interval in intervals):
            raise RuntimeError(f"Synthetic kernels did not overlap: {run}")
    return {"passed": True, "lrc_enabled": lrc_enabled, "concurrent": concurrent,
            "source_counts": counts, "source_records_by_opcode": dict(observed_opcodes),
            "source_records_by_kernel": dict(by_kernel), "source_cycle_intervals": first_last,
            "consistency_counts": report["counts"], "mixed_source_groups": report["mixed_source_groups"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path,
                        help="The same fully linked simulator binary used for the first gate")
    parser.add_argument("--gate", required=True, type=Path, help="Successful first-gate directory")
    parser.add_argument("--output", required=True, type=Path, help="New synthetic-results directory")
    parser.add_argument("--case", action="append",
                        choices=("sources-lrc-on", "sources-lrc-off", "sources-concurrent", "dirty-writebacks",
                                 "dirty-writebacks-dram-simple", "dirty-writebacks-dram-detailed",
                                 "write-allocate-simple", "write-allocate-detailed"),
                        help="Run only selected cases (repeatable); default runs all")
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.gate = args.gate.resolve(strict=True)
    args.output = args.output.resolve()
    if not (args.gate / "SUCCESS").is_file() or not json.loads((args.gate / "summary.json").read_text())["passed"]:
        parser.error("The complete kernel 8039 first gate must pass before synthetic runs")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"synthetic": True, "passed": False, "cases": {}, "binary_sha256": smoke.sha256(args.binary)}
    config = args.output / "config"
    shutil.copytree(args.gate / "config", config)
    try:
        for name, kernels, enabled, concurrent, writeback in (
            ("sources-lrc-on", [(990001, 0)], True, False, False),
            ("sources-lrc-off", [(990002, 0)], False, False, False),
            ("sources-concurrent", [(990003, 0), (990004, 1)], True, True, False),
            ("dirty-writebacks", [(990005, 0)], True, False, True),
            ("dirty-writebacks-dram-simple", [(990006, 0)], True, False, True),
            ("dirty-writebacks-dram-detailed", [(990007, 0)], True, False, True),
            ("write-allocate-simple", [(990008, 0)], False, False, False),
            ("write-allocate-detailed", [(990009, 0)], False, False, False),
        ):
            if args.case and name not in args.case:
                continue
            summary["cases"][name] = run_case(args, config, name, kernels, enabled, concurrent, writeback)
            smoke.write_json(args.output / "summary.json", summary)
        if smoke.sha256(args.binary) != summary["binary_sha256"]:
            raise RuntimeError("Binary changed during synthetic tests")
        summary["passed"] = True
    except Exception as error:
        summary["failure"] = str(error)
        print(str(error), file=sys.stderr)
    smoke.write_json(args.output / "summary.json", summary)
    (args.output / ("SUCCESS" if summary["passed"] else "FAILED")).write_text("Synthetic source tests\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
