#!/usr/bin/env python3
"""Run the complete Gemma 4 kernel 8039 with H100 and compare logging on/off.

The first gate intentionally does not claim coverage of ordinary L1 traffic,
bypass stores, atomics, or TMA reductions: those operations are absent or
predicated off in this trace. Results go into a new directory on every run.
"""

import argparse
import collections
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


FRAMEWORK = Path(__file__).resolve().parents[2]
GPU = FRAMEWORK / "gpu-simulator"
KERNEL_ID = 8039
TRACE_NAME = "kernel-8039-ctx_0x5c7e6b6cbd80.traceg.xz"
LOG_ENV = (
    "LOG_CACHE_TRACES", "L2_TRACE_DIR", "L1_TRACE_DIR", "L2_ACCESS_TRACE_FILE",
    "L2_TRAFFIC_AUDIT_FILE", "SCHED_TRACE_DIR", "LD_PRELOAD",
)
STAT_RE = re.compile(
    r"^(?:gpu_(?:tot_)?sim_(?:cycle|insn)|gpu_tot_issued_cta|gpu_stall_\w+|"
    r"gpgpu_n_mem_\w+|L[12][DICT]?_total_cache_\w+|"
    r"L[12][DICT]?_cache_stats_breakdown\[[^=]+|"
    r"LRC_subpartition_\S+|interchip_\w+)\s*="
)
UNCOVERED = [
    "L1 read misses (ordinary LDG instructions are all predicated off)",
    "L1 stores and writebacks",
    "non-TMA bypass stores",
    "ordinary global atomics and reductions",
    "TMA reductions",
    "concurrent-kernel attribution",
    "LRC disabled",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def trace_inventory(path):
    """Count active opcodes directly; opcode presence alone is not coverage."""
    header = {}
    active = collections.Counter()
    inactive = collections.Counter()
    with lzma.open(path, "rt") as trace:
        for line in trace:
            if line.startswith("-") and " = " in line:
                key, value = line[1:].strip().split(" = ", 1)
                header[key] = value
                continue
            words = line.split()
            if len(words) < 5:
                continue
            try:
                int(words[0], 16)
                mask = int(words[1], 16)
                opcode = words[3 + int(words[2])]
            except (ValueError, IndexError):
                continue
            if opcode.startswith(("LDG", "STG", "UTMA", "UBLK", "ATOM", "RED", "LDL", "STL")):
                (active if mask else inactive)[opcode] += 1
    if header.get("kernel id") != str(KERNEL_ID):
        raise ValueError(f"Expected kernel {KERNEL_ID}; found {header.get('kernel id')}")
    return {"header": header, "active_memory_opcodes": dict(active),
            "inactive_memory_opcodes": dict(inactive)}


def architectural_statistics(log):
    result = [line.strip() for line in log.splitlines() if STAT_RE.match(line.strip())]
    drain = [line.strip() for line in log.splitlines()
             if line.startswith("L2_TRAFFIC_FINAL_DRAIN ")]
    if len(drain) != 1 or not json.loads(drain[0].split(" ", 1)[1]).get("complete"):
        raise ValueError("Missing, duplicate, or incomplete final memory drain report")
    result.extend(drain)
    for name in ("gpu_tot_sim_cycle", "gpu_tot_sim_insn", "L2_total_cache_accesses",
                 "LRC_subpartition_num_lrc_to_l2_sectors.sum"):
        if not any(re.match(re.escape(name) + r"\s*=", line) for line in result):
            raise ValueError(f"Missing required simulator statistic: {name}")
    return result


def validate_source_instructions(run, trace_path, inventory):
    """Independently join every source row to its original active SASS record."""
    grid = [int(value) for value in inventory["header"]["grid dim"].strip("()").split(",")]
    instructions = {}
    cta = warp = None
    ordinal = 0
    with lzma.open(trace_path, "rt") as trace:
        for line in trace:
            if line.startswith("thread block = "):
                x, y, z = map(int, line.split("=", 1)[1].split(","))
                cta = x + grid[0] * (y + grid[1] * z)
            elif line.startswith("warp = "):
                warp = int(line.split("=", 1)[1])
                ordinal = 0
            else:
                words = line.split()
                if len(words) < 5:
                    continue
                try:
                    pc, mask = int(words[0], 16), int(words[1], 16)
                    opcode = words[3 + int(words[2])]
                except (ValueError, IndexError):
                    continue
                if mask and opcode.startswith(("LDG", "STG", "UTMALDG", "UTMASTG")) and opcode != "LDGDEPBAR":
                    if opcode.startswith("UTMALDG"):
                        source_kind = "tma_load"
                    elif opcode.startswith("UTMASTG"):
                        source_kind = "tma_store"
                    elif "BYPASS" in opcode:
                        source_kind = "l1_bypass_load" if opcode.startswith("LDG") else "l1_bypass_store"
                    else:
                        source_kind = "l1_miss" if opcode.startswith("LDG") else "l1_write"
                    instructions[(cta, warp, ordinal)] = (pc, opcode, source_kind)
                ordinal += 1
    assignments = {}
    ctas = set()
    for path in (run / "warp_assignments").glob("kernel_8039_*/shader_*.txt"):
        sm = int(path.stem.split("_")[-1])
        for line in path.read_text().splitlines():
            match = re.search(r"CTA \d+ \(ctaid=(\d+)\)", line)
            if match:
                cta = int(match[1])
                if cta in ctas:
                    raise ValueError(f"Duplicate CTA assignment {cta}")
                ctas.add(cta)
            match = re.search(r"warp_in_cta=(\d+) dynamic_warp_id=(\d+)\) -> Scheduler (\d+)", line)
            if match:
                key = (sm, int(match[2]))
                if key in assignments:
                    raise ValueError(f"Duplicate dynamic warp assignment {key}")
                assignments[key] = (cta, int(match[1]), int(match[3]))
    expected_ctas = grid[0] * grid[1] * grid[2]
    block = [int(value) for value in inventory["header"]["block dim"].strip("()").split(",")]
    expected_warps = expected_ctas * ((block[0] * block[1] * block[2] + 31) // 32)
    if len(ctas) != expected_ctas or len(assignments) != expected_warps:
        raise ValueError("Incomplete CTA/warp assignment coverage")
    matched = collections.Counter()
    matched_instructions = set()
    for filename in ("l1_to_l2_requests.txt", "l1_bypass_to_l2.txt", "tma_to_l2_requests.txt"):
        for path in (run / "l2_trace").rglob(filename):
            sm, scheduler = int(path.parent.parent.name.split("_")[1]), int(path.parent.name.split("_")[1])
            with path.open() as source_log:
                for line_number, line in enumerate(source_log, 1):
                    fields = dict(word.split("=", 1) for word in line.split())
                    cta, warp, expected_scheduler = assignments[(sm, int(fields["dynamic_warp"]))]
                    key = (cta, warp, int(fields["inst_ordinal"]))
                    expected = instructions.get(key)
                    if expected is None or int(fields["PC"], 16) != expected[0]:
                        raise ValueError(f"Unmatched or incorrect instruction attribution: {path}:{line_number}")
                    if expected_scheduler != scheduler or fields.get("source", expected[2]) != expected[2]:
                        raise ValueError(f"Incorrect source/scheduler attribution: {path}:{line_number}")
                    matched[expected[1]] += 1
                    matched_instructions.add(key)
    missing = set(instructions) - matched_instructions
    if missing:
        raise ValueError(f"{len(missing)} active source instructions have no source rows; examples: {sorted(missing)[:8]}")
    return {"passed": True, "source_rows_matched": sum(matched.values()),
            "active_source_instructions": len(instructions),
            "matched_source_instructions": len(matched_instructions),
            "missing_source_instructions": 0, "source_rows_by_opcode": dict(matched),
            "cta_assignments": len(ctas), "warp_assignments": len(assignments)}


def library_directory(args):
    if args.library_dir:
        return args.library_dir.resolve(strict=True)
    flavor = args.binary.parent.name
    candidates = sorted((GPU / "gpgpu-sim/lib").glob(f"*/cuda-*/{flavor}/libcudart.so"))
    if len(candidates) != 1:
        raise ValueError("Cannot select simulator libcudart unambiguously; provide --library-dir")
    return candidates[0].parent.resolve()


def run_simulator(args, config, source, library, enabled):
    run = args.output / (f"kernel-{KERNEL_ID}" + ("" if enabled else "-logging-off"))
    run.mkdir()
    (run / source.name).symlink_to(source)
    (run / "kernelslist.g").write_text(source.name + "\n")
    shutil.copy2(config / "config_hopper_islip.icnt", run)
    environment = os.environ.copy()
    for name in LOG_ENV:
        environment.pop(name, None)
    settings = {"LD_LIBRARY_PATH": str(library) + ":/usr/local/cuda/lib64"}
    if enabled:
        settings.update(
            LOG_CACHE_TRACES="1", L2_TRACE_DIR=str(run / "l2_trace"),
            L2_ACCESS_TRACE_FILE=str(run / "l2_accesses.txt"),
            L2_TRAFFIC_AUDIT_FILE=str(run / "l2_traffic_audit.jsonl"),
        )
    environment.update(settings)
    argv = [str(args.binary), "-config", str(config / "gpgpusim.config"),
            "-config", str(config / "trace.config"), "-trace", str(run / "kernelslist.g"),
            "-gpgpu_concurrent_kernel_sm", "0", "-gpgpu_max_concurrent_kernel", "1",
            "-gpgpu_perf_sim_memcpy", "0", "-trace_enabled", "0"]
    write_json(run / "command.json", {"argv": argv, "cwd": str(run), "environment": settings,
                                     "unset_inherited_environment": list(LOG_ENV)})
    print(f"Running full kernel {KERNEL_ID}, logging={'on' if enabled else 'off'}", flush=True)
    start = time.monotonic()
    timed_out = False
    with (run / "simulator.log").open("w") as log:
        try:
            process = subprocess.run(argv, cwd=run, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, timeout=args.timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
    log = (run / "simulator.log").read_text(errors="replace")
    completed = (returncode == 0 and "GPGPU-Sim: *** exit detected ***" in log
                 and "deadlock detected" not in log and "break due to reaching" not in log)
    status = {"kernel_completed": completed, "returncode": returncode, "timed_out": timed_out,
              "wall_seconds": time.monotonic() - start, "logging_enabled": enabled}
    write_json(run / "status.json", status)
    if not completed:
        raise RuntimeError(f"Kernel failed or incomplete; see {run / 'simulator.log'}")
    stats = architectural_statistics(log)
    write_json(run / "architectural-statistics.json", stats)
    if not enabled:
        unexpected = [name for name in ("l2_trace", "l2_accesses.txt", "l2_traffic_audit.jsonl",
                                       "warp_assignments", "dcache_status", "icache_status")
                      if (run / name).exists()]
        if unexpected:
            raise RuntimeError(f"Logging-disabled run unexpectedly emitted: {unexpected}")
    return run, status, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hopper-model", choices=("SM90_H100", "SM90_H200"), default="SM90_H100",
                        help="Full-size Hopper config to exercise (default: SM90_H100)")
    parser.add_argument("--binary", required=True, type=Path, help="Newly rebuilt accel-sim.out")
    parser.add_argument("--output", required=True, type=Path, help="New directory; must not already exist")
    parser.add_argument("--library-dir", type=Path, help="Directory of rebuilt simulator libcudart.so")
    parser.add_argument("--trace", type=Path,
                        default=FRAMEWORK / "data/gemma4-h100/traces" / TRACE_NAME)
    parser.add_argument("--timeout", type=float, default=900, help="Timeout per simulation, seconds")
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.output = args.output.resolve()
    source = args.trace.resolve(strict=True)
    library = library_directory(args)
    library_file = library / "libcudart.so"
    validator = Path(__file__).resolve().parent / "validator/validate_l2_traffic.py"
    if not library_file.is_file() or not validator.is_file():
        parser.error("Simulator library and traffic validator must exist before running the gate")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "INCOMPLETE").write_text(f"Running kernel 8039 {args.hopper_model} logging gate\n")
    summary = {"kernel_id": KERNEL_ID, "hopper_model": args.hopper_model, "passed": False, "uncovered_categories": UNCOVERED}
    try:
        config = args.output / "config"
        config.mkdir()
        h100 = GPU / "gpgpu-sim/configs/tested-cfgs" / args.hopper_model
        for name in ("gpgpusim.config", "config_hopper_islip.icnt"):
            shutil.copy2(h100 / name, config / name)
        shutil.copy2(GPU / "configs/tested-cfgs" / args.hopper_model / "trace.config", config / "trace.config")
        hashes = {"binary": sha256(args.binary), "library": sha256(library_file),
                  "trace": sha256(source)}
        provenance = {"binary": str(args.binary), "library": str(library_file),
                      "trace": str(source), "sha256": hashes,
                      "configs": {path.name: sha256(path) for path in config.iterdir()},
                      "runner_sha256": sha256(Path(__file__)), "validator_sha256": sha256(validator)}
        for label, repository in (("framework", FRAMEWORK), ("gpgpusim", GPU / "gpgpu-sim")):
            commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                                    text=True, capture_output=True)
            provenance[label + "_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
            provenance[label + "_status"] = (subprocess.check_output(
                ["git", "status", "--short"], cwd=repository, text=True)
                if commit.returncode == 0 else "Isolated source snapshot")
        provenance["validator_modules"] = {
            str(path.relative_to(validator.parent)): sha256(path)
            for path in sorted(validator.parent.rglob("*.py"))
            if "tests" not in path.relative_to(validator.parent).parts}
        write_json(args.output / "provenance.json", provenance)
        summary["trace_inventory"] = trace_inventory(source)
        write_json(args.output / "summary.json", summary)
        on_run, on_status, on_stats = run_simulator(args, config, source, library, True)
        summary["logging_on"] = on_status
        off_run, off_status, off_stats = run_simulator(args, config, source, library, False)
        summary["logging_off"] = off_status
        summary["logging_on_off_match"] = on_stats == off_stats
        summary["final_drain"] = json.loads(next(line.split(" ", 1)[1] for line in on_stats
                                                if line.startswith("L2_TRAFFIC_FINAL_DRAIN ")))
        summary["architectural_statistics_compared"] = len(on_stats)
        if on_stats != off_stats:
            summary["statistics_difference"] = {
                "logging_on_only": sorted((collections.Counter(on_stats) - collections.Counter(off_stats)).elements()),
                "logging_off_only": sorted((collections.Counter(off_stats) - collections.Counter(on_stats)).elements()),
            }
        for name, path in (("binary", args.binary), ("library", library_file), ("trace", source)):
            if sha256(path) != hashes[name]:
                raise RuntimeError(f"Input changed while running: {path}")
        argv = [sys.executable, str(validator), "--run", str(on_run),
                "--json-out", str(on_run / "traffic-validation.json")]
        with (on_run / "validator.log").open("w") as log:
            validation = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
        summary["validator_returncode"] = validation.returncode
        if (on_run / "traffic-validation.json").is_file():
            summary["traffic_validation"] = json.loads((on_run / "traffic-validation.json").read_text())
        if validation.returncode:
            raise RuntimeError(f"Traffic consistency validation failed; see {on_run / 'validator.log'}")
        report = summary["traffic_validation"]
        if not report.get("passed"):
            raise RuntimeError("Traffic validator did not report passed=true")
        counts = report["source_counts"]
        required = ("l1_bypass_load", "tma_load", "tma_store")
        for source_kind in required:
            if counts.get(source_kind, 0) <= 0:
                raise RuntimeError(f"Missing expected kernel 8039 source: {source_kind}")
        unexpected = {name: value for name, value in counts.items() if value and name not in required}
        if unexpected:
            raise RuntimeError(f"Unexpected sources for kernel 8039: {unexpected}")
        summary["coverage"] = {"observed_source_counts": counts,
                               "l1_source_log_expected_empty": True,
                               "mixed_source_merges_observed": report.get("mixed_source_groups", 0),
                               "split_departures_observed": report.get("counts", {}).get("split_departures", 0)}
        if not summary["coverage"]["mixed_source_merges_observed"]:
            summary["uncovered_categories"].append("mixed-source LRC merging")
        if not summary["coverage"]["split_departures_observed"]:
            summary["uncovered_categories"].append("multi-sector departure splitting")
        instruction_report = validate_source_instructions(on_run, source, summary["trace_inventory"])
        write_json(on_run / "instruction-validation.json", instruction_report)
        summary["instruction_validation"] = instruction_report
        for name in ("gpu_tot_sim_cycle", "gpu_tot_sim_insn"):
            summary[name] = int(next(line.split("=", 1)[1] for line in on_stats
                                     if line.startswith(name + " ")))
        if on_stats != off_stats:
            raise RuntimeError("Logging changed cycle, instruction, memory, cache, or LRC statistics")
        summary["passed"] = True
        (args.output / "INCOMPLETE").rename(args.output / "SUCCESS")
    except Exception as error:
        summary["failure"] = str(error)
        (args.output / "INCOMPLETE").rename(args.output / "FAILED")
        print(str(error), file=sys.stderr)
    finally:
        write_json(args.output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("trace_inventory", "traffic_validation")}, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
