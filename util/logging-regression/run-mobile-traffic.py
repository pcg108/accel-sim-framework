#!/usr/bin/env python3
"""Replay a trace collection as isolated kernels and validate L2 traffic logs.

Each kernel starts with cold simulator state; this is a logging collection, not
an end-to-end inference timing measurement or a logging on/off comparison.
Failed kernels retain their logs and do not prevent later kernels from running.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time


HERE = Path(__file__).resolve().parent
FRAMEWORK = HERE.parents[1]
VALIDATOR = HERE / "validator/validate_l2_traffic.py"
LOG_ENV = (
    "LOG_CACHE_TRACES", "L2_TRACE_DIR", "L1_TRACE_DIR", "L2_ACCESS_TRACE_FILE",
    "L2_TRAFFIC_AUDIT_FILE", "SCHED_TRACE_DIR", "LD_PRELOAD",
)
FAILURE = re.compile(
    r"Assertion .* failed|Program received signal|deadlock detected|break due to reaching|"
    r"simulation (?:terminated|stopped).*limit", re.I)
KERNEL_NAME = re.compile(r"kernel-(\d+)(?:-[^/]+)?\.traceg(?:\.xz|\.gz)?$")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validator_manifest(directory):
    """Fingerprint the full importable validator package, not just its entrypoint."""
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*.py"))
            if "__pycache__" not in path.parts and "tests" not in path.relative_to(directory).parts}


def manifest_fingerprint(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def pin_tools(output, binary, library_dir):
    """Every collection executes immutable local simulator and validator copies."""
    target = output / "tools"
    target.mkdir()
    shutil.copy2(binary, target / "accel-sim.out")
    shutil.copy2(Path(__file__), target / "run-mobile-traffic.py")
    shutil.copytree(VALIDATOR.parent, target / "validator",
                    ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"))
    if library_dir:
        (target / "lib").mkdir()
        for path in sorted(library_dir.glob("*.so*")):
            if path.is_file():
                shutil.copy2(path, target / "lib" / path.name)
    files = {str(path.relative_to(target)): sha256(path)
             for path in sorted(target.rglob("*")) if path.is_file()}
    write_json(target / "manifest.json", files)
    return target


def check_pinned_tools(target):
    for name, digest in json.loads((target / "manifest.json").read_text()).items():
        if sha256(target / name) != digest:
            raise ValueError(f"Pinned collection tool changed: {name}")


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def selected_ids(values):
    result = set()
    for value in values or []:
        for item in value.split(","):
            match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
            if not match:
                raise ValueError(f"Invalid kernel selection: {item!r}")
            first, last = int(match[1]), int(match[2] or match[1])
            if last < first:
                raise ValueError(f"Reversed kernel range: {item}")
            result.update(range(first, last + 1))
    return result


def trace_selection(directory, wanted):
    entries, seen, skipped = [], set(), Counter()
    for line_number, line in enumerate((directory / "kernelslist.g").read_text().splitlines(), 1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith(("Memcpy", "cudaMemcpy")):
            skipped[entry.split(",", 1)[0]] += 1
            continue
        relative = Path(entry)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError(f"kernelslist.g:{line_number}: expected a local trace filename")
        match = KERNEL_NAME.fullmatch(entry)
        if not match:
            raise ValueError(f"kernelslist.g:{line_number}: unrecognized entry {entry!r}")
        kernel = int(match[1])
        if kernel in seen:
            raise ValueError(f"Duplicate kernel ID {kernel} in kernelslist.g")
        seen.add(kernel)
        if not wanted or kernel in wanted:
            source = (directory / entry).resolve(strict=True)
            if not source.is_file():
                raise ValueError(f"Not a trace file: {source}")
            entries.append((kernel, source))
    if wanted - seen:
        raise ValueError(f"Selected kernels missing from kernelslist.g: {sorted(wanted - seen)}")
    if not entries:
        raise ValueError("No selected kernels")
    return entries, dict(skipped)


def trace_header(path):
    opener = lzma.open if path.suffix == ".xz" else gzip.open if path.suffix == ".gz" else open
    header = {}
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("#BEGIN_TB"):
                break
            if line.startswith("-") and " = " in line:
                key, value = line[1:].strip().split(" = ", 1)
                header[key] = value
    return header


def config_option(path, option, default):
    value = default
    for line in path.read_text().splitlines():
        words = line.split("#", 1)[0].split()
        if len(words) >= 2 and words[0] == option:
            value = words[1]
    return value


def simulation_log(run):
    completed, failures, statistics = False, [], {}
    path = run / "simulator.log"
    if not path.is_file():
        return completed, failures, statistics
    with path.open(errors="replace") as log:
        for line in log:
            completed |= "GPGPU-Sim: *** exit detected ***" in line
            if FAILURE.search(line) and len(failures) < 8:
                failures.append(line.strip())
            match = re.match(r"\s*(gpu_(?:tot_)?sim_(?:cycle|insn)|gpu_tot_issued_cta)\s*=\s*(\d+)", line)
            if match:
                statistics[match[1]] = int(match[2])
    return completed, failures, statistics


def resume_status(run, kernel, source):
    """Recover completed work, preserving terminal failures and interrupted files."""
    try:
        status = json.loads((run / "status.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        status = {"kernel_id": kernel, "run": str(run), "trace": str(source),
                  "kernel_completed": False, "passed": False}
    if status.get("kernel_completed") or status.get("invalid_input"):
        return status
    completed, failures, statistics = simulation_log(run)
    returncode = status.get("returncode")
    interrupted_signals = (-signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL, -signal.SIGHUP)
    terminal_exit = returncode is not None and returncode != 0 and returncode not in interrupted_signals
    if failures or status.get("timed_out") or terminal_exit:
        status.update(kernel_completed=False, passed=False)
        status.pop("retryable", None)
        if failures:
            status["failure"] = failures
        write_json(run / "status.json", status)
        return status
    if completed and returncode in (None, 0):
        header = trace_header(source)
        if "grid dim" in header:
            dimensions = [int(value) for value in header["grid dim"].strip("()").split(",")]
            expected_ctas = dimensions[0] * dimensions[1] * dimensions[2]
            if statistics.get("gpu_tot_issued_cta") != expected_ctas:
                failures.append(f"Issued CTAs={statistics.get('gpu_tot_issued_cta')}, expected={expected_ctas}")
        provenance_path = run / "trace-provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            if provenance.get("sha256") != sha256(source):
                failures.append("Input trace changed since simulation")
        status.update(kernel_completed=not failures, passed=False, returncode=0,
                      statistics=statistics, recovered_from_simulator_log=True)
        status.pop("retryable", None)
        if failures:
            status["failure"] = failures
        else:
            status.pop("failure", None)
        write_json(run / "status.json", status)
        return status
    # A killed worker can leave complete-looking output files, but without a
    # successful simulation they must never count as a finished kernel.
    archive_root = run.parent / "interrupted-attempts"
    archive_root.mkdir(exist_ok=True)
    archive = archive_root / f"{run.name}-{time.time_ns()}"
    run.rename(archive)
    print(f"REPLAY kernel {kernel}: preserved interrupted attempt at {archive}", flush=True)
    return None


def run_kernel(args, kernel, source, config, trace_config, max_merged):
    run = args.output / f"kernel-{kernel}"
    if str(kernel) in args.rejected_inputs:
        run.mkdir(exist_ok=True)
        status = {"kernel_id": kernel, "trace": str(source), "run": str(run),
                  "kernel_completed": False, "passed": False, "invalid_input": True,
                  "failure": args.rejected_inputs[str(kernel)]}
        write_json(run / "status.json", status)
        return status
    if run.exists():
        status = resume_status(run, kernel, source)
        if status is not None and not status.get("kernel_completed"):
            return status
        if (status is not None and status.get("passed") and not args.revalidate
                and status.get("validator_sha256") == args.validator_sha256):
            return status
    if not run.exists():
        print(f"START kernel {kernel}: {source.name}", flush=True)
        run.mkdir()
        (run / source.name).symlink_to(source)
        (run / "kernelslist.g").write_text(source.name + "\n")
        # The simulator lazily opens these outputs. Empty memory traffic is a
        # valid outcome, and still requires normal kernel completion below.
        (run / "l2_trace").mkdir()
        (run / "l2_accesses.txt").touch()
        (run / "l2_traffic_audit.jsonl").touch()
        header = trace_header(source)
        initial_stat = source.stat()
        write_json(run / "trace-provenance.json", {
            "trace": str(source), "sha256": sha256(source), "header": header,
            "size_bytes": initial_stat.st_size, "mtime_ns": initial_stat.st_mtime_ns})
        environment = os.environ.copy()
        for name in LOG_ENV:
            environment.pop(name, None)
        settings = {
            "LOG_CACHE_TRACES": "1", "L2_TRACE_DIR": str(run / "l2_trace"),
            "L2_ACCESS_TRACE_FILE": str(run / "l2_accesses.txt"),
            "L2_TRAFFIC_AUDIT_FILE": str(run / "l2_traffic_audit.jsonl"),
        }
        if args.library_dir:
            settings["LD_LIBRARY_PATH"] = str(args.library_dir) + ":/usr/local/cuda/lib64"
        environment.update(settings)
        command = [str(args.binary), "-config", str(config)]
        if trace_config:
            command += ["-config", str(trace_config)]
        command += ["-trace", str(run / "kernelslist.g"),
                    "-gpgpu_concurrent_kernel_sm", "0", "-gpgpu_max_concurrent_kernel", "1",
                    "-gpgpu_perf_sim_memcpy", "0", "-trace_enabled", "0"]
        if args.max_cycles:
            command += ["-gpgpu_max_cycle", str(args.max_cycles)]
        write_json(run / "command.json", {
            "argv": command, "cwd": str(run), "environment": settings,
            "unset_inherited_environment": list(LOG_ENV)})
        start = time.monotonic()
        timed_out = False
        with (run / "simulator.log").open("w") as log:
            try:
                process = subprocess.run(command, cwd=run, env=environment, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=args.timeout)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                timed_out, returncode = True, None
        completed, failures, statistics = simulation_log(run)
        interrupted = (not failures and not timed_out and returncode in
                       (-signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL, -signal.SIGHUP))
        if "grid dim" in header:
            dimensions = [int(value) for value in header["grid dim"].strip("()").split(",")]
            expected_ctas = dimensions[0] * dimensions[1] * dimensions[2]
            if statistics.get("gpu_tot_issued_cta") != expected_ctas:
                failures.append(f"Issued CTAs={statistics.get('gpu_tot_issued_cta')}, expected={expected_ctas}")
        final_stat = source.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            failures.append("Input trace changed during simulation")
        completed = completed and returncode == 0 and not failures and not timed_out
        status = {
            "kernel_id": kernel, "run": str(run), "trace": str(source), "passed": False,
            "kernel_completed": completed, "returncode": returncode, "timed_out": timed_out,
            "simulation_seconds": time.monotonic() - start, "statistics": statistics,
        }
        if not completed:
            status["failure"] = failures or ["Timed out" if timed_out else "Missing normal completion or nonzero exit"]
            if interrupted:
                status["retryable"] = True
        write_json(run / "status.json", status)
        if not completed:
            return status
    command = [sys.executable, str(getattr(args, "validator_path", VALIDATOR)), "--run", str(run),
               "--json-out", str(run / "traffic-validation.json"), "--lrc-max-merged", str(max_merged)]
    with args.validation_semaphore:
        start = time.monotonic()
        with (run / "validator.log").open("w") as log:
            validation = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    status["passed"] = False
    status["validation_seconds"] = time.monotonic() - start
    status["validator_returncode"] = validation.returncode
    status["validator_sha256"] = args.validator_sha256
    if (run / "traffic-validation.json").is_file():
        report = json.loads((run / "traffic-validation.json").read_text())
        status["passed"] = validation.returncode == 0 and report.get("passed", False)
        for key in ("source_counts", "counts", "consistency", "mixed_source_groups", "issues",
                    "timing_coverage", "final_drain"):
            status[key] = report.get(key)
    if not status["passed"]:
        status["failure"] = f"Traffic validation failed; see {run / 'validator.log'}"
    else:
        status.pop("failure", None)
    write_json(run / "status.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trace-config", type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory, unless --resume")
    parser.add_argument("--kernel-ids", nargs="+", help="IDs or inclusive ranges, separated by spaces or commas")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--validation-jobs", type=int, default=2, help="Maximum simultaneous memory-intensive validators")
    parser.add_argument("--timeout", type=float, default=1800, help="Seconds per simulation")
    parser.add_argument("--max-cycles", type=int, default=0, help="Optional safety bound; hitting it fails the kernel")
    parser.add_argument("--library-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse passed results and completed simulations; replay interrupted kernels")
    parser.add_argument("--revalidate", action="store_true", help="With --resume, also revalidate previously passed kernels")
    parser.add_argument("--rejected-inputs", type=Path, help="JSON object mapping known invalid kernel IDs to reasons/evidence")
    args = parser.parse_args()
    if args.revalidate and not args.resume:
        parser.error("--revalidate requires --resume")
    if args.jobs < 1 or args.validation_jobs < 1 or args.timeout <= 0 or args.max_cycles < 0:
        parser.error("jobs and timeout must be positive; max-cycles must be nonnegative")
    args.validation_semaphore = threading.Semaphore(args.validation_jobs)
    args.validator_modules = validator_manifest(VALIDATOR.parent)
    args.validator_sha256 = manifest_fingerprint(args.validator_modules)
    args.rejected_inputs = json.loads(args.rejected_inputs.read_text()) if args.rejected_inputs else {}
    if not isinstance(args.rejected_inputs, dict):
        parser.error("--rejected-inputs must contain a JSON object")
    for field in ("binary", "config", "trace_config", "trace_dir", "library_dir"):
        value = getattr(args, field)
        if value:
            setattr(args, field, value.resolve(strict=True))
    args.output = args.output.resolve()
    try:
        entries, skipped = trace_selection(args.trace_dir, selected_ids(args.kernel_ids))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    identity = {
        "binary": str(args.binary), "binary_sha256": sha256(args.binary),
        "runner_sha256": sha256(Path(__file__)),
        "validator_modules": args.validator_modules,
        "library_sha256": ({path.name: sha256(path) for path in sorted(args.library_dir.glob("*.so*"))
                            if path.is_file()} if args.library_dir else {}),
        "config": str(args.config), "config_sha256": sha256(args.config),
        "trace_config": str(args.trace_config) if args.trace_config else None,
        "trace_config_sha256": sha256(args.trace_config) if args.trace_config else None,
        "trace_dir": str(args.trace_dir), "kernelslist_sha256": sha256(args.trace_dir / "kernelslist.g"),
        "kernel_ids": [kernel for kernel, _ in entries], "max_cycles": args.max_cycles,
        "library_dir": str(args.library_dir) if args.library_dir else None,
        "rejected_inputs": args.rejected_inputs,
    }
    if args.resume:
        previous = json.loads((args.output / "provenance.json").read_text())
        if previous["identity"] != identity:
            parser.error("Resume input identity differs from the original run")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config").mkdir()
        shutil.copy2(args.config, args.output / "config/gpgpusim.config")
        if args.trace_config:
            shutil.copy2(args.trace_config, args.output / "config/trace.config")
        write_json(args.output / "provenance.json", {
            "identity": identity, "runner_sha256": sha256(Path(__file__)),
            "validator_sha256": args.validator_sha256, "argv": sys.argv,
            "skipped_non_kernel_entries": skipped,
            "semantics": "Independent cold-start kernels; logging on only; memcpy simulation disabled"})
    pinned = args.output / "tools" if args.resume else pin_tools(args.output, args.binary, args.library_dir)
    check_pinned_tools(pinned)
    args.binary = pinned / "accel-sim.out"
    args.validator_path = pinned / "validator/validate_l2_traffic.py"
    if args.library_dir:
        args.library_dir = pinned / "lib"
    config = args.output / "config/gpgpusim.config"
    trace_config = args.output / "config/trace.config" if args.trace_config else None
    max_merged = int(config_option(config, "-gpgpu_lrc_max_merged", 4))
    summary = {
        "complete": False, "passed": False, "selected_kernels": len(entries), "finished_kernels": 0,
        "passed_kernels": 0, "failed_kernels": 0, "unfinished_kernels": 0, "completed_simulations": 0, "kernels": [],
        "validated_source_counts": {}, "validated_record_counts": {},
        "run_semantics": "Each kernel is replayed independently with logging on and cold caches",
    }
    # A prior terminal marker describes the old attempt, not this resumed run.
    for marker in ("SUCCESS", "FAILED"):
        (args.output / marker).unlink(missing_ok=True)
    (args.output / "INCOMPLETE").write_text("Mobile traffic collection in progress\n")
    write_json(args.output / "summary.json", summary)
    sources, counts = Counter(), Counter()
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_kernel, args, kernel, source, config, trace_config, max_merged): kernel
                   for kernel, source in entries}
        for future in as_completed(futures):
            kernel = futures[future]
            try:
                status = future.result()
            except Exception as error:
                status = {"kernel_id": kernel, "kernel_completed": False, "passed": False,
                          "retryable": True, "failure": f"Runner error: {type(error).__name__}: {error}"}
                run = args.output / f"kernel-{kernel}"
                if run.is_dir():
                    write_json(run / "status.json", status)
            summary["kernels"].append(status)
            summary["finished_kernels"] += not status.get("retryable", False)
            summary["completed_simulations"] += bool(status.get("kernel_completed"))
            summary["unfinished_kernels" if status.get("retryable") else
                    "passed_kernels" if status.get("passed") else "failed_kernels"] += 1
            if status.get("passed"):
                sources.update(status.get("source_counts") or {})
                counts.update(status.get("counts") or {})
            summary["validated_source_counts"] = dict(sources)
            summary["validated_record_counts"] = dict(counts)
            summary["wall_seconds"] = time.monotonic() - start
            write_json(args.output / "summary.json", summary)
            print(f"{'PASS' if status.get('passed') else 'FAIL'} kernel {kernel} "
                  f"[{summary['finished_kernels']}/{len(entries)}]: "
                  f"{status.get('counts', status.get('failure', ''))}", flush=True)
    summary["complete"] = summary["finished_kernels"] == len(entries)
    summary["passed"] = summary["complete"] and summary["failed_kernels"] == 0
    if sha256(args.binary) != identity["binary_sha256"]:
        summary["passed"] = False
        summary["failure"] = "Simulator binary changed during collection"
    try:
        check_pinned_tools(pinned)
    except (ValueError, OSError) as error:
        summary["passed"] = False
        summary["failure"] = str(error)
    summary["kernels"].sort(key=lambda status: status["kernel_id"])
    write_json(args.output / "summary.json", summary)
    if summary["complete"]:
        (args.output / "INCOMPLETE").replace(args.output / ("SUCCESS" if summary["passed"] else "FAILED"))
    print(f"Finished: {summary['passed_kernels']} passed, {summary['failed_kernels']} failed; "
          f"{args.output / 'summary.json'}", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
