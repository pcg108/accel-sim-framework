#!/usr/bin/env python3
"""Create and finalize machine-readable Accel-Sim run manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(resolved), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    commit = git("rev-parse", "HEAD")
    if commit.returncode != 0:
        return {"root": str(resolved), "commit": None, "dirty": None}
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    full_status = git("status", "--porcelain", "--untracked-files=normal")
    remote = git("remote", "get-url", "origin")
    branch = git("branch", "--show-current")
    return {
        "root": str(resolved),
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() or None,
        "dirty": bool(full_status.stdout.strip()) if full_status.returncode == 0 else None,
        "dirty_tracked": (
            bool(tracked_status.stdout.strip()) if tracked_status.returncode == 0 else None
        ),
        "origin": remote.stdout.strip() if remote.returncode == 0 else None,
    }


def read_argv(path: Path) -> list[str]:
    raw = path.read_bytes()
    if not raw.endswith(b"\0"):
        raise ValueError(f"command argv file is not NUL terminated: {path}")
    return [item.decode("utf-8") for item in raw[:-1].split(b"\0")]


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def init_manifest(args: argparse.Namespace) -> int:
    simulator = args.simulator.resolve(strict=True)
    config = args.config.resolve(strict=True)
    input_selection = json.loads(args.input_selection.read_text(encoding="utf-8"))
    command = read_argv(args.command_argv)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "complete": False,
        "started_at": utc_now(),
        "finished_at": None,
        "exit_code": None,
        "workload": {
            "name": args.workload,
            "selected_kernel_ids": input_selection["selected_kernel_ids"],
            "selection_expression": args.selected_kernel_ids,
            "source_kernelslist": str(args.source_kernelslist.resolve(strict=True)),
            "staged_kernelslist": str(args.staged_kernelslist.resolve(strict=True)),
        },
        "logging": {
            "cache_traces": args.cache_traces == "on",
            "scheduler_trace": args.scheduler_trace == "on",
            "scheduler_sampling_core": 0 if args.scheduler_trace == "on" else None,
            "outputs": {
                "l1_trace": str(args.working_directory / "l1_trace"),
                "icache_status": str(args.working_directory / "icache_status"),
                "dcache_status": str(args.working_directory / "dcache_status"),
                "warp_assignments": str(args.working_directory / "warp_assignments"),
                "scheduler_traces": (
                    str(args.working_directory / "scheduler_traces")
                    if args.scheduler_trace == "on"
                    else None
                ),
            },
        },
        "command": {
            "argv": command,
            "working_directory": str(args.working_directory.resolve(strict=True)),
            "stdout": str(args.stdout),
            "stderr": str(args.stderr),
            "environment": {
                "LOG_CACHE_TRACES": "1" if args.cache_traces == "on" else None,
                "L1_TRACE_DIR": (
                    str(args.working_directory / "l1_trace")
                    if args.cache_traces == "on"
                    else None
                ),
                "SCHED_TRACE_DIR": (
                    str(args.working_directory / "scheduler_traces")
                    if args.scheduler_trace == "on"
                    else None
                ),
            },
        },
        "artifacts": {
            "output_directory": str(args.output_dir.resolve(strict=True)),
            "simulator": {"path": str(simulator), "sha256": sha256_file(simulator)},
            "config": {"path": str(config), "sha256": sha256_file(config)},
            "input_selection": input_selection,
        },
        "sources": {
            "framework": git_metadata(args.framework_root),
            "gpgpusim": git_metadata(args.gpgpusim_root),
            "harness": git_metadata(args.harness_root),
        },
        "container": {
            "image_ref": args.image_ref,
            "container_hostname": socket.gethostname(),
        },
        "host": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    write_json_atomic(args.manifest, manifest)
    return 0


def finalize_manifest(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest["exit_code"] = args.exit_code
    manifest["finished_at"] = utc_now()
    manifest["complete"] = args.exit_code == 0
    manifest["status"] = "succeeded" if args.exit_code == 0 else "failed"
    write_json_atomic(args.manifest, manifest)
    return 0


def abort_manifest(args: argparse.Namespace) -> int:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed",
        "complete": False,
        "started_at": None,
        "finished_at": utc_now(),
        "exit_code": args.exit_code,
        "failure_phase": args.failure_phase,
        "workload": {"name": args.workload},
        "artifacts": {"output_directory": str(args.output_dir.resolve(strict=True))},
        "container": {
            "image_ref": args.image_ref,
            "container_hostname": socket.gethostname(),
        },
    }
    write_json_atomic(args.manifest, manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--manifest", type=Path, required=True)
    init.add_argument("--workload", required=True)
    init.add_argument("--selected-kernel-ids", required=True)
    init.add_argument("--output-dir", type=Path, required=True)
    init.add_argument("--working-directory", type=Path, required=True)
    init.add_argument("--simulator", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--framework-root", type=Path, required=True)
    init.add_argument("--gpgpusim-root", type=Path, required=True)
    init.add_argument("--harness-root", type=Path, required=True)
    init.add_argument("--image-ref", required=True)
    init.add_argument("--source-kernelslist", type=Path, required=True)
    init.add_argument("--staged-kernelslist", type=Path, required=True)
    init.add_argument("--input-selection", type=Path, required=True)
    init.add_argument("--command-argv", type=Path, required=True)
    init.add_argument("--cache-traces", choices=("on", "off"), required=True)
    init.add_argument("--scheduler-trace", choices=("on", "off"), required=True)
    init.add_argument("--stdout", type=Path, required=True)
    init.add_argument("--stderr", type=Path, required=True)
    init.set_defaults(handler=init_manifest)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--exit-code", type=int, required=True)
    finalize.set_defaults(handler=finalize_manifest)

    abort = subparsers.add_parser("abort")
    abort.add_argument("--manifest", type=Path, required=True)
    abort.add_argument("--exit-code", type=int, required=True)
    abort.add_argument("--workload", required=True)
    abort.add_argument("--output-dir", type=Path, required=True)
    abort.add_argument("--image-ref", required=True)
    abort.add_argument("--failure-phase", required=True)
    abort.set_defaults(handler=abort_manifest)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
