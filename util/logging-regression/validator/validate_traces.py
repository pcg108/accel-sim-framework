#!/usr/bin/env python3
"""Stream-validate Accel-Sim logging output against a golden run.

The validator deliberately compares structure and invariants rather than exact
event ordering or cycle values.  Accel-Sim 2.0 can legitimately change those
while the logging contract remains intact.
"""

from __future__ import annotations

import argparse
import json
import lzma
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


VERSION = "1.0.0"
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

KERNEL_DIR_RE = re.compile(r"^kernel_(\d+)_(.+)$")
SHADER_DIR_RE = re.compile(r"^shader_(\d+)$")
SCHEDULER_DIR_RE = re.compile(r"^scheduler_(\d+)$")
WARP_FILE_RE = re.compile(r"^shader_(\d+)\.txt$")
TRACE_FILE_RE = re.compile(r"^kernel[-_](\d+)\.(?:traceg(?:\.xz)?|tracez)$")

CTA_RE = re.compile(
    r"^CTA\s+(\d+)\s+\(ctaid=(\d+)\)\s+assigned to shader core\s+(\d+):$"
)
WARP_RE = re.compile(
    r"^\s*Warp\s+(\d+)\s+\(warp_in_cta=(\d+)\s+dynamic_warp_id=(\d+)\)"
    r"\s+->\s+Scheduler\s+(\d+)\s*$"
)
DIM_RE = re.compile(r"^-(grid|block) dim\s*=\s*\((\d+),(\d+),(\d+)\)\s*$")
TRACE_KERNEL_ID_RE = re.compile(r"^-kernel id\s*=\s*(\d+)\s*$")

CACHE_STATUSES = {
    "HIT",
    "HIT_RESERVED",
    "MISS",
    "RESERVATION_FAIL",
    "SECTOR_MISS",
    "MSHR_HIT",
    "WRITE_ALLOCATED",
}
ACCESS_TYPES = {"READ", "WRITE", "WRITEBACK"}

L1_REQUEST_FIELDS = {
    "request_uid",
    "PC",
    "dynamic_warp",
    "inst_ordinal",
    "addr",
    "subpartition",
    "set_index",
    "tag",
    "sector_mask",
    "size",
    "type",
    "l1_to_l2_cycle",
}
L2_TIMING_LEGACY_FIELDS = {
    "request_uid",
    "start_cycle",
    "end_cycle",
    "elapsed_cycle",
}
L2_TIMING_EXTENDED_FIELDS = L2_TIMING_LEGACY_FIELDS | {
    "transport_uid",
    "service_uid",
    "addr",
    "sector_mask",
    "size",
}
ICACHE_FIELDS = {"PC", "warp", "status"}
DCACHE_FIELDS = {"request_uid", "PC", "warp", "addr", "size", "status"}

FAMILIES = ("l2_trace", "icache_status", "dcache_status", "warp_assignments")


@dataclass
class Issues:
    max_samples: int = 100
    error_count: int = 0
    warning_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.error_count += 1
        if len(self.errors) < self.max_samples:
            self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warning_count += 1
        if len(self.warnings) < self.max_samples:
            self.warnings.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "error_samples": self.errors,
            "warning_samples": self.warnings,
            "samples_truncated": {
                "errors": self.error_count > len(self.errors),
                "warnings": self.warning_count > len(self.warnings),
            },
        }


@dataclass(frozen=True)
class Limits:
    num_shaders: int
    num_schedulers: int
    max_warps_per_shader: int
    join_policy: str


def kernel_info(name: str) -> tuple[int, str] | None:
    match = KERNEL_DIR_RE.fullmatch(name)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def parse_kernel_ids(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            kernel_id = int(item, 10)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid kernel id {item!r}") from exc
        if kernel_id < 0:
            raise argparse.ArgumentTypeError("kernel IDs must be non-negative")
        result.add(kernel_id)
    if not result:
        raise argparse.ArgumentTypeError("kernel ID list is empty")
    return result


def resolve_family(root: Path, requested: str | None, defaults: tuple[str, ...]) -> Path:
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else root / path
    for dirname in defaults:
        path = root / dirname
        if path.is_dir():
            return path
    return root / defaults[0]


def discover_kernel_dirs(root: Path, side: str, family: str, issues: Issues) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not root.is_dir():
        issues.error(f"{side}:{family}: missing family directory: {root}")
        return result
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        parsed = kernel_info(path.name)
        if parsed is None:
            issues.error(f"{side}:{family}: invalid kernel directory name: {path.name}")
            continue
        kernel_id, kernel_name = parsed
        if kernel_id in (UINT32_MAX, UINT64_MAX):
            issues.error(f"{side}:{family}: sentinel kernel ID in {path.name}")
        if "unknown_kernel" in kernel_name.lower() or kernel_name.strip() == "":
            issues.error(f"{side}:{family}: sentinel/empty kernel name in {path.name}")
        if kernel_id in result:
            issues.error(
                f"{side}:{family}: duplicate directories for kernel {kernel_id}: "
                f"{result[kernel_id].name!r} and {path.name!r}"
            )
        else:
            result[kernel_id] = path
    return result


def collect_files(root: Path, selected: set[int], side: str, family: str, issues: Issues) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not root.is_dir():
        return files
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if not rel.parts:
            continue
        parsed = kernel_info(rel.parts[0])
        if parsed is None:
            issues.error(f"{side}:{family}: file outside a kernel directory: {rel}")
            continue
        if parsed[0] not in selected:
            continue
        files[rel.as_posix()] = path
    return files


def parse_kv_line(
    line: str,
    required: set[str],
    optional: set[str],
    location: str,
    issues: Issues,
) -> dict[str, str] | None:
    values: dict[str, str] = {}
    for token in line.split():
        if "=" not in token:
            issues.error(f"{location}: expected key=value token, got {token!r}")
            return None
        key, value = token.split("=", 1)
        if not key or not value:
            issues.error(f"{location}: malformed token {token!r}")
            return None
        if key in values:
            issues.error(f"{location}: duplicate field {key!r}")
            return None
        values[key] = value
    missing = required - values.keys()
    extra = values.keys() - required - optional
    if missing:
        issues.error(f"{location}: missing fields: {', '.join(sorted(missing))}")
        return None
    if extra:
        issues.error(f"{location}: unexpected fields: {', '.join(sorted(extra))}")
        return None
    return values


def uint_value(value: str, field_name: str, location: str, issues: Issues, *, hexadecimal: bool = False) -> int | None:
    if hexadecimal and not re.fullmatch(r"0[xX][0-9a-fA-F]+", value):
        issues.error(f"{location}: {field_name} must be hexadecimal, got {value!r}")
        return None
    try:
        number = int(value, 0 if value.lower().startswith("0x") else 10)
    except ValueError:
        issues.error(f"{location}: {field_name} is not an unsigned integer: {value!r}")
        return None
    if number < 0:
        issues.error(f"{location}: {field_name} is negative: {number}")
        return None
    return number


def update_minmax(stats: dict[str, Any], prefix: str, value: int) -> None:
    min_key = f"{prefix}_min"
    max_key = f"{prefix}_max"
    stats[min_key] = value if min_key not in stats else min(stats[min_key], value)
    stats[max_key] = value if max_key not in stats else max(stats[max_key], value)


def add_hist(stats: dict[str, Any], key: str, value: str | int) -> None:
    hist = stats.setdefault(key, Counter())
    hist[str(value)] += 1


def new_l1_kernel() -> dict[str, Any]:
    return {
        "files": 0,
        "request_records": 0,
        "timing_records": 0,
        "type_hist": Counter(),
        "size_hist": Counter(),
        "sector_mask_hist": Counter(),
        "address_offset_128b_hist": Counter(),
        "timing_schema_hist": Counter(),
        "timing_identity_hist": Counter(),
        "timing_size_hist": Counter(),
        "timing_sector_mask_hist": Counter(),
        "timing_address_offset_128b_hist": Counter(),
        "join": {
            "unique_requests": 0,
            "matched_requests": 0,
            "missing_requests": 0,
            "orphan_timing_records": 0,
            "timing_records": 0,
        },
    }


def new_cache_kernel() -> dict[str, Any]:
    return {
        "files": 0,
        "records": 0,
        "status_hist": Counter(),
        "size_hist": Counter(),
        "address_offset_128b_hist": Counter(),
    }


def new_warp_kernel() -> dict[str, Any]:
    return {
        "files": 0,
        "ctas": 0,
        "warps": 0,
        "shader_cta_hist": Counter(),
        "shader_warp_hist": Counter(),
        "scheduler_hist": Counter(),
    }


def parse_path_context(
    rel: str,
    expected_basename: str,
    side: str,
    family: str,
    limits: Limits,
    issues: Issues,
) -> tuple[int, int, int] | None:
    parts = Path(rel).parts
    if len(parts) != 4 or parts[-1] != expected_basename:
        issues.error(
            f"{side}:{family}:{rel}: expected "
            f"kernel_*/shader_*/scheduler_*/{expected_basename}"
        )
        return None
    parsed_kernel = kernel_info(parts[0])
    shader_match = SHADER_DIR_RE.fullmatch(parts[1])
    scheduler_match = SCHEDULER_DIR_RE.fullmatch(parts[2])
    if parsed_kernel is None or shader_match is None or scheduler_match is None:
        issues.error(f"{side}:{family}:{rel}: malformed kernel/shader/scheduler path")
        return None
    shader = int(shader_match.group(1))
    scheduler = int(scheduler_match.group(1))
    if shader >= limits.num_shaders:
        issues.error(f"{side}:{family}:{rel}: shader {shader} is outside [0,{limits.num_shaders - 1}]")
    if scheduler >= limits.num_schedulers:
        issues.error(
            f"{side}:{family}:{rel}: scheduler {scheduler} is outside "
            f"[0,{limits.num_schedulers - 1}]"
        )
    return parsed_kernel[0], shader, scheduler


def parse_l1_request(
    line: str, location: str, stats: dict[str, Any], issues: Issues
) -> int | None:
    values = parse_kv_line(line, L1_REQUEST_FIELDS, set(), location, issues)
    if values is None:
        return None
    decimal_fields = (
        "request_uid",
        "dynamic_warp",
        "inst_ordinal",
        "subpartition",
        "set_index",
        "size",
        "l1_to_l2_cycle",
    )
    parsed: dict[str, int] = {}
    for key in decimal_fields:
        number = uint_value(values[key], key, location, issues)
        if number is None:
            return None
        parsed[key] = number
    for key in ("PC", "addr", "tag", "sector_mask"):
        number = uint_value(values[key], key, location, issues, hexadecimal=True)
        if number is None:
            return None
        parsed[key] = number
    if parsed["inst_ordinal"] in (UINT32_MAX, UINT64_MAX):
        issues.error(f"{location}: sentinel inst_ordinal={parsed['inst_ordinal']}")
    if parsed["size"] == 0:
        issues.error(f"{location}: size must be positive")
    if parsed["sector_mask"] == 0:
        issues.error(f"{location}: sector_mask must be nonzero")
    if values["type"] not in ACCESS_TYPES:
        issues.error(f"{location}: invalid access type {values['type']!r}")

    stats["request_records"] += 1
    add_hist(stats, "type_hist", values["type"])
    add_hist(stats, "size_hist", parsed["size"])
    add_hist(stats, "sector_mask_hist", values["sector_mask"].lower())
    add_hist(stats, "address_offset_128b_hist", parsed["addr"] & 0x7F)
    update_minmax(stats, "address", parsed["addr"])
    update_minmax(stats, "l1_to_l2_cycle", parsed["l1_to_l2_cycle"])
    return parsed["request_uid"]


def parse_l2_timing(
    line: str, location: str, stats: dict[str, Any], issues: Issues
) -> int | None:
    values = parse_kv_line(
        line,
        L2_TIMING_LEGACY_FIELDS,
        L2_TIMING_EXTENDED_FIELDS - L2_TIMING_LEGACY_FIELDS,
        location,
        issues,
    )
    if values is None:
        return None
    fields = set(values)
    if fields == L2_TIMING_LEGACY_FIELDS:
        schema = "legacy"
    elif fields == L2_TIMING_EXTENDED_FIELDS:
        schema = "extended"
    else:
        missing_extended = sorted(L2_TIMING_EXTENDED_FIELDS - fields)
        unexpected = sorted(fields - L2_TIMING_EXTENDED_FIELDS)
        issues.error(
            f"{location}: timing fields must exactly match the legacy or extended schema; "
            f"missing_extended={missing_extended}, unexpected={unexpected}"
        )
        return None
    parsed: dict[str, int] = {}
    for key in ("request_uid", "start_cycle", "end_cycle", "elapsed_cycle"):
        number = uint_value(values[key], key, location, issues)
        if number is None:
            return None
        parsed[key] = number
    if parsed["request_uid"] in (UINT32_MAX, UINT64_MAX):
        issues.error(f"{location}: sentinel request_uid={parsed['request_uid']}")
    if schema == "extended":
        for key in ("transport_uid", "service_uid", "size"):
            number = uint_value(values[key], key, location, issues)
            if number is None:
                return None
            parsed[key] = number
        for key in ("addr", "sector_mask"):
            number = uint_value(values[key], key, location, issues, hexadecimal=True)
            if number is None:
                return None
            parsed[key] = number
        for key in ("transport_uid", "service_uid"):
            if parsed[key] in (UINT32_MAX, UINT64_MAX):
                issues.error(f"{location}: sentinel {key}={parsed[key]}")
        if parsed["size"] == 0:
            issues.error(f"{location}: size must be positive")
        if parsed["sector_mask"] == 0:
            issues.error(f"{location}: sector_mask must be nonzero")
    if parsed["end_cycle"] < parsed["start_cycle"]:
        issues.error(
            f"{location}: end_cycle {parsed['end_cycle']} precedes "
            f"start_cycle {parsed['start_cycle']}"
        )
    expected_elapsed = parsed["end_cycle"] - parsed["start_cycle"]
    if parsed["end_cycle"] >= parsed["start_cycle"] and parsed["elapsed_cycle"] != expected_elapsed:
        issues.error(
            f"{location}: elapsed_cycle={parsed['elapsed_cycle']}, expected {expected_elapsed}"
        )
    stats["timing_records"] += 1
    add_hist(stats, "timing_schema_hist", schema)
    if schema == "extended":
        if parsed["request_uid"] == parsed["transport_uid"] == parsed["service_uid"]:
            identity = "ordinary"
        elif parsed["service_uid"] != parsed["transport_uid"]:
            identity = "shared_service"
        else:
            identity = "sector_split"
        add_hist(stats, "timing_identity_hist", identity)
        add_hist(stats, "timing_size_hist", parsed["size"])
        add_hist(stats, "timing_sector_mask_hist", values["sector_mask"].lower())
        add_hist(stats, "timing_address_offset_128b_hist", parsed["addr"] & 0x7F)
        update_minmax(stats, "timing_address", parsed["addr"])
    update_minmax(stats, "start_cycle", parsed["start_cycle"])
    update_minmax(stats, "end_cycle", parsed["end_cycle"])
    update_minmax(stats, "elapsed_cycle", parsed["elapsed_cycle"])
    return parsed["request_uid"]


def iter_nonblank(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if line:
                yield line_number, line


def validate_l1(
    root: Path,
    selected: set[int],
    side: str,
    limits: Limits,
    issues: Issues,
) -> dict[str, Any]:
    files = collect_files(root, selected, side, "l2_trace", issues)
    result: dict[str, Any] = {
        "root": str(root),
        "files": len(files),
        "file_records": {},
        "kernels": defaultdict(new_l1_kernel),
    }
    request_rels: list[str] = []
    timing_rels: set[str] = set()
    for rel in sorted(files):
        basename = Path(rel).name
        if basename == "l1_to_l2_requests.txt":
            request_rels.append(rel)
        elif basename == "l2_to_icnt_timing.txt":
            timing_rels.add(rel)
        else:
            issues.error(f"{side}:l2_trace:{rel}: unexpected file")

    processed_timing: set[str] = set()
    for request_rel in request_rels:
        context = parse_path_context(
            request_rel, "l1_to_l2_requests.txt", side, "l2_trace", limits, issues
        )
        if context is None:
            continue
        kernel_id, _, _ = context
        stats = result["kernels"][kernel_id]
        stats["files"] += 1
        request_uids: set[int] = set()
        valid_records = 0
        try:
            for line_number, line in iter_nonblank(files[request_rel]):
                uid = parse_l1_request(
                    line, f"{side}:l2_trace:{request_rel}:{line_number}", stats, issues
                )
                if uid is None:
                    continue
                valid_records += 1
                if uid in request_uids:
                    issues.error(
                        f"{side}:l2_trace:{request_rel}:{line_number}: duplicate request_uid={uid}"
                    )
                request_uids.add(uid)
        except (OSError, UnicodeError) as exc:
            issues.error(f"{side}:l2_trace:{request_rel}: cannot read file: {exc}")
        result["file_records"][request_rel] = valid_records

        timing_rel = str(Path(request_rel).with_name("l2_to_icnt_timing.txt").as_posix())
        matched_uids: set[int] = set()
        orphan_records = 0
        timing_records = 0
        if timing_rel in files:
            processed_timing.add(timing_rel)
            stats["files"] += 1
            try:
                for line_number, line in iter_nonblank(files[timing_rel]):
                    uid = parse_l2_timing(
                        line,
                        f"{side}:l2_trace:{timing_rel}:{line_number}",
                        stats,
                        issues,
                    )
                    if uid is None:
                        continue
                    timing_records += 1
                    if uid in request_uids:
                        matched_uids.add(uid)
                    else:
                        orphan_records += 1
            except (OSError, UnicodeError) as exc:
                issues.error(f"{side}:l2_trace:{timing_rel}: cannot read file: {exc}")
            result["file_records"][timing_rel] = timing_records
        else:
            issues.error(f"{side}:l2_trace:{request_rel}: paired timing file is missing")

        missing_requests = len(request_uids - matched_uids)
        join = stats["join"]
        join["unique_requests"] += len(request_uids)
        join["matched_requests"] += len(matched_uids)
        join["missing_requests"] += missing_requests
        join["orphan_timing_records"] += orphan_records
        join["timing_records"] += timing_records
        if missing_requests or orphan_records:
            message = (
                f"{side}:l2_trace:{Path(request_rel).parent}: incomplete UID join: "
                f"missing_requests={missing_requests}, orphan_timing_records={orphan_records}"
            )
            if limits.join_policy == "strict":
                issues.error(message)
            else:
                issues.warning(message)

    # Timing files without a request peer still need schema validation.
    for timing_rel in sorted(timing_rels - processed_timing):
        context = parse_path_context(
            timing_rel, "l2_to_icnt_timing.txt", side, "l2_trace", limits, issues
        )
        if context is None:
            continue
        kernel_id, _, _ = context
        stats = result["kernels"][kernel_id]
        stats["files"] += 1
        records = 0
        try:
            for line_number, line in iter_nonblank(files[timing_rel]):
                uid = parse_l2_timing(
                    line, f"{side}:l2_trace:{timing_rel}:{line_number}", stats, issues
                )
                if uid is not None:
                    records += 1
        except (OSError, UnicodeError) as exc:
            issues.error(f"{side}:l2_trace:{timing_rel}: cannot read file: {exc}")
        result["file_records"][timing_rel] = records
        message = f"{side}:l2_trace:{timing_rel}: request peer is missing; {records} orphan records"
        if limits.join_policy == "strict":
            issues.error(message)
        else:
            issues.warning(message)

    result["relative_files"] = set(files)
    return result


def validate_cache(
    root: Path,
    selected: set[int],
    side: str,
    family: str,
    limits: Limits,
    issues: Issues,
) -> dict[str, Any]:
    is_icache = family == "icache_status"
    expected_basename = "icache_access.txt" if is_icache else "dcache_access.txt"
    required = ICACHE_FIELDS if is_icache else DCACHE_FIELDS
    files = collect_files(root, selected, side, family, issues)
    result: dict[str, Any] = {
        "root": str(root),
        "files": len(files),
        "file_records": {},
        "kernels": defaultdict(new_cache_kernel),
        "relative_files": set(files),
    }
    for rel, path in sorted(files.items()):
        context = parse_path_context(rel, expected_basename, side, family, limits, issues)
        if context is None:
            continue
        kernel_id, _, _ = context
        stats = result["kernels"][kernel_id]
        stats["files"] += 1
        records = 0
        try:
            for line_number, line in iter_nonblank(path):
                location = f"{side}:{family}:{rel}:{line_number}"
                values = parse_kv_line(line, required, set(), location, issues)
                if values is None:
                    continue
                pc = uint_value(values["PC"], "PC", location, issues, hexadecimal=True)
                warp = uint_value(values["warp"], "warp", location, issues)
                if pc is None or warp is None:
                    continue
                if warp >= limits.max_warps_per_shader:
                    issues.error(
                        f"{location}: warp {warp} is outside "
                        f"[0,{limits.max_warps_per_shader - 1}]"
                    )
                status = values["status"]
                if status not in CACHE_STATUSES:
                    issues.error(f"{location}: invalid cache status {status!r}")
                if not is_icache:
                    uid = uint_value(values["request_uid"], "request_uid", location, issues)
                    addr = uint_value(values["addr"], "addr", location, issues, hexadecimal=True)
                    size = uint_value(values["size"], "size", location, issues)
                    if uid is None or addr is None or size is None:
                        continue
                    if size == 0:
                        issues.error(f"{location}: size must be positive")
                    add_hist(stats, "size_hist", size)
                    add_hist(stats, "address_offset_128b_hist", addr & 0x7F)
                    update_minmax(stats, "address", addr)
                records += 1
                stats["records"] += 1
                add_hist(stats, "status_hist", status)
                update_minmax(stats, "pc", pc)
        except (OSError, UnicodeError) as exc:
            issues.error(f"{side}:{family}:{rel}: cannot read file: {exc}")
        result["file_records"][rel] = records
    return result


def validate_warps(
    root: Path,
    selected: set[int],
    side: str,
    limits: Limits,
    issues: Issues,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    files = collect_files(root, selected, side, "warp_assignments", issues)
    result: dict[str, Any] = {
        "root": str(root),
        "files": len(files),
        "file_records": {},
        "kernels": defaultdict(new_warp_kernel),
        "relative_files": set(files),
    }
    details: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "cta_ids": set(),
            "assignments": set(),
            "dynamic_warps": set(),
            "warps_per_cta": Counter(),
        }
    )
    for rel, path in sorted(files.items()):
        parts = Path(rel).parts
        file_match = WARP_FILE_RE.fullmatch(parts[-1]) if len(parts) == 2 else None
        parsed_kernel = kernel_info(parts[0]) if parts else None
        if len(parts) != 2 or file_match is None or parsed_kernel is None:
            issues.error(
                f"{side}:warp_assignments:{rel}: expected kernel_*/shader_*.txt"
            )
            continue
        kernel_id = parsed_kernel[0]
        shader = int(file_match.group(1))
        if shader >= limits.num_shaders:
            issues.error(
                f"{side}:warp_assignments:{rel}: shader {shader} is outside "
                f"[0,{limits.num_shaders - 1}]"
            )
        stats = result["kernels"][kernel_id]
        kernel_details = details[kernel_id]
        stats["files"] += 1
        records = 0
        current_ctaid: int | None = None
        current_has_warp = False
        try:
            with path.open("r", encoding="utf-8", errors="strict") as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    line = raw_line.rstrip("\r\n")
                    if not line.strip():
                        continue
                    location = f"{side}:warp_assignments:{rel}:{line_number}"
                    cta_match = CTA_RE.fullmatch(line)
                    if cta_match is not None:
                        if current_ctaid is not None and not current_has_warp:
                            issues.error(
                                f"{location}: previous CTA {current_ctaid} has no warp records"
                            )
                        _, ctaid_text, core_text = cta_match.groups()
                        ctaid = int(ctaid_text)
                        core = int(core_text)
                        if core != shader:
                            issues.error(
                                f"{location}: header core {core} does not match file shader {shader}"
                            )
                        if ctaid in kernel_details["cta_ids"]:
                            issues.error(f"{location}: duplicate ctaid={ctaid}")
                        else:
                            kernel_details["cta_ids"].add(ctaid)
                            stats["ctas"] += 1
                            add_hist(stats, "shader_cta_hist", shader)
                        current_ctaid = ctaid
                        current_has_warp = False
                        continue
                    warp_match = WARP_RE.fullmatch(line)
                    if warp_match is None:
                        issues.error(f"{location}: malformed warp-assignment line")
                        continue
                    if current_ctaid is None:
                        issues.error(f"{location}: warp record appears before a CTA header")
                        continue
                    warp_id, warp_in_cta, dynamic_warp, scheduler = map(
                        int, warp_match.groups()
                    )
                    if warp_id >= limits.max_warps_per_shader:
                        issues.error(
                            f"{location}: warp {warp_id} is outside "
                            f"[0,{limits.max_warps_per_shader - 1}]"
                        )
                    if scheduler >= limits.num_schedulers:
                        issues.error(
                            f"{location}: scheduler {scheduler} is outside "
                            f"[0,{limits.num_schedulers - 1}]"
                        )
                    assignment = (current_ctaid, warp_in_cta)
                    if assignment in kernel_details["assignments"]:
                        issues.error(
                            f"{location}: duplicate assignment for ctaid={current_ctaid}, "
                            f"warp_in_cta={warp_in_cta}"
                        )
                    else:
                        kernel_details["assignments"].add(assignment)
                    dynamic_key = (shader, dynamic_warp)
                    if dynamic_key in kernel_details["dynamic_warps"]:
                        issues.error(
                            f"{location}: duplicate dynamic_warp_id={dynamic_warp} on shader {shader}"
                        )
                    else:
                        kernel_details["dynamic_warps"].add(dynamic_key)
                    kernel_details["warps_per_cta"][current_ctaid] += 1
                    stats["warps"] += 1
                    add_hist(stats, "shader_warp_hist", shader)
                    add_hist(stats, "scheduler_hist", scheduler)
                    records += 1
                    current_has_warp = True
                if current_ctaid is not None and not current_has_warp:
                    issues.error(
                        f"{side}:warp_assignments:{rel}: final CTA {current_ctaid} has no warp records"
                    )
        except (OSError, UnicodeError) as exc:
            issues.error(f"{side}:warp_assignments:{rel}: cannot read file: {exc}")
        result["file_records"][rel] = records
    return result, details


def trace_text_lines(path: Path) -> Iterator[str]:
    if path.name.endswith(".xz"):
        with lzma.open(path, "rt", encoding="utf-8", errors="strict") as handle:
            yield from handle
        return
    if path.suffix == ".tracez":
        zstd = shutil.which("zstd")
        if zstd is None:
            raise OSError("zstd executable is required to inspect .tracez headers")
        process = subprocess.Popen(
            [zstd, "-dc", "--", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        assert process.stdout is not None
        try:
            yield from process.stdout
        finally:
            process.stdout.close()
            process.terminate()
            process.wait(timeout=5)
        return
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        yield from handle


def read_trace_metadata(path: Path, expected_kernel_id: int) -> dict[str, Any]:
    trace_kernel_id: int | None = None
    grid: tuple[int, int, int] | None = None
    block: tuple[int, int, int] | None = None
    for line_index, raw_line in enumerate(trace_text_lines(path)):
        if line_index > 100:
            break
        line = raw_line.strip()
        id_match = TRACE_KERNEL_ID_RE.fullmatch(line)
        if id_match:
            trace_kernel_id = int(id_match.group(1))
        dim_match = DIM_RE.fullmatch(line)
        if dim_match:
            dims = tuple(int(dim_match.group(index)) for index in (2, 3, 4))
            if dim_match.group(1) == "grid":
                grid = dims
            else:
                block = dims
        if trace_kernel_id is not None and grid is not None and block is not None:
            break
    if grid is None or block is None:
        raise ValueError("trace header lacks grid/block dimensions")
    if trace_kernel_id is not None and trace_kernel_id != expected_kernel_id:
        raise ValueError(
            f"trace header kernel ID {trace_kernel_id} does not match filename ID {expected_kernel_id}"
        )
    threads = math.prod(block)
    return {
        "path": str(path),
        "grid": list(grid),
        "block": list(block),
        "expected_ctas": math.prod(grid),
        "expected_warps_per_cta": (threads + 31) // 32,
        "expected_warps": math.prod(grid) * ((threads + 31) // 32),
    }


def discover_trace_metadata(
    golden_root: Path,
    requested_trace_dir: str | None,
    selected: set[int],
    enabled: bool,
    issues: Issues,
) -> dict[int, dict[str, Any]]:
    if not enabled:
        return {}
    trace_root = Path(requested_trace_dir) if requested_trace_dir else golden_root / "traces"
    if not trace_root.is_dir():
        issues.warning(f"trace-grid check skipped: trace directory does not exist: {trace_root}")
        return {}
    candidates: dict[int, list[Path]] = defaultdict(list)
    for path in trace_root.rglob("kernel-*"):
        if not path.is_file():
            continue
        match = TRACE_FILE_RE.fullmatch(path.name)
        if match is not None and int(match.group(1)) in selected:
            candidates[int(match.group(1))].append(path)
    metadata: dict[int, dict[str, Any]] = {}
    for kernel_id in sorted(selected):
        paths = candidates.get(kernel_id, [])
        if not paths:
            issues.warning(f"trace-grid check: no trace file found for kernel {kernel_id}")
            continue
        if len(paths) > 1:
            issues.warning(
                f"trace-grid check: multiple trace files for kernel {kernel_id}; "
                f"using {sorted(paths)[0]}"
            )
        path = sorted(paths)[0]
        try:
            metadata[kernel_id] = read_trace_metadata(path, kernel_id)
        except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
            issues.error(f"trace-grid check: cannot read {path}: {exc}")
    return metadata


def check_warp_grid(
    side: str,
    details: dict[int, dict[str, Any]],
    metadata: dict[int, dict[str, Any]],
    issues: Issues,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for kernel_id, expected in sorted(metadata.items()):
        actual = details.get(
            kernel_id,
            {"cta_ids": set(), "assignments": set(), "warps_per_cta": Counter()},
        )
        cta_ids = actual["cta_ids"]
        assignments = actual["assignments"]
        expected_ctas = expected["expected_ctas"]
        expected_warps_per_cta = expected["expected_warps_per_cta"]
        missing_ctas = sorted(set(range(expected_ctas)) - cta_ids)
        out_of_range_ctas = sorted(ctaid for ctaid in cta_ids if ctaid >= expected_ctas)
        wrong_warp_ctas = sorted(
            ctaid
            for ctaid, count in actual["warps_per_cta"].items()
            if count != expected_warps_per_cta
        )
        passed = (
            len(cta_ids) == expected_ctas
            and len(assignments) == expected["expected_warps"]
            and not missing_ctas
            and not out_of_range_ctas
            and not wrong_warp_ctas
        )
        if not passed:
            issues.error(
                f"{side}:warp_assignments:kernel_{kernel_id}: grid mismatch: "
                f"ctas={len(cta_ids)}/{expected_ctas}, "
                f"warps={len(assignments)}/{expected['expected_warps']}, "
                f"missing_ctas={missing_ctas[:10]}, out_of_range_ctas={out_of_range_ctas[:10]}, "
                f"wrong_warp_count_ctas={wrong_warp_ctas[:10]}"
            )
        checks[str(kernel_id)] = {
            **expected,
            "actual_ctas": len(cta_ids),
            "actual_warps": len(assignments),
            "missing_ctas_count": len(missing_ctas),
            "out_of_range_ctas_count": len(out_of_range_ctas),
            "wrong_warp_count_ctas": len(wrong_warp_ctas),
            "passed": passed,
        }
    return checks


def compare_file_sets(
    family: str,
    candidate: dict[str, Any],
    golden: dict[str, Any],
    issues: Issues,
) -> dict[str, Any]:
    candidate_files = candidate["relative_files"]
    golden_files = golden["relative_files"]
    missing = sorted(golden_files - candidate_files)
    extra = sorted(candidate_files - golden_files)
    if missing or extra:
        issues.error(
            f"comparison:{family}: relative file set differs: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    empty_where_golden_nonempty: list[str] = []
    for rel in sorted(candidate_files & golden_files):
        if golden["file_records"].get(rel, 0) > 0 and candidate["file_records"].get(rel, 0) == 0:
            empty_where_golden_nonempty.append(rel)
    if empty_where_golden_nonempty:
        issues.error(
            f"comparison:{family}: {len(empty_where_golden_nonempty)} candidate files are empty "
            "where golden files contain records"
        )
    return {
        "passed": not missing and not extra and not empty_where_golden_nonempty,
        "missing_files_count": len(missing),
        "extra_files_count": len(extra),
        "missing_file_samples": missing[:100],
        "extra_file_samples": extra[:100],
        "empty_where_golden_nonempty_count": len(empty_where_golden_nonempty),
        "empty_where_golden_nonempty_samples": empty_where_golden_nonempty[:100],
    }


def compare_kernel_dirs(
    family: str,
    candidate_dirs: dict[int, Path],
    golden_dirs: dict[int, Path],
    selected: set[int],
    issues: Issues,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for kernel_id in sorted(selected):
        candidate = candidate_dirs.get(kernel_id)
        golden = golden_dirs.get(kernel_id)
        passed = candidate is not None and golden is not None and candidate.name == golden.name
        if not passed:
            issues.error(
                f"comparison:{family}: kernel {kernel_id} directory mismatch: "
                f"candidate={candidate.name if candidate else None!r}, "
                f"golden={golden.name if golden else None!r}"
            )
        comparison[str(kernel_id)] = {
            "candidate": candidate.name if candidate else None,
            "golden": golden.name if golden else None,
            "passed": passed,
        }
    return comparison


def numeric_delta(candidate: int | None, golden: int | None) -> dict[str, Any]:
    delta = None if candidate is None or golden is None else candidate - golden
    percent = None
    if delta is not None and golden not in (None, 0):
        percent = 100.0 * delta / golden
    return {
        "candidate": candidate,
        "golden": golden,
        "delta": delta,
        "percent_delta": percent,
    }


def histogram_delta(candidate: Any, golden: Any) -> dict[str, Any]:
    candidate_counter = Counter(candidate or {})
    golden_counter = Counter(golden or {})
    return {
        str(key): numeric_delta(candidate_counter.get(key, 0), golden_counter.get(key, 0))
        for key in sorted(candidate_counter.keys() | golden_counter.keys(), key=str)
    }


def semantic_comparison(
    family: str,
    candidate: dict[str, Any],
    golden: dict[str, Any],
    selected: set[int],
) -> dict[str, Any]:
    if family == "l2_trace":
        metrics = (
            "request_records",
            "timing_records",
            "address_min",
            "address_max",
            "l1_to_l2_cycle_min",
            "l1_to_l2_cycle_max",
            "elapsed_cycle_min",
            "elapsed_cycle_max",
        )
        histograms = (
            "type_hist",
            "size_hist",
            "sector_mask_hist",
            "address_offset_128b_hist",
            "timing_schema_hist",
            "timing_identity_hist",
            "timing_size_hist",
            "timing_sector_mask_hist",
            "timing_address_offset_128b_hist",
        )
    elif family == "icache_status":
        metrics = ("records", "pc_min", "pc_max")
        histograms = ("status_hist",)
    elif family == "dcache_status":
        metrics = ("records", "pc_min", "pc_max", "address_min", "address_max")
        histograms = ("status_hist", "size_hist", "address_offset_128b_hist")
    else:
        metrics = ("ctas", "warps")
        histograms = ("shader_cta_hist", "shader_warp_hist", "scheduler_hist")

    result: dict[str, Any] = {}
    for kernel_id in sorted(selected):
        candidate_stats = candidate["kernels"].get(kernel_id, {})
        golden_stats = golden["kernels"].get(kernel_id, {})
        result[str(kernel_id)] = {
            "metrics": {
                metric: numeric_delta(candidate_stats.get(metric), golden_stats.get(metric))
                for metric in metrics
            },
            "histograms": {
                histogram: histogram_delta(
                    candidate_stats.get(histogram), golden_stats.get(histogram)
                )
                for histogram in histograms
            },
        }
        if family == "l2_trace":
            candidate_join = candidate_stats.get("join", {})
            golden_join = golden_stats.get("join", {})
            result[str(kernel_id)]["join"] = {
                key: numeric_delta(candidate_join.get(key), golden_join.get(key))
                for key in (
                    "unique_requests",
                    "matched_requests",
                    "missing_requests",
                    "orphan_timing_records",
                    "timing_records",
                )
            }
    return result


def plain(value: Any) -> Any:
    if isinstance(value, Counter):
        return {str(key): value[key] for key in sorted(value, key=str)}
    if isinstance(value, defaultdict):
        return {str(key): plain(item) for key, item in sorted(value.items())}
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def kernel_metric(family: str, stats: dict[str, Any]) -> str:
    if family == "l2_trace":
        return f"{stats.get('request_records', 0)}/{stats.get('timing_records', 0)}"
    if family in ("icache_status", "dcache_status"):
        return str(stats.get("records", 0))
    return f"{stats.get('ctas', 0)}/{stats.get('warps', 0)}"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Accel-Sim trace validation",
        "",
        f"**Result:** {'PASS' if report['passed'] else 'FAIL'}  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**Candidate:** `{report['inputs']['candidate_root']}`  ",
        f"**Golden:** `{report['inputs']['golden_root']}`  ",
        f"**Kernel IDs:** {', '.join(map(str, report['inputs']['kernel_ids']))}",
        "",
        "## Structural summary",
        "",
        "| Family | Candidate files | Golden files | File set |",
        "|---|---:|---:|:---:|",
    ]
    for family in FAMILIES:
        candidate = report["datasets"]["candidate"][family]
        golden = report["datasets"]["golden"][family]
        comparison = report["comparisons"][family]["files"]
        lines.append(
            f"| {family} | {candidate['files']} | {golden['files']} | "
            f"{'PASS' if comparison['passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "Metrics below are requests/timings for L1, events for caches, and CTAs/warps "
            "for assignments. Event-count differences are informational.",
            "",
            "| Kernel | L1 candidate | L1 golden | I-cache candidate | I-cache golden | "
            "D-cache candidate | D-cache golden | Warp candidate | Warp golden |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for kernel_id in report["inputs"]["kernel_ids"]:
        key = str(kernel_id)
        values: list[str] = []
        for family in FAMILIES:
            candidate_stats = report["datasets"]["candidate"][family]["kernels"].get(key, {})
            golden_stats = report["datasets"]["golden"][family]["kernels"].get(key, {})
            values.extend((kernel_metric(family, candidate_stats), kernel_metric(family, golden_stats)))
        lines.append(f"| {kernel_id} | " + " | ".join(values) + " |")

    lines.extend(["", "## Candidate L1/L2 correlation", "", "| Kernel | Matched | Missing | Orphan timings | Coverage |", "|---:|---:|---:|---:|---:|"])
    l1_kernels = report["datasets"]["candidate"]["l2_trace"]["kernels"]
    for kernel_id in report["inputs"]["kernel_ids"]:
        join = l1_kernels.get(str(kernel_id), {}).get("join", {})
        unique = join.get("unique_requests", 0)
        matched = join.get("matched_requests", 0)
        coverage = matched / unique if unique else 1.0
        lines.append(
            f"| {kernel_id} | {matched} | {join.get('missing_requests', 0)} | "
            f"{join.get('orphan_timing_records', 0)} | {coverage:.4%} |"
        )

    issue_data = report["issues"]
    lines.extend(
        [
            "",
            f"## Issues ({issue_data['error_count']} errors, {issue_data['warning_count']} warnings)",
            "",
        ]
    )
    if issue_data["error_samples"]:
        lines.append("### Error samples")
        lines.append("")
        lines.extend(f"- {message}" for message in issue_data["error_samples"])
        lines.append("")
    if issue_data["warning_samples"]:
        lines.append("### Warning samples")
        lines.append("")
        lines.extend(f"- {message}" for message in issue_data["warning_samples"])
        lines.append("")
    if not issue_data["error_samples"] and not issue_data["warning_samples"]:
        lines.append("No issues found.")
        lines.append("")
    lines.append("Detailed histograms and mismatch samples are in the JSON report.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Accel-Sim cache/warp logs against a golden run"
    )
    parser.add_argument("--candidate", required=True, type=Path, help="candidate run root")
    parser.add_argument("--golden", required=True, type=Path, help="golden run root")
    parser.add_argument(
        "--kernel-ids",
        type=parse_kernel_ids,
        help="comma-separated subset (default: all kernels found in either dataset)",
    )
    parser.add_argument(
        "--candidate-l2-dir", "--candidate-l1-dir", dest="candidate_l1_dir",
        help="candidate L2 trace directory, absolute or relative to candidate root",
    )
    parser.add_argument(
        "--golden-l2-dir", "--golden-l1-dir", dest="golden_l1_dir",
        help="golden L2 trace directory, absolute or relative to golden root",
    )
    parser.add_argument(
        "--trace-dir",
        help="trace directory used for grid checks (default: GOLDEN/traces)",
    )
    parser.add_argument(
        "--no-trace-grid-check", action="store_true", help="skip trace grid/block validation"
    )
    parser.add_argument(
        "--join-policy",
        choices=("report", "strict"),
        default="report",
        help="whether incomplete L1/L2 UID joins warn or fail (default: report)",
    )
    parser.add_argument("--num-shaders", type=int, default=8)
    parser.add_argument("--num-schedulers", type=int, default=4)
    parser.add_argument("--max-warps-per-shader", type=int, default=64)
    parser.add_argument("--max-issue-samples", type=int, default=100)
    parser.add_argument("--json-out", type=Path, default=Path("validation-report.json"))
    parser.add_argument("--markdown-out", type=Path, default=Path("validation-report.md"))
    parser.add_argument("--version", action="version", version=VERSION)
    return parser


def validate(args: argparse.Namespace) -> dict[str, Any]:
    issues = Issues(max_samples=args.max_issue_samples)
    limits = Limits(
        num_shaders=args.num_shaders,
        num_schedulers=args.num_schedulers,
        max_warps_per_shader=args.max_warps_per_shader,
        join_policy=args.join_policy,
    )
    candidate_root = args.candidate.resolve()
    golden_root = args.golden.resolve()
    candidate_roots = {
        "l2_trace": resolve_family(
            candidate_root, args.candidate_l1_dir, ("l2_trace", "l1_trace")
        ),
        "icache_status": candidate_root / "icache_status",
        "dcache_status": candidate_root / "dcache_status",
        "warp_assignments": candidate_root / "warp_assignments",
    }
    golden_roots = {
        "l2_trace": resolve_family(
            golden_root, args.golden_l1_dir, ("l2_trace", "l1_trace")
        ),
        "icache_status": golden_root / "icache_status",
        "dcache_status": golden_root / "dcache_status",
        "warp_assignments": golden_root / "warp_assignments",
    }

    all_dirs: dict[str, dict[str, dict[int, Path]]] = {"candidate": {}, "golden": {}}
    for side, roots in (("candidate", candidate_roots), ("golden", golden_roots)):
        for family, root in roots.items():
            all_dirs[side][family] = discover_kernel_dirs(root, side, family, issues)

    if args.kernel_ids:
        selected = set(args.kernel_ids)
    else:
        selected = set()
        for side in ("candidate", "golden"):
            for family in FAMILIES:
                selected.update(all_dirs[side][family])
    if not selected:
        issues.error("no kernel IDs were selected or discovered")

    directory_comparisons: dict[str, Any] = {}
    for family in FAMILIES:
        directory_comparisons[family] = compare_kernel_dirs(
            family,
            all_dirs["candidate"][family],
            all_dirs["golden"][family],
            selected,
            issues,
        )

    raw_datasets: dict[str, dict[str, Any]] = {"candidate": {}, "golden": {}}
    warp_details: dict[str, dict[int, dict[str, Any]]] = {}
    for side, roots in (("candidate", candidate_roots), ("golden", golden_roots)):
        raw_datasets[side]["l2_trace"] = validate_l1(
            roots["l2_trace"], selected, side, limits, issues
        )
        raw_datasets[side]["icache_status"] = validate_cache(
            roots["icache_status"], selected, side, "icache_status", limits, issues
        )
        raw_datasets[side]["dcache_status"] = validate_cache(
            roots["dcache_status"], selected, side, "dcache_status", limits, issues
        )
        warp_result, side_warp_details = validate_warps(
            roots["warp_assignments"], selected, side, limits, issues
        )
        raw_datasets[side]["warp_assignments"] = warp_result
        warp_details[side] = side_warp_details

    trace_metadata = discover_trace_metadata(
        golden_root,
        args.trace_dir,
        selected,
        not args.no_trace_grid_check,
        issues,
    )
    grid_checks = {
        side: check_warp_grid(side, warp_details[side], trace_metadata, issues)
        for side in ("candidate", "golden")
    }

    comparisons: dict[str, Any] = {}
    for family in FAMILIES:
        comparisons[family] = {
            "kernel_directories": directory_comparisons[family],
            "files": compare_file_sets(
                family,
                raw_datasets["candidate"][family],
                raw_datasets["golden"][family],
                issues,
            ),
            "semantics": semantic_comparison(
                family,
                raw_datasets["candidate"][family],
                raw_datasets["golden"][family],
                selected,
            ),
        }

    warp_identity: dict[str, Any] = {}
    for kernel_id in sorted(selected):
        candidate_assignments = warp_details["candidate"].get(kernel_id, {}).get(
            "assignments", set()
        )
        golden_assignments = warp_details["golden"].get(kernel_id, {}).get(
            "assignments", set()
        )
        missing = golden_assignments - candidate_assignments
        extra = candidate_assignments - golden_assignments
        passed = not missing and not extra
        if not passed:
            issues.error(
                f"comparison:warp_assignments:kernel_{kernel_id}: assignment identities differ: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        warp_identity[str(kernel_id)] = {
            "passed": passed,
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_samples": [list(item) for item in sorted(missing)[:100]],
            "extra_samples": [list(item) for item in sorted(extra)[:100]],
        }
    comparisons["warp_assignments"]["assignment_identity"] = warp_identity

    # Remove internal file lists before serializing; mismatch samples are kept
    # in the comparison section.
    for side in ("candidate", "golden"):
        for family in FAMILIES:
            raw_datasets[side][family].pop("relative_files", None)

    report = {
        "format_version": 1,
        "validator_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": issues.error_count == 0,
        "inputs": {
            "candidate_root": str(candidate_root),
            "golden_root": str(golden_root),
            "candidate_family_roots": {key: str(value) for key, value in candidate_roots.items()},
            "golden_family_roots": {key: str(value) for key, value in golden_roots.items()},
            "kernel_ids": sorted(selected),
            "kernel_filter_explicit": args.kernel_ids is not None,
            "join_policy": args.join_policy,
            "num_shaders": args.num_shaders,
            "num_schedulers": args.num_schedulers,
            "max_warps_per_shader": args.max_warps_per_shader,
        },
        "datasets": plain(raw_datasets),
        "trace_grid_checks": plain(grid_checks),
        "comparisons": plain(comparisons),
        "issues": issues.as_dict(),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("num_shaders", "num_schedulers", "max_warps_per_shader", "max_issue_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    report = validate(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(markdown_report(report), encoding="utf-8")
    print(
        f"{'PASS' if report['passed'] else 'FAIL'}: "
        f"{report['issues']['error_count']} errors, "
        f"{report['issues']['warning_count']} warnings; "
        f"reports: {args.json_out}, {args.markdown_out}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
