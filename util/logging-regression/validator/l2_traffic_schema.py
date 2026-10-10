"""Shared, strict line parsers for source, finalized LRC, and completion logs."""

import json
import re


SOURCE_FILES = {
    "l1_to_l2_requests.txt": {"l1_miss", "l1_write"},
    "l1_bypass_to_l2.txt": {"l1_bypass_load", "l1_bypass_store"},
    "tma_to_l2_requests.txt": {"tma_load", "tma_store"},
}
NEW_FILES = {"l1_bypass_to_l2.txt", "tma_to_l2_requests.txt", "lrc_to_l2_requests.txt"}
TRACKED_SOURCES = set().union(*SOURCE_FILES.values())
WRITE_SOURCES = {"l1_write", "l1_bypass_store", "tma_store"}
HEX_FIELDS = {"PC", "addr", "tag", "sector_mask", "byte_mask"}
SOURCE_FIELDS = {
    "request_uid", "PC", "dynamic_warp", "inst_ordinal", "addr",
    "subpartition", "set_index", "tag", "sector_mask", "size", "type",
}
LRC_FIELDS = {
    "service_uid", "addr", "subpartition", "set_index", "tag", "sector_mask",
    "size", "lrc_to_l2_cycle", "contributor_count", "contributors",
}
CONTRIBUTOR_FIELDS = {
    "request_uid", "transport_uid", "source", "access_type", "kernel_uid", "sm",
    "scheduler", "dynamic_warp", "inst_ordinal", "PC", "source_cycle",
}


def number(value, key):
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an unsigned integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", value):
        result = int(value, 16 if value.lower().startswith("0x") else 10)
    else:
        raise ValueError(f"{key} must be an unsigned integer, got {value!r}")
    if result < 0 or result >= 1 << (128 if key == "byte_mask" else 64):
        raise ValueError(f"{key} out of unsigned range: {result}")
    return result


def key_values(line):
    result = {}
    for token in line.split():
        if "=" not in token:
            raise ValueError(f"expected key=value, got {token!r}")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            raise ValueError(f"empty or duplicate field: {key!r}")
        result[key] = value
    return result


def exact_fields(values, expected):
    missing, extra = expected - values.keys(), values.keys() - expected
    if missing or extra:
        raise ValueError(f"schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def parse_numeric(values, nonnumeric):
    result = dict(values)
    for key, value in values.items():
        if key in nonnumeric:
            continue
        if key in HEX_FIELDS and not (isinstance(value, str) and re.fullmatch(r"0[xX][0-9a-fA-F]+", value)):
            raise ValueError(f"{key} must be hexadecimal")
        result[key] = number(value, key)
    return result


def check_request(record):
    if not record["size"]:
        raise ValueError("size must be positive")
    if not 0 < record["sector_mask"] < 16:
        raise ValueError("sector_mask must select one or more of four sectors")


def parse_source_line(line, filename):
    values = key_values(line)
    legacy = filename == "l1_to_l2_requests.txt"
    cycle = "l1_to_l2_cycle" if legacy else "source_to_l2_cycle"
    exact_fields(values, SOURCE_FIELDS | {cycle} | (set() if legacy else {"source"}))
    record = parse_numeric(values, {"type", "source"})
    if record["type"] not in {"READ", "WRITE", "WRITEBACK"}:
        raise ValueError(f"invalid type: {record['type']!r}")
    if legacy:
        record["source"] = "l1_miss" if record["type"] == "READ" else "l1_write"
    if record["source"] not in SOURCE_FILES[filename]:
        raise ValueError(f"source={record['source']} belongs in a different source file")
    if (record["type"] != "READ") != (record["source"] in WRITE_SOURCES):
        raise ValueError("source label disagrees with read/write type")
    check_request(record)
    record["source_cycle"] = record.pop(cycle)
    return record


def parse_lrc_line(line):
    values = key_values(line)
    exact_fields(values, LRC_FIELDS)
    record = parse_numeric(values, {"contributors"})
    check_request(record)
    try:
        contributors = json.loads(record["contributors"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid contributors JSON: {exc}") from exc
    if not isinstance(contributors, list) or not contributors:
        raise ValueError("contributors must be a nonempty JSON array")
    if len(contributors) != record["contributor_count"]:
        raise ValueError("contributor_count disagrees with contributors array")
    parsed = []
    seen = set()
    for contributor in contributors:
        if not isinstance(contributor, dict):
            raise ValueError("contributor must be an object")
        exact_fields(contributor, CONTRIBUTOR_FIELDS)
        member = parse_numeric(contributor, {"source", "access_type"})
        if member["source"] not in (TRACKED_SOURCES - WRITE_SOURCES) | {"other_read"}:
            raise ValueError(f"invalid read contributor source: {member['source']!r}")
        if not isinstance(member["access_type"], str) or not member["access_type"]:
            raise ValueError("contributor access_type must be a nonempty string")
        if member["transport_uid"] in seen:
            raise ValueError(f"duplicate contributor transport_uid={member['transport_uid']}")
        seen.add(member["transport_uid"])
        parsed.append(member)
    if parsed[0]["transport_uid"] != record["service_uid"]:
        raise ValueError("first contributor must be the service representative")
    if not any(member["source"] in TRACKED_SOURCES for member in parsed):
        raise ValueError("LRC group must include a tracked source")
    record["contributors"] = parsed
    return record


TIMING_LEGACY_FIELDS = {"request_uid", "start_cycle", "end_cycle", "elapsed_cycle"}
TIMING_FIELDS = TIMING_LEGACY_FIELDS | {
    "transport_uid", "service_uid", "addr", "sector_mask", "size",
}


def parse_timing_line(line, *, allow_legacy=False):
    """Parse completion records without truncating 64-bit timestamps."""
    values = key_values(line)
    legacy = allow_legacy and values.keys() == TIMING_LEGACY_FIELDS
    exact_fields(values, TIMING_LEGACY_FIELDS if legacy else TIMING_FIELDS)
    record = parse_numeric(values, set())
    if not legacy:
        check_request(record)
    for key in ("request_uid", "transport_uid", "service_uid"):
        if key in record and record[key] in {(1 << 32) - 1, (1 << 64) - 1}:
            raise ValueError(f"sentinel {key}={record[key]}")
    if record["end_cycle"] < record["start_cycle"]:
        raise ValueError("end_cycle precedes start_cycle")
    if record["elapsed_cycle"] != record["end_cycle"] - record["start_cycle"]:
        raise ValueError("elapsed_cycle disagrees with end_cycle - start_cycle")
    return record
