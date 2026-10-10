#!/usr/bin/env python3
"""Reconcile source logs, finalized LRC groups, and independent simulator audits.

This is a completed-run validator, not a golden-run comparison. Every logged
source request is checked against the interconnect departure audit, then against
sector ingress, LRC membership, accepting L2 lookups, and service completion. The optional audit is
required here so two production logs cannot agree by omitting the same traffic.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from l2_traffic_schema import (
    SOURCE_FILES, TRACKED_SOURCES, WRITE_SOURCES, key_values, number,
    parse_lrc_line, parse_source_line, parse_timing_line,
)
from validate_traces import Issues, KERNEL_DIR_RE, SHADER_DIR_RE, SCHEDULER_DIR_RE, WARP_RE

ATTRIBUTION = ("kernel_uid", "sm", "scheduler", "dynamic_warp", "inst_ordinal", "PC")
AUDIT_NUMBERS = {
    "source_uid", "cycle", "source_cycle", "addr", "size", "data_size", "sector_mask",
    "byte_mask", *ATTRIBUTION,
}
SENTINELS = {(1 << 32) - 1, (1 << 64) - 1}
ELIGIBLE_ACCESS_TYPES = {
    "GLOBAL_ACC_R", "GLOBAL_ACC_W", "LOCAL_ACC_R", "LOCAL_ACC_W",
    "L1_WRBK_ACC", "L1_WR_ALLOC_R",
}


class TrafficIssues(Issues):
    """Keep complete defect counts even when diagnostic samples are capped."""

    def __init__(self):
        super().__init__()
        self.duplicate_errors = 0

    def error(self, message):
        if "duplicate" in message or "expected one primary lookup" in message:
            self.duplicate_errors += 1
        super().error(message)


def nonblank(path, issues):
    try:
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if line.strip():
                    yield f"{path}:{lineno}", line.strip()
    except (OSError, UnicodeError) as exc:
        issues.error(f"cannot read {path}: {exc}")


def context(path, root):
    parts = path.relative_to(root).parts
    if len(parts) != 4:
        raise ValueError("expected kernel_*/shader_*/scheduler_*/filename")
    matches = [pattern.fullmatch(part) for pattern, part in zip(
        (KERNEL_DIR_RE, SHADER_DIR_RE, SCHEDULER_DIR_RE), parts)]
    if not all(matches):
        raise ValueError("malformed kernel/shader/scheduler directory")
    return dict(zip(("kernel_uid", "sm", "scheduler"), (int(m[1]) for m in matches)))


def compare(record, expected, fields, label, issues):
    for key in fields:
        if record.get(key) != expected.get(key):
            issues.error(f"{label}: {key}={record.get(key)!r}, expected {expected.get(key)!r}")


def insert_unique(records, key, record, label, issues):
    if key in records:
        issues.error(f"{label}: duplicate UID={key}")
    else:
        records[key] = record


def load_production(root, issues):
    sources, groups = {}, {}
    if not root.is_dir():
        issues.error(f"missing source log directory: {root}")
        return sources, groups
    for path in sorted(root.rglob("*.txt")):
        if path.name not in SOURCE_FILES and path.name != "lrc_to_l2_requests.txt":
            continue
        try:
            attribution = context(path, root)
        except ValueError as exc:
            issues.error(f"{path}: {exc}")
            continue
        for location, line in nonblank(path, issues):
            try:
                if path.name in SOURCE_FILES:
                    record = parse_source_line(line, path.name)
                    record.update(attribution)
                    for key in ATTRIBUTION:
                        if record[key] in SENTINELS:
                            issues.error(f"{location}: unknown {key} in tracked departure")
                    insert_unique(sources, record["request_uid"], record, location, issues)
                else:
                    record = parse_lrc_line(line)
                    representative = record["contributors"][0]
                    compare(representative, attribution, attribution.keys(), location, issues)
                    insert_unique(groups, record["service_uid"], record, location, issues)
            except (ValueError, TypeError, KeyError) as exc:
                issues.error(f"{location}: {exc}")
    return sources, groups


def load_audit(path, issues):
    departures, ingress = {}, {}
    for location, line in nonblank(path, issues):
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("audit row must be an object")
            event = record.get("event")
            if event not in {"departure", "ingress"}:
                raise ValueError(f"invalid event={event!r}")
            required = AUDIT_NUMBERS | {"event", "source", "subpartition", "access_type"}
            if event == "ingress":
                required |= {"transport_uid", "subpartition", "lrc_enabled", "is_write", "multicast_destinations"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"missing audit fields: {sorted(missing)}")
            for key in AUDIT_NUMBERS:
                record[key] = number(record[key], key)
            record["subpartition"] = number(record["subpartition"], "subpartition")
            if record["source"] not in TRACKED_SOURCES | {"other_read"}:
                raise ValueError(f"unknown source label={record['source']!r}")
            if not isinstance(record["access_type"], str) or not record["access_type"]:
                raise ValueError("access_type must be a nonempty string")
            if record["source"] == "other_read" and record["access_type"] in ELIGIBLE_ACCESS_TYPES:
                issues.error(f"{location}: missing source tagging for eligible data request")
            if "multicast_destinations" in record:
                record["multicast_destinations"] = number(record["multicast_destinations"], "multicast_destinations")
                if record["source"] in TRACKED_SOURCES and not record["source"].startswith("tma_") and record["multicast_destinations"] != 1:
                    issues.error(f"{location}: non-TMA request has multicast_destinations != 1")
            if event == "ingress":
                for key in ("transport_uid", "subpartition", "multicast_destinations"):
                    record[key] = number(record[key], key)
                if not record["multicast_destinations"]:
                    raise ValueError("multicast_destinations must be positive")
                for key in ("lrc_enabled", "is_write"):
                    if not isinstance(record[key], bool):
                        raise ValueError(f"{key} must be a JSON boolean")
                insert_unique(ingress, record["transport_uid"], record, location, issues)
            else:
                if record["source"] not in TRACKED_SOURCES:
                    raise ValueError("departure audit must describe a tracked source")
                if record["cycle"] != record["source_cycle"]:
                    issues.error(f"{location}: departure cycle disagrees with source_cycle")
                insert_unique(departures, record["source_uid"], record, location, issues)
        except (ValueError, TypeError, KeyError) as exc:
            issues.error(f"{location}: {exc}")
    return departures, ingress


def reconcile_sources(sources, departures, ingress, issues):
    for uid in departures.keys() - sources.keys():
        issues.error(f"missing source row for audited departure UID={uid}")
    for uid in sources.keys() - departures.keys():
        issues.error(f"orphan source row UID={uid}: no audited departure")
    for uid in sources.keys() & departures.keys():
        compare(sources[uid], departures[uid],
                (*ATTRIBUTION, "source", "source_cycle", "addr", "size", "sector_mask", "subpartition"),
                f"source UID={uid}", issues)
    children = defaultdict(list)
    for uid, child in ingress.items():
        if child["source"] not in TRACKED_SOURCES:
            continue
        parent = departures.get(child["source_uid"])
        if parent is None:
            issues.error(f"ingress UID={uid}: no parent departure UID={child['source_uid']}")
            continue
        compare(child, parent, (*ATTRIBUTION, "source", "source_cycle"), f"ingress UID={uid}", issues)
        if child["cycle"] < parent["cycle"]:
            issues.error(f"ingress UID={uid} precedes its source departure")
        if child["is_write"] != (child["source"] in WRITE_SOURCES):
            issues.error(f"ingress UID={uid}: write flag disagrees with source")
        children[child["source_uid"]].append(child)
    split_count = 0
    for uid, parent in departures.items():
        rows = children.get(uid, [])
        if not rows:
            issues.error(f"departure UID={uid}: no sector ingress records")
            continue
        if len(rows) == 1 and rows[0]["transport_uid"] == uid:
            compare(rows[0], parent, ("addr", "size", "data_size", "sector_mask", "byte_mask"),
                    f"unsplit request UID={uid}", issues)
            continue
        split_count += 1
        # Sector expansion consumes the packet's data_size, which can differ
        # from the original memory access size (for example, an L1 line fill).
        expected_mask = parent["sector_mask"]
        constant_pair = parent["data_size"] == 64 and expected_mask in (0, 15)
        if parent["data_size"] == 128:
            expected_mask = 15
        elif constant_pair:
            expected_mask = 3 if parent["addr"] % 128 == 0 else 12
        actual_mask, actual_bytes = 0, 0
        for child in rows:
            mask = child["sector_mask"]
            if not 0 < mask < 16 or mask.bit_count() != 1:
                issues.error(f"split child UID={child['transport_uid']}: invalid sector mask={mask}")
                continue
            if actual_mask & mask:
                issues.error(f"departure UID={uid}: duplicate child sector mask={mask}")
            actual_mask |= mask
            sector = mask.bit_length() - 1
            expected_addr = parent["addr"] if constant_pair else parent["addr"] + 32 * sector
            compare(child, {"addr": expected_addr, "size": 32, "data_size": 32},
                    ("addr", "size", "data_size"), f"split child UID={child['transport_uid']}", issues)
            expected_bytes = parent["byte_mask"] & (((1 << 32) - 1) << (32 * sector))
            if child["byte_mask"] != expected_bytes:
                issues.error(f"split child UID={child['transport_uid']}: byte-mask coverage mismatch")
            actual_bytes |= child["byte_mask"]
        if actual_mask != expected_mask:
            issues.error(f"departure UID={uid}: child sector coverage={actual_mask:#x}, expected={expected_mask:#x}")
        if actual_bytes != parent["byte_mask"]:
            issues.error(f"departure UID={uid}: missing or extra bytes after sector splitting")
    return split_count


def reconcile_groups(groups, ingress, max_merged, issues):
    membership = {}
    mixed, sizes = 0, Counter()
    for service_uid, group in groups.items():
        members = group["contributors"]
        sizes[len(members)] += 1
        mixed += len({m["source"] for m in members}) > 1
        if len(members) > max_merged:
            issues.error(f"service UID={service_uid}: exceeds configured merge limit={max_merged}")
        representative = ingress.get(service_uid)
        if representative is None:
            issues.error(f"service UID={service_uid}: no representative ingress")
        else:
            compare(group, representative, ("addr", "subpartition", "sector_mask", "size"),
                    f"service UID={service_uid}", issues)
            if group["lrc_to_l2_cycle"] != representative["cycle"]:
                issues.error(f"service UID={service_uid}: forwarding cycle differs from representative ingress")
        for member in members:
            uid = member["transport_uid"]
            if uid in membership:
                issues.error(f"duplicate LRC membership UID={uid}")
            membership[uid] = service_uid
            original = ingress.get(uid)
            if original is None:
                issues.error(f"LRC contributor UID={uid}: no ingress audit")
                continue
            if original["is_write"] or not original["lrc_enabled"]:
                issues.error(f"LRC contributor UID={uid}: write or LRC-disabled request")
            compare(member, original, (*ATTRIBUTION, "source", "source_cycle"),
                    f"LRC contributor UID={uid}", issues)
            if member["request_uid"] != original["source_uid"]:
                issues.error(f"LRC contributor UID={uid}: incorrect parent request UID")
            if "access_type" in original:
                compare(member, original, ("access_type",), f"LRC contributor UID={uid}", issues)
            # LRC keys by address, not sector mask. The legacy special-64B
            # splitter gives two children the same address but different
            # masks, and they can merge. Their individual masks/bytes were
            # checked against the parent in reconcile_sources; only the
            # representative must match the result's mask exactly.
            compare(original, group, ("addr", "subpartition", "size"),
                    f"LRC contributor UID={uid}", issues)
            # A late contributor is expected to arrive AFTER the first request
            # was forwarded. Only the representative fixes forwarding time.
            if original["cycle"] < group["lrc_to_l2_cycle"]:
                issues.error(f"LRC contributor UID={uid} arrived before its representative")
    for uid, row in ingress.items():
        if row["source"] in TRACKED_SOURCES and row["lrc_enabled"] and not row["is_write"] and uid not in membership:
            issues.error(f"missing LRC membership for ingress UID={uid}")
    return membership, mixed, dict(sorted(sizes.items()))


def accepted_records(path, issues):
    accepted = defaultdict(list)
    for location, line in nonblank(path, issues):
        try:
            record = key_values(line)
            for key in ("request_uid", "cycle", "addr", "subpartition", "size", "sector_mask", "kernel_uid", "sm", "dynamic_warp", "inst_ordinal"):
                record[key] = number(record[key], key)
            if record.get("status") == "RESERVATION_FAIL":
                raise ValueError("accepted-access file contains a reservation failure")
            if record.get("type") not in {"READ", "WRITE", "WRITEBACK"}:
                raise ValueError("invalid accepted-access type")
            accepted[record["request_uid"]].append(record)
        except (ValueError, TypeError, KeyError) as exc:
            issues.error(f"{location}: {exc}")
    return accepted


def separate_auxiliary_l2(accepted, ingress, issues):
    """Internal allocation reads can perform peer L2 lookups without ingress.

    They are created by L2 after a tracked write lookup and never replace that
    source service. Keep this exclusion explicit: unrelated orphan lookups
    still fail the normal ingress reconciliation.
    """
    source_services = {}
    auxiliary_count = 0
    for uid, rows in accepted.items():
        regular = []
        for row in rows:
            if row.get("access_type") != "L2_WR_ALLOC_R":
                regular.append(row)
                continue
            auxiliary_count += 1
            if row["type"] != "READ":
                issues.error(f"auxiliary L2 UID={uid}: L2_WR_ALLOC_R must have type=READ")
            if uid in ingress:
                issues.error(f"auxiliary L2 UID={uid}: allocation read cannot stand in for an ingress service")
        if regular:
            source_services[uid] = regular
    return source_services, auxiliary_count


def reconcile_l2(accepted, ingress, groups, membership, issues):
    expected = {uid for uid, row in ingress.items() if row["is_write"] or not row["lrc_enabled"]}
    expected |= groups.keys()
    pure_other_services = set()
    secondary = 0
    for uid, records in accepted.items():
        original = ingress.get(uid)
        if original is None:
            issues.error(f"accepted L2 UID={uid}: no ingress audit")
            continue
        if uid in membership and membership[uid] != uid:
            issues.error(f"merged contributor UID={uid} unexpectedly performed its own L2 lookup")
        primary = [row for row in records if row.get("access_type") != "CHIPLET_ACC_R"]
        secondary_rows = [row for row in records if row.get("access_type") == "CHIPLET_ACC_R"]
        secondary += len(secondary_rows)
        if len(primary) != 1:
            issues.error(f"accepted L2 UID={uid}: expected one primary lookup, found={len(primary)}")
        if len(secondary_rows) > 1:
            issues.error(f"accepted L2 UID={uid}: duplicate secondary chiplet lookup")
        if uid not in expected:
            if original["source"] == "other_read" and uid not in membership and original["lrc_enabled"] and not original["is_write"]:
                pure_other_services.add(uid)
            else:
                issues.error(f"accepted L2 UID={uid}: no corresponding forwarded request")
        for row in records:
            compare(row, original, ("addr", "size", "sector_mask", "kernel_uid", "sm", "dynamic_warp", "inst_ordinal"),
                    f"accepted L2 UID={uid}", issues)
            if row["cycle"] < original["cycle"]:
                issues.error(f"accepted L2 UID={uid} precedes ingress")
            if (row["type"] != "READ") != original["is_write"]:
                issues.error(f"accepted L2 UID={uid}: type disagrees with ingress")
        if primary:
            compare(primary[0], original, ("subpartition",), f"primary L2 UID={uid}", issues)
        if primary and secondary_rows and secondary_rows[0]["cycle"] < primary[0]["cycle"]:
            issues.error(f"accepted L2 UID={uid}: secondary lookup precedes primary")
    for uid in expected - accepted.keys():
        issues.error(f"missing accepted L2 lookup for forwarded UID={uid}")
    return pure_other_services, secondary


def check_warps(root, sources, issues):
    if not root.is_dir():
        issues.warning("warp_assignments missing: input warp membership was not checked")
        return False
    assignments = set()
    for path in root.glob("kernel_*/shader_*.txt"):
        kernel, shader = KERNEL_DIR_RE.fullmatch(path.parent.name), re.fullmatch(r"shader_(\d+)\.txt", path.name)
        if not kernel or not shader:
            continue
        for _, line in nonblank(path, issues):
            warp = WARP_RE.fullmatch(line)
            if warp:
                assignments.add((int(kernel[1]), int(shader[1]), int(warp[4]), int(warp[3])))
    for uid, row in sources.items():
        key = tuple(row[key] for key in ("kernel_uid", "sm", "scheduler", "dynamic_warp"))
        if key not in assignments:
            issues.error(f"source UID={uid}: no matching warp assignment {key}")
    return True


def load_timings(root, issues):
    timings, duplicates = {}, Counter()
    for path in sorted(root.rglob("l2_to_icnt_timing.txt")):
        try:
            attribution = context(path, root)
        except ValueError as exc:
            issues.error(f"{path}: {exc}")
            continue
        for location, line in nonblank(path, issues):
            try:
                record = parse_timing_line(line)
                record.update(attribution)
                uid = record["transport_uid"]
                if uid in timings:
                    duplicates[uid] += 1
                insert_unique(timings, uid, record, location, issues)
            except (ValueError, TypeError, KeyError) as exc:
                issues.error(f"{location}: {exc}")
    return timings, duplicates


def reconcile_timings(timings, duplicates, sources, ingress, membership, accepted, drain, issues):
    expected = {uid: row for uid, row in ingress.items() if row["source"] in TRACKED_SOURCES}
    missing = expected.keys() - timings.keys()
    orphan = timings.keys() - expected.keys()
    for uid in missing:
        issues.error(f"missing timing for ingress transport UID={uid}")
    for uid in orphan:
        issues.error(f"orphan timing transport UID={uid}: no tracked ingress")
    valid = set()
    service_ends = defaultdict(set)
    service_members = defaultdict(set)
    for uid in expected.keys() & timings.keys():
        timing, child = timings[uid], expected[uid]
        before = issues.error_count
        label = f"timing transport UID={uid}"
        compare(timing, child, ("kernel_uid", "sm", "scheduler", "addr", "sector_mask", "size"), label, issues)
        compare(timing, {"request_uid": child["source_uid"], "start_cycle": child["source_cycle"],
                         "service_uid": membership.get(uid, uid)},
                ("request_uid", "start_cycle", "service_uid"), label, issues)
        if timing["end_cycle"] < child["cycle"]:
            issues.error(f"{label}: completion precedes ingress")
        service_uid = membership.get(uid, uid)
        for lookup in accepted.get(service_uid, []):
            if timing["end_cycle"] < lookup["cycle"]:
                issues.error(f"{label}: completion precedes accepted L2 service UID={service_uid}")
        if drain and timing["end_cycle"] > drain["end_cycle"]:
            issues.error(f"{label}: completion exceeds final drain end_cycle")
        service_ends[service_uid].add(timing["end_cycle"])
        service_members[service_uid].add(uid)
        if before == issues.error_count and uid not in duplicates:
            valid.add(uid)
    for service_uid, ends in service_ends.items():
        if len(ends) != 1:
            issues.error(f"service UID={service_uid}: contributors disagree on completion cycle")
            valid.difference_update(service_members[service_uid])
    parents = defaultdict(set)
    for uid, row in expected.items():
        parents[row["source_uid"]].add(uid)
    completed = {parent for parent, children in parents.items() if children <= valid}
    coverage = {}
    for source in sorted(TRACKED_SOURCES):
        children = {uid for uid, row in expected.items() if row["source"] == source}
        source_parents = {uid for uid, row in sources.items() if row["source"] == source}
        # An orphan can still have a known source parent; expose that category.
        source_orphans = {uid for uid in orphan if timings[uid]["request_uid"] in source_parents}
        coverage[source] = {
            "expected_transports": len(children),
            "matched_transports": len(children & timings.keys()),
            "valid_transports": len(children & valid),
            "missing_transports": len(children & missing),
            "duplicate_records": sum(duplicates[uid] for uid in children),
            "orphan_records": sum(1 + duplicates[uid] for uid in source_orphans),
            "source_requests": len(source_parents),
            "fully_completed_requests": len(source_parents & completed),
        }
    return {
        "records": len(timings) + sum(duplicates.values()),
        "unique_transports": len(timings), "expected_transports": len(expected),
        "matched_transports": len(expected.keys() & timings.keys()),
        "valid_transports": len(valid), "missing_transports": len(missing),
        "orphan_transports": len(orphan), "duplicate_records": sum(duplicates.values()),
        "fully_completed_requests": len(completed & sources.keys()),
        "source_requests": len(sources), "per_source": coverage,
        "latency_boundary": "source departure to memory-side service completion",
        "parent_aggregation": "max(child end_cycle) - source departure cycle",
    }


def final_drain(text, issues):
    rows = re.findall(r"^L2_TRAFFIC_FINAL_DRAIN\s+(.+)$", text, re.M)
    if len(rows) != 1:
        issues.error(f"simulator.log: expected one final drain report, found={len(rows)}")
        return {}
    try:
        record = json.loads(rows[0])
        expected = {"finish_cycle", "end_cycle", "drain_cycles", "complete",
                    "lrc_incoming_sectors", "lrc_outgoing_sectors"}
        if not isinstance(record, dict) or record.keys() != expected:
            raise ValueError("invalid final drain report schema")
        if not isinstance(record["complete"], bool):
            raise ValueError("final drain complete must be a JSON boolean")
        for key in expected - {"complete"}:
            record[key] = number(record[key], key)
        if record["end_cycle"] < record["finish_cycle"]:
            raise ValueError("final drain end_cycle precedes finish_cycle")
        if record["drain_cycles"] != record["end_cycle"] - record["finish_cycle"]:
            raise ValueError("final drain drain_cycles disagrees with end_cycle - finish_cycle")
        if not record["complete"]:
            issues.error("simulator.log: final memory drain is incomplete")
        return record
    except (ValueError, TypeError, KeyError) as exc:
        issues.error(f"simulator.log: {exc}")
        return {}


def check_stats(path, ingress, groups, pure_other_services, issues):
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        issues.error(f"cannot read {path}: {exc}")
        return {}, {}
    if "GPGPU-Sim: *** exit detected ***" not in text:
        issues.error("simulator.log: normal completion marker is missing")
    if re.search(r"(?:Assertion .* failed|Program received signal|deadlock detected|break due to reaching|simulation (?:terminated|stopped).*limit)", text, re.I):
        issues.error("simulator.log: failure/deadlock/early-termination marker present")
    incoming = sum(row["sector_mask"].bit_count() * row["multicast_destinations"]
                   for row in ingress.values() if row["lrc_enabled"] and not row["is_write"])
    forwarded = sum(group["sector_mask"].bit_count() for group in groups.values())
    forwarded += sum(ingress[uid]["sector_mask"].bit_count() for uid in pure_other_services)
    result = {"incoming_audit_weighted": incoming, "forwarded_results_and_excluded": forwarded,
              "excluded_read_services": len(pure_other_services)}
    drain = final_drain(text, issues)
    for label, expected in (("ICNT to LRC", incoming), ("LRC to L2", forwarded)):
        key = label.lower().replace(" ", "_")
        actual = drain.get("lrc_incoming_sectors" if label == "ICNT to LRC" else "lrc_outgoing_sectors")
        result[key] = actual
        if actual is None:
            issues.error(f"simulator.log: missing final {label} sector counter")
        elif actual is not None and actual != expected:
            issues.error(f"{label} sector counter={actual}, expected={expected}")
    return result, drain


def validate(run, max_merged=4):
    issues = TrafficIssues()
    sources, groups = load_production(run / "l2_trace", issues)
    departures, ingress = load_audit(run / "l2_traffic_audit.jsonl", issues)
    split_count = reconcile_sources(sources, departures, ingress, issues)
    membership, mixed, sizes = reconcile_groups(groups, ingress, max_merged, issues)
    all_accepted = accepted_records(run / "l2_accesses.txt", issues)
    accepted, auxiliary_count = separate_auxiliary_l2(all_accepted, ingress, issues)
    pure_other, secondary = reconcile_l2(accepted, ingress, groups, membership, issues)
    warp_checked = check_warps(run / "warp_assignments", sources, issues)
    counters, drain = check_stats(run / "simulator.log", ingress, groups, pure_other, issues)
    timings, duplicate_timings = load_timings(run / "l2_trace", issues)
    timing_coverage = reconcile_timings(timings, duplicate_timings, sources, ingress, membership, accepted, drain, issues)
    source_counts = Counter(row["source"] for row in sources.values())
    tracked_ingress = {uid: row for uid, row in ingress.items() if row["source"] in TRACKED_SOURCES}
    expected_members = {uid for uid, row in tracked_ingress.items() if row["lrc_enabled"] and not row["is_write"]}
    expected_services = groups.keys() | {uid for uid, row in ingress.items() if row["is_write"] or not row["lrc_enabled"]}
    return {
        "passed": not issues.error_count,
        "run": str(run),
        "issues": issues.as_dict(),
        "source_counts": dict(sorted(source_counts.items())),
        "counts": {"source_records": len(sources), "departure_audit_records": len(departures),
                   "ingress_records": len(ingress), "lrc_results": len(groups),
                   "lrc_contributors": len(membership), "accepted_l2_records": sum(map(len, all_accepted.values())),
                   "auxiliary_l2_lookups": auxiliary_count,
                   "secondary_chiplet_lookups": secondary, "split_departures": split_count,
                   "timing_records": timing_coverage["records"]},
        "merge_size_distribution": sizes,
        "mixed_source_groups": mixed,
        "consistency": {
            "missing_source_records": len(departures.keys() - sources.keys()),
            "orphan_source_records": len(sources.keys() - departures.keys()),
            "departures_without_ingress": len(departures.keys() - {row["source_uid"] for row in tracked_ingress.values()}),
            "ingress_without_parent": sum(row["source_uid"] not in departures for row in tracked_ingress.values()),
            "missing_lrc_memberships": len(expected_members - membership.keys()),
            "orphan_lrc_memberships": len(membership.keys() - ingress.keys()),
            "missing_l2_services": len(expected_services - accepted.keys()),
            "orphan_l2_access_uids": len(accepted.keys() - ingress.keys()),
            "duplicate_errors": issues.duplicate_errors,
            "missing_timing_transports": timing_coverage["missing_transports"],
            "orphan_timing_transports": timing_coverage["orphan_transports"],
            "duplicate_timing_records": timing_coverage["duplicate_records"],
        },
        "counter_reconciliation": counters,
        "final_drain": drain,
        "timing_coverage": timing_coverage,
        "coverage": {"warp_assignments_checked": warp_checked,
                     "absent_source_categories": sorted(TRACKED_SOURCES - source_counts.keys()),
                     "lrc_enabled_observed": sorted({row["lrc_enabled"] for row in ingress.values()}),
                     "input_instruction_trace_check": "performed by the smoke runner; this validator checks immutable audit attribution",
                     "cache_set_tag_check": "schema only; cache indexing geometry is not independently recomputed"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--lrc-max-merged", type=int, default=4)
    args = parser.parse_args()
    if args.lrc_max_merged < 1:
        parser.error("--lrc-max-merged must be positive")
    report = validate(args.run, args.lrc_max_merged)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {report['issues']['error_count']} errors; {args.json_out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
