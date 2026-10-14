import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validate_l2_traffic import validate
from l2_traffic_schema import parse_lrc_line, TRACKED_SOURCES


class Fixture:
    def __init__(self, root):
        self.root = root
        self.sources = []
        self.audit = []
        self.groups = []
        self.accepted = []
        self.timings = None
        self.drain = None
        self.incoming = 0
        self.forwarded = 0
        self.attrs = dict(kernel_uid=1, sm=0, scheduler=0, dynamic_warp=0, inst_ordinal=7, PC="0x20")

    def source(self, uid, source, *, cycle=100, addr=0x1000, size=32, mask=1):
        write = source in {"l1_write", "l1_bypass_store", "tma_store"}
        filename = ("l1_to_l2_requests.txt" if source.startswith("l1_") and source in {"l1_miss", "l1_write"}
                    else "tma_to_l2_requests.txt" if source.startswith("tma_") else "l1_bypass_to_l2.txt")
        row = dict(request_uid=uid, PC="0x20", dynamic_warp=0, inst_ordinal=7,
                   addr=hex(addr), subpartition=0, set_index=0, tag="0x1000",
                   sector_mask=hex(mask), size=size, type="WRITE" if write else "READ")
        row["l1_to_l2_cycle" if filename == "l1_to_l2_requests.txt" else "source_to_l2_cycle"] = cycle
        if filename != "l1_to_l2_requests.txt":
            row["source"] = source
        self.sources.append((filename, row))
        byte_mask = sum(((1 << 32) - 1) << (32 * sector) for sector in range(4) if mask & (1 << sector))
        audit = dict(event="departure", source_uid=uid, transport_uid=uid, source=source,
                     cycle=cycle, source_cycle=cycle, addr=hex(addr), size=size, data_size=size,
                     sector_mask=hex(mask), byte_mask=hex(byte_mask),
                     multicast_destinations=1, access_type="GLOBAL_ACC_W" if write else "GLOBAL_ACC_R",
                     is_write=write, lrc_enabled=True, subpartition=0, **self.attrs)
        self.audit.append(audit)
        return audit

    def ingress(self, parent, *, uid=None, cycle=130, addr=None, size=None, mask=None, lrc=True, multicast=1):
        row = copy.deepcopy(parent)
        row.update(event="ingress", transport_uid=parent["source_uid"] if uid is None else uid,
                   cycle=cycle, lrc_enabled=lrc, multicast_destinations=multicast)
        if addr is not None:
            row["addr"] = hex(addr)
        if size is not None:
            row["size"] = row["data_size"] = size
        if mask is not None:
            row["sector_mask"] = hex(mask)
            row["byte_mask"] = hex(int(parent["byte_mask"], 16) & sum(((1 << 32) - 1) << (32 * sector) for sector in range(4) if mask & (1 << sector)))
        self.audit.append(row)
        if lrc and not row["is_write"]:
            self.incoming += int(row["sector_mask"], 16).bit_count() * multicast
        return row

    def group(self, members):
        base = members[0]
        contributors = []
        for row in members:
            member = {key: row[key] for key in ("source", "access_type", "kernel_uid", "sm", "scheduler", "dynamic_warp", "inst_ordinal", "PC", "source_cycle")}
            member.update(request_uid=row["source_uid"], transport_uid=row["transport_uid"])
            contributors.append(member)
        record = dict(service_uid=base["transport_uid"], addr=base["addr"], subpartition=0,
                      set_index=0, tag="0x1000", sector_mask=base["sector_mask"], size=base["size"],
                      lrc_to_l2_cycle=base["cycle"], contributor_count=len(members), contributors=contributors)
        self.groups.append(record)
        self.forwarded += int(base["sector_mask"], 16).bit_count()
        return record

    def accept(self, row, *, chiplet=False):
        record = dict(cycle=200 if not chiplet else 220, addr=row["addr"],
                      type="WRITE" if row["is_write"] else "READ", access_type="CHIPLET_ACC_R" if chiplet else row["access_type"],
                      status="MISS", request_uid=row["transport_uid"], kernel_uid=1, stream_id=0,
                      sm=0, dynamic_warp=0, inst_ordinal=7, subpartition=1 if chiplet else 0,
                      size=row["size"], sector_mask=row["sector_mask"], attributed=1)
        self.accepted.append(record)
        return record

    def mixed(self):
        members = []
        for uid, source in enumerate(("l1_miss", "l1_bypass_load", "tma_load"), 1):
            parent = self.source(uid, source, cycle=100 + uid * 10)
            members.append(self.ingress(parent, cycle=130 + uid * 10))
        self.group(members)
        self.accept(members[0])
        self.accept(members[0], chiplet=True)
        return members

    def complete(self):
        """Provide completion fixtures independently of the production parser."""
        membership = {m["transport_uid"]: group["service_uid"] for group in self.groups
                      for m in group["contributors"]}
        service_ends = {}
        for row in self.audit:
            if row["event"] == "ingress":
                service = membership.get(row["transport_uid"], row["transport_uid"])
                service_ends[service] = max(service_ends.get(service, 300), row["cycle"] + 100)
        self.timings = []
        for child in self.audit:
            if child["event"] != "ingress" or child["source"] not in TRACKED_SOURCES:
                continue
            service = membership.get(child["transport_uid"], child["transport_uid"])
            end = service_ends[service]
            row = dict(request_uid=child["source_uid"], transport_uid=child["transport_uid"],
                       service_uid=service, addr=child["addr"], sector_mask=child["sector_mask"],
                       size=child["size"], start_cycle=child["source_cycle"], end_cycle=end,
                       elapsed_cycle=end-child["source_cycle"])
            attrs = {key: child[key] for key in ("kernel_uid", "sm", "scheduler")}
            self.timings.append((attrs, row))
        return self.timings

    def write(self):
        common = self.root / "l2_trace/kernel_1_test/shader_0/scheduler_0"
        common.mkdir(parents=True, exist_ok=True)
        files = {}
        for filename, row in self.sources:
            files.setdefault(filename, []).append(row)
        if self.groups:
            files["lrc_to_l2_requests.txt"] = self.groups
        for filename, rows in files.items():
            (common / filename).write_text("".join(encode(row) + "\n" for row in rows))
        (self.root / "l2_traffic_audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.audit))
        (self.root / "l2_accesses.txt").write_text("".join(encode(row) + "\n" for row in self.accepted))
        if self.timings is None:
            self.complete()
        timing_files = {}
        for attrs, row in self.timings:
            kernel_name = "test" if attrs["kernel_uid"] == 1 else "other"
            path = (self.root / f'l2_trace/kernel_{attrs["kernel_uid"]}_{kernel_name}' /
                    f'shader_{attrs["sm"]}/scheduler_{attrs["scheduler"]}/l2_to_icnt_timing.txt')
            timing_files.setdefault(path, []).append(row)
        for path, rows in timing_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(encode(row) + "\n" for row in rows))
        end = max([300] + [row["end_cycle"] for _, row in self.timings])
        drain = self.drain if self.drain is not None else dict(
            finish_cycle=250, end_cycle=end, drain_cycles=end - 250, complete=True,
            lrc_incoming_sectors=self.incoming, lrc_outgoing_sectors=self.forwarded)
        (self.root / "simulator.log").write_text(
            f"Number of ICNT to LRC sectors: {self.incoming}\n"
            f"Number of LRC to L2 sectors: {self.forwarded}\n"
            "L2_TRAFFIC_FINAL_DRAIN " + json.dumps(drain) + "\n"
            "GPGPU-Sim: *** exit detected ***\n")
        warps = self.root / "warp_assignments/kernel_1_test"
        warps.mkdir(parents=True, exist_ok=True)
        (warps / "shader_0.txt").write_text("CTA 0 (ctaid=0) assigned to shader core 0:\n  Warp 0 (warp_in_cta=0 dynamic_warp_id=0) -> Scheduler 0\n")


def encode(row):
    return " ".join(f"{key}={json.dumps(value, separators=(',', ':')) if isinstance(value, list) else value}" for key, value in row.items())


class TrafficValidatorTests(unittest.TestCase):
    def run_fixture(self, configure, error=None, max_merged=4):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            configure(fixture)
            fixture.write()
            report = validate(fixture.root, max_merged)
            if error is None:
                self.assertTrue(report["passed"], report["issues"])
            else:
                self.assertFalse(report["passed"])
                self.assertTrue(any(error in e for e in report["issues"]["error_samples"]), report["issues"])
            return report

    def test_mixed_sources_late_contributors_and_secondary_lookup(self):
        report = self.run_fixture(lambda f: f.mixed())
        self.assertEqual(report["mixed_source_groups"], 1)
        self.assertEqual(report["counts"]["secondary_chiplet_lookups"], 1)
        self.assertEqual(report["merge_size_distribution"], {3: 1})

    def test_same_source_merge(self):
        def case(f):
            members = [f.ingress(f.source(uid, "l1_bypass_load"), cycle=130 + uid)
                       for uid in (1, 2)]
            f.group(members)
            f.accept(members[0])
        report = self.run_fixture(case)
        self.assertEqual(report["mixed_source_groups"], 0)
        self.assertEqual(report["merge_size_distribution"], {2: 1})

    def test_other_read_contributor_kept_in_tracked_group(self):
        def case(f):
            tracked = f.ingress(f.source(1, "l1_miss"))
            other = f.source(2, "l1_bypass_load")
            f.sources.pop()
            f.audit.pop()
            other.update(source="other_read", source_uid=(1 << 32) - 1,
                         source_cycle=0, access_type="INST_ACC_R")
            child = f.ingress(other, uid=2, cycle=140)
            f.group([tracked, child])
            f.accept(tracked)
        report = self.run_fixture(case)
        self.assertEqual(report["counts"]["lrc_contributors"], 2)

    def test_pure_other_read_group_is_excluded_but_counters_reconcile(self):
        def case(f):
            f.mixed()
            other = f.source(99, "l1_bypass_load", addr=0x2000)
            f.sources.pop()
            f.audit.pop()
            other.update(source="other_read", source_uid=(1 << 32) - 1,
                         source_cycle=0, access_type="INST_ACC_R")
            child = f.ingress(other, uid=99)
            f.accept(child)
            f.forwarded += 1
        report = self.run_fixture(case)
        self.assertEqual(report["counter_reconciliation"]["excluded_read_services"], 1)

    def test_cross_kernel_merge_preserves_member_directory_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.mixed()
            attrs = dict(kernel_uid=2, sm=1, scheduler=2, dynamic_warp=3)
            for row in fixture.audit:
                if row["source_uid"] == 2:
                    row.update(attrs)
            fixture.sources[1][1]["dynamic_warp"] = 3
            fixture.groups[0]["contributors"][1].update(attrs)
            fixture.write()
            original = fixture.root / "l2_trace/kernel_1_test/shader_0/scheduler_0/l1_bypass_to_l2.txt"
            moved = fixture.root / "l2_trace/kernel_2_other/shader_1/scheduler_2/l1_bypass_to_l2.txt"
            moved.parent.mkdir(parents=True, exist_ok=True)
            original.rename(moved)
            warp = fixture.root / "warp_assignments/kernel_2_other/shader_1.txt"
            warp.parent.mkdir(parents=True)
            warp.write_text("CTA 0 (ctaid=0) assigned to shader core 1:\n  Warp 1 (warp_in_cta=0 dynamic_warp_id=3) -> Scheduler 2\n")
            report = validate(fixture.root)
            self.assertTrue(report["passed"], report["issues"])

    def test_missing_source_is_detected_independently(self):
        def case(f):
            f.mixed()
            f.sources.pop()
        self.run_fixture(case, "missing source row")

    def test_omitted_source_tag_and_row_are_not_silently_excluded(self):
        def case(f):
            f.mixed()
            f.sources.pop()
            for row in f.audit:
                if row["source_uid"] == 3:
                    row["source"] = "other_read"
            f.groups[0]["contributors"][-1]["source"] = "other_read"
        self.run_fixture(case, "missing source tagging for eligible data request")

    def test_duplicate_departure(self):
        def case(f):
            f.mixed()
            f.audit.append(copy.deepcopy(f.audit[0]))
        self.run_fixture(case, "duplicate UID")

    def test_wrong_instruction_attribution(self):
        def case(f):
            f.mixed()
            f.sources[1][1]["inst_ordinal"] = 8
        self.run_fixture(case, "inst_ordinal=8")

    def test_wrong_source_subpartition(self):
        def case(f):
            f.mixed()
            f.sources[1][1]["subpartition"] = 9
        self.run_fixture(case, "subpartition=9")

    def test_unknown_pc_does_not_count_as_complete_attribution(self):
        def case(f):
            f.mixed()
            f.sources[1][1]["PC"] = "0xffffffffffffffff"
            for row in f.audit:
                if row["source_uid"] == 2:
                    row["PC"] = "0xffffffffffffffff"
            f.groups[0]["contributors"][1]["PC"] = "0xffffffffffffffff"
        self.run_fixture(case, "unknown PC")

    def test_missing_lrc_contributor(self):
        def case(f):
            f.mixed()
            f.groups[0]["contributors"].pop()
            f.groups[0]["contributor_count"] -= 1
        self.run_fixture(case, "missing LRC membership")

    def test_duplicate_service(self):
        def case(f):
            f.mixed()
            f.groups.append(copy.deepcopy(f.groups[0]))
        self.run_fixture(case, "duplicate UID")

    def test_merge_limit(self):
        self.run_fixture(lambda f: f.mixed(), "configured merge limit", max_merged=2)

    def test_multisector_parent_multiple_singleton_services(self):
        def case(f):
            parent = f.source(1, "l1_miss", size=64, mask=3)
            for sector in range(2):
                child = f.ingress(parent, uid=10 + sector, addr=0x1000 + 32 * sector, size=32, mask=1 << sector)
                f.group([child])
                f.accept(child)
        report = self.run_fixture(case)
        self.assertEqual(report["counts"]["split_departures"], 1)
        self.assertEqual(report["merge_size_distribution"], {1: 2})

    def test_special_64b_children_merge_by_same_address_with_distinct_masks(self):
        def case(f):
            parent = f.source(1, "l1_bypass_load", size=64, mask=15)
            parent["byte_mask"] = hex((1 << 64) - 1)
            children = [f.ingress(parent, uid=10 + sector, addr=0x1000,
                                  size=32, mask=1 << sector)
                        for sector in range(2)]
            f.group(children)
            f.accept(children[0])
        report = self.run_fixture(case)
        self.assertEqual(report["counts"]["split_departures"], 1)
        self.assertEqual(report["counts"]["lrc_contributors"], 2)
        self.assertEqual(report["counts"]["lrc_results"], 1)

    def test_lrc_representative_mask_must_still_match_ingress(self):
        def case(f):
            f.mixed()
            f.groups[0]["sector_mask"] = "0x2"
        self.run_fixture(case, "service UID=1: sector_mask=2, expected 1")

    def test_missing_split_sector(self):
        def case(f):
            parent = f.source(1, "l1_miss", size=64, mask=3)
            child = f.ingress(parent, uid=10, addr=0x1000, size=32, mask=1)
            f.group([child])
            f.accept(child)
        self.run_fixture(case, "child sector coverage")

    def test_overlapping_split_byte_masks(self):
        def case(f):
            parent = f.source(1, "l1_miss", size=64, mask=3)
            for sector in range(2):
                child = f.ingress(parent, uid=10 + sector, addr=0x1000 + 32 * sector, size=32, mask=1 << sector)
                f.group([child])
                f.accept(child)
                if sector:
                    child["byte_mask"] = parent["byte_mask"]
        self.run_fixture(case, "byte-mask coverage mismatch")

    def test_disabled_lrc_and_bypass_stores(self):
        def case(f):
            for uid, source in enumerate(("l1_bypass_load", "l1_bypass_store", "tma_store"), 1):
                parent = f.source(uid, source)
                child = f.ingress(parent, lrc=False)
                f.accept(child)
        report = self.run_fixture(case)
        self.assertEqual(report["counts"]["lrc_results"], 0)

    def test_missing_accepted_access(self):
        def case(f):
            f.mixed()
            f.accepted.clear()
        self.run_fixture(case, "missing accepted L2 lookup")

    def test_internal_allocation_lookup_is_counted_without_source_or_timing(self):
        def case(f):
            child = f.ingress(f.source(1, "l1_bypass_store"), lrc=False)
            primary = f.accept(child)
            auxiliary = dict(primary, request_uid=99, access_type="L2_WR_ALLOC_R",
                             type="READ", cycle=250, subpartition=42)
            f.accepted.append(auxiliary)
        report = self.run_fixture(case)
        self.assertEqual(report["counts"]["accepted_l2_records"], 2)
        self.assertEqual(report["counts"]["auxiliary_l2_lookups"], 1)
        self.assertEqual(report["consistency"]["orphan_l2_access_uids"], 0)
        self.assertEqual(report["timing_coverage"]["expected_transports"], 1)

    def test_ordinary_orphan_lookup_still_fails(self):
        def case(f):
            f.mixed()
            f.accepted.append(dict(f.accepted[0], request_uid=99))
        report = self.run_fixture(case, "accepted L2 UID=99: no ingress audit")
        self.assertEqual(report["consistency"]["orphan_l2_access_uids"], 1)
        self.assertEqual(report["counts"]["auxiliary_l2_lookups"], 0)

    def test_auxiliary_allocation_requires_read_type(self):
        def case(f):
            child = f.ingress(f.source(1, "l1_bypass_store"), lrc=False)
            primary = f.accept(child)
            f.accepted.append(dict(primary, request_uid=99, access_type="L2_WR_ALLOC_R"))
        self.run_fixture(case, "L2_WR_ALLOC_R must have type=READ")

    def test_auxiliary_uid_cannot_replace_source_service(self):
        def case(f):
            child = f.ingress(f.source(1, "l1_bypass_store"), lrc=False)
            f.accept(child).update(request_uid=99, access_type="L2_WR_ALLOC_R", type="READ")
        report = self.run_fixture(case, "missing accepted L2 lookup for forwarded UID=1")
        self.assertEqual(report["consistency"]["missing_l2_services"], 1)

    def test_auxiliary_cannot_reuse_ingress_uid_to_supply_service(self):
        def case(f):
            child = f.ingress(f.source(1, "l1_bypass_load"), lrc=False)
            f.accept(child)["access_type"] = "L2_WR_ALLOC_R"
        report = self.run_fixture(case, "allocation read cannot stand in for an ingress service")
        self.assertEqual(report["consistency"]["missing_l2_services"], 1)

    def test_duplicate_accepted_primary_access(self):
        def case(f):
            f.mixed()
            f.accepted.append(copy.deepcopy(f.accepted[0]))
        self.run_fixture(case, "expected one primary lookup")

    def test_multicast_counter_weighting(self):
        def case(f):
            parent = f.source(1, "tma_load")
            parent["multicast_destinations"] = 4
            child = f.ingress(parent, multicast=4)
            f.group([child])
            f.accept(child)
        report = self.run_fixture(case)
        self.assertEqual(report["counter_reconciliation"]["icnt_to_lrc"], 4)

    def test_ordinary_load_cannot_be_multicast(self):
        def case(f):
            parent = f.source(1, "l1_bypass_load")
            child = f.ingress(parent, multicast=4)
            f.group([child])
            f.accept(child)
        self.run_fixture(case, "non-TMA request has multicast_destinations")

    def test_all_source_categories_have_completion_coverage(self):
        def case(f):
            for uid, source in enumerate(sorted(TRACKED_SOURCES), 1):
                child = f.ingress(f.source(uid, source), lrc=False)
                f.accept(child)
        report = self.run_fixture(case)
        self.assertEqual(report["timing_coverage"]["fully_completed_requests"], 6)
        for row in report["timing_coverage"]["per_source"].values():
            self.assertEqual(row["expected_transports"], 1)
            self.assertEqual(row["valid_transports"], 1)
            self.assertEqual(row["fully_completed_requests"], 1)

    def test_untracked_representative_services_tracked_contributor(self):
        def case(f):
            other = f.source(1, "l1_bypass_load")
            f.sources.pop()
            f.audit.pop()
            other.update(source="other_read", source_uid=(1 << 32) - 1,
                         source_cycle=0, access_type="INST_ACC_R")
            representative = f.ingress(other, uid=1)
            tracked = f.ingress(f.source(2, "tma_load"), cycle=140)
            f.group([representative, tracked])
            f.accept(representative)
        report = self.run_fixture(case)
        self.assertEqual(report["timing_coverage"]["unique_transports"], 1)
        self.assertEqual(report["timing_coverage"]["valid_transports"], 1)

    def test_missing_split_completion_invalidates_parent(self):
        def case(f):
            parent = f.source(1, "l1_miss", size=64, mask=3)
            for sector in range(2):
                child = f.ingress(parent, uid=10 + sector, addr=0x1000 + 32 * sector,
                                  size=32, mask=1 << sector)
                f.group([child])
                f.accept(child)
            f.complete().pop()
        report = self.run_fixture(case, "missing timing for ingress transport")
        self.assertEqual(report["timing_coverage"]["matched_transports"], 1)
        self.assertEqual(report["timing_coverage"]["fully_completed_requests"], 0)

    def test_missing_all_new_source_timings_cannot_pass(self):
        def case(f):
            f.mixed()
            f.complete()
            f.timings = f.timings[:1]
        report = self.run_fixture(case, "missing timing")
        self.assertEqual(report["timing_coverage"]["missing_transports"], 2)

    def test_duplicate_completion_is_rejected(self):
        def case(f):
            f.mixed()
            f.complete().append(copy.deepcopy(f.timings[0]))
        report = self.run_fixture(case, "duplicate UID")
        self.assertEqual(report["timing_coverage"]["duplicate_records"], 1)
        self.assertEqual(report["timing_coverage"]["fully_completed_requests"], 2)

    def test_orphan_completion_is_rejected(self):
        def case(f):
            f.mixed()
            f.complete()[0][1]["transport_uid"] = 999
        report = self.run_fixture(case, "orphan timing transport")
        self.assertEqual(report["timing_coverage"]["orphan_transports"], 1)

    def test_wrong_timing_fields(self):
        for field, value, error in (
                ("request_uid", 99, "request_uid=99"),
                ("service_uid", 99, "service_uid=99"),
                ("addr", "0x1020", "addr=4128"),
                ("sector_mask", "0x2", "sector_mask=2"),
                ("size", 64, "size=64"),
                ("start_cycle", 1, "start_cycle=1")):
            with self.subTest(field=field):
                def case(f):
                    f.mixed()
                    row = f.complete()[0][1]
                    row[field] = value
                    row["elapsed_cycle"] = row["end_cycle"] - row["start_cycle"]
                self.run_fixture(case, error)

    def test_wrong_timing_directory(self):
        def case(f):
            f.mixed()
            f.complete()[0][0]["scheduler"] = 9
        self.run_fixture(case, "scheduler=9")

    def test_lrc_contributors_must_share_completion(self):
        def case(f):
            f.mixed()
            row = f.complete()[1][1]
            row["end_cycle"] -= 1
            row["elapsed_cycle"] -= 1
        report = self.run_fixture(case, "contributors disagree on completion")
        self.assertEqual(report["timing_coverage"]["fully_completed_requests"], 0)

    def test_completion_cannot_precede_ingress_or_l2_acceptance(self):
        for end, error in ((130, "completion precedes ingress"),
                           (199, "completion precedes accepted L2 service")):
            with self.subTest(end=end):
                def case(f):
                    f.mixed()
                    for _, row in f.complete():
                        row["end_cycle"] = end
                        row["elapsed_cycle"] = end - row["start_cycle"]
                self.run_fixture(case, error)

    def test_malformed_timing_is_not_counted_complete(self):
        for field, value, error in (("size", 0, "size must be positive"),
                                    ("elapsed_cycle", 1, "elapsed_cycle disagrees"),
                                    ("end_cycle", -1, "end_cycle must be an unsigned"),
                                    ("end_cycle", 1 << 64, "out of unsigned range")):
            with self.subTest(field=field, value=value):
                def case(f):
                    f.mixed()
                    f.complete()[0][1][field] = value
                self.run_fixture(case, error)

    def test_legacy_timing_cannot_pass_strict_completion_validation(self):
        def case(f):
            f.mixed()
            for _, row in f.complete():
                for field in ("transport_uid", "service_uid", "addr", "size", "sector_mask"):
                    row.pop(field)
        self.run_fixture(case, "schema mismatch")

    def test_64_bit_timestamps_remain_exact(self):
        def case(f):
            base = (1 << 32) + 123
            child = f.ingress(f.source(1, "l1_bypass_store", cycle=base), cycle=base + 30, lrc=False)
            f.accept(child)["cycle"] = base + 50
            f.complete()
        report = self.run_fixture(case)
        self.assertEqual(report["timing_coverage"]["valid_transports"], 1)
        self.assertGreater(report["final_drain"]["end_cycle"], 1 << 32)

    def test_final_counters_override_kernel_finish_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.mixed()
            fixture.write()
            path = fixture.root / "simulator.log"
            path.write_text(path.read_text().replace("ICNT to LRC sectors: 3", "ICNT to LRC sectors: 0"))
            report = validate(fixture.root)
            self.assertTrue(report["passed"], report["issues"])

    def test_missing_or_duplicate_final_drain_fails(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                fixture.mixed()
                fixture.write()
                path = fixture.root / "simulator.log"
                lines = path.read_text().splitlines(True)
                marker = next(line for line in lines if line.startswith("L2_TRAFFIC_FINAL_DRAIN"))
                path.write_text("".join(lines) + marker if duplicate else "".join(line for line in lines if line != marker))
                report = validate(fixture.root)
                self.assertFalse(report["passed"])
                self.assertTrue(any("expected one final drain report" in e for e in report["issues"]["error_samples"]))

    def test_incomplete_or_malformed_final_drain_fails(self):
        for field, value, error in (("complete", False, "final memory drain is incomplete"),
                                    ("complete", 1, "must be a JSON boolean"),
                                    ("drain_cycles", 2, "drain_cycles disagrees"),
                                    ("end_cycle", 249, "precedes finish_cycle"),
                                    ("lrc_incoming_sectors", 99, "counter=99")):
            with self.subTest(field=field):
                def case(f):
                    f.mixed()
                    f.drain = dict(finish_cycle=250, end_cycle=300, drain_cycles=50, complete=True,
                                   lrc_incoming_sectors=3, lrc_outgoing_sectors=1)
                    f.drain[field] = value
                self.run_fixture(case, error)

    def test_completion_cannot_exceed_final_drain(self):
        def case(f):
            f.mixed()
            f.drain = dict(finish_cycle=250, end_cycle=299, drain_cycles=49, complete=True,
                           lrc_incoming_sectors=3, lrc_outgoing_sectors=1)
        self.run_fixture(case, "completion exceeds final drain")

    def test_invalid_contributor_json_is_a_validation_error(self):
        with self.assertRaises(ValueError):
            parse_lrc_line("service_uid=1 addr=0x1000 subpartition=0 set_index=0 tag=0x1000 sector_mask=0x1 size=32 lrc_to_l2_cycle=1 contributor_count=1 contributors=[broken]")

    def test_aborted_run_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.mixed()
            fixture.write()
            (fixture.root / "simulator.log").write_text("Program received signal SIGABRT\n")
            report = validate(fixture.root)
            self.assertFalse(report["passed"])
            self.assertTrue(any("normal completion" in e for e in report["issues"]["error_samples"]))

    def test_cycle_limited_run_with_normal_exit_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.mixed()
            fixture.write()
            with (fixture.root / "simulator.log").open("a") as handle:
                handle.write("GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **\n")
            report = validate(fixture.root)
            self.assertFalse(report["passed"])
            self.assertTrue(any("early-termination" in e for e in report["issues"]["error_samples"]))


if __name__ == "__main__":
    unittest.main()
