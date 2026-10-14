import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_validate_l2_traffic as legacy
import validate_l2_traffic as core
import validate_l2_traffic_bounded as bounded


class BoundedTrafficValidatorTests(legacy.TrafficValidatorTests):
    """Run the complete strict traffic regression matrix through partitioning."""
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.serial = 0
        def check(run, max_merged=4):
            self.serial += 1
            return bounded.validate(run, max_merged, partitions=7,
                                    work=Path(self.work.name) / str(self.serial),
                                    subprocess_partitions=False)
        self.mock = patch.object(legacy, 'validate', check)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.work.cleanup()


class PartitionTests(unittest.TestCase):
    def test_cross_partition_uid_references_and_equivalent_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            fixture = legacy.Fixture(run)
            fixture.mixed()
            for uid in [101, 204, 309, 512]:
                child = fixture.ingress(fixture.source(uid, 'l1_bypass_store'), lrc=False)
                fixture.accept(child)
            fixture.write()
            ordinary = core.validate(run)
            result = bounded.validate(run, partitions=7, work=Path(tmp) / 'work')
            for key in ['passed', 'source_counts', 'counts', 'consistency', 'timing_coverage',
                        'counter_reconciliation', 'final_drain', 'merge_size_distribution',
                        'mixed_source_groups', 'coverage', 'issues']:
                self.assertEqual(result[key], ordinary[key], key)
            self.assertGreater(result['validation_resources']['partitions'], 1)

    def test_compressed_audit_sources_and_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            fixture = legacy.Fixture(run)
            fixture.mixed()
            fixture.write()
            expected = core.validate(run)
            for path in [run / 'l2_traffic_audit.jsonl', run / 'l2_accesses.txt',
                         *(run / 'l2_trace').rglob('*.txt')]:
                with gzip.open(path.with_name(path.name + '.gz'), 'wb') as out:
                    out.write(path.read_bytes())
                path.unlink()
            direct = core.validate(run)
            result = bounded.validate(run, partitions=5, work=Path(tmp) / 'work',
                                      subprocess_partitions=False)
            for actual in [direct, result]:
                for key in ['passed', 'counts', 'timing_coverage', 'consistency']:
                    self.assertEqual(actual[key], expected[key])

    def test_truncated_compressed_input_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            fixture = legacy.Fixture(run)
            fixture.mixed()
            fixture.write()
            audit = run / 'l2_traffic_audit.jsonl'
            compressed = audit.with_name(audit.name + '.gz')
            compressed.write_bytes(gzip.compress(audit.read_bytes())[:-8])
            audit.unlink()
            result = bounded.validate(run, partitions=5, work=Path(tmp) / 'work',
                                      subprocess_partitions=False)
            self.assertFalse(result['passed'])
            self.assertTrue(any('cannot read' in x for x in result['issues']['error_samples']))

    def test_bad_global_counter_not_hidden_by_partitioning(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            fixture = legacy.Fixture(run)
            fixture.mixed()
            fixture.write()
            log = run / 'simulator.log'
            log.write_text(log.read_text().replace('"lrc_incoming_sectors": 3', '"lrc_incoming_sectors": 4'))
            result = bounded.validate(run, partitions=5, work=Path(tmp) / 'work',
                                      subprocess_partitions=False)
            self.assertFalse(result['passed'])
            self.assertTrue(any('ICNT to LRC sector counter' in x for x in result['issues']['error_samples']))

    def test_uid_index_limit_fails_without_partial_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            fixture = legacy.Fixture(run)
            fixture.mixed()
            fixture.write()
            with self.assertRaisesRegex(ValueError, 'UID index exceeds'):
                bounded.validate(run, partitions=7, work=Path(tmp) / 'work',
                                 max_index_uids=1, subprocess_partitions=False)

    def test_connected_component_cannot_exceed_memory_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            fixture = legacy.Fixture(run)
            fixture.mixed()
            fixture.write()
            with self.assertRaisesRegex(ValueError, 'exceeds the byte limit'):
                bounded.validate(run, partitions=128, work=Path(tmp) / 'work',
                                 max_partition_bytes=1, subprocess_partitions=False)

    def test_wrong_service_ids_stay_visible_across_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            fixture = legacy.Fixture(run)
            fixture.mixed()
            child = fixture.ingress(fixture.source(9001, 'l1_bypass_store'), lrc=False)
            fixture.accept(child)
            fixture.complete()
            fixture.timings[0][1]['service_uid'] = 9001
            fixture.write()
            result = bounded.validate(run, partitions=13, work=Path(tmp) / 'work',
                                      subprocess_partitions=False)
            self.assertFalse(result['passed'])
            self.assertTrue(any('service_uid' in x for x in result['issues']['error_samples']))


if __name__ == '__main__':
    unittest.main()
