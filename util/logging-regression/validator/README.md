# Accel-Sim logging validator

`validate_traces.py` streams Accel-Sim L1/L2, I-cache, D-cache, and warp-assignment
logs and compares a candidate run with a golden run. It validates record schemas,
IDs and topology bounds, timing arithmetic, cache-status values, warp/CTA
uniqueness, trace grid dimensions, relative file sets, and nonempty-output
expectations. Event counts and bounded histograms are reported but are not
required to be byte-for-byte equal.

Both L2 timing formats are accepted. The historical format contains
`request_uid`, `start_cycle`, `end_cycle`, and `elapsed_cycle`. The corrected
format additionally requires `transport_uid`, `service_uid`, `addr`,
`sector_mask`, and `size`. Partial mixtures are rejected. In both formats,
`request_uid` joins to the union of L1, non-TMA bypass, and TMA source files;
`transport_uid` identifies a sector/transport request and `service_uid`
identifies the request that supplied shared L2 service.

The join uses the source's kernel/shader/scheduler directory and departure
cycle. Bypass/TMA-only directories can have completion files without an L1
file. Extended timings reject duplicate transport IDs. The historical UID join
can confirm that a parent has at least one completion; it cannot prove that
all split children completed without the independent audit.

Both datasets use the `l2_trace` family. Historical `l1_trace` directories are
accepted as a fallback. Use `--candidate-l2-dir` or `--golden-l2-dir` for custom
paths; the old `--candidate-l1-dir` and `--golden-l1-dir` options remain aliases.

## Rodinia

```bash
python3 validate_traces.py \
  --candidate /scratch/prashanth/accelsim2-results/rodinia \
  --golden /scratch/prashanth/gpu_model/accel-sim-data-rodinia_nn_smaller \
  --join-policy strict \
  --json-out /scratch/prashanth/accelsim2-results/rodinia/validation.json \
  --markdown-out /scratch/prashanth/accelsim2-results/rodinia/validation.md
```

## ResNet representative subset

```bash
python3 validate_traces.py \
  --candidate /scratch/prashanth/accelsim2-results/resnet-subset \
  --golden /scratch/prashanth/gpu_model/accel-sim-data-resnet50-smaller \
  --kernel-ids 1,13,23 \
  --join-policy strict \
  --json-out /scratch/prashanth/accelsim2-results/resnet-subset/validation.json \
  --markdown-out /scratch/prashanth/accelsim2-results/resnet-subset/validation.md
```

Use the default `--join-policy report` for an archived pre-fix L2-timing run;
incomplete UID joins then produce warnings. Use `strict` for the corrected
logger and acceptance runs. The exit status is zero only when no validation
errors are found.

If the run used a different topology, pass `--num-shaders`,
`--num-schedulers`, and `--max-warps-per-shader`. Trace grid/block metadata is
located recursively under `GOLDEN/traces`; it can be overridden with
`--trace-dir` or disabled with `--no-trace-grid-check`.

Run the self-tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Independent completion acceptance

```bash
python3 validate_l2_traffic.py --run /path/to/run --json-out /path/to/report.json
```

This strict validator requires the three source families as applicable,
finalized LRC groups when enabled, `l2_traffic_audit.jsonl`, `l2_accesses.txt`,
and `simulator.log`. Every tracked audited ingress transport requires exactly
one nine-field `l2_to_icnt_timing.txt` row. Missing, duplicate, malformed, or
orphan rows fail. It verifies parent/service identity, directory attribution,
address/mask/size, source start, ordering, 64-bit elapsed arithmetic, and common
completion cycles for LRC contributors, including untracked representatives.

`simulator.log` must contain one `L2_TRAFFIC_FINAL_DRAIN` JSON record with
`finish_cycle`, `end_cycle`, `drain_cycles`, `complete`,
`lrc_incoming_sectors`, and `lrc_outgoing_sectors`. An incomplete drain fails,
and final LRC counters must agree with the audit and groups. Existing kernel
statistics may precede memory completion and are not used for this comparison.

`timing_coverage` reports expected, matched, valid, missing, duplicate and
orphan transports and fully completed parents, including per-source counts.
A parent is complete only when every audited child has one valid completion.
Old logs without all-source timing and final-drain evidence cannot pass this
validator, even if their source/ingress counts agree.

For latency plots, use `max(child end_cycle) - source departure cycle` for each
fully completed parent. The end is memory-side service completion, not SM
arrival or register readiness. See `../PORT_NOTES.md` for the full contract.

### Large completed runs

`validate_l2_traffic_bounded.py --run RUN --json-out REPORT.json` runs the same
strict traffic checks using UID-connected partitions. Its compact UID index
joins each source parent, split transport, LRC contributor, service and timing
reference before assigning records to a partition. Duplicate and incorrect
cross-references stay visible to the checker. It checks the original simulator
completion/drain metadata and reconciles global LRC counters against the sum of
all partitions before reporting a pass.

The default is 128 partitions, validated serially in separate processes.
`--work-dir` selects a fresh scratch directory; `--partitions` changes the
partition count. `--max-partition-bytes` rejects oversized partitions (default
2 GiB of uncompressed input) rather than risking an uncontrolled allocation.
The UID index scales with the number of unique identifiers and is capped by
`--max-index-uids` (default 64 million); exceeding either limit fails validation
without partial acceptance. The large parsed-record dictionaries exist for
only one partition at a time. Progress,
input SHA256 hashes, partition reports and peak RSS measurements are retained
in the work directory. Partition reports are intermediate results, not complete
run acceptance. Successful temporary partitions are removed; failed partitions
are kept for diagnosis.

Both strict entry points accept gzip source/LRC/timing files and use `.gz`
audit/access files when the corresponding uncompressed file is absent. Keep
only one source/LRC/timing representation per file: retaining both is detected
as duplicate records. A lossless archive should record the original SHA256 and
verify decompression before removing its uncompressed copy. These changes do
not modify simulator logging or relax any completion requirement.
