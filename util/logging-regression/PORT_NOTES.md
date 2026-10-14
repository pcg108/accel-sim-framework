# Accel-Sim 2.0 logging port notes

## Supported flow

This port targets Accel-Sim's trace-driven frontend (`kernelslist.g` plus
`.traceg`/`.traceg.xz`/`.tracez`) and preserves these output families:

- `l2_trace`
- `icache_status`
- `dcache_status`
- `warp_assignments`
- optional `scheduler_traces`

Set `LOG_CACHE_TRACES=1` to enable the four required families and set
`L2_TRACE_DIR` to select the L1/L2 directory. The variable is presence-based,
so `LOG_CACHE_TRACES=0` also enables it; unset it to disable cache logging.
Scheduler logging is independently controlled by the simulator trace options
and `SCHED_TRACE_DIR`.

Native, non-trace-driven GPGPU-Sim execution is not an accepted mode: native
instructions do not currently carry the kernel provenance populated by the
Accel-Sim trace frontend.

## L2 timing record

The corrected record is:

```text
request_uid=<source parent> transport_uid=<sector request> service_uid=<L2 service request> addr=<hex> sector_mask=<hex> size=<bytes> start_cycle=<u64> end_cycle=<u64> elapsed_cycle=<u64>
```

`request_uid` joins to the union of `l1_to_l2_requests.txt`,
`l1_bypass_to_l2.txt`, and `tma_to_l2_requests.txt`. The source category comes
from that join. `transport_uid` distinguishes sector children; `service_uid`
identifies the request that supplied L2 service. Exactly one timing row is
required for every tracked post-split ingress transport. A TMA multicast has
one timing per transport, irrespective of destination count.

The interval is **source departure to memory-side service completion**. It
does not include the return interconnect or register readiness. Starts come
from immutable source provenance, and all timestamps are unsigned 64-bit.
LRC contributors retain individual source starts and share the representative's
service UID and completion cycle. Their timings are emitted before reply
ownership transfers, so interconnect retries do not duplicate completions.
An untracked representative can still service tracked contributors.

LRC sends the representative reply after the other contributors. This keeps
the source L2 queue from retaining a pointer whose ownership has transferred
to the interconnect while other replies wait for space. Fixing that existing
lifetime hazard can change reply ordering and simulation cycles relative to
older binaries. The behavior is identical with completion logging on or off.

Write allocation uses the original write's transport UID as its logical
service UID and the actual allocation-completion cycle. L1 writebacks receive
completion records whether L2 absorbs them or DRAM retires them. Internally
generated L2 writebacks and auxiliary chiplet forwarding copies are excluded.
Chiplet reads complete at the source chiplet after remote service returns.

For plotting, aggregate a parent request as:

```text
parent completion = max(child end_cycle)
parent latency = parent completion - source departure cycle
```

Do not average child latencies to estimate the time when a parent is complete.
Require every independently audited child before including a parent in a
complete-request latency plot.

`validate_traces.py` also accepts the archived four-field historical schema.
The strict audit validator requires the full nine fields and rejects old logs
without complete transport coverage.

### Final memory drain

After the final kernel finishes, the trace frontend advances outstanding
memory traffic at normal clock rates without issuing new instructions or
forcing cache flushes. Draining applies with logging enabled or disabled.
Kernel-finish statistics retain their existing boundary; a separate stdout
record reports the drain:

```text
L2_TRAFFIC_FINAL_DRAIN {"finish_cycle":100,"end_cycle":120,"drain_cycles":20,"complete":true,"lrc_incoming_sectors":8,"lrc_outgoing_sectors":6}
```

The strict validator requires exactly one report, validates its cycle
arithmetic and `complete` flag, and reconciles these final LRC counters instead
of the earlier kernel-finish counters. Every completion must fall at or before
`end_cycle`. Simulation limits or deadlock can interrupt draining; those runs
fail completeness validation and receive no synthetic completion timestamps.

## L2 source and LRC result records

With `LOG_CACHE_TRACES` set, the existing kernel/shader/scheduler hierarchy
also contains these lazily created files:

| File | Boundary and contents |
| --- | --- |
| `l1_to_l2_requests.txt` | Existing L1D departures; format unchanged |
| `l1_bypass_to_l2.txt` | Non-TMA loads and stores injected directly into the interconnect |
| `tma_to_l2_requests.txt` | TMA loads, stores, and reductions injected into the interconnect |
| `lrc_to_l2_requests.txt` | Finalized results of LRC forwarding, after sector splitting and merging |

The source files are mutually exclusive. Ordinary global atomics/reductions
follow the simulator's bypass-load classification. TMA reductions are stores.
Source request records are captured **before sector splitting**, at injection;
splitting occurs on the receiving memory subpartition before LRC insertion.

The two new source files use the existing L1 fields with `l1_to_l2_cycle`
replaced by `source_to_l2_cycle`, plus `source`. The labels are
`l1_bypass_load`, `l1_bypass_store`, `tma_load`, and `tma_store`. Their
`set_index` and `tag` are computed using L2 configuration. The existing L1 file
retains its original L1 configuration interpretation and schema.
When L2 is disabled, the new source/result logs use explicit unavailable
geometry values: `set_index=4294967295` (`UINT_MAX`) and
`tag=0xffffffffffffffff` (`UINT64_MAX`). They do not query an uninitialized
cache configuration.

```text
request_uid=<original departure> PC=<hex> dynamic_warp=<id> inst_ordinal=<id> addr=<hex> subpartition=<id> set_index=<id> tag=<hex> sector_mask=<hex> size=<bytes> type=<READ|WRITE|WRITEBACK> source=<label> source_to_l2_cycle=<u64>
```

Each LRC row identifies one forwarded request and its complete contributor
list, including singleton results. It is written once, before reply
distribution, when merging has closed. Its timestamp is the earlier
forwarding cycle, so file order is completion order rather than forwarding
order. Incomplete simulations can lack finalized outstanding groups.

```text
service_uid=<representative sector UID> addr=<hex> subpartition=<id> set_index=<L2 set> tag=<L2 tag hex> sector_mask=<hex> size=<bytes> lrc_to_l2_cycle=<u64> contributor_count=<count> contributors=<compact JSON array>
```

Each contributor JSON object contains `request_uid` (the original departure),
`transport_uid` (its sector), `source`, `access_type`, `kernel_uid`, `sm`,
`scheduler`, `dynamic_warp`, `inst_ordinal`, `PC` (hex string), and
`source_cycle`. L1 read contributors are labeled `l1_miss`, including
write-allocation reads. LRC can combine sources, warps, and kernels; the result
lives under the representative's directory while each contributor retains its
own immutable attribution. Additional instructions merged within L1's MSHRs
are outside this logging scope.

All tracked sources remain in their original departure files after merging.
Join using original request and sector UIDs; do not sum source rows and LRC
rows as independent traffic. Writes bypass LRC, and no LRC result file is
created with LRC disabled. Pure instruction/constant/texture groups are not
logged, but such members of a group containing tracked data traffic remain
visible as `other_read`. LRC merge limits come from configuration.

### Independent completeness validation

Set `L2_TRAFFIC_AUDIT_FILE` to an absolute JSONL output path for verification.
This optional audit captures actual interconnect departures independently of
the production source writers, and all post-split ingress requests before
LRC insertion. Each event records source/sector identity, immutable instruction
metadata, access and packet sizes, sector and 128-bit byte masks, source/event
cycles, write/LRC flags, and multicast destination count. Untracked ingress
uses `other_read` with an explicit unknown source UID.

For a completed run containing `l2_trace/`, `l2_accesses.txt`,
`l2_traffic_audit.jsonl`, and `simulator.log`:

```bash
python3 util/logging-regression/validator/validate_l2_traffic.py \
  --run /path/to/run --json-out /path/to/report.json --lrc-max-merged 4
```

The strict validator reconciles source/audit bijections, attribution, sector
and byte coverage, complete LRC membership, accepted L2 accesses (including
secondary `CHIPLET_ACC_R` lookups), exactly one timing per audited ingress
transport, and final simulator LRC counters. Timing checks include source,
transport and service IDs; directory attribution; address, mask and size;
source start, ordering and elapsed arithmetic; and common LRC completion cycles. The incoming
counter weights real TMA multicast by destination count; contributor count
alone does not reproduce it. Missing records, duplicate service/member IDs,
incorrect metadata, and incomplete or undrained runs fail validation. Stores-only
dirty-eviction tests exercise posted writebacks that outlive kernel execution.
The report includes per-source timing coverage and fully completed parent
counts; one missing split child prevents its parent from counting as complete.
Internal `L2_WR_ALLOC_R` reads can perform auxiliary peer-L2 lookups. These
must have type `READ`, are counted as `auxiliary_l2_lookups`, and do not require
source, ingress, or completion rows. They cannot satisfy a tracked source's
accepted-service requirement; other orphan accepted accesses remain errors.
Warp assignments are checked when available; the H100 smoke runner additionally checks originating
instructions against the input trace. The report lists absent source
categories explicitly instead of counting them as covered.

`validate_traces.py` accepts old trees and joins timings to all three source
files. New-source-only scheduler directories may contain timings without an L1
file. Archived golden trees need not contain the new source/result files or
such timing files. Historical L1 file comparisons remain strict. Its UID join
checks parent coverage; use the standalone audit validator for independent
post-split completeness and final-drain acceptance.

### First H100 kernel gate

After building the release simulator, run the complete Gemma 4 kernel 8039
with the checked-in H100 configs. The output directory must not already exist:

```bash
python3 util/logging-regression/run-l2-traffic-smoke.py \
  --binary gpu-simulator/build/logging-release/accel-sim.out \
  --library-dir gpu-simulator/build/logging-release/gpgpu-sim \
  --output data/gemma4-h100/l2-traffic-smoke
```

`--library-dir` points directly to the matching rebuilt simulator
`libcudart.so` in the CMake build tree. The runner
records input/config/binary/library hashes and commands, compares logging on
and off, invokes both audit and originating-instruction validation, and writes
`summary.json` plus `SUCCESS` or `FAILED`.

The first passing kernel 8039 gate checked 504,000 source departures:
192 bypass LDGSTS reads, 491,520 TMA reads (including multicast), and 12,288
TMA stores. All 491,712 read contributors resolved to 234,956 LRC results;
including the stores and 106,592 secondary chiplet lookups reconciled all
353,836 accepted L2 accesses. Every source row mapped to its active originating
instruction; 3,504 such instructions across 48 CTAs and 576 warps were covered.
No missing, orphan, or duplicate records remained. All 133 compared simulator
statistics matched with logging enabled and disabled, including 12,984 cycles
and 19,840,968 instructions.

That kernel exercises no L1 source departures, non-TMA stores, ordinary
atomics/reductions, TMA reductions, concurrent kernels, or disabled LRC. It
also produced **zero sector splits and zero mixed-source LRC groups**; those
behaviors are not validated by this kernel gate. Targeted synthetic simulator
cases and validator unit fixtures follow this first gate for those paths.

The gate exposed preexisting uninitialized multicast fields on ordinary
`warp_inst_t` instructions, which caused bogus, nondeterministic ICNT-to-LRC
counts for non-TMA LDGSTS reads. Constructors now initialize those fields to
false/zero. The corrected incoming counter (including legitimate TMA multicast
weighting) is 786,624 sectors for this kernel, identically with logging on and
off. This fix does not alter the kernel's simulated cycles or instruction count.

### Targeted runtime fixture

After the H100 gate, exercise the real memory-subpartition ingress and sector
splitter using the existing CMake build (with `compile_commands.json`):

```bash
python3 util/logging-regression/tests/run-l2-traffic-runtime.py \
  --build-dir gpu-simulator/build/logging-release \
  --output data/gemma4-h100/l2-traffic-runtime
```

The passing fixture checks 9 source requests, 18 resulting sector requests,
and 14 finalized LRC results containing all 18 contributors. It covers sector
splitting and source provenance independently of kernel 8039. It invokes the
actual public ingress/splitter path, but does not simulate cache completion;
the existing special 64-byte same-address merge behavior is preserved.
Those children can have distinct sector masks while sharing the LRC address
key. Validation retains each child's audited mask and requires the result's
mask to match its representative, without requiring all contributor masks to
be identical.

This fixture exposed a preexisting cumulative byte-mask bug in the 128-byte
and special 64-byte splitting loops: each later child could retain byte bits
from earlier sectors. Initializing the temporary mask inside each sector
iteration fixes the overlap. The repeated fixture passes both provenance and
per-sector byte coverage checks with no findings.

## Build

Inside the logging container:

```bash
export ACCELSIM_CONFIG=debug
cmake -S gpu-simulator -B gpu-simulator/build/logging-debug
cmake --build gpu-simulator/build/logging-debug -j8
cmake --install gpu-simulator/build/logging-debug
```

Use `ACCELSIM_CONFIG=release` and a separate `logging-release` build directory
for the release binary.

## Known boundaries

- `GPGPUSIM_LOG_SHADER_CYCLE_DISTRO` is valid only for sequential kernels.
- Fully buffered trace tails may be lost if the simulator terminates through
  `_exit` after a crash; normal completion and per-kernel close flush files.
- The upstream LRC multicast retry path retains its base request across partial
  sends; prolonged interconnect backpressure remains an upstream risk.
- The custom 8-SM regression configuration does not model an H100. It is used
  only to compare logging structure and invariants on established traces.
