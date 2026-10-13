# Accel-Sim 2.0 regression harness

This harness runs the logging regressions in a new, immutable result directory.
It never writes into the golden trace trees. Selected traces are staged as
symbolic links, and every referenced input is SHA-256 hashed before the
simulator starts. The golden mounts themselves should be read-only.

## Common setup

Run the harness inside `accel-sim-v2-logging` with the source, golden, and
result mounts available. Use the immutable ID of the derived image, not merely
its mutable tag:

```bash
export ACCEL_SIM_ROOT=/accel-sim/accel-sim-framework
export GPGPUSIM_ROOT="$ACCEL_SIM_ROOT/gpu-simulator/gpgpu-sim"
export ACCEL_SIM_BINARY="$ACCEL_SIM_ROOT/gpu-simulator/bin/debug/accel-sim.out"
export ACCEL_SIM_CONFIG="$GPGPUSIM_ROOT/configs/model-cfgs/gpgpusim_8sm.config"
export ACCEL_SIM_IMAGE_REF='accel-sim-v2-logging@sha256:REPLACE_WITH_DERIVED_IMAGE_DIGEST'
export ACCEL_SIM_RESULTS_ROOT=/results
```

`LOG_CACHE_TRACES` is enabled by presence in the simulator, so setting it to
`0` does **not** disable it. The runner removes inherited logging variables and
sets them only for the selected mode.

The shader-cycle distribution (`shader-cycle distro`) logger is valid only for
sequential kernel execution. Do not interpret its per-kernel distribution for
traces that overlap kernels or exercise concurrent-kernel execution.

## Regression commands

Rodinia NN, all four kernels:

```bash
./run-regression.sh rodinia \
  --framework-root "$ACCEL_SIM_ROOT" \
  --gpgpusim-root "$GPGPUSIM_ROOT" \
  --simulator "$ACCEL_SIM_BINARY" \
  --config "$ACCEL_SIM_CONFIG" \
  --image-ref "$ACCEL_SIM_IMAGE_REF" \
  --output-dir "$ACCEL_SIM_RESULTS_ROOT/rodinia-$(date -u +%Y%m%dT%H%M%SZ)"
```

Representative ResNet-50 kernels 1, 13, and 23:

```bash
./run-regression.sh resnet-subset \
  --framework-root "$ACCEL_SIM_ROOT" \
  --gpgpusim-root "$GPGPUSIM_ROOT" \
  --simulator "$ACCEL_SIM_BINARY" \
  --config "$ACCEL_SIM_CONFIG" \
  --image-ref "$ACCEL_SIM_IMAGE_REF" \
  --output-dir "$ACCEL_SIM_RESULTS_ROOT/resnet-subset-$(date -u +%Y%m%dT%H%M%SZ)"
```

Full ResNet-50, all 50 kernels:

```bash
./run-regression.sh resnet-full \
  --framework-root "$ACCEL_SIM_ROOT" \
  --gpgpusim-root "$GPGPUSIM_ROOT" \
  --simulator "$ACCEL_SIM_BINARY" \
  --config "$ACCEL_SIM_CONFIG" \
  --image-ref "$ACCEL_SIM_IMAGE_REF" \
  --output-dir "$ACCEL_SIM_RESULTS_ROOT/resnet-full-$(date -u +%Y%m%dT%H%M%SZ)"
```

Scheduler logging smoke test, Rodinia kernel 1 and core 0 only:

```bash
./run-regression.sh scheduler-smoke \
  --framework-root "$ACCEL_SIM_ROOT" \
  --gpgpusim-root "$GPGPUSIM_ROOT" \
  --simulator "$ACCEL_SIM_BINARY" \
  --config "$ACCEL_SIM_CONFIG" \
  --image-ref "$ACCEL_SIM_IMAGE_REF" \
  --output-dir "$ACCEL_SIM_RESULTS_ROOT/scheduler-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
```

Normal regressions explicitly pass `-trace_enabled 0`. The scheduler smoke
passes `-trace_enabled 1 -trace_components WARP_SCHEDULER
-trace_sampling_core 0` and sets `SCHED_TRACE_DIR` to the result tree. You can
also add `--scheduler-trace on` to another workload, but doing so can produce a
large output.

Use `--cache-traces off` for the clean-v2/logging-v2 observational comparison.
In that mode `LOG_CACHE_TRACES`, `L2_TRACE_DIR`, and the legacy
`L1_TRACE_DIR` are absent. `L2_TRACE_DIR` selects the traffic log directory;
the simulator accepts `L1_TRACE_DIR` only when `L2_TRACE_DIR` is unset.

## Result contract

Each run creates exactly one new directory and refuses to reuse even an empty
one. It contains:

```text
data/                         simulator working directory and trace outputs
  l2_trace/                   source, LRC result, and completion records
  icache_status/
  dcache_status/
  warp_assignments/
  scheduler_traces/           only when scheduler tracing is enabled
inputs/trace/                 staged kernelslist plus links to golden traces
metadata/manifest.json        source SHAs, hashes, command, environment, status
metadata/input-selection.json selected IDs and per-trace hashes
metadata/inputs.sha256        human-readable checksum list
metadata/command.txt          shell-escaped exact command
metadata/simulator.*.log      captured stdout and stderr
```

The runner requires an immutable `sha256:...` image ID or `name@sha256:...`
repository digest and Git commits for both source roots.

`INCOMPLETE` exists from initialization until the simulator exits. It is
atomically renamed to `SUCCESS` on exit code zero or `FAILED` otherwise. The
manifest is finalized in either case and records the simulator exit code.

## Harness tests

The tests use tiny fake traces and a fake simulator; they do not launch an
Accel-Sim workload:

```bash
bash -n run-regression.sh tests/test-run-regression.sh
python3 -m py_compile lib/stage_inputs.py lib/manifest.py
tests/test-run-regression.sh
```

## Complete L2 timing coverage

`l2_to_icnt_timing.txt` covers regular L1 requests, non-TMA bypass loads/stores,
and TMA loads/stores. Join `request_uid` against the three source files and
use `transport_uid` for individual post-split requests. LRC contributors have
individual starts and share a service UID and memory-side completion cycle.
Parent latency is `max(child end_cycle) - source departure cycle`, after
confirming every audited child completed. This interval excludes the return
interconnect and register readiness.

Completed trace runs drain outstanding memory after the final kernel and emit
`L2_TRAFFIC_FINAL_DRAIN` in `simulator.log`, separately from kernel-finish
statistics. Strict acceptance requires a complete drain, reconciled final LRC
counters, and exactly one valid timing per tracked ingress transport. Reports
include per-source timing coverage and fully completed parent counts. See
[validator/README.md](validator/README.md) and [PORT_NOTES.md](PORT_NOTES.md).

When refreshing existing collections, preserve old outputs and pin the binary,
runner, and all validator modules to the refreshed collection. Old source-only
validation reports do not establish completion coverage. The mobile refresh
is stored under `/results/mobile-traffic-20261007/completion-timing-v2/`.

## L2 source logging first gate

Replay the complete Gemma 4 kernel 8039 with the unchanged H100 configuration
before broader testing. Use the newly rebuilt simulator and its library:

```bash
python3 util/logging-regression/run-l2-traffic-smoke.py \
  --binary gpu-simulator/build/logging-release/accel-sim.out \
  --library-dir gpu-simulator/build/logging-release/gpgpu-sim \
  --output /results/l2-traffic-8039-first
```

The output directory must be new. The runner records input/configuration
hashes, runs logging on and off, compares cycles, instructions, memory/cache
statistics and LRC counters, and invokes `validator/validate_l2_traffic.py`.
It also independently joins every source record's PC and instruction ordinal
through warp assignments to the original trace, checking that every active
source instruction is represented. `summary.json` reports the outcome and
unexercised categories; `SUCCESS` requires all checks to pass.

Kernel 8039 exercises bypass LDGSTS reads, TMA reads (including multicast),
and TMA stores. Its ordinary LDG instructions are predicated off, so an absent
L1 departure file is expected. Do not claim coverage of missing categories
from a successful run of this kernel alone.

After this gate passes, run the small, explicitly synthetic source cases:

```bash
python3 util/logging-regression/tests/run-l2-traffic-synthetic.py \
  --binary gpu-simulator/build/logging-release/accel-sim.out \
  --gate /results/l2-traffic-8039-first \
  --output /results/l2-traffic-synthetic
```

These generated traces cover ordinary L1 loads/stores, bypass loads/stores,
global atomics/reductions, TMA reductions, mixed-source merging, LRC disabled,
and attribution across two overlapping kernels. A separate dirty-writeback
case applies test-only overrides: a 1 KiB direct-mapped write-back L1,
adaptive L1 disabled, and dirty-eviction threshold zero. Conflict stores force
eviction, followed by a load that keeps the CTA alive while posted writebacks
drain. Without that final load, this simulator can finish a stores-only kernel
before its posted eviction traffic reaches L2; the validator rejects the
result as incomplete.

The cases use the fully linked
simulator binary and retain each command, generated trace, consistency report,
and summary. They are logging tests, not captured workloads or hardware
performance measurements. Sector-splitting tests are provided separately by
the C++ fixture. Use repeatable `--case` options to rerun only selected synthetic
cases; the default runs all four. The first gate always retains the H100
write-through configuration unchanged.

## Mobile workload traffic collection

`run-mobile-traffic.py` collects the four L2 traffic log types for every selected
entry in a capture's `kernelslist.g`. It runs each kernel in a separate simulator
process with cold caches, logging enabled, one active kernel, memcpy simulation
disabled, and scheduler tracing disabled. This is a per-kernel logging and
consistency check; it does not measure end-to-end inference or compare logging
on/off performance.

The available captures used for the mobile collection have three pairings:

| Capture | Mobile model config | Trace timing config |
|---|---|---|
| ResNet-50, 50 listed kernels | `gpgpusim_8sm.config` | `SM86_RTX3070/trace.config` |
| Gemma 4 Ampere capture, one projection kernel | `gpgpusim_8sm.config` | `SM86_RTX3070/trace.config` |
| Gemma 4 Hopper capture, 6,539 listed kernels | `gpgpusim_8sm_hopper.config` | `SM90_H100/trace.config` |

Model configs are under `gpu-simulator/gpgpu-sim/configs/model-cfgs`; trace
timings are under `gpu-simulator/configs/tested-cfgs`. Use the matching capture
and configuration pair. For example, to collect the available Hopper capture:

```bash
python3 util/logging-regression/run-mobile-traffic.py \
  --binary gpu-simulator/build/logging-release/accel-sim.out \
  --config gpu-simulator/gpgpu-sim/configs/model-cfgs/gpgpusim_8sm_hopper.config \
  --trace-config gpu-simulator/configs/tested-cfgs/SM90_H100/trace.config \
  --trace-dir data/gemma4-h100/traces \
  --output /results/mobile-hopper-gemma \
  --jobs 12 --validation-jobs 2 --timeout 1800
```

The output directory must be new. `--kernel-ids 6564,6565 6578` selects a subset;
omitting it selects every listed kernel. `--jobs` controls parallel kernel
workers, and `--validation-jobs` separately caps the memory-intensive validators.
Each completed simulation gets `traffic-validation.json`, reconciling source
records, independent interconnect audits, LRC contributors, and accepted L2
accesses. Kernel directories also retain source hashes, commands, simulator
logs, and status. `summary.json` is atomically updated after each kernel;
aggregate record counts include only kernels that pass validation.

Failed, timed-out, cycle-limited, and invalid-input kernels remain explicit
failures while later kernels continue. A known malformed input can be recorded
without replaying it using `--rejected-inputs FILE`, where `FILE` contains a JSON
object such as `{"50": {"reason": "truncated instruction", "line": 187663628}}`.
Retain the evidence for such rejections. `--max-cycles` is an optional safety
bound; reaching it never counts as success.

`--resume` requires the same binary, configuration, trace selection, and input
identity. It revalidates completed simulations with the current validator and
runs any kernels whose directories have not yet been created. It retains prior
incomplete simulations and input rejections as failures; use a fresh output
directory to replay those kernels. Missing source log categories are expected
when a kernel does not generate that traffic; Ampere with LRC disabled should
have no LRC result records.


### Versioned collection tools and capture handoff

`run-mobile-traffic.py` snapshots its executable, simulator shared libraries,
runner, and all importable validator modules into each collection's `tools/`
directory. `tools/manifest.json` hashes every copied file; resume checks the full
validator package and execution identity. Successful kernel statuses include
`timing_coverage` and `final_drain` alongside the source and consistency counts.

`run-traffic-collections.py --manifest /path/to/collection.json` supervises
multiple collections. Each manifest entry supplies `name`, `output`, and the
runner `argv`; the top level supplies `cwd`. An optional `capture_status` path
holds a collection until capture reports both `state=complete` and `passed=true`.
The supervisor writes an atomic `summary.json`, preserves terminal failures,
and terminates its child process groups on interruption. A refreshed collection
must use a new output directory; keep prior collection outputs as historical
evidence.

## TMA multicast destination regressions

`tests/run-tma-multicast.py` tests empty masks (including out-of-bounds zero
fill), inactive instructions, remote-only delivery without an issuing-CTA
barrier, self-only delivery, and delivery to both CTAs. Each case is replayed
from text and `.tracez`, checks actual CTA execution and final draining, and
checks transport/completion identities. Pass `--disable-lrc` to exercise the
direct L2 path. It accepts `--binary`, `--library-dir`, `--config` (a directory
containing `gpgpusim.config` and `trace.config`), and a fresh `--output` directory.

The simulator interprets the multicast mask as the complete destination set,
as specified for [Hopper TMA multicast](https://docs.nvidia.com/cuda/archive/12.0.1/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
An active instruction with an empty destination set still retires, but produces
no modeled data transfers or mbarrier transaction-byte completions. The issuing
CTA receives multicast completion only if its bit is set. This also applies to
out-of-bounds zero fill; an empty mask cannot signal a local barrier.

The strict traffic validator continues to reject zero-destination ingress.
Older logs containing those requests must be regenerated with corrected
simulation; accepting or deleting their timing rows would hide incorrect
traffic and barrier effects. These regressions verify simulator semantics,
not hardware measurements of any speculative memory-system activity.
