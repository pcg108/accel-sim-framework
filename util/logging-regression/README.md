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
In that mode both `LOG_CACHE_TRACES` and `L1_TRACE_DIR` are absent.

## Result contract

Each run creates exactly one new directory and refuses to reuse even an empty
one. It contains:

```text
data/                         simulator working directory and trace outputs
  l1_trace/                   L1-to-L2 and L2-to-ICNT records
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
