# Accel-Sim 2.0 logging port notes

## Supported flow

This port targets Accel-Sim's trace-driven frontend (`kernelslist.g` plus
`.traceg`/`.traceg.xz`/`.tracez`) and preserves these output families:

- `l1_trace`
- `icache_status`
- `dcache_status`
- `warp_assignments`
- optional `scheduler_traces`

Set `LOG_CACHE_TRACES=1` to enable the four required families and set
`L1_TRACE_DIR` to select the L1/L2 directory. The variable is presence-based,
so `LOG_CACHE_TRACES=0` also enables it; unset it to disable cache logging.
Scheduler logging is independently controlled by the simulator trace options
and `SCHED_TRACE_DIR`.

Native, non-trace-driven GPGPU-Sim execution is not an accepted mode: native
instructions do not currently carry the kernel provenance populated by the
Accel-Sim trace frontend.

## L2 timing record

The corrected record is:

```text
request_uid=<L1 parent> transport_uid=<sector request> service_uid=<L2 service request> addr=<hex> sector_mask=<hex> size=<bytes> start_cycle=<u64> end_cycle=<u64> elapsed_cycle=<u64>
```

`request_uid` joins to `l1_to_l2_requests.txt`. `transport_uid` distinguishes
sector children, and `service_uid` identifies the request that supplied a
coalesced L2 service. Completion timestamps remain 64-bit for long workloads.
Completed LRC entries are closed to late merges, remote-chiplet requests emit
only at final/source-side completion, and writebacks absorbed by L2 still emit
a timing row. These rules give one valid service interval for every logged L1
departure.

The validator accepts this schema and the archived four-field historical
schema, but rejects partial mixtures.

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
