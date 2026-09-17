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
`request_uid` is the identity joined to the corresponding L1 request;
`transport_uid` identifies a sector/transport request and `service_uid`
identifies the request that supplied shared L2 service.

The candidate family `l1_trace` is automatically mapped to the golden family
`l2_trace`. Use `--candidate-l1-dir` when `L1_TRACE_DIR` used another name.

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
