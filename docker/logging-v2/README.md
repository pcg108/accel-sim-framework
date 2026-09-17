# Accel-Sim 2.0 logging container

This directory defines the reproducible build and safe launch boundary for the
Accel-Sim 2.0 logging migration. Source code and generated data are bind-mounted;
they are not stored in the container overlay.

Exact upstream commits, the parent image digest, and the accepted nested
GPGPU-Sim port commit are recorded in `source-lock.env`.

## Build

The `Dockerfile` pins the exact parent-image digest. Building may still contact
the Ubuntu package mirrors for the explicitly listed build dependencies.

```bash
cd /scratch/prashanth/accelsim2/accel-sim-framework/docker/logging-v2
docker build --tag accel-sim-v2-logging:2.0 .
```

## Launch

The launcher expects the framework checkout at
`/scratch/prashanth/accelsim2/accel-sim-framework`.
It creates the results and Hugging Face cache directories when absent, mounts
both golden datasets read-only, and runs with the caller's UID and primary GID.
The launcher resolves the selected image tag to its immutable local content ID
and exports it as `ACCEL_SIM_IMAGE_REF` for run manifests. It also exports the
container-local golden trace roots:

- `RODINIA_TRACE_DIR=/goldens/rodinia/traces/nn-rodinia`
- `RESNET_TRACE_DIR=/goldens/resnet50/traces/resnet50`

```bash
/scratch/prashanth/accelsim2/accel-sim-framework/docker/logging-v2/launch.sh
```

For a persistent detached development container:

```bash
ACCELSIM2_DETACH=1 /scratch/prashanth/accelsim2/accel-sim-framework/docker/logging-v2/launch.sh
```

An optional command can replace the default interactive shell:

```bash
/scratch/prashanth/accelsim2/accel-sim-framework/docker/logging-v2/launch.sh /bin/bash -lc 'nvidia-smi'
```

The locations, image, name, UID, and GID can be overridden with the corresponding
`ACCELSIM2_*` variables defined near the top of `launch.sh`. The launcher refuses
to replace an existing `accel-sim-v2-logging` container; container removal is an
explicit operator action.
