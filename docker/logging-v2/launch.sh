#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEFAULT_IMAGE="accel-sim-v2-logging:2.0"
readonly DEFAULT_CONTAINER_NAME="accel-sim-v2-logging"
readonly DEFAULT_SOURCE_DIR="/scratch/prashanth/accelsim2/accel-sim-framework"
readonly DEFAULT_RESULTS_DIR="/scratch/prashanth/accelsim2-results"
readonly DEFAULT_HF_CACHE_DIR="/scratch/prashanth/accelsim2-hf-cache"
readonly DEFAULT_RODINIA_GOLDEN_DIR="/scratch/prashanth/gpu_model/accel-sim-data-rodinia_nn_smaller"
readonly DEFAULT_RESNET_GOLDEN_DIR="/scratch/prashanth/gpu_model/accel-sim-data-resnet50-smaller"

die() {
    echo "launch.sh: error: $*" >&2
    exit 1
}

require_directory() {
    local description="$1"
    local path="$2"
    [[ -d "$path" ]] || die "$description directory does not exist: $path"
}

canonical_directory() {
    local path="$1"
    realpath --canonicalize-existing "$path"
}

validate_mount_path() {
    local path="$1"
    [[ "$path" != *","* && "$path" != *$'\n'* ]] \
        || die "mount paths may not contain commas or newlines: $path"
}

command -v docker >/dev/null 2>&1 || die "docker is not installed or is not on PATH"
command -v realpath >/dev/null 2>&1 || die "realpath is not installed or is not on PATH"

image="${ACCELSIM2_IMAGE:-$DEFAULT_IMAGE}"
container_name="${ACCELSIM2_CONTAINER_NAME:-$DEFAULT_CONTAINER_NAME}"
source_dir="${ACCELSIM2_SOURCE_DIR:-$DEFAULT_SOURCE_DIR}"
results_dir="${ACCELSIM2_RESULTS_DIR:-$DEFAULT_RESULTS_DIR}"
hf_cache_dir="${ACCELSIM2_HF_CACHE_DIR:-$DEFAULT_HF_CACHE_DIR}"
rodinia_golden_dir="${ACCELSIM2_RODINIA_GOLDEN_DIR:-$DEFAULT_RODINIA_GOLDEN_DIR}"
resnet_golden_dir="${ACCELSIM2_RESNET_GOLDEN_DIR:-$DEFAULT_RESNET_GOLDEN_DIR}"
host_uid="${ACCELSIM2_UID:-$(id -u)}"
host_gid="${ACCELSIM2_GID:-$(id -g)}"
detach="${ACCELSIM2_DETACH:-0}"

[[ "$container_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || die "invalid Docker container name: $container_name"
[[ "$host_uid" =~ ^[0-9]+$ ]] || die "ACCELSIM2_UID must be numeric"
[[ "$host_gid" =~ ^[0-9]+$ ]] || die "ACCELSIM2_GID must be numeric"
[[ "$detach" == "0" || "$detach" == "1" ]] \
    || die "ACCELSIM2_DETACH must be 0 or 1"

docker info >/dev/null 2>&1 \
    || die "cannot access the Docker daemon; check socket permissions"

if docker container inspect "$container_name" >/dev/null 2>&1; then
    die "container '$container_name' already exists; start or remove it explicitly rather than replacing it"
fi

image_ref="$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null)" \
    || die "image '$image' is not present; build it explicitly before launching"
[[ "$image_ref" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "Docker returned an invalid immutable image ID for '$image': $image_ref"

require_directory "Accel-Sim source" "$source_dir"
require_directory "Rodinia golden" "$rodinia_golden_dir"
require_directory "ResNet golden" "$resnet_golden_dir"

umask 0002
mkdir -p "$results_dir" "$hf_cache_dir"
mkdir -p "$results_dir/.container-home" "$results_dir/.cache"

[[ -w "$source_dir" ]] || die "Accel-Sim source directory is not writable: $source_dir"
[[ -w "$results_dir" ]] || die "results directory is not writable: $results_dir"
[[ -w "$hf_cache_dir" ]] || die "Hugging Face cache directory is not writable: $hf_cache_dir"

source_dir="$(canonical_directory "$source_dir")"
results_dir="$(canonical_directory "$results_dir")"
hf_cache_dir="$(canonical_directory "$hf_cache_dir")"
rodinia_golden_dir="$(canonical_directory "$rodinia_golden_dir")"
resnet_golden_dir="$(canonical_directory "$resnet_golden_dir")"

for path in \
    "$source_dir" \
    "$results_dir" \
    "$hf_cache_dir" \
    "$rodinia_golden_dir" \
    "$resnet_golden_dir"; do
    validate_mount_path "$path"
done

tty_args=()
if [[ "$detach" == "1" ]]; then
    tty_args=(-d)
elif [[ -t 0 && -t 1 ]]; then
    tty_args=(-it)
elif [[ -t 0 ]]; then
    tty_args=(-i)
fi

container_command=("$@")
if (( ${#container_command[@]} == 0 )); then
    if [[ "$detach" == "1" ]]; then
        container_command=(sleep infinity)
    else
        container_command=(/bin/bash)
    fi
fi

exec docker run \
    "${tty_args[@]}" \
    --name "$container_name" \
    --hostname "$container_name" \
    --init \
    --gpus all \
    --shm-size 16g \
    --ulimit nofile=1048576:1048576 \
    --user "$host_uid:$host_gid" \
    --env "HOME=/results/.container-home" \
    --env "HOST_UID=$host_uid" \
    --env "HOST_GID=$host_gid" \
    --env "HF_HOME=/hf-cache" \
    --env "TRANSFORMERS_CACHE=/hf-cache/transformers" \
    --env "XDG_CACHE_HOME=/results/.cache" \
    --env "ACCEL_SIM_IMAGE_REF=$image_ref" \
    --env "RODINIA_TRACE_DIR=/goldens/rodinia/traces/nn-rodinia" \
    --env "RESNET_TRACE_DIR=/goldens/resnet50/traces/resnet50" \
    --mount "type=bind,src=$source_dir,dst=/accel-sim/accel-sim-framework" \
    --mount "type=bind,src=$results_dir,dst=/results" \
    --mount "type=bind,src=$hf_cache_dir,dst=/hf-cache" \
    --mount "type=bind,src=$rodinia_golden_dir,dst=/goldens/rodinia,readonly" \
    --mount "type=bind,src=$resnet_golden_dir,dst=/goldens/resnet50,readonly" \
    --workdir /accel-sim/accel-sim-framework \
    "$image" \
    "${container_command[@]}"
