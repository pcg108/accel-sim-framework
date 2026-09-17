#!/usr/bin/env bash

set -Eeuo pipefail

readonly HARNESS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly DEFAULT_RODINIA_TRACE_DIR="/scratch/prashanth/gpu_model/accel-sim-data-rodinia_nn_smaller/traces/nn-rodinia"
readonly DEFAULT_RESNET_TRACE_DIR="/scratch/prashanth/gpu_model/accel-sim-data-resnet50-smaller/traces/resnet50"

usage() {
  cat <<'EOF'
Usage:
  run-regression.sh WORKLOAD --simulator PATH --config PATH \
    --framework-root PATH --gpgpusim-root PATH --image-ref REF \
    --output-dir ABSOLUTE_PATH [OPTIONS]

Workloads:
  rodinia          All kernels in the Rodinia NN golden trace set
  resnet-subset    ResNet-50 kernels 1, 13, and 23
  resnet-full      All kernels in the ResNet-50 golden trace set
  scheduler-smoke  Rodinia NN kernel 1 with scheduler tracing on core 0

Required options:
  --simulator PATH       accel-sim.out to execute
  --config PATH          GPGPU-Sim configuration file
  --framework-root PATH  Accel-Sim framework Git checkout
  --gpgpusim-root PATH   GPGPU-Sim Git checkout
  --image-ref REF        Immutable container image reference or image ID
  --output-dir PATH      New absolute run directory; it must not exist

Options:
  --rodinia-trace-dir PATH  Override the Rodinia trace directory
  --resnet-trace-dir PATH   Override the ResNet trace directory
  --cache-traces on|off     Enable the four cache/warp outputs (default: on)
  --scheduler-trace on|off  Enable WARP_SCHEDULER on core 0 (default: off;
                            scheduler-smoke requires on)
  --sim-arg VALUE           Append one simulator option/value token; repeatable
  -h, --help                Show this help

Environment defaults:
  ACCEL_SIM_BINARY, ACCEL_SIM_CONFIG, ACCEL_SIM_ROOT, GPGPUSIM_ROOT,
  ACCEL_SIM_IMAGE_REF, RODINIA_TRACE_DIR, RESNET_TRACE_DIR

LOG_CACHE_TRACES is presence-based in the instrumented simulator. This harness
always removes inherited logging variables before selectively setting them.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

require_value() {
  local option="$1"
  local value="${2-}"
  [[ -n "$value" ]] || die "$option requires a value"
}

canonical_existing_file() {
  local path="$1"
  [[ -f "$path" ]] || die "file does not exist: $path"
  realpath -e -- "$path"
}

canonical_existing_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "directory does not exist: $path"
  realpath -e -- "$path"
}

shell_join() {
  local arg
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

workload="${1-}"
if [[ -z "$workload" || "$workload" == "-h" || "$workload" == "--help" ]]; then
  usage
  [[ -n "$workload" ]] && exit 0
  exit 2
fi
shift

case "$workload" in
  rodinia | resnet-subset | resnet-full | scheduler-smoke) ;;
  *) die "unknown workload '$workload'" ;;
esac

simulator="${ACCEL_SIM_BINARY-}"
config="${ACCEL_SIM_CONFIG-}"
framework_root="${ACCEL_SIM_ROOT-}"
gpgpusim_root="${GPGPUSIM_ROOT-}"
image_ref="${ACCEL_SIM_IMAGE_REF-}"
output_dir=""
rodinia_trace_dir="${RODINIA_TRACE_DIR-$DEFAULT_RODINIA_TRACE_DIR}"
resnet_trace_dir="${RESNET_TRACE_DIR-$DEFAULT_RESNET_TRACE_DIR}"
cache_traces="on"
scheduler_trace="off"
declare -a extra_sim_args=()

if [[ "$workload" == "scheduler-smoke" ]]; then
  scheduler_trace="on"
fi

while (($#)); do
  case "$1" in
    --simulator)
      require_value "$1" "${2-}"
      simulator="$2"
      shift 2
      ;;
    --config)
      require_value "$1" "${2-}"
      config="$2"
      shift 2
      ;;
    --framework-root)
      require_value "$1" "${2-}"
      framework_root="$2"
      shift 2
      ;;
    --gpgpusim-root)
      require_value "$1" "${2-}"
      gpgpusim_root="$2"
      shift 2
      ;;
    --image-ref)
      require_value "$1" "${2-}"
      image_ref="$2"
      shift 2
      ;;
    --output-dir)
      require_value "$1" "${2-}"
      output_dir="$2"
      shift 2
      ;;
    --rodinia-trace-dir)
      require_value "$1" "${2-}"
      rodinia_trace_dir="$2"
      shift 2
      ;;
    --resnet-trace-dir)
      require_value "$1" "${2-}"
      resnet_trace_dir="$2"
      shift 2
      ;;
    --cache-traces)
      require_value "$1" "${2-}"
      cache_traces="$2"
      shift 2
      ;;
    --scheduler-trace)
      require_value "$1" "${2-}"
      scheduler_trace="$2"
      shift 2
      ;;
    --sim-arg)
      require_value "$1" "${2-}"
      extra_sim_args+=("$2")
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_sim_args+=("$@")
      break
      ;;
    *) die "unknown option '$1' (use --sim-arg to pass simulator arguments)" ;;
  esac
done

[[ "$cache_traces" == "on" || "$cache_traces" == "off" ]] ||
  die "--cache-traces must be 'on' or 'off'"
[[ "$scheduler_trace" == "on" || "$scheduler_trace" == "off" ]] ||
  die "--scheduler-trace must be 'on' or 'off'"
if [[ "$workload" == "scheduler-smoke" && "$scheduler_trace" != "on" ]]; then
  die "scheduler-smoke requires --scheduler-trace on"
fi

[[ -n "$simulator" ]] || die "--simulator (or ACCEL_SIM_BINARY) is required"
[[ -n "$config" ]] || die "--config (or ACCEL_SIM_CONFIG) is required"
[[ -n "$framework_root" ]] || die "--framework-root (or ACCEL_SIM_ROOT) is required"
[[ -n "$gpgpusim_root" ]] || die "--gpgpusim-root (or GPGPUSIM_ROOT) is required"
[[ -n "$image_ref" ]] || die "--image-ref (or ACCEL_SIM_IMAGE_REF) is required"
[[ -n "$output_dir" ]] || die "--output-dir is required"
[[ "$output_dir" == /* ]] || die "--output-dir must be an absolute path"
[[ "$image_ref" == sha256:* || "$image_ref" == *@sha256:* ]] ||
  die "--image-ref must be an immutable sha256 image ID or repo digest"

simulator="$(canonical_existing_file "$simulator")"
[[ -x "$simulator" ]] || die "simulator is not executable: $simulator"
config="$(canonical_existing_file "$config")"
framework_root="$(canonical_existing_dir "$framework_root")"
gpgpusim_root="$(canonical_existing_dir "$gpgpusim_root")"
rodinia_trace_dir="$(canonical_existing_dir "$rodinia_trace_dir")"
resnet_trace_dir="$(canonical_existing_dir "$resnet_trace_dir")"
git -C "$framework_root" rev-parse --verify HEAD >/dev/null 2>&1 ||
  die "framework root is not a Git checkout with a commit: $framework_root"
git -C "$gpgpusim_root" rev-parse --verify HEAD >/dev/null 2>&1 ||
  die "GPGPU-Sim root is not a Git checkout with a commit: $gpgpusim_root"

output_dir="$(realpath -m -- "$output_dir")"
[[ "$output_dir" != "/" ]] || die "refusing to use / as an output directory"
[[ ! -e "$output_dir" && ! -L "$output_dir" ]] ||
  die "output directory already exists; choose a fresh path: $output_dir"
output_parent="$(dirname -- "$output_dir")"
[[ -d "$output_parent" ]] || die "output parent does not exist: $output_parent"
[[ -w "$output_parent" ]] || die "output parent is not writable: $output_parent"

case "$workload" in
  rodinia)
    source_trace_dir="$rodinia_trace_dir"
    selected_ids="all"
    ;;
  resnet-subset)
    source_trace_dir="$resnet_trace_dir"
    selected_ids="1,13,23"
    ;;
  resnet-full)
    source_trace_dir="$resnet_trace_dir"
    selected_ids="all"
    ;;
  scheduler-smoke)
    source_trace_dir="$rodinia_trace_dir"
    selected_ids="1"
    ;;
esac

source_kernelslist="$source_trace_dir/kernelslist.g"
[[ -f "$source_kernelslist" ]] || die "missing kernels list: $source_kernelslist"

mkdir -- "$output_dir"
readonly output_dir
readonly metadata_dir="$output_dir/metadata"
readonly inputs_dir="$output_dir/inputs"
readonly data_dir="$output_dir/data"
readonly incomplete_marker="$output_dir/INCOMPLETE"
mkdir -- "$metadata_dir" "$inputs_dir" "$data_dir"
printf 'run started; see metadata/manifest.json\n' >"$incomplete_marker"

run_started=1
finalized=0
current_phase="staging-inputs"
finalize_run() {
  local rc="$?"
  if ((run_started == 1 && finalized == 0)); then
    finalized=1
    set +e
    printf '%s\n' "$rc" >"$metadata_dir/exit_code"
    if [[ -f "$metadata_dir/manifest.json" ]]; then
      python3 "$HARNESS_DIR/lib/manifest.py" finalize \
        --manifest "$metadata_dir/manifest.json" \
        --exit-code "$rc"
    else
      python3 "$HARNESS_DIR/lib/manifest.py" abort \
        --manifest "$metadata_dir/manifest.json" \
        --exit-code "$rc" \
        --workload "$workload" \
        --output-dir "$output_dir" \
        --image-ref "$image_ref" \
        --failure-phase "$current_phase"
    fi
    if ((rc == 0)); then
      mv -- "$incomplete_marker" "$output_dir/SUCCESS"
    else
      mv -- "$incomplete_marker" "$output_dir/FAILED"
    fi
  fi
  exit "$rc"
}
trap finalize_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 "$HARNESS_DIR/lib/stage_inputs.py" \
  --source-dir "$source_trace_dir" \
  --destination "$inputs_dir/trace" \
  --kernel-ids "$selected_ids" \
  --selection-json "$metadata_dir/input-selection.json" \
  --checksums "$metadata_dir/inputs.sha256"

staged_kernelslist="$inputs_dir/trace/kernelslist.g"
command_argv_file="$metadata_dir/command.argv0"
command_text_file="$metadata_dir/command.txt"

declare -a simulator_args=(
  "$simulator"
  -trace "$staged_kernelslist"
  -config "$config"
)
simulator_args+=("${extra_sim_args[@]}")

if [[ "$scheduler_trace" == "on" ]]; then
  simulator_args+=(
    -trace_enabled 1
    -trace_components WARP_SCHEDULER
    -trace_sampling_core 0
  )
else
  simulator_args+=(-trace_enabled 0)
fi

# env -u is intentional: LOG_CACHE_TRACES=0 still enables cache logging.
declare -a command=(
  env
  -u LOG_CACHE_TRACES
  -u L1_TRACE_DIR
  -u SCHED_TRACE_DIR
)
if [[ "$cache_traces" == "on" ]]; then
  command+=(
    LOG_CACHE_TRACES=1
    "L1_TRACE_DIR=$data_dir/l1_trace"
  )
fi
if [[ "$scheduler_trace" == "on" ]]; then
  command+=("SCHED_TRACE_DIR=$data_dir/scheduler_traces")
fi
command+=("${simulator_args[@]}")

printf '%s\0' "${command[@]}" >"$command_argv_file"
shell_join "${command[@]}" >"$command_text_file"

python3 "$HARNESS_DIR/lib/manifest.py" init \
  --manifest "$metadata_dir/manifest.json" \
  --workload "$workload" \
  --selected-kernel-ids "$selected_ids" \
  --output-dir "$output_dir" \
  --working-directory "$data_dir" \
  --simulator "$simulator" \
  --config "$config" \
  --framework-root "$framework_root" \
  --gpgpusim-root "$gpgpusim_root" \
  --harness-root "$HARNESS_DIR" \
  --image-ref "$image_ref" \
  --source-kernelslist "$source_kernelslist" \
  --staged-kernelslist "$staged_kernelslist" \
  --input-selection "$metadata_dir/input-selection.json" \
  --command-argv "$command_argv_file" \
  --cache-traces "$cache_traces" \
  --scheduler-trace "$scheduler_trace" \
  --stdout "$metadata_dir/simulator.stdout.log" \
  --stderr "$metadata_dir/simulator.stderr.log"

current_phase="simulator"
printf 'Starting %s; output: %s\n' "$workload" "$output_dir"
printf 'Command: '
shell_join "${command[@]}"

set +e
(
  cd -- "$data_dir"
  "${command[@]}"
) >"$metadata_dir/simulator.stdout.log" 2>"$metadata_dir/simulator.stderr.log"
simulator_rc=$?
set -e

exit "$simulator_rc"
