#!/usr/bin/env bash

set -Eeuo pipefail

readonly TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly HARNESS_DIR="$(dirname -- "$TEST_DIR")"
readonly TEMP_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_ROOT"' EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_file() {
  [[ -f "$1" ]] || fail "expected file: $1"
}

assert_dir() {
  [[ -d "$1" ]] || fail "expected directory: $1"
}

assert_not_exists() {
  [[ ! -e "$1" && ! -L "$1" ]] || fail "did not expect path: $1"
}

assert_contains() {
  local expected="$1"
  local path="$2"
  grep -Fq -- "$expected" "$path" || fail "'$expected' not found in $path"
}

framework="$TEMP_ROOT/framework"
gpgpusim="$framework/gpu-simulator/gpgpu-sim"
rodinia="$TEMP_ROOT/rodinia"
resnet="$TEMP_ROOT/resnet"
mkdir -p -- "$gpgpusim" "$rodinia" "$resnet"

git -C "$framework" init -q
git -C "$framework" -c user.name=Test -c user.email=test@example.invalid \
  commit --allow-empty -qm initial
git -C "$gpgpusim" init -q
git -C "$gpgpusim" -c user.name=Test -c user.email=test@example.invalid \
  commit --allow-empty -qm initial

for kernel_id in 1 2 3 4; do
  printf 'rodinia trace %s\n' "$kernel_id" >"$rodinia/kernel-$kernel_id.traceg"
done
for kernel_id in $(seq 1 50); do
  printf 'resnet trace %s\n' "$kernel_id" >"$resnet/kernel-$kernel_id.traceg"
done
{
  for kernel_id in 1 2 3 4; do
    printf 'MemcpyHtoD,0x1000,32\n'
    printf 'kernel-%s.traceg\n' "$kernel_id"
  done
} >"$rodinia/kernelslist.g"
{
  for kernel_id in $(seq 1 50); do
    printf 'kernel-%s.traceg\n' "$kernel_id"
  done
} >"$resnet/kernelslist.g"

config="$TEMP_ROOT/gpgpusim_8sm.config"
printf '%s\n' '-trace_enabled 1' >"$config"
fake_simulator="$TEMP_ROOT/fake-accel-sim.out"
cat >"$fake_simulator" <<'EOF'
#!/usr/bin/env bash
set -eu
{
  printf 'LOG_CACHE_TRACES=%s\n' "${LOG_CACHE_TRACES-<unset>}"
  printf 'L1_TRACE_DIR=%s\n' "${L1_TRACE_DIR-<unset>}"
  printf 'SCHED_TRACE_DIR=%s\n' "${SCHED_TRACE_DIR-<unset>}"
  printf 'ARGS='
  printf '<%s>' "$@"
  printf '\n'
} >fake-observation.txt
if [[ -n "${L1_TRACE_DIR-}" ]]; then
  mkdir -p -- "$L1_TRACE_DIR" icache_status dcache_status warp_assignments
fi
if [[ -n "${SCHED_TRACE_DIR-}" ]]; then
  mkdir -p -- "$SCHED_TRACE_DIR"
fi
exit "${FAKE_EXIT_CODE-0}"
EOF
chmod +x "$fake_simulator"

common_args=(
  --simulator "$fake_simulator"
  --config "$config"
  --framework-root "$framework"
  --gpgpusim-root "$gpgpusim"
  --image-ref 'test-image@sha256:0123456789abcdef'
  --rodinia-trace-dir "$rodinia"
  --resnet-trace-dir "$resnet"
)

rodinia_output="$TEMP_ROOT/out-rodinia"
LOG_CACHE_TRACES=0 L1_TRACE_DIR=/bad SCHED_TRACE_DIR=/bad \
  "$HARNESS_DIR/run-regression.sh" rodinia "${common_args[@]}" \
  --output-dir "$rodinia_output"
assert_file "$rodinia_output/SUCCESS"
assert_not_exists "$rodinia_output/INCOMPLETE"
assert_contains 'LOG_CACHE_TRACES=1' "$rodinia_output/data/fake-observation.txt"
assert_contains 'SCHED_TRACE_DIR=<unset>' "$rodinia_output/data/fake-observation.txt"
assert_contains '<-trace_enabled><0>' "$rodinia_output/data/fake-observation.txt"
assert_dir "$rodinia_output/data/l1_trace"

subset_output="$TEMP_ROOT/out-subset"
"$HARNESS_DIR/run-regression.sh" resnet-subset "${common_args[@]}" \
  --output-dir "$subset_output"
mapfile -t subset_lines <"$subset_output/inputs/trace/kernelslist.g"
[[ "${subset_lines[*]}" == 'kernel-1.traceg kernel-13.traceg kernel-23.traceg' ]] ||
  fail "unexpected ResNet subset: ${subset_lines[*]}"
[[ -L "$subset_output/inputs/trace/kernel-13.traceg" ]] ||
  fail "subset trace was not staged as a symlink"
python3 - "$subset_output/metadata/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["status"] == "succeeded"
assert manifest["complete"] is True
assert manifest["exit_code"] == 0
assert manifest["workload"]["selected_kernel_ids"] == [1, 13, 23]
assert manifest["sources"]["framework"]["commit"]
assert manifest["sources"]["gpgpusim"]["commit"]
assert len(manifest["artifacts"]["input_selection"]["trace_files"]) == 3
PY

full_output="$TEMP_ROOT/out-full"
"$HARNESS_DIR/run-regression.sh" resnet-full "${common_args[@]}" \
  --output-dir "$full_output" --cache-traces off
[[ "$(grep -c '^kernel-' "$full_output/inputs/trace/kernelslist.g")" == 50 ]] ||
  fail "full ResNet list does not contain 50 kernels"
assert_contains 'LOG_CACHE_TRACES=<unset>' "$full_output/data/fake-observation.txt"
assert_contains 'L1_TRACE_DIR=<unset>' "$full_output/data/fake-observation.txt"

scheduler_output="$TEMP_ROOT/out-scheduler"
"$HARNESS_DIR/run-regression.sh" scheduler-smoke "${common_args[@]}" \
  --output-dir "$scheduler_output"
assert_contains 'SCHED_TRACE_DIR=' "$scheduler_output/data/fake-observation.txt"
assert_contains '<-trace_enabled><1>' "$scheduler_output/data/fake-observation.txt"
assert_contains '<-trace_components><WARP_SCHEDULER>' \
  "$scheduler_output/data/fake-observation.txt"
assert_contains '<-trace_sampling_core><0>' "$scheduler_output/data/fake-observation.txt"
[[ "$(grep -c '^kernel-' "$scheduler_output/inputs/trace/kernelslist.g")" == 1 ]] ||
  fail "scheduler smoke should stage one kernel"

set +e
"$HARNESS_DIR/run-regression.sh" rodinia "${common_args[@]}" \
  --output-dir "$rodinia_output" >/dev/null 2>&1
existing_rc=$?
set -e
[[ "$existing_rc" == 2 ]] || fail "existing output should fail with exit 2"

failed_output="$TEMP_ROOT/out-failed"
set +e
FAKE_EXIT_CODE=7 "$HARNESS_DIR/run-regression.sh" rodinia "${common_args[@]}" \
  --output-dir "$failed_output"
failed_rc=$?
set -e
[[ "$failed_rc" == 7 ]] || fail "expected fake simulator exit 7, got $failed_rc"
assert_file "$failed_output/FAILED"
assert_not_exists "$failed_output/INCOMPLETE"
python3 - "$failed_output/metadata/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["status"] == "failed"
assert manifest["complete"] is False
assert manifest["exit_code"] == 7
PY

broken_rodinia="$TEMP_ROOT/broken-rodinia"
mkdir -- "$broken_rodinia"
printf 'kernel-1.traceg\n' >"$broken_rodinia/kernelslist.g"
staging_failure_output="$TEMP_ROOT/out-staging-failure"
set +e
"$HARNESS_DIR/run-regression.sh" rodinia "${common_args[@]}" \
  --rodinia-trace-dir "$broken_rodinia" \
  --output-dir "$staging_failure_output" >/dev/null 2>&1
staging_failure_rc=$?
set -e
[[ "$staging_failure_rc" == 2 ]] ||
  fail "expected input staging exit 2, got $staging_failure_rc"
assert_file "$staging_failure_output/FAILED"
python3 - "$staging_failure_output/metadata/manifest.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["status"] == "failed"
assert manifest["complete"] is False
assert manifest["exit_code"] == 2
assert manifest["failure_phase"] == "staging-inputs"
PY

printf 'All harness tests passed.\n'
