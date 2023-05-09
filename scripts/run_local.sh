#!/bin/bash
# Runs experiments on a local computer.
#
# This script can have multiple instances running at once, though the instances
# will need to have different seeds to prevent port conflicts.
#
# Usage:
#   bash scripts/run_local.sh CONFIG SEED NUM_WORKERS [RELOAD_PATH]
# Example:
#   # 8 workers with configuration config/foo.gin and seed 1.
#   bash scripts/run_local.sh config/foo.gin 1 8

#   # 8 workers with configuration config/foo.gin and seed 1, and binding the
#   `/project/` directory.
#   bash scripts/run_local.sh config/foo.gin 1 8 -p /project
#
#   # 4 workers with configuration config/foo.gin and seed 2, and reloading from
#   # old_dir/.
#   bash scripts/run_local.sh config/foo.gin 2 4 old_dir/

print_header() {
  echo
  echo "------------- $1 -------------"
}

# Prints "=" across an entire line.
print_thick_line() {
  printf "%0.s=" $(seq 1 `tput cols`)
  echo
}

#
# Parse command line flags.
#
USAGE="Usage: bash scripts/run_local.sh CONFIG SEED NUM_WORKERS [-p PROJECT_DIR] [-r RELOAD_PATH]"

CONFIG="$1"
SEED="$2"
NUM_WORKERS="$3"
shift 3

while getopts "p:r:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      r) RELOAD_PATH=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

echo "RELOAD_PATH: ${RELOAD_PATH}"

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:/project"
fi
echo "Singularity opts: ${SINGULARITY_OPTS}"

if [ -z "${NUM_WORKERS}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -n "$RELOAD_PATH" ]; then
  RELOAD_ARG="--reload ${RELOAD_PATH}"
else
  RELOAD_ARG=""
fi

set -u  # Uninitialized vars are error.

#
# Run the experiment.
#

SCHEDULER_FILE=".$(date +'%Y-%m-%d_%H-%M-%S')_scheduler_info.json"
PIDS_TO_KILL=()

# Use different port number so multiple jobs can start on one node. 8786
# is the default port. Then add offset of 10 and then the seed.
SCHEDULER_PORT=$((8786 + 10 + $SEED))

print_header "Starting Dask scheduler on port $SCHEDULER_PORT"
# shellcheck disable=SC2086
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
  dask-scheduler \
    --port $SCHEDULER_PORT \
    --scheduler-file $SCHEDULER_FILE &
PIDS_TO_KILL+=("$!")
sleep 2 # Wait for scheduler to start.

print_header "Starting Dask workers"
# shellcheck disable=SC2086
singularity exec ${SINGULARITY_OPTS} --nv singularity/ubuntu_warehouse.sif \
  dask-worker \
    --memory-limit="4 GiB" \
    --scheduler-file $SCHEDULER_FILE \
    --nprocs $NUM_WORKERS \
    --nthreads 1 &
PIDS_TO_KILL+=("$!")
sleep 5

print_header "Running experiment"
echo
print_thick_line
# shellcheck disable=SC2086
singularity exec ${SINGULARITY_OPTS} --nv singularity/ubuntu_warehouse.sif \
  python env_search/main.py \
    --config "$CONFIG" \
    --address "127.0.0.1:$SCHEDULER_PORT" \
    --seed "$SEED" \
    $RELOAD_ARG
print_thick_line

#
# Clean Up.
#

print_header "Cleanup"
for pid in ${PIDS_TO_KILL[*]}
do
  kill -9 "${pid}"
done

rm $SCHEDULER_FILE
