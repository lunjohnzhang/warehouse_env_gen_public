#!/bin/bash
# Runs scripts on an HPC with Slurm installed.
#
# This script sources an HPC_CONFIG, which is a bash script with the following
# variables defined:
# - HPC_SLURM_ACCOUNT (the slurm username for running the jobs)
# - HPC_SLURM_TIME (time limit for the jobs, specified as HH:MM:SS)
# - HPC_SLURM_NUM_NODES (number of slurm nodes to run)
# - HPC_SLURM_CPUS_PER_NODE (number of CPUs per node; note we always run one
#   task per node)
# - HPC_SLURM_MEM_PER_CPU (memory per CPU)
# - HPC_MASTER_WORKERS (number of additional workers to allocate on the master
#   node, so that the main script can start immediately without having to wait
#   for additional worker nodes to join - note that the master node already
#   allocates 2 CPUs, for the scheduler and main experiment script)
# - HPC_MASTER_GPU (set to any string to indicate that a GPU should be used on
#   the master node; otherwise leave it out or set it to empty string)
#
# For instance, a file might look like:
#
#   HPC_SLURM_ACCOUNT=account_123
#   HPC_SLURM_TIME=20:00:00
#   HPC_SLURM_NUM_NODES=10
#   HPC_SLURM_CPUS_PER_NODE=12
#   HPC_SLURM_MEM_PER_CPU=2GB
#   HPC_MASTER_WORKERS=2
#   HPC_MASTER_GPU=true
#
# Other options:
# - Pass -d to perform a dry run (i.e. don't submit any scripts).
# - Pass -r LOGDIR to reload from an existing logging directory and continue an
#   experiment.
#
# NOTE: Keep the SEED small, as we add it to the default port number (8786) so
# that we have a (somewhat) unique port and can start multiple jobs on one node.
#
# NOTE: If you do not have a /project directory on your cluster, comment out the
# PROJECT_DIR variable below.
#
# Usage:
#   bash scripts/run_slurm.sh CONFIG SEED HPC_CONFIG [-d] [-r LOGDIR]
#
# Example:
#   bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/foo.sh
#
#   # Dry run version of the above.
#   bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/foo.sh -d
#
#   # Run the above, with reloading from old_dir/
#   bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/foo.sh -r old_dir/

print_header() {
  echo
  echo "------------- $1 -------------"
}

#
# Set singularity opts -- comment out PROJECT_DIR if you do not have /project on
# your cluster.
#

# PROJECT_DIR="/project"
SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:/project"
fi
echo "Singularity opts: ${SINGULARITY_OPTS}"

#
# Parse command line flags.
#

CONFIG="$1"
SEED="$2"
HPC_CONFIG="$3"
shift 3  # Remove first 3 parameters so getopts does not see them.

if [ -z "$HPC_CONFIG" ]
then
  echo "Usage: bash scripts/run_slurm.sh CONFIG SEED HPC_CONFIG [-d] [-r LOGDIR]"
  exit 1
fi

# For more info on getopts, see https://bytexd.com/using-getopts-in-bash/
DRY_RUN=""
RELOAD_ARG=""
while getopts "dr:" opt; do
  case $opt in
    d)
      echo "Using DRY RUN"
      DRY_RUN="1"
      ;;
    r)
      echo "Using RELOAD: $OPTARG"
      RELOAD_ARG="--reload $OPTARG"
      ;;
  esac
done

#
# Parse HPC config.
#

# Defines HPC_SLURM_ACCOUNT, HPC_SLURM_TIME, HPC_SLURM_NUM_NODES,
# HPC_SLURM_CPUS_PER_NODE, HPC_SLURM_MEM_PER_CPU, and maybe HPC_MASTER_GPU.
source "$HPC_CONFIG"

if [ -z "$HPC_SLURM_ACCOUNT" ] ||
   [ -z "$HPC_SLURM_TIME" ] ||
   [ -z "$HPC_SLURM_NUM_NODES" ] ||
   [ -z "$HPC_SLURM_CPUS_PER_NODE" ] ||
   [ -z "$HPC_SLURM_MEM_PER_CPU" ] ||
   [ -z "$HPC_MASTER_WORKERS" ]
then
echo "\
HPC_CONFIG must have the following variables defined:
- HPC_SLURM_ACCOUNT
- HPC_SLURM_TIME
- HPC_SLURM_NUM_NODES
- HPC_SLURM_CPUS_PER_NODE
- HPC_MASTER_WORKERS"
  exit 1
fi

if [ -z "$HPC_MASTER_GPU" ]; then
  HPC_MASTER_GPU=""  # Make sure HPC_MASTER_GPU is initialized.
fi

set -u  # Uninitialized vars are error.

#
# Build and submit slurm scripts.
#

# Global storage holding all job ids. Newline-delimited string, where each line
# holds name;job_id.
JOB_IDS=""

# Submits a script and records it in JOB_IDS.
submit_script() {
  name="$1"
  slurm_script="$2"
  output=$(sbatch --parsable "$slurm_script")
  IFS=';' read -ra tokens <<< "$output"
  job_id="${tokens[0]}"
  JOB_IDS="${JOB_IDS}${name};${job_id}\n"
  echo "Submitted $job_id ($name)"
}

print_header "Create logging directory"
DATE="$(date +'%Y-%m-%d_%H-%M-%S')"
LOGDIR="slurm_logs/slurm_${DATE}"
echo "SLURM Log directory: ${LOGDIR}"

# Save config.
mkdir -p "$LOGDIR/config"
cp "$HPC_CONFIG" "$LOGDIR/config/"

print_header "Submitting scheduler"
SCHEDULER_SCRIPT="${LOGDIR}/scheduler.slurm"
SCHEDULER_OUTPUT="${LOGDIR}/scheduler.out"
SCHEDULER_FILE="${LOGDIR}/scheduler_info.json"
# 1 CPU for scheduler, 1 CPU for main script, and a couple extra workers.
SCHEDULER_CPUS=$(( 2 + $HPC_MASTER_WORKERS ))
# Use different port number so multiple jobs can start on one node. 8786
# is the default port. Then add offset of 10 and then the seed.
SCHEDULER_PORT=$((8786 + 10 + $SEED))
echo "Starting scheduler from: ${SCHEDULER_SCRIPT}"

echo "\
#!/bin/bash
#SBATCH --job-name=${DATE}_scheduler
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=$SCHEDULER_CPUS
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=$HPC_SLURM_TIME
#SBATCH --account=$HPC_SLURM_ACCOUNT
#SBATCH --output $SCHEDULER_OUTPUT
#SBATCH --error $SCHEDULER_OUTPUT
$(if [ -n "$HPC_MASTER_GPU" ]; then echo -e "#SBATCH --partition=gpu\n#SBATCH --gres=gpu:1"; fi)

echo
echo \"========== Start ==========\"
date

$(if [ -n "$HPC_MASTER_GPU" ]; then echo "module load cuda/10.2.89"; fi)

# Start the scheduler.
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \\
  dask-scheduler \\
    --port $SCHEDULER_PORT \\
    --scheduler-file $SCHEDULER_FILE &

sleep 10  # Wait for scheduler to start.

# Parse address from scheduler file.
address=\$(singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif python -c \"\\
import json
with open('$SCHEDULER_FILE', 'r') as file:
  print(json.load(file)['address'])
\")

# Start main experiment in the background.
singularity exec ${SINGULARITY_OPTS} $(if [ -n "$HPC_MASTER_GPU" ]; then echo "--nv"; fi) \\
  singularity/ubuntu_warehouse.sif \\
  python env_search/main.py \\
    --config $CONFIG $RELOAD_ARG \\
    --address \$address \\
    --slurm-logdir $LOGDIR \\
    --seed $SEED &

# Start some workers so that the scheduler always has a few workers with it and
# does not block.
# Important to not run in background, as the worker must block so that the slurm
# job does not terminate.
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \\
  dask-worker \\
    --scheduler-file $SCHEDULER_FILE \\
    --nprocs $HPC_MASTER_WORKERS \\
    --nthreads 1

echo
echo \"========== Done ==========\"
date" > "$SCHEDULER_SCRIPT"

# Submit the scheduler script.
if [ -z "$DRY_RUN" ]; then submit_script "scheduler" "$SCHEDULER_SCRIPT"; fi

print_header "Submitting workers"
for (( worker_id = 0; worker_id < $HPC_SLURM_NUM_NODES; worker_id++ ))
do

  WORKER_SCRIPT="${LOGDIR}/worker-${worker_id}.slurm"
  WORKER_OUTPUT="${LOGDIR}/worker-${worker_id}.out"
  echo "Starting worker-${worker_id} from: ${WORKER_SCRIPT}"

  # Write the worker script. Note that we do not need to worry about cleaning up
  # workers. When the schedulers die, the workers eventually die too ;)
  echo "\
#!/bin/bash
#SBATCH --job-name=${DATE}_worker-${worker_id}
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=$HPC_SLURM_CPUS_PER_NODE
#SBATCH --mem-per-cpu=$HPC_SLURM_MEM_PER_CPU
#SBATCH --time=$HPC_SLURM_TIME
#SBATCH --account=$HPC_SLURM_ACCOUNT
#SBATCH --output $WORKER_OUTPUT
#SBATCH --error $WORKER_OUTPUT

echo
echo \"========== Start ==========\"
date

# Important to not run in background, as the worker must block so that the slurm
# job does not terminate.
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \\
  dask-worker \\
    --scheduler-file $SCHEDULER_FILE \\
    --nprocs $HPC_SLURM_CPUS_PER_NODE \\
    --nthreads 1

echo
echo \"========== Done ==========\"
date" > "$WORKER_SCRIPT"

  # Submit the worker script.
  if [ -z "$DRY_RUN" ]; then submit_script "worker-${worker_id}" "$WORKER_SCRIPT"; fi

done

#
# Print monitoring instructions.
#

print_header "Monitoring Instructions"
echo "\
To view output from the scheduler and main script, run:

  tail -f $SCHEDULER_OUTPUT
"

#
# Print cancellation instructions.
#

if [ -n "$DRY_RUN" ]
then
  print_header "Skipping cancellation, dashboard, postprocessing instructions"
  exit 0
fi

# Record job ids in logging directory. This can be picked up by
# scripts/slurm_cancel.sh in order to cancel the job.
echo -n -e "$JOB_IDS" > "${LOGDIR}/job_ids.txt"

print_header "Canceling"
echo "\
To cancel this job, run:

  bash scripts/slurm_cancel.sh $LOGDIR
"

#
# Print Dask dashboard instructions.
#

print_header "Dask Dashboard"
echo "Waiting for scheduler to start..."

# Wait for scheduler to start.
while [ ! -e $SCHEDULER_FILE ]; do
  sleep 1
done