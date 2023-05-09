#!/bin/bash
# Cancels all slurm jobs associated with a logging directory.
#
# Usage:
#   bash scripts/slurm_cancel.sh [SLURM_DIR]
#
# Example:
#   bash scripts/slurm_cancel.sh slurm_logs/slurm_.../
SLURM_DIR="$1"
ID_FILE="$SLURM_DIR/job_ids.txt"
while read line; do
  IFS=';' read -ra tokens <<< "$line"
  name="${tokens[0]}"
  job_id="${tokens[1]}"
  echo "Cancelling $job_id ($name)"
  scancel "$job_id"
done < $ID_FILE
