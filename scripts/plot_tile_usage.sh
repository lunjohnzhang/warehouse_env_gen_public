#!/bin/bash

USAGE="Usage: bash scripts/plot_tile_usage.sh LOGDIR MODE"

LOGDIR="$1"
LOGDIR_TYPE="$2"
MODE="$3"

if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${LOGDIR_TYPE}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${MODE}" ]
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/tile_usage.py \
        --logdir "$LOGDIR" \
        --logdir-type "$LOGDIR_TYPE" \
        --mode "$MODE"