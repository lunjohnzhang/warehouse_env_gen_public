#!/bin/bash

USAGE="Usage: bash scripts/plot_heatmap.sh LOGDIR HEATMAP_ONLY"

LOGDIR="$1"
MODE="$2"
HEATMAP_ONLY="$3"

if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${HEATMAP_ONLY}" ]
then
  HEATMAP_ONLY="False"
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/heatmap.py \
        --logdir "$LOGDIR" \
        --mode "$MODE" \
        --kiva \
        --heatmap_only "$HEATMAP_ONLY"