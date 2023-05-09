#!/bin/bash

USAGE="Usage: bash scripts/plot_figures.sh LOGDIR"

MANIFEST="$1"
MODE="$2"

collect="collect"
comparison="comparison"


if [ -z "${MANIFEST}" ];
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"

if [ "${MODE}" = "${collect}" ];
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/figures.py \
            collect \
            --reps 1 "$MANIFEST"
fi

if [ "${MODE}" = "${comparison}" ];
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/figures.py \
            comparison \
            "figure_data.json"
fi