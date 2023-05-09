#!/bin/bash

USAGE="Usage: bash scripts/plot_throughput.sh LOGDIR MODE"

LOGDIR="$1"
MODE="$2"
LOGDIR2="$3"

if [ -z "${LOGDIR}" ]
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
if [ "${MODE}" = "n_agents" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_vs_n_agents.py \
            --logdirs_plot "$LOGDIR"
fi

if [ "${MODE}" = "thr_time" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_thr_time.py \
            --logdirs_plot "$LOGDIR"
fi

if [ "${MODE}" = "cross_n_agents" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_vs_n_agents_cross.py \
            --all_logdirs_plot "$LOGDIR"
fi

if [ "${MODE}" = "cross_thr_time" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_thr_time_cross.py \
            --all_logdirs_plot "$LOGDIR"
fi

if [ "${MODE}" = "cross_thr_time_w_n_agents" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/throughput_thr_time_w_n_agents_cross.py \
            --all_thr_time_logdirs_plot "$LOGDIR" \
            --all_n_agents_logdirs_plot "$LOGDIR2"
fi