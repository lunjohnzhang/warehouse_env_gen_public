#!/bin/bash

USAGE="Usage: bash scripts/run_single_sim.sh SIM_CONFIG MAP_FILE AGENT_NUM N_EVALS MODE N_SIM"

SIM_CONFIG="$1"
MAP_FILE="$2"
AGENT_NUM="$3"
N_EVALS="$4"
MODE="$5"
N_SIM="$6"

if [ "${MODE}" = "inc_agents" ]
then
    singularity exec --cleanenv singularity/ubuntu_warehouse.sif \
        python env_search/warehouse/module.py \
            --warehouse-config "$SIM_CONFIG" \
            --map-filepath "$MAP_FILE" \
            --agent-num "$AGENT_NUM" \
            --n_evals "$N_EVALS" \
            --mode "$MODE" \
            --n_sim "$N_SIM"
    sleep 2
fi

if [ "${MODE}" = "constant" ]
then
    singularity exec --cleanenv singularity/ubuntu_warehouse.sif \
    python env_search/warehouse/module.py \
        --warehouse-config "$SIM_CONFIG" \
        --map-filepath "$MAP_FILE" \
        --agent-num "$AGENT_NUM" \
        --n_evals "$N_EVALS" \
        --mode "$MODE"
fi