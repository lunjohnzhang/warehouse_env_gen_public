#!/bin/bash

TO_PLOT="$1"

for DIR in ${TO_PLOT}/*;
do
    # echo "${DIR}"
    for LOG_DIR in ${DIR}/*
    do
        echo "${LOG_DIR}"
        bash scripts/plot_tile_usage.sh "$LOG_DIR" qd extreme
    done
done