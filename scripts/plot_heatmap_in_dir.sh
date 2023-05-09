#!/bin/bash

TO_PLOT="$1"

for DIR in ${TO_PLOT}/*;
do
    # echo "${DIR}"
    for LOG_DIR in ${DIR}/*
    do
        bash scripts/plot_heatmap.sh "$LOG_DIR" single
    done
done