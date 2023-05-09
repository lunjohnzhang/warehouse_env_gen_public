#!/bin/bash

TO_PLOT="$1"

for DIR in ${TO_PLOT}/*;
do
    echo "${DIR}"
    bash scripts/plot_throughput.sh "$DIR" cross_thr_time
done