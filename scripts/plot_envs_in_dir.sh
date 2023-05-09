#!/bin/bash

TO_PLOT="$1"

mkdir -p "${TO_PLOT}/img"

for ENV in ${TO_PLOT}/*.json;
do
    python env_search/analysis/visualize_env.py \
        --map-filepath "${ENV}" \
        --store_dir "${TO_PLOT}/img"
done