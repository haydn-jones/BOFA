#!/bin/bash
runai submit \
    --name hpo \
    -v $PWD:/workspace \
    --node-type a6000 \
    -i haydnj/bla \
    -g 0.125 \
    -p jacobrg \
    -e WANDB_API_KEY=63a0bf041f9307207232d2b595719e2610dd202c \
    --backoff-limit 0 \
    --parallelism 24 --completions 96 \
    -- python -m scripts.molecule_optimization