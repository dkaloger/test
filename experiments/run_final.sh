#!/bin/bash
cd "$(dirname "$0")/.."
run() { python3 experiments/train_exp.py --iters 3000 --threads 1 "$@" > experiments/logs/$2.log 2>&1; }
run --name final_baseline &
run --name final_best --modern --muon --ngram-kd 0.1 --kd-anneal --save checkpoints/model_v2.pt &
run --name final_best_seed2 --modern --muon --ngram-kd 0.1 --kd-anneal --seed 2 &
run --name final_modern_muon_seed2 --modern --muon --seed 2 &
wait
echo ALL_DONE
