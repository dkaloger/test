#!/bin/bash
# Screening: 2000 steps each, 4 single-thread jobs at a time.
cd "$(dirname "$0")/.."
run() { python3 experiments/train_exp.py --iters 2000 --threads 1 "$@" > experiments/logs/$2.log 2>&1; }
# wave 1
run --name baseline &
run --name baseline_seed2 --seed 2 &
run --name modern --modern &
run --name muon --muon &
wait
# wave 2
run --name ema --ema 0.998 &
run --name input_noise --input-noise 0.05 &
run --name ngram_kd --ngram-kd 0.3 &
run --name mtp --mtp 0.3 &
wait
echo ALL_DONE
