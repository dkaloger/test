#!/bin/bash
cd "$(dirname "$0")/.."
run() { python3 experiments/train_exp.py --iters 2000 --threads 1 "$@" > experiments/logs/$2.log 2>&1; }
run --name modern_muon --modern --muon &
run --name modern_muon_mtp --modern --muon --mtp 0.3 &
run --name modern_muon_kdfade --modern --muon --ngram-kd 0.1 --kd-anneal &
run --name modern_muon_drop02 --modern --muon --dropout 0.2 &
wait
echo ALL_DONE
