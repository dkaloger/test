# Training-improvement experiments

All runs: 4-layer, 256-dim, ~3.2M-param char-level GPT on Tiny Shakespeare, batch 32 x 128,
1 CPU thread each. **Metric: mean cross-entropy over the entire validation split** (deterministic).
Reproduce any row with `python experiments/train_exp.py --name X <flags>`.

## Screening (2,000 steps)

| Run | Flags | Val loss | Δ vs baseline |
|---|---|---|---|
| baseline | – | 1.5527 | – |
| baseline, seed 2 | `--seed 2` | 1.5505 | −0.002 (seed noise) |
| Modern arch | `--modern` (RMSNorm + RoPE + SwiGLU) | 1.4858 | **−0.067** |
| Muon optimizer | `--muon` | 1.4902 | **−0.063** |
| EMA of weights | `--ema 0.998` | 1.5822 | +0.030 |
| Input-char noise | `--input-noise 0.05` | 1.5946 | +0.042 |
| Trigram-teacher KD | `--ngram-kd 0.3` | 1.5896 | +0.037 |
| Multi-token prediction | `--mtp 0.3` | 1.5542 | +0.002 |
| Modern + Muon | `--modern --muon` | 1.4692 | −0.084 |
| + fading trigram KD | `... --ngram-kd 0.1 --kd-anneal` | **1.4601** | **−0.093** |
| + multi-token prediction | `... --mtp 0.3` | 1.4754 | −0.077 |
| + dropout 0.2 | `... --dropout 0.2` | 1.5245 | −0.028 |

## Final (3,000 steps, same length as the original model)

| Run | Seed | Val loss | Δ | Train CE (last batch) | Time |
|---|---|---|---|---|---|
| baseline | 1337 | 1.5056 | – | 1.365 | 51 min |
| Modern + Muon + fading KD | 1337 | 1.4606 | −0.045 | 1.218 | 67 min |
| Modern + Muon + fading KD | 2 | **1.4465** | **−0.059** | 1.175 | 67 min |
| Modern + Muon | 2 | 1.4565 | −0.049 | 1.167 | 67 min |

## Takeaways
- **Modern architecture and Muon are real wins** (each ~30x the seed noise), but overlap: combined is not additive.
- **Fading trigram-teacher distillation** (the least standard idea) gave −0.009 and −0.010 in the two same-seed
  paired comparisons. Seed-to-seed spread of the final recipe is 0.014, so treat this as suggestive, not proven.
  At full strength (0.3, no fade) it hurts: the teacher is much weaker than the student.
- **Regularizers (EMA, input noise, dropout 0.2, MTP) did not help** at this training length.
- **The gain shrinks with longer training** (−0.093 at 2k steps → −0.045..−0.059 at 3k) because the new model
  overfits sooner (train/val gap 0.24 vs 0.14). More data is the next lever, not more tricks.
- **Wall clock:** the new recipe is ~33% slower per step on CPU, but still wins at equal time:
  2,000 new-recipe steps (45 min, 1.494) beat 3,000 baseline steps (51 min, 1.506).
