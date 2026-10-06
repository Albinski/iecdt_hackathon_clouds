# Run log

One row per experiment. Numbers are from
`uv run python -m iecdt_hackathon.evaluate` on the **validation** split, which
scores the four published tasks; the leaderboard scores ten, so a row here is an
indicator and not a prediction of rank.

**Seed noise is ±0.008 overall and ±0.017 on `task_6`** (`d256` vs `d256_seed1`).
Treat anything smaller than that as no change.

Conventions:

- **Run** is the `--out` directory under `runs/`, and also the wandb run name and
  the `--name` of its `.npz` under `embeddings/val/`. One string identifies the
  checkpoint, the embedding and the graphs.
- **Key settings** are the `--set` overrides from the config default
  (`configs/default.yaml` for the autoencoder, `configs/ijepa.yaml` for I-JEPA).
  Blank means the config as committed.
- **D** is the embedding width the probes actually see.
- Smoke and shape-check runs are not logged here; only runs whose numbers mean
  something.

## Results

| Run | Model | D | Steps | Key settings | task_4 | task_5 | task_6 | task_7 | Overall |
|---|---|---|---|---|---|---|---|---|---|
| `d32` | ConvAE | 32 | 5,000 | `model.embedding_dim=32` | 0.789 | 0.855 | 0.334 | 0.812 | 0.697 |
| `d64` | ConvAE | 64 | 5,000 | `model.embedding_dim=64` | 0.816 | 0.859 | 0.380 | 0.901 | 0.739 |
| `d128` | ConvAE | 128 | 5,000 | `model.embedding_dim=128` | 0.871 | 0.905 | 0.411 | 0.922 | 0.777 |
| `d256` | ConvAE | 256 | 5,000 | — (config default) | 0.888 | 0.913 | 0.435 | 0.931 | 0.792 |
| `d256_seed1` | ConvAE | 256 | 5,000 | `training.seed=1` | 0.885 | 0.905 | 0.418 | 0.928 | 0.784 |
| `d512_w64` | ConvAE | 512 | 5,000 | `model.embedding_dim=512` `model.width=64` | 0.926 | 0.928 | 0.415 | 0.944 | **0.803** |
| `handcrafted` | 55 per-tile statistics, no training | 55 | — | — | 0.914 | 0.923 | **0.516** | 0.886 | **0.810** |
| `physical` | 144 physical features, no training | 144 | — | `scripts/physical_features.py` | 0.949 | 0.958 | **0.547** | 0.960 | **0.853** |
| `ae512_phys` | ConvAE ⊕ `physical` | 656 | 5,000 | `scripts/concat_embeddings.py` | 0.954 | 0.968 | **0.553** | 0.969 | **0.861** |
| `ae-d512` | ConvAE | 512 | 20,000 | `model.embedding_dim=512` `model.width=64` | 0.925 | 0.928 | 0.417 | 0.944 | 0.8036 |
| `ijepa-s16_mean` | I-JEPA ViT-S/16 | 384 | 50,000 | — (config default) | 0.9604 | 0.9545 | 0.4879 | 0.9749 | 0.8444 |
| `ijepa-s16_meanstd` | I-JEPA ViT-S/16 | 768 | 50,000 | — (config default) | 0.9620 | 0.9579 | 0.4875 | 0.9766 | 0.8460 |
| `ijepa-s16_mean4` | I-JEPA ViT-S/16 | 1536 | 50,000 | — (config default) | 0.9627 | 0.9595 | 0.4838 | **0.9793** | **0.8463** |
| `ijepa-s16_mean4std` | I-JEPA ViT-S/16 | 3072 | 50,000 | — (config default) | 0.9556 | 0.9553 | 0.4787 | 0.9757 | 0.8413 |
| `ijepa-s16-cache_mean` | I-JEPA ViT-S/16 | 384 | 150,000 | `data.cache=ram_uint8` | 0.9766 | 0.9558 | 0.5266 | 0.9730 | 0.8580 |
| `ijepa-s16-cache_meanstd` | I-JEPA ViT-S/16 | 768 | 150,000 | `data.cache=ram_uint8` | 0.9778 | 0.9591 | 0.4830 | 0.9749 | 0.8487 |
| `ijepa-s16-cache_mean4` | I-JEPA ViT-S/16 | 1536 | 150,000 | `data.cache=ram_uint8` | 0.9811 | 0.9674 | 0.5385 | 0.9810 | 0.8670 |
| `ijepa-s16-cache_mean4std` | I-JEPA ViT-S/16 | 3072 | 150,000 | `data.cache=ram_uint8` | 0.9757 | 0.9624 | 0.5257 | 0.9777 | 0.8604 |
| `ijepa-ti16-cache_mean` | I-JEPA ViT-Ti/16 | 192 | 200,000 | `model.arch=vit_tiny` `model.pred_emb_dim=96` `data.cache=ram_uint8` | 0.9673 | 0.9373 | 0.4781 | 0.9355 | 0.8296 |
| `ijepa-ti16-cache_meanstd` | I-JEPA ViT-Ti/16 | 384 | 200,000 | `model.arch=vit_tiny` `model.pred_emb_dim=96` `data.cache=ram_uint8` | 0.9717 | 0.9468 | 0.4801 | 0.9493 | 0.8369 |
| `ijepa-ti16-cache_mean4` | I-JEPA ViT-Ti/16 | 768 | 200,000 | `model.arch=vit_tiny` `model.pred_emb_dim=96` `data.cache=ram_uint8` | 0.9776 | 0.9624 | 0.5171 | 0.9736 | 0.8577 |
| `ijepa-ti16-cache_mean4std` | I-JEPA ViT-Ti/16 | 1536 | 200,000 | `model.arch=vit_tiny` `model.pred_emb_dim=96` `data.cache=ram_uint8` | 0.9775 | 0.9633 | 0.4825 | 0.9750 | 0.8496 |
| `ijepa-b16-cache_mean` | I-JEPA ViT-B/16 | 768 | 70,000 | `model.arch=vit_base` `model.pred_emb_dim=384` `model.pred_num_heads=6` `data.cache=ram_uint8` | 0.9618 | 0.9604 | 0.4877 | 0.9749 | 0.8462 |
| `ijepa-b16-cache_meanstd` | I-JEPA ViT-B/16 | 1536 | 70,000 | `model.arch=vit_base` `model.pred_emb_dim=384` `model.pred_num_heads=6` `data.cache=ram_uint8` | 0.9605 | 0.9633 | 0.4636 | 0.9742 | 0.8404 |
| `ijepa-b16-cache_mean4` | I-JEPA ViT-B/16 | 3072 | 70,000 | `model.arch=vit_base` `model.pred_emb_dim=384` `model.pred_num_heads=6` `data.cache=ram_uint8` | 0.9507 | 0.9536 | 0.4662 | 0.9694 | 0.8350 |
| `ijepa-s16-lr15` | I-JEPA ViT-S/16 | 768 | 150,000 | `data.cache=ram_uint8` `training.lr=1.5e-4` `training.start_lr=3.0e-5` | — | — | — | — | *blocked, not run* |
| `ae512_phys*` | `ae-d512` ⊕ `physical` | 656 | — | `concat_embeddings.py` | 0.9546 | 0.9679 | **0.5709** | 0.9692 | 0.8656 |
| `ijepa_phys` | `ijepa-s16_meanstd` ⊕ `physical` | 912 | — | `concat_embeddings.py` | 0.9688 | 0.9704 | 0.5275 | 0.9802 | 0.8617 |
| `ijepa4_phys` | `ijepa-s16_mean4` ⊕ `physical` | 1680 | — | `concat_embeddings.py` | 0.9685 | 0.9692 | 0.5455 | **0.9812** | **0.8661** |
| `ijepa_ae_phys` | `ijepa-s16_meanstd` ⊕ `ae-d512` ⊕ `physical` | 1424 | — | `concat_embeddings.py` | 0.9672 | 0.9714 | 0.5359 | 0.9802 | 0.8637 |
| `ijepa-s16-cache_mean4_phys` | `ijepa-s16-cache_mean4` ⊕ `physical` | 1680 | — | `compare_fusion.py` | 0.9824 | 0.9734 | 0.5623 | 0.9837 | 0.8754 |
| `ijepa-ti16-cache_mean4_phys` | `ijepa-ti16-cache_mean4` ⊕ `physical` | 912 | — | `compare_fusion.py` | 0.9796 | 0.9728 | 0.5639 | 0.9802 | 0.8741 |
| `ijepa-ti16-cache_meanstd_phys` | `ijepa-ti16-cache_meanstd` ⊕ `physical` | 528 | — | `compare_fusion.py` | 0.9758 | 0.9695 | 0.5705 | 0.9764 | 0.8730 |
| `ijepa-b16-cache_mean_phys` | `ijepa-b16-cache_mean` ⊕ `physical` | 912 | — | `compare_fusion.py` | 0.9663 | 0.9698 | 0.5762 | 0.9787 | 0.8727 |
| `ijepa-ti16-cache_mean_phys` | `ijepa-ti16-cache_mean` ⊕ `physical` | 336 | — | `compare_fusion.py` | 0.9739 | 0.9682 | 0.5710 | 0.9743 | 0.8718 |
| `ijepa-s16-cache_mean_phys` | `ijepa-s16-cache_mean` ⊕ `physical` | 528 | — | `compare_fusion.py` | 0.9795 | 0.9725 | 0.5535 | 0.9812 | 0.8717 |
| `ijepa-ti16-cache_mean4std_phys` | `ijepa-ti16-cache_mean4std` ⊕ `physical` | 1680 | — | `compare_fusion.py` | 0.9793 | 0.9707 | 0.5521 | 0.9802 | 0.8706 |
| `ijepa-s16-cache_mean4std_phys` | `ijepa-s16-cache_mean4std` ⊕ `physical` | 3216 | — | `compare_fusion.py` | 0.9773 | 0.9678 | 0.5535 | 0.9797 | 0.8696 |
| `ijepa-s16-cache_meanstd_phys` | `ijepa-s16-cache_meanstd` ⊕ `physical` | 912 | — | `compare_fusion.py` | 0.9802 | 0.9726 | 0.5434 | 0.9812 | 0.8693 |
| `ijepa-b16-cache_mean4_phys` | `ijepa-b16-cache_mean4` ⊕ `physical` | 3216 | — | `compare_fusion.py` | 0.9546 | 0.9607 | 0.5376 | 0.9726 | 0.8564 |
| `ijepa-b16-cache_meanstd_phys` | `ijepa-b16-cache_meanstd` ⊕ `physical` | 1680 | — | `compare_fusion.py` | 0.9644 | 0.9694 | 0.4991 | 0.9777 | 0.8526 |

## The three cached runs, and what more training did not buy

All three finished their configured schedules on 6 Oct — no walltime
truncation, so the LR, WD and EMA anneals all completed. `ijepa-s16-lr15` has not
run: its 8 h request cannot fit the gap before the `Oct2026PatchDay`
maintenance reservation (06 Oct 05:00–23:00), so it sits `PENDING` with reason
`ReqNodeNotAvail` and cannot start before 23:00 on 6 Oct. It was only ever
insurance against a collapse at `lr=3e-4` that did not happen in either healthy
run, so there is no reason left to wait for it.

| Run | Steps | Elapsed | `feature_std` | `token_std` | `eff_rank` | held-out JEPA loss |
|---|---|---|---|---|---|---|
| `ijepa-s16-cache` | 150,000 | 5:23 | 0.943 | 0.791 | 87.5 / 384 | 0.093 |
| `ijepa-ti16-cache` | 200,000 | 6:00 | 0.989 | 1.213 | 73.6 / 192 | 0.055 |
| `ijepa-b16-cache` | 70,000 | 4:16 | 0.351 | 0.281 | 53.5 / 768 | 0.030 |

**The held-out JEPA loss ranks them backwards, which is the point of never
selecting on it.** `b16` has the lowest loss of the three and by far the lowest
effective rank relative to its width — 53.5 of 768 dimensions, 7%, against 23%
for `s16` and 38% for `ti16`. It is also the most pooling-sensitive model in the
log: its `mean` pool fuses to 0.8727 (4th of 16) while its `meanstd` and `mean4`
pools are 16th and 14th. A representation spread thinly over few directions
survives a `mean` but degrades as soon as more columns are stacked on it.

Its diagnostics say why, and they say something more specific than "collapsed":

```
step    500   feature_std 0.365  token_std 0.115  eff_rank  3.7
step  7_500   feature_std 0.620  token_std 0.190  eff_rank 10.1
step 28_500   feature_std 0.326  token_std 0.146  eff_rank 18.5   <- trough
step 35_500   feature_std 0.280  token_std 0.157  eff_rank 25.0
step 70_000   feature_std 0.351  token_std 0.281  eff_rank 53.5   <- still climbing
```

It fell towards collapse through the first ~30k steps and was climbing out
steeply when its schedule ended. So ViT-B here is *undertrained and recovering*,
not a verdict on the architecture — and it is the only run whose fused `task_6`
improved beyond that task's 0.017 noise band between the mid-run snapshot and
the end (0.5550 → 0.5762, the best fused `task_6` of any I-JEPA model in the
log). If any run deserves more steps, it is this one.

### Training to completion changed nothing measurable

Each cached run was scored twice with the identical probe, folds, seed and
`physical.npz`: once from a mid-run snapshot (~70k for `s16`, ~90k for `ti16`)
and once from the final checkpoint. `results/comparison-mid/comparison.json`
and `results/comparison-final/comparison.json` hold both.

| config | alone, mid→final | fused, mid→final | `task_6`, mid→final |
|---|---|---|---|
| `s16-cache_mean` | 0.8478 → 0.8580 **+0.0102** | 0.8783 → 0.8717 −0.0066 | 0.5849 → 0.5535 **−0.0314** |
| `s16-cache_mean4` | 0.8616 → 0.8670 +0.0054 | 0.8753 → 0.8754 +0.0001 | 0.5682 → 0.5623 −0.0059 |
| `s16-cache_meanstd` | 0.8420 → 0.8487 +0.0067 | 0.8672 → 0.8693 +0.0021 | 0.5397 → 0.5434 +0.0037 |
| `s16-cache_mean4std` | 0.8524 → 0.8604 +0.0080 | 0.8631 → 0.8696 +0.0064 | 0.5329 → 0.5535 **+0.0206** |
| `ti16-cache_mean` | 0.8405 → 0.8296 **−0.0109** | 0.8760 → 0.8718 −0.0041 | 0.5885 → 0.5710 **−0.0175** |
| `ti16-cache_meanstd` | 0.8388 → 0.8369 −0.0019 | 0.8760 → 0.8730 −0.0029 | 0.5857 → 0.5705 −0.0151 |
| `ti16-cache_mean4` | 0.8624 → 0.8577 −0.0047 | 0.8745 → 0.8741 −0.0004 | 0.5671 → 0.5639 −0.0032 |
| `ti16-cache_mean4std` | 0.8547 → 0.8496 −0.0051 | 0.8693 → 0.8706 +0.0013 | 0.5495 → 0.5521 +0.0026 |
| `b16-cache_mean` | 0.8448 → 0.8462 +0.0014 | 0.8671 → 0.8727 +0.0056 | 0.5550 → 0.5762 **+0.0212** |
| `b16-cache_mean4` | 0.8424 → 0.8350 −0.0075 | 0.8601 → 0.8564 −0.0037 | 0.5536 → 0.5376 −0.0161 |
| `b16-cache_meanstd` | 0.8382 → 0.8404 +0.0022 | 0.8574 → 0.8526 −0.0048 | 0.5177 → 0.4991 **−0.0186** |

**Every fused delta is inside the 0.008 significance threshold** — the largest
is 0.0066. `s16` more than doubled its steps (70k → 150k) and `ti16` roughly
doubled (~90k → 200k) for no measurable fused gain.

Two guards make that readable as a result rather than as drift. The comparison
also re-scored five files that did not change between the two runs — `ae-d512`
and the four `ijepa-s16_*` — and they came back bit-identical, `+0.0000` on
every column; and the `physical` reference reproduced to 4 d.p. in both. So the
pipeline is the same pipeline, and the eleven non-zero deltas above are real
differences between checkpoints.

What *did* move, above threshold, is the standalone score — and in both
directions (`s16-cache_mean` +0.0102, `ti16-cache_mean` −0.0109). The extra
training kept reshaping the encoders without making them better partners for
`physical`. In probe terms these models had converged by ~70k steps; the
remaining 80–110k steps bought a different representation of the same quality.

**Consequence for the submission.** The best final-checkpoint fusion is
`ijepa-s16-cache_mean4_phys` at 0.8754, which is 0.0029 *below* the 0.8783 that
the mid-run `s16-cache--best_mean` ⊕ `physical` scored and that produced the
+0.8062 test result. That is a tie inside noise, so there is nothing here worth
spending a submission on. Had the final checkpoint been scored first, it would
have been the one submitted, and the leaderboard number would almost certainly
have been the same.

### `std` pooling still loses, and width still is not the lever

The no-`std` ordering from the mid-run comparison mostly holds on the final
checkpoints. The best fused pool is `mean4` for `s16` (0.8754) and `ti16`
(0.8741) and `mean` for `b16` (0.8727), and four of the five matched pairs put
the `*std` variant below its non-`std` sibling:

| arch | non-`std` | `*std` | effect of `std` |
|---|---|---|---|
| `s16` `mean` / `meanstd` | 0.8717 | 0.8693 | −0.0023 |
| `s16` `mean4` / `mean4std` | 0.8754 | 0.8696 | −0.0058 |
| `ti16` `mean` / `meanstd` | 0.8718 | 0.8730 | **+0.0012** |
| `ti16` `mean4` / `mean4std` | 0.8741 | 0.8706 | −0.0036 |
| `b16` `mean` / `meanstd` | 0.8727 | 0.8526 | −0.0201 |

The one exception, `ti16` `mean`→`meanstd`, is +0.0012 — well inside noise, and
the same pair was an exact tie in the mid-run comparison (0.8760 both). So the
honest statement is that `std` pooling is harmful at every width above 192 and
neutral at 192, not that it is harmful everywhere. The mechanism is unchanged:
`physical` already carries per-band spreads, percentiles, gradients and
multiscale variability, so token-`std` adds correlated columns that an
unregularised probe pays variance for — and the penalty grows with the number
of columns added, which is why `b16` (768 extra) is hurt eight times as much as
`ti16` (192 extra).

ViT-Ti remains the efficiency result: `ti16-cache_mean_phys` reaches 0.8718 at
**336 dims**, within 0.004 of the 1,680-dim `s16` winner, from a 5.6M-parameter
encoder. And `ti16` carries the highest `token_std` of the three despite being
the narrowest — token-level heterogeneity is what `task_6` wants, and the tiny
model has the most of it.

## Notes

- **`d32`–`d512_w64`** — the dimension sweep (`sweep_dim.sh`), all at 5,000
  steps rather than the config's 20,000. Scores rise steadily with D and show no
  overfitting penalty even at 512, despite the probes being unregularised.
- **`d256_seed1`** — a seed repeat, run only to measure run-to-run noise. That is
  where the ±0.008 threshold above comes from.
- **`d512_w64`** — width and D moved together, so the two effects are not
  separable from this row alone.
- **`handcrafted`** — 55 radiance statistics (`scripts/explore_tasks.py`).
  Superseded by `physical`, but kept because the ±0.008 threshold and the
  fusion argument were both derived from it: its margin was entirely on `task_6`
  (+0.10 over the best autoencoder) while it lost `task_7` (−0.06).
- **`physical`** — **the bar to beat is now 0.853, not 0.810.** The same 55
  statistics plus 89 physically motivated features: band-1 reflectance with a
  sun-angle correction, inverse-Planck brightness temperatures for bands 29/31/32,
  a 7×7 reflectance × temperature regime histogram, height above the local sea
  surface (warmest decile as the surface), the 11–12 µm split window and 8.5–11 µm
  phase difference, connected-component statistics for bright and for cold cloud,
  and multiscale block-mean variability at 4/16/64 px. Untrained, and it beats
  every autoencoder **on every one of the four tasks** — including `task_7`,
  where the 55-dim version lost. That removes the complementarity that made
  fusion with the autoencoder attractive.
- **`ae512_phys`** — stacking gains only +0.008 over `physical` alone, which is
  exactly the seed-noise threshold, and its `task_6` *balanced accuracy* is
  **lower** than `physical` alone (0.57 vs 0.62). So the autoencoder contributes
  almost nothing once the physical features are present. **This is the result
  I-JEPA now has to beat: not 0.853 on its own, but +0.008 on top of
  `physical`.**
- **`task_6` plateaus at ≈0.42 for every D ≥ 128** while the regressions keep
  climbing. The limit is the objective, not the capacity — this is the
  hypothesis I-JEPA is meant to test, so report `task_6` separately rather than
  only the mean.
- **`ijepa-s16`** — the hypothesis confirmed. 0.8463 against the best
  autoencoder's 0.803 is **+0.043, five times seed noise**, and `task_6` moves
  0.415 → 0.488 where quadrupling the autoencoder's training moved it 0.002. The
  limit really was the reconstruction objective.
- **Pooling is a non-lever alone and a large lever fused, in the opposite
  direction.** Standalone, 384 → 1536 spans 0.8444–0.8463, i.e. 0.002, inside
  noise, and `mean4std` at 3072 regresses to 0.8413 as the plan predicted. But
  concatenated with the 144 physical features the ordering inverts and the
  spread grows fivefold: `mean` (384+144) reaches **0.8720** while
  `meanstd` (768+144) gets 0.8617 and `mean4` (1536+144) 0.8661. `task_6` drives
  it — 0.5703, 0.5275, 0.5455 — which is what an unregularised
  `LogisticRegression(C=inf)` on 10,000 rows should do as the column count
  climbs. So **the narrowest pooling is the one to fuse**, and the earlier
  "drop the 3072 variant" conclusion stands for a different reason than the one
  I gave: width is not free once something is concatenated onto it.
- **I-JEPA and `physical` are complementary, unlike the autoencoder.** I-JEPA
  wins all three regressions (`task_4` +0.013, `task_7` +0.019, `task_5` level)
  and loses `task_6` by 0.059. The autoencoder lost to `physical` on all four.
  That difference is why fusion was worth building here and was not there.
- **`ae512_phys*`** — rebuilt locally as a control, and it does **not** exactly
  reproduce the `ae512_phys` row above: 0.8656 against 0.861. The cause is
  identified rather than mysterious — I stacked `ae-d512` (20,000 steps) because
  `d512_w64`'s (5,000 steps) `.npz` is not on this disk. The two score 0.8036 and
  0.803 standalone, and three of the four fused tasks agree to three decimals;
  the whole 0.005 gap is `task_6` (0.5709 vs 0.553), a swing of 0.018 — which is
  exactly that task's noise threshold. Treat `ae512_phys*` as the calibrated
  comparator for the rows below it, not as a reproduction.
- **The headline, and it is a negative one: fused I-JEPA and fused autoencoder
  are indistinguishable.** `ijepa4_phys` 0.8661 against `ae512_phys*` 0.8656 is
  a gap of **0.0005**, one-sixteenth of seed noise. I-JEPA is +0.043 better
  standalone and that advantage almost entirely disappears once the 144 physical
  features are present — both encoders add about +0.013 on top of them. The
  honest reading is that `physical` already captures most of what either learned
  encoder knows, and the remaining headroom on these four tasks is small.
- **The val proxy slightly under-weights I-JEPA.** It scores 3 regressions and 1
  classification (75% regression); the leaderboard scores 8 regressions and 2
  classifications (80%). Since I-JEPA's gain is entirely in the regressions and
  its deficit entirely in `task_6`, re-weighting the fused rows 80/20 moves
  `ijepa4_phys` to 0.8875 and `ae512_phys*` to 0.8853 — still a 0.002 gap, so
  this changes the ranking's sign but not the conclusion that they are tied.
- **Each I-JEPA row is really four.** Pooling lives in `model_cfg`, so one
  training run writes `best_mean.pt`, `best_meanstd.pt`, `best_mean4.pt` and
  `best_mean4std.pt` from the same weights, and the job scores all of them as
  `<run>_mean`, `<run>_meanstd`, … The **D** column above is the `mean+std`
  default; the others are ×0.5, ×2 and ×4 of it. ViT-B's `mean4std` would be
  6,144 wide against the 4,096 submission cap, so it is skipped automatically.
  Report whichever variant wins, and say which it was.

## The first sweep

Submitted 2026-10-05 18:30 (`./sweep_ijepa.sh`), six jobs, one question each.

| Job | Run | Question it answers |
|---|---|---|
| 57375250 | `ijepa-s16` | Does latent prediction beat reconstruction at all? Also measures the true step rate. |
| 57383968 | `ijepa-s16-cache` | Sample-limited or capacity-limited? The RAM cache removes the I/O bottleneck. |
| 57383969 | `ijepa-ti16-cache` | Is ViT-S already too big for 100k tiles? |
| 57383970 | `ijepa-b16-cache` | The other direction — worth a slot only with the cache on. |
| 57383971 | `ijepa-s16-lr15` | The least-confident hyperparameter. 3.0e-4 is sqrt-scaled from the reference's batch-2048 peak, which is a rule of thumb, not a measurement. |
| 57375685 | `ae-d512` ✓ | The fair comparator. Every autoencoder row above is at 5,000 steps, so without this "I-JEPA beats the AE" compares against an undertrained baseline. |

**gpuhost004 took three of these and failed the GPU preflight on all three**
(jobs 57375251, 57375254, 57375255, each dead in 2–7 s). `nvidia-smi -L` printed
the A100's UUID while `torch.cuda.device_count()` returned 0 — the same fault
that cost a 12 h slot on gpuhost005 earlier. They were resubmitted with
`EXCLUDE=gpuhost004`, which is why three job IDs above are from the second
batch. Worth reporting to JASMIN support; the node will keep accepting jobs and
handing the slot straight back.

Note the orchid QOS allows **four concurrent jobs per user**, so a six-run
sweep runs in two waves.

### What the first attempt taught

**`ae-d512` is the most informative completed run, and it is bad news for the
autoencoder.** At 20,000 steps it scores 0.8036 — against `d512_w64`'s 0.803 at
*5,000* steps. Four times the training changed nothing, and `task_6` moved from
0.415 to 0.417, well inside the ±0.017 noise on that task. The autoencoder is
**saturated, not undertrained**, which removes the obvious objection to the
I-JEPA comparison and sharpens the hypothesis: 0.803 is what pixel
reconstruction is worth here, whatever you spend on it.

**All four cached runs died ~4 minutes in, and it was my bug, not the cluster.**
`quantisation_range` returned a float64 upper bound — `.astype(np.float32)` does
not rebind the name, so adding the un-cast `lo` back re-promoted it — which made
every cached image a double and the first conv raise *"Input type (double) and
bias type (c10::BFloat16) should be the same"*. Three things let it through and
all three are now fixed:

- **The cache path had no test at all**, and every smoke run used
  `cache: none`, so a real job was the first thing to exercise it.
  `tests/test_cache.py` now covers the dtypes, the quantisation fidelity, and a
  cached batch going through `IJepa.forward` — the exact step that failed.
- **`share_memory_()` doubled peak RSS.** It allocates a second 39 GB block in
  /dev/shm and copies into it, and both halves count against the job's `--mem`
  cgroup: measured at 96.7 GB of a 120 GB limit. Dropped in favour of fork
  copy-on-write, which gives the same sharing for one copy, with a guard that
  refuses the `spawn` start method (which would pickle 39 GB per worker).
- **`_sample_block_mask` could loop forever.** On a grid too small to satisfy
  `min_keep` it spun with no bound. A hung job is indistinguishable from a slow
  one and would have consumed its whole allocation silently; it now raises after
  a bounded number of attempts and names the fix.

Resubmitted as 57383968 / 57383969 / 57383970 / 57383971 at
`WALLTIME=08:00:00 MAX_SECONDS=23400`, shortened again to clear the 05:00
maintenance window, and still excluding gpuhost004.

Two scheduling notes, both of which shaped the numbers above:

- **A maintenance reservation (`Oct2026PatchDay`, 06 Oct 05:00–23:00) covers
  every orchid GPU host.** A 12 h job will not start within 12 h of a
  reservation — SLURM parks it until the window *ends* — so these run at
  `--time=09:00:00` with `training.max_seconds=25200`. Check
  `scontrol show reservation` before a sweep and pass `WALLTIME=`/`MAX_SECONDS=`
  to suit.
- **Step budgets are set to finish, not to fill the slot.** The LR, weight-decay
  and EMA schedules are all defined over the *total* step count, so a run cut
  off by `max_seconds` never anneals and underperforms badly rather than
  slightly.

**Measured once the sweep was running, replacing the estimates it was sized on:**

| | estimated | measured |
|---|---|---|
| ViT-S/16 uncached, 14 workers | 3.5 steps/s | **6.05 steps/s** (774 tiles/s) |
| cache pre-pass | ~258 tiles/s, ~5 min | **~490 tiles/s, ~3.5 min** |

So the loader was about twice as fast as the 8-worker benchmark suggested, and
the I/O bottleneck is roughly 3× rather than 8×. The cache is therefore a
smaller lever than projected — still worth the two runs testing it, but the
uncached configuration is not as starved as it looked. `ijepa-s16` will finish
50,000 steps in about 2.3 h, so the next sweep can afford materially more steps.

**One thing to watch:** `collapse/effective_rank` is around 5 on a batch of 128
at step 1,000. That is lower than a ViT on natural images would show, and it is
the metric to check first if the probe scores disappoint — though MODIS cloud
scenes genuinely have fewer degrees of freedom than photographs, and the
autoencoder reaches R² ≈ 0.9 on some tasks from 256 dimensions, so low rank is
not by itself a failure. `feature_std` 0.52 and `token_std` 0.29 are healthy.

## Adding a row

```bash
# 1. train (wandb run name = the --out basename)
sbatch train_ijepa.sbatch                        # or: --out runs/<name> --wandb

# 2. embed the validation split under the same name
uv run python -m iecdt_hackathon.embed \
  --checkpoint runs/<name>/best_meanstd.pt \
  --data-dir /gws/ssde/j25b/iecdt/modis_hackathon/val \
  --out embeddings/val --name <name> --overwrite

# 3. score it next to whatever you are comparing against
uv run python -m iecdt_hackathon.evaluate --out results/<name> \
  --embeddings embeddings/val/<name>.npz embeddings/val/d256.npz
```

`--overwrite` in step 2 is not optional: `embed.py` keeps an existing `.npz`
rather than recomputing it, so without it a retrained model is silently scored
on its predecessor's embeddings.

Then copy the four per-task numbers and the overall into the table. Runs log to
wandb live by default; only a run forced to `wandb_mode: offline` needs a push
afterwards, and it leaves the command in `runs/<name>/wandb_sync.txt`.
