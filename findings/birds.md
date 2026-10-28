# Birds findings — why CUB-200 replaced Oxford 102 Flowers

The small/large classification pair (MobileNetV3-Large / ConvNeXt-Base) was
first built on Oxford 102 Flowers. A pilot compared it against CUB-200-2011
birds to see which dataset leaves more for an adaptive router to gain. That is
why the project moved to birds.

This file has two kinds of numbers, and they differ in how reproducible they
are:

- **Flowers and the pilot's birds figures** are copied from the pilot's output.
  That output lived in the gitignored `training/scratch/`, which has since been
  cleared. The flowers half cannot be regenerated at all (its dataset and
  checkpoints were deleted). The pilot's birds checkpoints are not the ones
  committed today, so the pilot's birds figures cannot be regenerated either.
- **The committed-checkpoint birds tables** (the **Birds, committed
  checkpoints** section below) are reproducible.
  They come from the committed birds checkpoints and from committed code that
  reproduces them. Per-photo predictions are recorded in the `bird_predictions`
  table, and the tables are computed from those records.

## Regenerating the committed-checkpoint tables

Requires migration `0004` applied (the user runs migrations). Then, from
`training/`:

```
python -m birds.evaluation.predictions   # record per-photo predictions
python -m birds.evaluation.summary       # print the tables below
```

The records hold 23,576 rows in `bird_predictions`: 11,788 photos per model
(5,400 train / 594 val / 5,794 test), each row carrying its checkpoint's
training run id. The tables below use the **test split only**. The test
accuracies derived from the records match what
`python -m birds.evaluation.evaluate` reports independently (0.7798 small,
0.8699 large), which is the cross-check that the records are right. Two
consecutive summary runs produced byte-identical output.

## Pilot settings

- Single run, no repeats, so treat differences as indicative rather than exact.
- Flowers: the committed flowers checkpoints, evaluated on the test split
  (6,149 photos).
- Birds: the same two models fine-tuned on CUB-200-2011 with the flowers
  recipe unchanged (small: 30 epochs, lr 1e-3, batch 64; large: 15 epochs,
  lr 1e-4, batch 32; AdamW, cosine schedule, label smoothing 0.1, bf16
  autocast).
- 224 px images for both datasets.
- Birds validation is a stratified 10% of the official train split (seed 42:
  5,400 fit / 594 val); the test set is the official 5,794 photos.
- Pilot validation bests on birds: small 0.8047 (epoch 26), large 0.8855
  (epoch 9).
- The flowers dataset and checkpoints were later deleted, so the flowers
  numbers cannot be re-run without re-downloading and re-training.

## Flowers vs. birds (pilot figures, not reproducible)

The decision to replace flowers was made on these numbers. The birds column
comes from the pilot's checkpoints (small 0.7727 / large 0.8676), not from the
committed checkpoints in the next section.

| | Flowers-102 (n=6,149) | CUB-200 birds, pilot (n=5,794) |
|---|---|---|
| Small accuracy | 0.9221 | 0.7727 |
| Large accuracy | 0.9467 | 0.8676 |
| Gap (large - small) | +0.0246 | +0.0949 |
| Both right | 0.9031 | 0.7421 |
| Only large right | 0.0436 | 0.1255 |
| Only small right | 0.0190 | 0.0305 |
| Both wrong | 0.0343 | 0.1018 |
| Oracle (whichever is right) | 0.9657 | 0.8982 |
| Confidence AUC, max probability | 0.926 | 0.844 |
| Confidence AUC, top-2 margin | 0.933 | 0.854 |

Confidence AUC is how well the small model's confidence separates its correct
predictions from its wrong ones.

Pilot cascade, escalating the small model's least-confident predictions (by
max softmax probability) to the large model. "Gap recovered" is the share of
the small-to-large accuracy gap regained.

| Escalated | Flowers accuracy | Flowers gap recovered | Birds accuracy | Birds gap recovered |
|---|---|---|---|---|
| 0% | 0.9221 | 0.0% | 0.7727 | 0.0% |
| 5% | 0.9312 | 37.1% | 0.7941 | 22.5% |
| 10% | 0.9393 | 70.2% | 0.8119 | 41.3% |
| 15% | 0.9447 | 92.1% | 0.8241 | 54.2% |
| 20% | 0.9462 | 98.0% | 0.8369 | 67.6% |
| 30% | 0.9475 | 103.3% | 0.8517 | 83.3% |
| 50% | 0.9465 | 99.3% | 0.8657 | 98.0% |
| 100% | 0.9467 | 100.0% | 0.8676 | 100.0% |

Label smoothing 0.1 compresses the small model's probabilities, so read the
confidence ordering rather than the absolute confidence values.

## Birds, committed checkpoints (reproducible)

Test split, 5,794 photos. Regenerate with the two commands above.

### Small vs. large

| | CUB-200 birds (n=5,794) |
|---|---|
| Small accuracy | 0.7798 |
| Large accuracy | 0.8699 |
| Gap (large - small) | 0.0901 |
| Both right | 0.7497 |
| Only large right | 0.1201 |
| Only small right | 0.0300 |
| Both wrong | 0.1001 |
| Oracle (whichever is right) | 0.8999 |

Confidence AUC (correct vs. incorrect predictions):

| Model | Max probability | Top-2 margin |
|---|---|---|
| Small | 0.8324 | 0.8443 |
| Large | 0.8816 | 0.8864 |

The large model's own AUCs are new; the pilot only computed the small model's.

### Cascade

Escalated counts are floor (`n * percent // 100`), which is why 5% is 289
photos of 5,794 rather than 290.

| Escalated | Photos | Max-probability accuracy | Gap recovered | Top-2-margin accuracy | Gap recovered |
|---|---|---|---|---|---|
| 0% | 0 | 0.7798 | 0.0000 | 0.7798 | 0.0000 |
| 5% | 289 | 0.7979 | 0.2011 | 0.7981 | 0.2031 |
| 10% | 579 | 0.8122 | 0.3602 | 0.8148 | 0.3889 |
| 15% | 869 | 0.8260 | 0.5134 | 0.8293 | 0.5498 |
| 20% | 1158 | 0.8359 | 0.6226 | 0.8416 | 0.6858 |
| 30% | 1738 | 0.8523 | 0.8046 | 0.8562 | 0.8487 |
| 50% | 2897 | 0.8654 | 0.9502 | 0.8668 | 0.9655 |
| 100% | 5794 | 0.8699 | 1.0000 | 0.8699 | 1.0000 |

Top-2 margin beats max probability at every escalation level (for example
0.3889 vs. 0.3602 of the gap recovered at 10%, 0.8487 vs. 0.8046 at 30%).

## Interpretation

On the pilot figures, flowers' small-to-large gap is only 2.46 points and
about 15% escalation recovers most of it, so there is little for a router to
decide. Birds had a gap of 9.49 points and needed far more escalation to
recover it, leaving a larger prize, while confidence remained informative
(AUC 0.844 to 0.854).

The committed checkpoints tell the same story for birds: a gap of 9.01
points, with 50% escalation needed to recover 0.9502 of it (by max
probability), and confidence still informative on both models. The
flowers-vs-birds comparison itself rests on the pilot's numbers only; no
flowers figure was recomputed for this feature.
