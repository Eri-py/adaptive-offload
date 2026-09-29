# Birds findings — why CUB-200 replaced Oxford 102 Flowers

The small/large classification pair (MobileNetV3-Large / ConvNeXt-Base) was
first built on Oxford 102 Flowers. A pilot compared it against CUB-200-2011
birds to see which dataset leaves more for an adaptive router to gain. Numbers
below are copied from the pilot's output. That output lived in the gitignored
`training/scratch/`, which has since been cleared, so this file is now the
record: the flowers half cannot be regenerated at all (its dataset and
checkpoints were deleted), and the birds half would need a re-run.

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

## Small vs. large

| | Flowers-102 (n=6,149) | CUB-200 birds (n=5,794) |
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

## Cascade

Escalate the small model's least-confident predictions (by max softmax
probability) to the large model. "Gap recovered" is the share of the
small-to-large accuracy gap regained.

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

## Interpretation

Flowers' small-to-large gap is only 2.46 points and about 15% escalation
recovers most of it, so there is little for a router to decide. Birds has a
gap of 9.49 points and needs far more escalation to recover it, leaving a
larger prize, while confidence remains informative (AUC 0.844 to 0.854).
