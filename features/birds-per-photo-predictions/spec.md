# Birds Per-Photo Predictions

## Overview

The bird attempt's two models are only comparable in aggregate today: evaluation
reports each model's accuracy and mean latency, and nothing records what either
model predicted for an individual photo. The cascade table in
`findings/birds.md` — the result that motivated choosing birds over flowers —
came from a script in the cleared, gitignored `training/scratch/`, so nothing
committed can reproduce it, and its numbers came from pilot checkpoints that no
longer match the committed ones. This feature records both models' per-photo
predictions and confidence to Postgres, derives the summary tables from those
records, and rewrites the findings file's cascade section from reproducible
output. Every later routing or cascade result depends on this record existing.

## Requirements

- **Per-photo records exist for both models across all three splits** of
  CUB-200-2011: the official test photos (5,794), the validation photos carved
  from the official training set (594), and the remaining training photos
  (5,400). Each model is run with its own evaluation preprocessing, over the
  existing deterministic split — the split definition is not changed or
  re-derived here.
- **Each record carries** the split it belongs to, the photo, the photo's true
  species, which model produced the prediction, that model's top-1 predicted
  species, whether that prediction was correct, the model's highest class
  probability, and the model's margin between its two highest classes. Both
  models get the confidence figures, not just the small one.
- **Each record identifies the checkpoint behind it** by the training run that
  produced that checkpoint, so a record can be traced back to the model that
  made it and recognised as stale after a retrain. This is the specific gap
  that made the earlier pilot numbers unreproducible.
- **Records live in the project's existing Postgres research database**, in a
  new table owned by the shared database package, introduced by a new
  migration. The migration is run by the user — no part of this feature runs
  it, and nothing creates or alters the database itself.
- **Re-running replaces rather than accumulates.** Running again for a given
  model and split overwrites that model and split's existing records and leaves
  every other model and split untouched, so refreshing after a retrain is just
  running it again.
- **Summary tables are derived from the stored records, for the test split
  only**, and cover:
  - the agreement breakdown: each model's accuracy, the gap between them, how
    often both are right, how often only the large model is right, how often
    only the small model is right, how often both are wrong, and the oracle
    (either one right)
  - how well each model's confidence separates its own correct predictions from
    its incorrect ones, reported for both confidence figures
  - the cascade: accuracy when the least-confident share of photos is escalated
    to the large model, at the same escalation points the findings file already
    reports (0, 5, 10, 15, 20, 30, 50 and 100 percent), given for each
    confidence figure, alongside the share of the small-to-large accuracy gap
    that escalation recovers
- **The summary tables are printed and recorded in experiment tracking** as
  their own tracked run, since they span both checkpoints and belong to neither
  model's training run.
- **The summary output is deterministic:** two consecutive runs against
  unchanged records produce identical output.
- **A missing prerequisite stops the run with an actionable message**, naming
  the command that fixes it, consistent with how the existing birds entry
  points behave for a missing dataset or a missing checkpoint. A run also
  refuses to proceed without a GPU, as the other birds entry points do.
- **`findings/birds.md`'s cascade and comparison sections are rewritten** from
  this feature's own output, replacing the pilot figures, and state that the
  numbers come from the committed checkpoints and from committed code that can
  reproduce them.

## Out of Scope

- Any change to what the existing evaluation entry point reports (aggregate
  accuracy, loss and mean latency) or how it reports it.
- Per-photo latency, on-device or server latency, and any cost model for what
  escalating costs. The cascade numbers here are accuracy-only.
- Choosing an escalation threshold, or building any router, cascade policy or
  decision layer that uses these records.
- Evaluating the cascade under a latency budget, and sweeping local or offload
  latency — a later spec.
- Retraining, changing either model, changing the training recipe, or changing
  the split definition or its seed.
- Any change to attempt 1 (COCO) code, its tables, or its findings file.
- Anything phone-side or server-side: no endpoint serves these predictions.
- Storing images, full probability vectors, or any per-photo model output
  beyond the fields listed above.
- Adding a second dataset, or making the dataset configurable.
- Running the migration, or any create/drop of a database.

## Acceptance Criteria

- Given the dataset and both checkpoints are present and the migration has been
  run, when the feature's entry point is run, then per-photo records exist for
  both models across all three splits, totalling two records per photo for
  11,788 photos.
- Given a stored record, when it is inspected, then it states its split, its
  photo, the photo's true species, its model, that model's predicted species,
  whether the prediction was correct, the model's highest class probability and
  the model's top-two margin.
- Given a stored record, when it is inspected, then it identifies the training
  run of the checkpoint that produced it.
- Given records already exist for a model and split, when the feature is run
  again for that model and split, then the earlier records for it are gone, the
  new ones are present, and records for every other model and split are
  unchanged.
- Given records for all three splits exist, when the summary tables are
  produced, then they are computed from the test split's records only, and the
  training and validation records do not affect any reported figure.
- Given the summary tables, when the agreement breakdown is read, then it
  reports both accuracies, the gap, both-right, only-large-right,
  only-small-right, both-wrong and the oracle.
- Given the summary tables, when the confidence separation is read, then a
  figure is reported for both confidence measures, for both models.
- Given the summary tables, when the cascade table is read, then it gives an
  accuracy and a share of the gap recovered at each of the eight escalation
  points, for both confidence measures, and the 0 percent row equals the small
  model's accuracy while the 100 percent row equals the large model's.
- Given each model's accuracy derived from the records, when it is compared
  against the accuracy the existing evaluation entry point reports for the test
  split, then the two agree.
- Given the summary run has completed, when experiment tracking is inspected,
  then the agreement, confidence and cascade figures are recorded against a
  single run covering both models.
- Given unchanged records, when the summary is produced twice, then both runs'
  output is identical.
- Given the dataset is missing, or a checkpoint is missing, or no GPU is
  available, when the feature is run, then it stops with a message that names
  what is missing and the command that resolves it, and writes no records.
- Given this feature is complete, when `findings/birds.md` is read, then its
  comparison and cascade sections carry this feature's numbers rather than the
  pilot's, and state that they come from the committed checkpoints and
  reproducible code.
- Given this feature is complete, then no migration has been run by it and the
  existing evaluation entry point's behaviour is unchanged.

## Open Questions

None.
