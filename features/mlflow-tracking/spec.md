# MLflow Experiment Tracking

## Overview

Training runs currently leave no durable record: their numbers survive only in
whatever log file happened to be kept, and one such folder was already cleared,
taking the pilot's logs with it. From now on every training experiment is
tracked in MLflow, so each run's settings and results are recorded
automatically and runs can be compared.

## Requirements

- **Every training run is recorded** with enough detail to interpret and
  compare it later:
  - its settings: which model, epochs, learning rate, weight decay, batch
    size, seed, dataset and number of classes
  - how it progressed: training loss and validation accuracy after each epoch
  - its outcome: the best validation accuracy and which epoch produced it
  - which version of the code produced it
- **Evaluation results are recorded against the run that produced the model**,
  so a model's test accuracy and its per-photo timing (with the device) sit
  alongside the training that produced it, rather than in a separate place.
- **Run history is stored locally** in a folder that git ignores. Nothing is
  uploaded anywhere, and nothing needs to be running for training to work.
- **Viewing the history is the user's own step.** Training never starts a
  server or opens a browser, and the project documents the command to run.
- **Model checkpoints are not stored in the run history.** They are hundreds
  of megabytes each and already saved on disk.
- **Tracking never breaks training.** If recording a run fails, training and
  evaluation still complete and still report their results.
- **Existing tests keep passing without writing run history.** Tests must not
  create tracking files or depend on a tracking store.
- **Both bird models are retrained under tracking**, so the recorded history
  starts complete rather than with a gap. Their results are expected to match
  what is already recorded.
- **The convention is written down** in `.claude/coding-guidelines.md`, so
  future training code tracks its runs the same way.

## Out of Scope

- A tracking server, a database-backed store, or anything shared beyond this
  machine.
- Storing checkpoints, datasets or figures as run artifacts.
- Tracking anything that is not a training or evaluation run: the simulator,
  the router experiments and the data-generation pipeline are unchanged.
- Hyperparameter search, automated comparison or model registry features.
- Retrofitting the COCO/router work in `training/coco/`.
- Any change to what training does or to the results it produces.

## Acceptance Criteria

- Given a training run completes, when its recorded history is inspected, then
  there is one run holding the settings, the per-epoch training loss and
  validation accuracy, the best validation accuracy and its epoch, and a
  reference to the code version.
- Given an evaluation completes, when the history is inspected, then each
  model's test accuracy, per-photo timing and device are recorded against the
  training run that produced that model, not as a separate unrelated run.
- Given a training or evaluation run, then the run history is written under a
  local folder that `git status` never reports.
- Given the run history exists, then the project documents how to view it, and
  no code starts a viewer or a server.
- Given a run completes, then no checkpoint file has been copied into the run
  history.
- Given recording a run fails for any reason, when training or evaluation
  runs, then it still finishes and still prints its results, and the failure
  is visible rather than silent.
- Given the test suite runs, then it passes without creating any run history,
  and no test depends on a tracking store existing.
- Given both bird models are retrained under tracking, then their test
  accuracies are within a point of the recorded 77.98% and 86.99%, and any
  larger difference is reported rather than accepted silently.
- Given this feature is complete, then `.claude/coding-guidelines.md` states
  that training runs are tracked, and the full training test suite passes with
  ruff clean and no new type errors.

## Open Questions

None.
