# Generalize run-simulation's Image Data Source — Implementation Plan

## Summary

`run-simulation`'s image source and dataset name are currently hardcoded to
one pre-downloaded COCO val2017 copy. This plan generalizes
`image_source.py` (the module formerly named `coco.py`, already relocated
to `training/datagen/sourcing/` by a separate module-regrouping change) with
fail-fast validation and a dataset-agnostic error message, and adds required
`--annotations`/`--images`/`--dataset` CLI arguments to `run-simulation` —
without touching `stub_inference`, the other CLI tools, or
`sync-coco-cache`'s own defaults.

> **Note:** `coco.py` has already been renamed to `image_source.py` and
> moved to `training/datagen/sourcing/` as part of an unrelated
> module-regrouping change (splitting `datagen/`'s flat core modules into
> `sourcing/`/`sampling/`/`simulate/` subpackages). All import paths in this
> plan already reflect that new location — there is no rename/move left to
> do here, only the content changes below.

## Approach & Key Decisions

- **No further file moves or splits.** `image_source.py` stays one file —
  `download_missing_images`/`fetch_image_bytes`/`COCO_VAL2017_BASE_URL`
  (genuinely COCO-specific download logic) stay alongside
  `load_image_index`/`resolve_image_path` (now dataset-agnostic) rather than
  splitting into two modules. The spec only requires generalizing the
  module's content; a new module boundary isn't asked for.
- **`run_simulation()`'s new `dataset` parameter follows the existing
  optional-tunable pattern** (`dataset: str | None = None`, defaulting
  internally to `config.DATASET_NAME`) rather than becoming a required
  function argument — this keeps every existing test that calls
  `run_simulation()` directly (without passing `dataset=`) passing
  unmodified. The spec's "no defaults" requirement is about the CLI's
  explicitness, enforced in `main()`'s argparse (`required=True`), not the
  core function's signature.
- **Fail-fast validation for a bad `--annotations` file lives inside
  `image_source.load_image_index()`** (missing file / invalid JSON /
  missing `images` key) rather than in `main()`, so it's unit-testable
  independent of the CLI. Fail-fast validation for a missing `--images`
  folder is a small dedicated helper in `run_simulation.py`, called from
  `main()` before any scoring or database work begins.
- **`resolve_image_path`'s per-file "not found" error message is
  generalized**, dropping its "run `sync_coco_cache`" suggestion — that
  suggestion is actively wrong once the images folder isn't guaranteed to
  be a COCO download.
- **`sync_coco_cache.py` is not renamed and its own COCO val2017 defaults
  are untouched.** Per the spec, it stays the one COCO-specific convenience
  CLI, unaffected by `run-simulation`'s new required arguments. (Its import
  of `image_source` was already updated by the module-regrouping change.)

## Out of Scope

- Any change to `stub_inference` or how latency/accuracy are computed.
- Any change to `score_complexity`, `preview_sample`, `preview_conditions`,
  or `relabel_run` — their own default dataset name and behavior are
  untouched.
- Dropping or renaming `ImageRecord.image_id`.
- Removing `config.DATASET_NAME` or `image_source.py`'s COCO val2017
  constants (`ANNOTATIONS_PATH`, `IMAGES_DIR`, `COCO_VAL2017_BASE_URL`) —
  these remain as `sync-coco-cache`'s own defaults.
- Any change to `download_missing_images`'s behavior.
- Real accuracy scoring against ground-truth bounding boxes.
- Renaming `sync_coco_cache.py` itself, or changing its docstring/invocation
  example.
- Any further reorganization of `training/datagen/`'s module layout
  (already done separately).

## Dependencies and Configuration

None — no new packages, no config entries, no DB migration.

## Files Changed

| Path | Action | Purpose | Why |
|------|--------|---------|-----|
| `training/datagen/sourcing/image_source.py` | edit | add fail-fast validation to `load_image_index`; generalize `resolve_image_path`'s error message; generalize the module docstring | core generalization work |
| `training/tests/datagen/sourcing/test_image_source.py` | edit | update two existing error-message assertions; add new validation tests | covers spec's annotations-error acceptance criterion |
| `training/datagen/cli/run_simulation.py` | edit | new required `--annotations`/`--images`/`--dataset` args; `dataset` param; images-dir validation | core of this feature |
| `training/tests/datagen/cli/test_run_simulation.py` | edit | new test for the images-dir validation helper | adds missing-images-folder coverage |

## Tasks

### Task 1 — Generalize `image_source.py`'s validation and error messages

- **Objective:** Make `image_source.load_image_index` fail clearly on a bad
  annotations file, and generalize `resolve_image_path`'s error message so
  it no longer assumes the images folder is a COCO download.
- **Files:** `training/datagen/sourcing/image_source.py`,
  `training/tests/datagen/sourcing/test_image_source.py`.
- **Details:**
  - Update the module's top-of-file docstring to describe it as
    COCO-format annotation loading + local image resolution (used by any
    COCO-format dataset), with the COCO val2017 download helpers
    (`download_missing_images`, `fetch_image_bytes`, `COCO_VAL2017_BASE_URL`)
    still living here as the one COCO-specific piece.
  - In `load_image_index`: raise `FileNotFoundError` with a clear message
    naming the path if `annotations_path` doesn't exist; catch
    `json.JSONDecodeError` and re-raise as `ValueError` with a clear
    message naming the path; raise `ValueError` with a clear message if
    the parsed JSON has no `"images"` key. Do this validation before/around
    the existing parsing logic — the return value and existing
    `ImageRecord` construction stay the same for valid input.
  - In `resolve_image_path`'s `FileNotFoundError` message: drop the
    `"COCO image ..."` framing and the `"Run python -m
    datagen.cli.sync_coco_cache"` suggestion (misleading once the images
    folder isn't necessarily a COCO download) — replace with a plain
    message naming the file and `images_dir`, no download suggestion.
  - In `training/tests/datagen/sourcing/test_image_source.py`: the two
    existing `FileNotFoundError` tests that `match=` the old
    `"datagen.cli.sync_coco_cache"` string need updating to match the new
    message instead. Add new unit tests (reusing the file's existing
    fake-annotations-writing helper pattern) for: missing annotations
    file, invalid JSON content, and JSON missing the `"images"` key —
    each asserting the specific exception type and that the error message
    names the bad path.
- **Success criteria:**
  - `cd training && pytest tests/datagen/sourcing/test_image_source.py -v`
    — all pass, including the three new validation tests.
  - `cd training && ruff check . && mypy .` clean.

### Task 2 — Add required `--annotations`/`--images`/`--dataset` args to `run-simulation`

- **Objective:** Make `run-simulation` take its image source and dataset
  name as required CLI arguments instead of hardcoded COCO defaults, with
  a clear fail-fast error for a missing images folder.
- **Files:** `training/datagen/cli/run_simulation.py`,
  `training/tests/datagen/cli/test_run_simulation.py`.
- **Details:**
  - Add a `dataset: str | None = None` keyword parameter to
    `run_simulation()`, positioned alongside its other optional tunables.
    Resolve it the same way the others are resolved (`resolved_dataset =
    dataset if dataset is not None else config.DATASET_NAME`) and replace
    every current `config.DATASET_NAME` reference inside the function body
    (the `get_known_complexity` call, both `store_complexity_scores`
    calls, and `RunConfig(dataset=...)`) with `resolved_dataset`.
  - Add a small module-level helper, e.g. `_require_images_dir(images_dir:
    Path) -> None`, that raises `FileNotFoundError(f"No such images
    folder: {images_dir}")` if `not images_dir.is_dir()`. Keep it a plain
    function (not inline in `main()`) so it's directly unit-testable.
  - In `main()`: add three required `argparse` arguments — `--annotations`
    (`type=Path`, `required=True`, help mentions it expects a COCO-format
    annotations file), `--images` (`type=Path`, `required=True`, help:
    folder to resolve image files from), `--dataset` (`type=str`,
    `required=True`, help: dataset name scene-complexity scores are stored
    under). After `args = parser.parse_args()` and before building the
    engine: call `_require_images_dir(args.images)`, then build
    `image_records = image_source.load_image_index(args.annotations)`
    (its own validation from Task 1 covers a bad annotations file) and
    `resolve_image = functools.partial(image_source.resolve_image_path,
    images_dir=args.images)` (add `import functools` at the top;
    `image_source` is already imported via `from datagen.sourcing import
    image_source`). Pass `image_records=image_records,
    resolve_image=resolve_image, dataset=args.dataset` into the
    `run_simulation(...)` call alongside the existing `engine, args.preset`.
  - Update the module's top-of-file docstring and `main()`'s
    `argparse.ArgumentParser(description=...)` to document the three new
    required arguments, and fix the stale invocation example (currently
    `python -m datagen.cli.run_simulation --preset baseline`, which
    predates this package's `[project.scripts]` entry points) to `
    run-simulation --preset baseline --annotations <path> --images <folder>
    --dataset <name>`.
  - Add a unit test for `_require_images_dir`: one case with an existing
    `tmp_path` directory (no exception), one with a nonexistent path
    (raises `FileNotFoundError`, message contains the path).
  - Do not change any existing `run_simulation(...)` call's arguments or
    assertions in the existing test file — none of them pass `dataset=`,
    so `resolved_dataset` still resolves to `config.DATASET_NAME` inside
    the function, matching the current `run.dataset == config.DATASET_NAME`
    / `filter_by(dataset=config.DATASET_NAME)` assertions unchanged.
- **Success criteria:**
  - `run-simulation --help` (from an activated venv) lists `--annotations`,
    `--images`, and `--dataset` as required arguments.
  - Running `run-simulation --preset baseline` with no other arguments
    exits with an argparse error (missing required arguments), making no
    database connection.
  - The new `_require_images_dir` unit tests pass.
  - Every existing test in `test_run_simulation.py` still passes
    unmodified in its assertions.
  - `cd training && ruff check . && mypy .` clean.

### Task 3 — Regression test run

- **Objective:** Run every test that exercises code added or modified in
  this plan, and confirm all pass.
- **Files:** none changed
- **Success criteria:**
  - `cd training && pytest` — full suite passes (ask the user to start
    Postgres first if the ephemeral-test-DB fixture can't connect; never
    start it yourself, per `CLAUDE.md`).
  - `cd training && ruff check . && mypy .` clean.
