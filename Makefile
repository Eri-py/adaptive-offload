# Makefile — adaptive-offload
#
# Convenience wrapper around the per-package quality gates documented in
# .claude/coding-guidelines.md. `database/`, `server/`, and `training/` each
# own their own pyproject.toml (and ruff/mypy config) but share one
# virtualenv at the repo root (.venv/). Every target here just does, from
# the repo root, exactly what you'd type by hand from inside a package
# directory with that shared venv activated.
#
# Postgres must already be running for `test`/`check` targets that touch
# database/ or training/ (their fixtures create/drop an ephemeral test DB).
# This Makefile never starts or stops it.

SHELL := /bin/bash

VENV := .venv
# Interpreter the existing .venv was built with (see .venv/pyvenv.cfg's
# `executable` field) — a uv-managed CPython, not the system python3.
VENV_PYTHON := /home/eriol/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
ACTIVATE := source $(VENV)/bin/activate

.PHONY: help venv \
	test lint check \
	test-database test-server test-training \
	lint-database lint-server lint-training

.DEFAULT_GOAL := help

# pytest exits 5 when a package has zero test files (e.g. server/ currently
# has none yet) — that's an empty suite, not a failure, so it's treated as
# a pass here. Any other nonzero exit (assertion failures, collection
# errors, etc.) still propagates and stops the aggregate target.
define run_pytest
	$(ACTIVATE) && cd $(1) && pytest; ec=$$?; \
	if [ $$ec -eq 5 ]; then \
		echo "$(1)/: no tests collected (pytest exit 5) — not treated as a failure"; \
		exit 0; \
	fi; \
	exit $$ec
endef

help: ## Show this help message
	@echo "adaptive-offload — available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the shared .venv if missing, editable-install database/server/training into it
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating venv at $(VENV) with $(VENV_PYTHON)..."; \
		$(VENV_PYTHON) -m venv $(VENV); \
	else \
		echo "$(VENV) already exists, skipping creation."; \
	fi
	$(ACTIVATE) && pip install -e "./database[dev]" -e "./server[dev]" -e "./training[dev]"

test: test-database test-server test-training ## Run pytest for database/, server/, and training/ (stops at first failure)

lint: lint-database lint-server lint-training ## Run ruff + mypy for database/, server/, and training/ (stops at first failure)

check: lint test ## Run lint then test across all three packages — the pre-commit gate

test-database: ## Run database/'s test suite
	$(call run_pytest,database)

test-server: ## Run server/'s test suite
	$(call run_pytest,server)

test-training: ## Run training/'s test suite
	$(call run_pytest,training)

lint-database: ## Run ruff check + mypy for database/
	$(ACTIVATE) && cd database && ruff check . && mypy .

lint-server: ## Run ruff check + mypy for server/
	$(ACTIVATE) && cd server && ruff check . && mypy .

lint-training: ## Run ruff check + mypy for training/
	$(ACTIVATE) && cd training && ruff check . && mypy .
