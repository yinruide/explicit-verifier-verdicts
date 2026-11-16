# Reproduction targets. `tables`, `figures` and `verify` read the released run records only:
# no API call, no GPU, no cost. Set DATA to the released data directory (or export RAFR_DATA_DIR).
#
#   make install
#   make tables  DATA=/path/to/release_data
#   make figures DATA=/path/to/release_data
#   make verify  DATA=/path/to/release_data
#
# Each target is one call into analysis/reproduce.py, so `python analysis/reproduce.py tables
# --data DIR` does the same without make.

PYTHON ?= python
DATA ?= $(RAFR_DATA_DIR)
OUT ?= out

.PHONY: help install tables figures verify all test lint rerun-plan clean

help:
	@echo "install    pip install -e '.[analysis,math,dev]'"
	@echo "tables     regenerate every table of the paper into $(OUT)/paper/tables.md   (no API, no GPU)"
	@echo "figures    regenerate the two operating-point figures into $(OUT)/figures/   (no API, no GPU)"
	@echo "verify     compare every regenerated number with the original outputs in DATA"
	@echo "all        tables + figures + verify"
	@echo "test       offline test suite"
	@echo "rerun-plan print the from-scratch pipeline, with the APIs, GPU and cost of each stage"

install:
	$(PYTHON) -m pip install -e ".[analysis,math,dev]"

tables:
	$(PYTHON) analysis/reproduce.py tables --data "$(DATA)" --out "$(OUT)"

figures:
	$(PYTHON) analysis/reproduce.py figures --data "$(DATA)" --out "$(OUT)"

verify:
	$(PYTHON) analysis/reproduce.py verify --data "$(DATA)" --out "$(OUT)"

all: tables figures verify

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

# ---- from scratch ------------------------------------------------------------------------------
# Rerunning the experiments calls paid APIs and needs a CUDA GPU; it is documented, not automated.
# See README.md, "Rerunning from scratch", for the commands. `make rerun-plan` prints the stages.
rerun-plan:
	@echo "Stage                                   APIs (keys in .env)                GPU            Cost at list price (recorded)"
	@echo "detox: data, G first attempts, E labels  OpenAI (G), Anthropic (E)          -              ~\$$80  (step 0 E prefilter \$$53, G \$$6, E \$$21)"
	@echo "detox: verifier scores (V1, V2)          -                                  1x A100 80GB   ~0.2 GPU-h"
	@echo "detox: thresholds, channel, gate         -                                  -              0"
	@echo "detox: Exp 1 + Exp 2 primary             OpenAI (M), Anthropic (E), xAI     1x A100 80GB   ~\$$12 API; ~0.5 GPU-h (Exp 2 loop)"
	@echo "detox: extensions (Exp 1, Exp 2)         OpenAI, Anthropic, xAI             1x A100 80GB   ~\$$42 API; ~2.3 GPU-h active"
	@echo "math:  G first attempts (9,000 problems) OpenAI (G); E is local Math-Verify  -              ~\$$106"
	@echo "math:  verifier scores (V1, V2)          -                                  1x A100 80GB   ~39 GPU-h (recorded session)"
	@echo "math:  Exp 1 + Exp 2 primary             OpenAI (M)                         1x A100 80GB   ~\$$33 API; ~7 GPU-h (Exp 2 loop)"
	@echo "math:  extensions + correction ceiling   OpenAI (M)                         1x A100 80GB   ~\$$101 API; ~17 GPU-h"
	@echo "Total recorded API spend for the paper's runs: detox ~\$$134, math ~\$$240 (list prices; smokes excluded)."

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('$(OUT)', '.pytest_cache', '.ruff_cache', 'build', 'dist')]"
