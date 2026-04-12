PYTHON ?= python

.PHONY: smoke test lint

smoke:
	$(PYTHON) -m nanocoop.cli offline --config configs/offline_smoke.yaml
	$(PYTHON) -m nanocoop.cli rlvr --config configs/rlvr_smoke.yaml
	$(PYTHON) -m nanocoop.cli prompt-opt --config configs/prompt_opt_smoke.yaml

test:
	pytest -q

lint:
	ruff check src tests
