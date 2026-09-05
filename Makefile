PYTHON ?= .venv/bin/python
PYTEST = $(PYTHON) -m pytest
export CUDA_VISIBLE_DEVICES =
export JAX_PLATFORMS = cpu
export OMP_NUM_THREADS = 2
export OPENBLAS_NUM_THREADS = 2
export MKL_NUM_THREADS = 2
export MPLBACKEND = Agg
export PYTEST_DISABLE_PLUGIN_AUTOLOAD = 1

.PHONY: test test-unit test-integration test-static test-e2e test-slow lint

test:
	$(PYTEST) --junitxml=temp/test-results/default.xml

test-unit:
	$(PYTEST) -m unit --junitxml=temp/test-results/unit.xml

test-integration:
	$(PYTEST) -m 'integration and not slow' --junitxml=temp/test-results/integration.xml

test-static:
	$(PYTEST) -m static --junitxml=temp/test-results/static.xml

test-e2e:
	$(PYTEST) tests/e2e -m e2e --junitxml=temp/test-results/e2e.xml

test-slow:
	$(PYTEST) -m slow --junitxml=temp/test-results/slow.xml

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
