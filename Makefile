.PHONY: help test test-unit test-lint lint demo examples version install clean check

# ── Default ───────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Credence — epistemic guard for AI-generated code"
	@echo ""
	@echo "  make test          Full test suite (same command CI runs)"
	@echo "  make test-unit     Unit tests only"
	@echo "  make lint          ruff"
	@echo "  make demo          The 30-second smoke test (no API key)"
	@echo "  make examples      Run every script in examples/"
	@echo "  make version       Print the installed version"
	@echo "  make check         lint + test + demo, in that order"
	@echo ""
	@echo "  make install       pip install -e \".[dev]\""
	@echo "  make clean         Remove __pycache__ and .pytest_cache"
	@echo ""

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	python3 -m pytest tests/ -q

test-unit:
	python3 -m pytest tests/unit/ -q

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	python3 -m ruff check credence/ tests/ examples/

# ── Runnable things ───────────────────────────────────────────────────────────

demo:
	python3 -m credence demo

examples:
	python3 examples/quickstart.py
	python3 examples/hook_demo.py

version:
	python3 -m credence --version

# ── Composite gate ────────────────────────────────────────────────────────────

check: lint test demo
	@echo ""
	@echo "check: lint, tests, and demo all passed"

# ── Setup and housekeeping ────────────────────────────────────────────────────

install:
	python3 -m pip install -e ".[dev]"

# Removes regenerable caches only. No source, no databases.
clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
	@echo "clean: removed __pycache__, .pytest_cache, .ruff_cache"
