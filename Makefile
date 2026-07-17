.PHONY: audit test lint

audit:
	bash scripts/check_deps.sh

test:
	pytest tests/ -v -m "not slow"

lint:
	ruff check .
