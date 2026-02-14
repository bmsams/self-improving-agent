.PHONY: test test-live lint typecheck bench run run-live clean help

help: ## Show this help message
	@echo "Self-Improving Agent — Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

test: ## Run unit tests only (no API calls)
	python -m pytest tests/test_agent.py tests/test_agentcore.py -v --tb=short

test-live: ## Run live integration tests (requires ANTHROPIC_API_KEY)
	python -m pytest tests/test_integration_live.py -v --tb=long -s

test-all: ## Run all tests (unit + live)
	python -m pytest tests/ -v --tb=short

lint: ## Run ruff linter
	python -m ruff check src/ tests/ main.py

lint-fix: ## Run ruff linter with auto-fix
	python -m ruff check src/ tests/ main.py --fix

typecheck: ## Run mypy type checker
	python -m mypy src/ --ignore-missing-imports

bench: ## Run benchmark suite
	python main.py bench

run: ## Run single generation with mock provider
	python main.py run --mock --auto-merge

run-live: ## Run single generation with real LLM
	python main.py run --auto-merge

loop: ## Run improvement loop (3 generations, mock)
	python main.py loop --max-gen 3 --mock --auto-merge

loop-live: ## Run improvement loop with real LLM (3 generations)
	python main.py loop --max-gen 3 --auto-merge

status: ## Show current agent state
	python main.py status

history: ## Show evolution history
	python main.py history

reset: ## Reset agent state to generation 0
	python main.py reset

clean: ## Remove generated files
	rm -rf .pytest_cache __pycache__ .mypy_cache .coverage coverage.json
	rm -rf .benchmark_history.json .agent_state.json .junit-results.xml
	rm -rf reports/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
