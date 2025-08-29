.PHONY: help install format lint check test clean run

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

format: ## Format code with Black and isort
	@./scripts/format_code.sh

lint: ## Run flake8 linting
	uv run flake8 backend/ main.py

check: ## Run all quality checks (format check, import check, lint)
	@./scripts/quality_check.sh

test: ## Run tests
	uv run pytest backend/tests/ -v

clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

run: ## Start the development server
	cd backend && uv run uvicorn app:app --reload --port 8000

quality: format lint test ## Run format, lint, and test in sequence