# One-command entry points for reviewers. Everything runs from the repository root.
PYTHON ?= python3
VENV := .venv/bin

.PHONY: help install test lint demo api dashboard eval examples previews clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install core + UI dependencies into .venv
	$(PYTHON) -m venv .venv
	$(VENV)/pip install -r requirements-ci.txt
	$(VENV)/pip install --no-deps -e .
	$(VENV)/pip install -r requirements-apps.txt -r projects/realtime-multimodal/requirements.txt

lint: ## Static checks
	$(VENV)/ruff check .

test: ## Full test suite
	$(VENV)/pytest -q

demo: ## Start the flagship app in labeled demo mode (no model downloads)
	RETRIEVAL_MODE=demo $(VENV)/uvicorn ai_portfolio.api:app --host 0.0.0.0 --port 8000

neural: ## Start the flagship app with real Ollama generation
	RETRIEVAL_MODE=neural $(VENV)/uvicorn ai_portfolio.api:app --host 0.0.0.0 --port 8000

dashboard: ## Start the RAG dashboard (expects an API on port 8000)
	$(VENV)/streamlit run projects/production-rag-application/app.py --server.port 8501

eval: ## Deterministic smoke evaluation against the reviewed baseline
	$(VENV)/ai-eval --mode smoke --baseline data/eval/smoke-baseline.json

examples: ## Validate the shipped example fixtures
	$(VENV)/pytest tests/test_examples.py -q

previews: ## Re-render the interface structure previews
	$(VENV)/python scripts/make_previews.py

clean: ## Remove local artifacts and caches
	rm -rf artifacts .pytest_cache .ruff_cache
