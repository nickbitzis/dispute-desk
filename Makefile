PORT ?= 8833
TXNS ?= 300000

.PHONY: install test lint format pipeline gate policy run sweep screenshot docker-build compose-up clean

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

pipeline:  ## generate, train, score, compare policies, analytics, seed the store (about 30 seconds)
	uv run disputedesk all --txns $(TXNS)

gate:  ## the CI gate: fail unless the model policy beats fight-everything on held-out disputes
	uv run disputedesk --data-dir data/ci all --txns $(TXNS) --gate

policy:
	uv run disputedesk policy

run:
	uv run disputedesk serve --port $(PORT)

sweep:  ## needs the server running: reminders, auto-accepts, issuer timeouts
	curl -s -X POST localhost:$(PORT)/api/sweep -H 'content-type: application/json' -d '{}'

screenshot:  ## needs the server running on $(PORT) and Google Chrome
	uv run python scripts/screenshot.py --port $(PORT)

docker-build:
	docker build -t dispute-desk .

compose-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache data
