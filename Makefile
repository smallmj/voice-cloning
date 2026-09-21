# VoiceClone 命令集（详见 README.md「命令集」）

.PHONY: test test-frontend typecheck lint format

test:
	cd sidecar && uv run pytest

test-frontend:
	cd app && npm test

typecheck:
	cd app && npm run typecheck

lint:
	cd sidecar && uv run ruff check .
	cd app && npm run lint

format:
	cd sidecar && uv run ruff check --fix . && uv run ruff format .
	cd app && npx prettier --write .
