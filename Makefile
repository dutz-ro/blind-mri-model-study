.PHONY: install test lint demo docker-build

install:
	./install.sh

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

demo:
	.venv/bin/mristudy demo run

docker-build:
	docker compose build
