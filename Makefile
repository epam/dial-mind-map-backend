PORT ?= 5000

.PHONY: all install build serve help

all: build

install:
	poetry install

build: install
	poetry build

serve: install
	poetry run uvicorn "general_mindmap.v2.app:app" --host "0.0.0.0" --reload --port $(PORT)

help:
	@echo '===================='
	@echo 'build                        - build the source and wheels archives'
	@echo '-- RUN --'
	@echo 'serve                        - run the dev server locally'
