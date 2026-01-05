REQUIREMENT = requirements.txt

VER  = $(word 2, $(shell python --version 2>&1))
SRC  = app.py app_test.py
PY36 = $(shell expr $(VER) \>= 3.6)

.PHONY: all docs deps pytest build test format clean

all: build test

docs:
	cd docs && make html

build:
	./build.sh

test: build
	@for t in build/src/basic/*/; do \
		name=$$(basename $$t); \
		[ -x "$$t$$name" ] && "$$t$$name" || true; \
	done
	@for t in build/src/raii/*/; do \
		name=$$(basename $$t); \
		[ -x "$$t$$name" ] && "$$t$$name" || true; \
	done

pytest: clean docs
	pycodestyle $(SRC)
	pydocstyle $(SRC)
	bandit app.py
	coverage run app_test.py && coverage report --fail-under=100 -m $(SRC)
ifeq ($(PY36), 1)
	black --quiet --diff --check --line-length 79 $(SRC)
endif

deps:
	pip install -r requirements.txt
ifeq ($(PY36), 1)
	pip install black==22.3.0
endif

format:
	find . -type f -name "*.cc" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" | xargs -I{} clang-format -style=file -i {}

clean:
	rm -rf build
	cd docs && make clean
