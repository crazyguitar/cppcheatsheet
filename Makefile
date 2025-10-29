REQUIREMENT = requirements.txt

VER  = $(word 2, $(shell python --version 2>&1))
SRC  = app.py app_test.py
PY36 = $(shell expr $(VER) \>= 3.6)

.PHONY: build deps test
build: html

%:
	cd docs && make $@

test: clean build
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
