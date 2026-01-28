VENV = .venv
PYTHON = python3.12

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install pandas seaborn matplotlib numpy

run:
	$(VENV)/bin/python main.py

clean:
	rm -rf $(VENV)

reinstall: clean install

.PHONY: install clean reinstall run
