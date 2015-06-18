.PHONY: clean nuke test coverage

PYENV = . env/bin/activate;
PYTHON = $(PYENV) python
PIP = $(PYENV) pip
EXTRAS_REQS := $(wildcard requirements-*.txt)

include analysis.mk

DISTRIBUTE = sdist bdist_wheel

release: env
	$(PYTHON) setup.py $(DISTRIBUTE) upload -r livefyre

package: env
	$(PYTHON) setup.py $(DISTRIBUTE)

test: extras
	$(PYTHON) `which nosetests` $(NOSEARGS)

extras: env/make.extras
env/make.extras: $(EXTRAS_REQS) | env
	rm -rf env/build
	$(PYENV) for req in $?; do pip install -r $$req; done
	touch $@

nuke: clean
	rm -rf *.egg *.egg-info env cover coverage.xml nosetests.xml

clean:
	python setup.py clean
	rm -rf dist build
	find . -path ./env -prune -o -type f -regex '.*\.pyc' -or -regex '.*\-theirs\..*' -exec rm {} \;

coverage: test
	open cover/index.html

ifeq ($(DOMINO_RUN),1)
VENV_OPTS="--system-site-packages"
else
VENV_OPTS="--no-site-packages"
endif

env: env/bin/activate
env/bin/activate: requirements.txt setup.py
	test -f $@ || virtualenv $(VENV_OPTS) env
	$(PYENV) easy_install -U pip
	$(PYENV) curl https://bootstrap.pypa.io/ez_setup.py | python
	$(PIP) install setuptools
	$(PIP) install distribute
	$(PIP) install wheel
	$(PIP) install numpy
	$(PIP) install -r $<
	touch $@
