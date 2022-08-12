SHELL = /bin/bash

# Let all commands of a receipe run in one shell,
# so that conda activate is effective for the consecutive commands.
.ONESHELL:

#==============================================================================
# Options
#==============================================================================

# Defaults
# --------
# Name of target environment
NAME = pytrajplot

# Default for pinned/unpinned installation
# Override the default with eiher 'make install PINNED=1'
# or 'make install PINNED=0' as appropriate.
PINNED = 1

# Settings
# --------
# Name of development environment
NAME_DEV = $(NAME)-dev
# Files defining environment
FILE_UNPINNED = requirements/requirements.txt
FILE_PINNED = environment.yml
FILE_DEV_UNPINNED = requirements/dev-requirements.txt
FILE_DEV_PINNED = dev-environment.yml
#------------------------------------------------------------------------------

# Command marker for help target
# (second line to add a blank without becoming a search target for grep)
MARKER=\#CMD
MARKER+=


# Command to activate conda environment ('conda activate' does not work in a makefile)
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Options for call to conda env
ifneq ($(PREFIX),)
  TARGET_ENV = --prefix $(PREFIX)
  NAME_OR_PREFIX = $(PREFIX)
  TARGET_ENV_DEV = --prefix $(PREFIX)
  NAME_OR_PREFIX_DEV = $(PREFIX)
else
  TARGET_ENV = --name $(NAME)
  NAME_OR_PREFIX = $(NAME)
  TARGET_ENV_DEV = --name $(NAME_DEV)
  NAME_OR_PREFIX_DEV = $(NAME_DEV)
endif

ifeq ($(PINNED),0)
  FILE = $(FILE_UNPINNED)
  FILE_DEV = $(FILE_DEV_UNPINNED)
else
  FILE = $(FILE_PINNED)
  FILE_DEV = $(FILE_DEV_PINNED)
endif

#==============================================================================
# Help
#==============================================================================
.PHONY: help    #CMD Print this help page.
help:
	@echo -e "\nAdd PINNED=0 to the 'make' command for unpinned installation."
	@echo -e "Add NAME=name_of_env to use a different environment name"
	@echo -e "or  PREFIX=/path/to/env to install to a different location."
	@echo -e "\nTargets:"
	@grep "$(MARKER)" Makefile | sed 's/.PHONY: //' | sed "s/$(MARKER)/\t/"

#==============================================================================
# Installation
#==============================================================================

.PHONY: install #CMD Install the package with runtime dependencies.
install: env
	@echo -e "\n[make install] installing the package with runtime dependencies."
	$(CONDA_ACTIVATE) $(NAME_OR_PREFIX)
	python -m pip install .
	pytrajplot -V
	@echo "To activate this environment, use: conda activate $(NAME_OR_PREFIX)"

.PHONY: install-dev #CMD Install the package as editable with runtime and development dependencies.
install-dev: env-dev
	@echo -e "\n[make install-dev] installing the package as editable with development dependencies"
	$(CONDA_ACTIVATE) $(NAME_OR_PREFIX_DEV)
	python -m pip install --editable .
	pytrajplot -V
	@echo "To activate this environment, use: conda activate $(NAME_OR_PREFIX_DEV)"

#==============================================================================
# Create Conda Environment
#==============================================================================

env: $(FILE)
	@echo -e "\n[make env] creating conda environment:"  $(NAME_OR_PREFIX)
	@echo -e "[make env] from file:" $(FILE)
	conda env create --force $(TARGET_ENV) --file $(FILE)
	@echo -e "\n[make env-dev] conda environment created:"  $(NAME_OR_PREFIX_DEV)

.PHONY: env-dev #CMD Add the development environment for the package.
env-dev: $(FILE) $(FILE_DEV)
	@echo -e "\n[make env-dev] creating conda environment:"  $(NAME_OR_PREFIX_DEV)
	@echo -e "[make env-dev] from file:" $(FILE)
	conda env create --force $(TARGET_ENV_DEV) --file $(FILE)
	@echo -e "[make env-dev] installing dev requirements:"  $(NAME_OR_PREFIX_DEV)
	@echo -e "[make env-dev] from file:" $(FILE) $(FILE_DEV)
	conda env create         $(TARGET_ENV_DEV) --file $(FILE_DEV)
	@echo -e "\n[make env-dev] conda environment created:"  $(NAME_OR_PREFIX_DEV)

.PHONY: pinned  #CMD Save the current environment for pinned installation.
pinned:
	@echo -e "\n[make pinned] creating file defining pinned conda environment"
	$(CONDA_ACTIVATE) $(NAME_OR_PREFIX)
	conda env export --no-builds | tail -n +2 | head -n -2 > $(FILE_PINNED)
	@echo "[make pinned] Pinned environment saved in" $(FILE_PINNED)

.PHONY: pinned-dev #CMD Save the current environment for pinned development installation.
pinned-dev:
	@echo -e "\n[make pinned-dev] creating file defining pinned conda environment"
	$(CONDA_ACTIVATE) $(NAME_OR_PREFIX)
	conda env export --no-builds | tail -n +2 | head -n -2 > $(FILE_DEV_PINNED)
	@echo "[make pinned-dev] Pinned environment saved in" $(FILE_DEV_PINNED)

#==============================================================================
# Run the tests
#==============================================================================

.PHONY: test    #CMD Run tests.
test:
	@echo -e "\n[make test] running all tests"
	$(CONDA_ACTIVATE) $(NAME_OR_PREFIX)
	pytest tests

#==============================================================================
