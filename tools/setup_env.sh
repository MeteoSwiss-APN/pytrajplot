#!/bin/bash
#
# Create conda environment with pinned or unpinned requirements
#
# - 2022-08 (D. Regenass) Write original script
# - 2022-09 (S. Ruedisuehli) Refactor; add some options
#

# Default env names
DEFAULT_ENV_NAME="pytrajplot"

# Default options
ENV_NAME="${DEFAULT_ENV_NAME}"
PYVERSION=3.11
PINNED=true
EXPORT=false
CONDA=conda
HELP=false

# Environment file (pinned dependencies, used with 'conda env create', YAML format)
ENV_PINNED=requirements/environment.yml
# Requirement file (unpinned dependencies, used with 'conda create', plain text format)
# Note: Unlike the blueprint, a plain text file is used here for unpinned requirements
ENV_UNPINNED=requirements/requirements.txt

help_msg="Usage: $(basename "${0}") [-n NAME] [-p VER] [-u] [-e] [-m] [-h]

Options:
 -n NAME    Env name [default: ${DEFAULT_ENV_NAME}
 -p VER     Python version [default: ${PYVERSION}]
 -u         Use unpinned requirements (minimal version restrictions)
 -e         Export environment files (requires -u)
 -m         Use mamba instead of conda
 -h         Print this help message and exit
"

# Eval command line options
while getopts n:p:defhimu flag; do
    case ${flag} in
        n) ENV_NAME=${OPTARG};;
        p) PYVERSION=${OPTARG};;
        e) EXPORT=true;;
        h) HELP=true;;
        m) CONDA=mamba;;
        u) PINNED=false;;
        ?) echo -e "\n${help_msg}" >&2; exit 1;;
    esac
done

if ${HELP}; then
    echo "${help_msg}"
    exit 0
fi

echo "Setting up environment for installation"
eval "$(conda shell.bash hook)" || exit  # NOT ${CONDA} (doesn't work with mamba)
conda activate || exit # NOT ${CONDA} (doesn't work with mamba)

# Create new env; pass -f to overwriting any existing one
echo "Creating ${CONDA} environment"
# Note: Unlike the blueprint, the environment is not created beforehand here,
#       because this led to numerous conflicts when trying to update the
#       environment with all dependencies in a second step
#${CONDA} create -n ${ENV_NAME} python=${PYVERSION} --yes || exit

# Install requirements in new env
if ${PINNED}; then
    echo "Pinned installation"
    # Note: Unlike the blueprint, the environment is created, not updated here
    ${CONDA} env create --force --name ${ENV_NAME} --file $ENV_PINNED || exit
else
    echo "Unpinned installation"
    # Note: Unlike the blueprint, a plain text file is used for unpinned requirements
    # Note: Unlike the blueprint, the environment is created, not updated here
    ${CONDA} create --force --name ${ENV_NAME} --file $ENV_UNPINNED --yes || exit
    if ${EXPORT}; then
        echo "Export pinned prod environment"
        ${CONDA} env export --name ${ENV_NAME} --no-builds | \grep -v '^prefix:' > $ENV_PINNED || exit
    fi
fi

# Note: Unlike the blueprint, print out a message that environment has been created
echo Environment created, install package with:

# Note: Unlike the blueprint, give instructions on how to install the package.
#       If package installation should be done automatically,
#       remove the leading 'echo' commands:
echo conda activate ${ENV_NAME}
echo python -m pip install .
