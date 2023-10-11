# PyTrajPlot

PyTrajPlot is a Python-based tool to visualize trajectories calculated with
[LAGRANTO](https://www.research-collection.ethz.ch/handle/20.500.11850/103598)
based on ECMWF's [IFS-HRES](https://www.ecmwf.int/en/forecasts/documentation-and-support)
(on an European or global domain) or the [COSMO](https://www.cosmo-model.org) model
(on a limited domain centered over Switzerland).

## Installation

PyTrajPlot is hosted on
[GitHub](https://github.com/MeteoSwiss-APN/pytrajplot). For the available
releases, see [Releases](https://github.com/MeteoSwiss-APN/pytrajplot/releases).

Create a local copy of the repository and continue installation from the root of
the `pytrajplot` directory.

    git clone https://github.com/MeteoSwiss-APN/pytrajplot.git
    cd pytrajplot

### With Conda

Having Conda installed is a prerequisite for the further installation
process. If that is not the case, install the latest Miniconda version with

    tools/setup_miniconda.sh

or install it manually from the
[miniconda webpage](https://docs.conda.io/en/latest/miniconda.html).
To activate the base environment from the installed miniconda run `source miniconda/bin/activate`.
Afterwards, follow the
instructions here below to clone the GitHub repository, set up a conda environment and test all
possible use-cases.

Create an environment and install the package dependencies with the provided script `setup_env.sh`.
Check available options with
```bash
tools/setup_env.sh -h
```
We distinguish pinned installations based on exported (reproducible) environments,
and free installations, where the installation
is based on top-level dependencies listed in `requirements/requirements.txt`.
A pinned installation is done with

    tools/setup_env.sh -n <package_env_name>

If you start developing, you might want to do an unpinned installation and export the environment:

    tools/setup_env.sh -u -e -n <package_env_name>

*Hint*: Use the flag `-m` to speed up the installation using mamba. You will of course have to
install mamba first. We recommend to install mamba into your base
environment `conda install -c conda-forge mamba`. If you install mamba in another (maybe dedicated)
environment, environments installed with mamba will be located
in `<miniconda_root_dir>/envs/mamba/envs`, which is not very practical.

The package itself is installed with `pip`.

    conda activate <package_env_name>
    pip install .

For development, install the package in editable mode:

    conda activate <package_env_name>
    pip install --editable .

*Warning:* Make sure you use the right pip, i.e. the one from the installed conda environment (`which pip` should point to something like `path/to/miniconda/envs/<package_env_name>/bin/pip`).

Once your package is installed, run the tests by typing:

```
conda activate <package_env_name>
pytest
```

If the tests pass, you are good to go.
If developing, make sure to update the requirements file
and export your environment after installation
every time you add new imports while developing.

### Shortcut using Makefile

Instead of intalling the environment and the package with the above commands,
you may alternatively use the available Makefile.
Using the Makefile is deprecated, because MeteoSwiss' blueprint stopped the support. It will be removed in a future version.
Please migrate to instruction in section [With Conda](#with-conda).
Use `make help` to see other available make targets. Use the `--dry-run`
option to see the commands `make` would run without executing them.

    make install
    make test

---
### Test results

If no errors occur, plots such as these should be saved in their respective folders in the `./local` directory.
![](https://i.imgur.com/Zp4F9Z7.jpg)
![](https://i.imgur.com/4WvLK1x.jpg)


## Usage

Activate the conda environment:
```conda activate pytrajplot```
To get a list of all available commands, just type:
```pytrajplot --help```
The possible options are as follows:

```
Usage: pytrajplot [OPTIONS] INPUT_DIR OUTPUT_DIR

Options:
  --start-prefix TEXT             Prefix for the start files. Default: startf_
  --traj-prefix TEXT              Prefix for the start files. Default:
                                  tra_geom_

  --info-name TEXT                Prefix for the plot info files. Default:
                                  plot_info

  --separator TEXT                Separator str between origin of trajectory
                                  and side trajectory index. Default: ~

  --language [en|english|de|ger|german|Deutsch]
                                  Choose language. Default: en
  --domain [ch|europe|centraleurope|alps|dynamic|dynamic_zoom]
                                  Choose domains for map plots. Default:
                                  centraleurope, europe, dynamic

  --datatype [eps|jpeg|jpg|pdf|pgf|png|ps|raw|rgba|svg|svgz|tif|tiff]
                                  Choose data type(s) of final result.
                                  Default: pdf

  -V, --version                   Print version and exit.
  --help                          Show this message and exit.
```

The only mandatory arguments are `INPUT_DIR` and `OUTPUT_DIR`. The input directory
specifies the path to the source files. In the input directory, there should be
at exactly **one plot_info** file, and for each trajectory file one
corresponding start file.

### File Nomenclature

Should the prefixes of the file names deviate from the default values
(*tra_geom_*, *startf_*, *plot_info*), it is possible to specify the prefix of
the start and trajectory files, as well as the name of the plot_info file.

The relevant part in the filename of the trajectory/start files, is the
*key*. In general, the *key* looks like: `XXX-YYYF/B`. It has to satisfy the
following *conditions*:

1. keys must match between start/trajectory file
```
traj_prefix+key <---> start_prefix+key
```
2. keys must end with **F** / **B** to determine the trajectories direction (forward/backward)
3. XXX refers to the start of the computation of trajectories (w.r.t the model base time,
   which is specified in the corresponding plot_info file)
4. YYY refers to the end-time of the trajectory computation (w.r.t to the model base time.
5. XXX and YYY are seperated by a dash
6. The difference of XXX and YYY equals the trajectory length (in hours).

Information in the header and footer of the output plots, is partially generated
from the information in the *key*.

#### Examples

Backward Trajectories; 33h in the past from model base time until model base time.

> startf_033-000B/tra_geom_033-000B

Forwart trajectories; 48h to the future from model base time.

> startf_000-048F/tra_geom_000-048F

### Code Overview

This part is a small step-by-step guide, how an exemplary `pytrajplot` command
runs through the code with references to the corresponding (Python) scripts and
functions.

#### Example

```
pytrajplot tests/test_hres/4_altitudes/ plots
--- Parsing Input Files
--- Assembling Ouput
--- Done.
```

##### 0. [cli.py](src/pytrajplot/cli.py)

Before the input files get parsed, the user inputs need to be parsed using the
function `interpret_options`.

##### 1. Parsing Input Files: [parse_data.py](src/pytrajplot/parse_data.py)

In the next step the `check_input_dir` function from the data parser script is
initialised.

###### Procedure

1. iterate through the directory and read the start & plot_info files. simultaneously collect all present keys
   *Remark:* The start file is parsed using the `read_startf` function and
   the plot_info file is parsed using the `PLOT_INFO` [class](src/pytrajplot/parsing/plot_info.py).
2. for each found key, parse corresponding trajectory file using the `read_trajectory` function.

There is a number of different helper-functions involved in the parsing of these
files. The code is well commented and the docstrings should provide further
information on the use of each function, see
[here](src/pytrajplot/parse_data.py).

Ultimately, the parsing-pipeline returns two *dictionaries*. The main
dictionary, containing all information, is the `trajectory_dict`. Each key
contains a `pandas dataframe`, with the combined information of the
corresponding start/trajectory file. The second dictionary contains the relevant
information of the plot_info file, which corresponds to all start/trajecotry
files.

##### 3. Assembling Ouput: [generate_pdf.py](src/pytrajplot/generate_pdf.py)

Once all the data from one directory is in this usable dictionary format, the
plotting pipeline is initialised. The first part of the
[generate_pdf](src/pytrajplot/generate_pdf.py) script iterates through this
dictionary, retrieves the dataframes and "parses" them. For each trajectory
origin, the plotting pipeline is called and one plot generated. Usually, there
are several trajectories/origins per dataframe.

*Fun Fact:* @MeteoSwiss approximately 2800 trajectory plots are generated each day for the IFS-HRES (over Europe and globally) and COSMO-1E models.

###### Procedure

1. iterate through `trajectory_dict`
2. retrieve `df` for current key
3. iterate through dataframe
4. for each origin, present in current dataframe, fill a new dictionary (`plot_dict`) with plot-specific information.
5. initialise pipeline with the plot_dict by calling `assemble_pdf`
    5.1 create output directory (if it doesn't exist)
    5.2 add altitude plot to figure
    5.3 add footer to figure
    5.4 add header to figure
    5.5 add map figure
6. save figure
7. repeat steps 3.-6. until all plots for the current dataframes/domains/datatypes have been generated
8. repeat steps 2.-7. until all figures for all start/trajectory files have been generated
9. return `-- done`

###### Remark

Again, this procedure outlines the inner workings of the plotting scripts. For
greater insight, it is recommended to read the scripts and pay special attention
to the comments and docstrings. All plotting-scripts are located
[here](src/pytrajplot/plotting).



## Future Developments

Possible developments in the future:
- [ ] Add debug statements and debug flag
- [ ] Write Class for COSMO trajectory files
- [ ] Write Class for HRES trajectory files
- [ ] Write Class for start files
- [ ] Make Code more efficient
- [ ] Fix Aspect Ratio of Map, for trajectories with unusual longitudinal/latitudinal expansions

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the MeteoSwiss
blueprint for the CSCS systems.
