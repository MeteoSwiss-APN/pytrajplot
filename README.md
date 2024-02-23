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

### Prerequisite

Having Conda installed is a prerequisite for the further installation
process. To install an appropriate Miniconda version, use

    tools/setup_miniconda.sh

or install the latest version manually from the
[miniconda webpage](https://docs.conda.io/en/latest/miniconda.html).
Then follow the
instructions here below to set up a conda environment and test multiple use-cases.

### Create environment

Create an environment and install the package dependencies with the  script `setup_env.sh` provided in the `tools` directory.
Check available options with

    tools/setup_env.sh -h

We distinguish pinned installations based on exported (reproducible) environments,
saved in `requirements/environment.yml`,
and free installations, where the installation
is based on top-level dependencies listed in `requirements/requirements.txt`.
A pinned installation in an conda environment
with the default name `pytrajplot` is done with

    tools/setup_env.sh

Add the option `-n <package_env_name>` to create an environment with a custom name.

If you start developing a new version, you might want to do an unpinned installation with option `-u` and export the environment with option `-e`:

    tools/setup_env.sh -u -e

*Note*: The flag `-m` can be used to use `mamba` as solver
instead of the built-in `conda`
solver. However since `conda` version 23, the mamba solver is the default solver
in conda an no speed up is achieved by this option,
thus we no longer recommend its use.

### Install Package

Activate the newly created environment with (replace `pytrajplot` by your custom
name `<package_env_name>` if you have used the `-n <package_env_name>` option).

    conda activate pytrajplot

The package itself is installed with `pip`. As all dependencies are already
installed by conda and should not be modified by pip, use the `--no-deps` flag.

    pip install --no-deps .

For development, install the package in editable mode:

    pip install --editable --no-deps .

*Warning:* Make sure you use the right pip, i.e. the one from the installed conda environment (`which pip` should point to something like `path/to/miniconda/envs/<package_env_name>/bin/pip`).

Once your package is installed, run the basic tests by typing:

    pytest

A more comprehensive set of tests can be exectuted by running the script

    tests/test_pytrajplot.sh

If developing, make sure to update the requirements file
and export your environment after installation
every time you add new imports while developing.

---

### Test results

If no errors occur, the above test script save plots such as these
here below in their respective folders in the `local` directory.
![](https://i.imgur.com/Zp4F9Z7.jpg)
![](https://i.imgur.com/4WvLK1x.jpg)

## Usage

Activate the conda environment (you might have chosen a different name for the environment than the default name `pytrajplot`):

    conda activate pytrajplot

To get a list of all available commands, just type:

    pytrajplot --help

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
at exactly **one `plot_info`** file, and for each trajectory file one
corresponding `start` file.

### File Nomenclature

Should the prefixes of the file names deviate from the default values
(*tra_geom_*, *startf_*, *plot_info*), it is possible to specify the prefix of
the start and trajectory files, as well as the name of the plot_info file.

The relevant part in the filename of the trajectory/start files, is the
*key*. In general, the *key* looks like: `XXX-YYYF/B`. It has to satisfy the
following *conditions*:

1. Keys must match between start/trajectory file

    ```
    traj_prefix+key <---> start_prefix+key
    ```

2. Keys must end with **F** / **B** to determine the trajectories direction (forward/backward)

3. XXX refers to the start of the computation of trajectories (w.r.t the model base time,
   which is specified in the corresponding plot_info file)

4. YYY refers to the end-time of the trajectory computation (w.r.t to the model base time.

5. XXX and YYY are seperated by a dash

6. The difference of XXX and YYY equals the trajectory length (in hours).

Information in the header and footer of the output plots, is partially generated
from the information in the *key*.

#### Examples

Backward Trajectories; 33 h in the past from model base time until model base time.

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

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the MeteoSwiss
blueprint for the CSCS systems.
