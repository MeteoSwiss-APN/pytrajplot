# PyTrajPlot
PyTrajPlot is a Python-based tool to visualize ICON & COSMO trajectory simulations results.

## Installation
PyFlexPlot is hosted on [GitHub](https://github.com/MeteoSwiss-APN/pytrajplot) Github. For the available releases, see [Releases](https://github.com/MeteoSwiss-APN/pytrajplot/releases).
### With Conda
Having Conda installed is a pre-requisit for the further installation process. If that is not the case, install the latest Miniconda version from [here](https://docs.conda.io/en/latest/miniconda.html). Afterwards, follow these instructions to clone the GitHub Repo; set up a conda environment and test all possible use-cases. Make sure to execute the following commands from the root of the `pytrajplot` directory.

1. `git clone https://github.com/MeteoSwiss-APN/pytrajplot.git`
2. `make venv install`
3. `conda activate pytrajplot`
4. `./tests/test_pytrajplot.sh`

---
If no errors occur, plost such as these should be saved in their respective folders, in the `test/` directory.
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
The only mandatory arguments are `INPUT_DIR` & `OUTPUT_DIR`. The input directory specifies the path to the source files. In the input directory, there should be at least one *plot_info* file, and for each trajectory file one corresponding start file.

### File Nomenclature
Should the prefixes of the file names deviate from the default values (*tra_geom_*, *startf_*, *plot_info*),  it is possible to specify the prefix of the start and trajectory files, as well as the name of the plot_info file.


The relevant part in the filename of the trajectory/start files, is the *key*. In general, the *key* looks like: `XXX-YYYF/B`. It has to satisfy the following *conditions*:

1. keys must match between start/trajectory file
```
traj_prefix+key <---> start_prefix+key
```
2. keys must end with **F** / **B** to determine the trajectories direction (forward/backward)
3. XXX refers to the initialisation time of the model (w.r.t the model base time, which is specified in the corresponding plot_info file)
5. YYY refers to the end-time of the model's computation (w.r.t to the model base time.
6. XXX and YYY are seperated by a dash
7. The difference of XXX and YYY equals the model's runtime.

Information in the header & footer of the output plots, is partially generated from the information in the *key*.

#### Examples
Backward Trajectories; 33h in the past from model base time until model base time.
> startf_033-000B/tra_geom_033-000B

Forwart trajectories; 48h to the future from model base time.
> startf_000-048F/tra_geom_000-048F

### Code Overview

This part is a small step-by-step guide, how an exemplary `pytrajplot` command runs through the code with references to the corresponding (Python) scripts and functions.

#### Exempli Gratia
```
pytrajplot tests/test_hres/4_altitudes/ plots
--- Parsing Input Files
--- Assembling Ouput
--- Done.
```

##### 0. [cli.py](src/pytrajplot/cli.py)
Before the input files get parsed, the user inputs need to be parsed using the function `interpret_options`.

##### 1. [parse_data.py](src/pytrajplot/parse_data.py)
In the next step the `check_input_dir` function from the data parser script is initialised.

#####
