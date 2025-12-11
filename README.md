# PyTrajPlot

PyTrajPlot is a Python tool to visualize trajectories calculated with
[LAGRANTO](https://www.research-collection.ethz.ch/handle/20.500.11850/103598) based on
model outputs from ECMWF's [IFS](https://www.ecmwf.int/en/forecasts/documentation-and-support)
model (European/global domains), the [COSMO](https://www.cosmo-model.org) model, or the
[ICON](https://www.icon-model.org) model (limited domain centered over Switzerland).

## Installation using Conda and Poetry

### Get source code

Create a local copy of the git repository and change working directory to it:

```bash
git clone https://github.com/MeteoSwiss-APN/pytrajplot.git
cd pytrajplot
```

### Create Conda environment

Create an environment with Conda and activate it

```bash
conda create -n pytrajplot python=3.13 poetry=1.8
conda activate pytrajplot
```

### Build the project (Poetry)
Alternative build with mchbuild see further below.

```bash
poetry install
```

### Run tests

```bash
poetry run pytest
```

If no errors occur, the tests save plots in their respective folders in the
`local` directory. Example output images:

![example1](https://i.imgur.com/Zp4F9Z7.jpg)
![example2](https://i.imgur.com/4WvLK1x.jpg)

### Run quality tools

```bash
poetry run pylint pytrajplot
poetry run mypy pytrajplot
```

### Build the project (mchbuild)

```bash
pipx install mchbuild
mchbuild local.build.install
# Optional
mchbuild local.build.format
mchbuild local.build.docs
mchbuild local.test.unit
mchbuild local.test.lint
mchbuild local.run
```

More information: see [`.mch-ci.yml`](.mch-ci.yml) and the [mchbuild documentation](https://meteoswiss.atlassian.net/wiki/x/YoM-Jg?atlOrigin=eyJpIjoiNDgxYmJjMDhmNDViNGIyNmI1OGU4NzY4NTFhNzViZWEiLCJwIjoiYyJ9).

## Usage

Get a short help with a list of available options:

```bash
pytrajplot --help
```

Output of the above command at the time of writing this documentation:


```
Usage: pytrajplot [OPTIONS] INPUT_DIR OUTPUT_DIR

Options:
  --start-prefix TEXT             Prefix for the start files. Default: startf_
  --traj-prefix TEXT              Prefix for the start files. Default:
                                  tra_geom_
  --info-name TEXT                Name of plot_info file. Default: plot_info
  --separator TEXT                Separator str between origin of trajectory
                                  and side trajectory index. Default: ~
  --language [en|english|de|ger|german|Deutsch|deutsch]
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

Only `INPUT_DIR` and `OUTPUT_DIR` are mandatory. The input directory must contain
exactly one `plot_info` file and one or more pairs of one trajectory file and one starting points file each.

### File naming convention

The name of the plot information file (default `plot_info`) and the prefixes of the 
trajectory data file (default `tra_geom_`) and the starting point file (default `startf_`)
can be specified with options.

The relevant part of the trajectory-data/starting-points filenames is the *key*, generally
formatted as `XXX-YYYF` or `XXX-YYYB`. See below for the meaning of the parts.

Naming requirements:

1.  Keys must match between start and trajectory files (i.e. `traj_prefix+key`
    matches `start_prefix+key`).
2.  Keys must end with `F` or `B` to indicate trajectory direction.
3.  `XXX` refers to the 3-digit start hour of the computed trajectory,
    relative to the base time (initial time) of the weather prediction model.
4.  `YYY` refers to the 3-digit end hour of the trajectory, relative to model base time.
5.  `XXX` and `YYY` are separated by a dash.
6.  The difference between `YYY` and `XXX` equals the trajectory length in
    hours. For backward trajectories, `XXX` is greater than `YYY`.

Information in the header and footer of output plots is partially generated
from the key.

### Examples

Backward trajectories (starting at 33 h in the future, going backwards to the model base time):

```
startf_033-000B/tra_geom_033-000B
```

Forward trajectories (from model base time 48 h into the future):

```
startf_000-048F/tra_geom_000-048F
```
## Code overview

This is a short guide showing how a `pytrajplot` invocation flows through the
codebase. See the referenced modules for implementation details.

Example output:

```
pytrajplot tests/test_hres/4_altitudes/ plots
--- Parsing Input Files
--- Assembling Output
--- Done.
```

1. `pytrajplot/cli.py` — argument parsing (function: `interpret_options`).

2.  Parsing input files:
    [`pytrajplot/parse_data.py`](pytrajplot/parse_data.py) — 
    the `check_input_dir` function
    scans the input directory and parses `start` and `plot_info` files. The
    `read_startf` and `PLOT_INFO` utilities parse those files; `read_trajectory`
    parses trajectory files.

    The parsing pipeline returns two dictionaries: `trajectory_dict` (main) where
    each key maps to a pandas DataFrame with combined start/trajectory data, and a
    second dictionary containing plot_info data.

3.  Assembling output:
    [`pytrajplot/generate_pdf.py`](pytrajplot/generate_pdf.py) —
    converts each DataFrame to plots by calling the plotting pipeline (function `assemble_pdf`), which:

   - creates the output directory (if needed),
   - adds altitude plots, header/footer and map figures,
   - saves figures.

See the [`src/pytrajplot/plotting`](src/pytrajplot/plotting) directory for plotting scripts.

> Fun fact: MeteoSwiss generates roughly 2800 trajectory plots per day for the
> IFS and ICON-CH1-EPS models.

## Release

Releases follow a GitOps flow and are triggered by creating Git tags. Tags must
follow semantic versioning (https://semver.org/) and be PEP 440 compatible
(https://peps.python.org/pep-0440/).

## Deploy

The continuous integration (CI) pipelines publish artifacts to the artifact registry. For Kubernetes (k8s) deployments,
create a deployment pipeline using `Jenkinsfile_k8s_deploy` as `Jenkinsfile`
and trigger it from Jenkins with the chosen branch or tag. For ACPM-style
deployments, update the artifact list in the deployment repository with the
new version.
