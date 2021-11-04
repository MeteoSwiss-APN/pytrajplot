"""Command line interface of pytrajplot."""
# Standard library
import csv
import datetime
import json
import logging
import os

# Third-party
import click

# Local
from . import __version__
from .get_data import *
from .plot_altitude import *
from .plot_map import *
from .utils import count_to_log_level


def interpret_options(start_prefix, traj_prefix, info_prefix, language):
    prefix_dict = {
        "start": start_prefix,
        "trajectory": traj_prefix,
        "plot_info": info_prefix,
    }

    if language[0] == "e":
        language = "en"
    else:
        language = "de"

    return prefix_dict, language


### DEFINE COMMAND LINE INPUTS
@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(exists=False))
@click.option(
    "--start-prefix",
    default="startf_",
    type=str,
    help="Prefix for the start files. Default: startf_",
)
@click.option(
    "--traj-prefix",
    default="tra_geom_",
    type=str,
    help="Prefix for the start files. Default: tra_geom_",
)
@click.option(
    "--info-prefix",
    default="plot_info",
    type=str,
    help="Prefix for the plot info files. Default: plot_info",
)
@click.option(
    "--separator",
    default="~",
    type=str,
    help="Separator str between origin of trajectory and side trajectory index. Default: ~",
)
@click.option(
    "--language",
    type=click.Choice(
        [
            "en",
            "english",
            "de",
            "ger",
            "german",
            "Deutsch",
        ],
        case_sensitive=False,
    ),
    multiple=False,
    default=("en"),
    help="Choose language. Default: en",
)
def main(
    *,
    info_prefix: str,
    start_prefix: str,
    traj_prefix: str,
    input_dir: str,
    output_dir: str,
    separator: str,
    language: str,
) -> None:

    prefix_dict, language = interpret_options(
        start_prefix=start_prefix,
        traj_prefix=traj_prefix,
        info_prefix=info_prefix,
        language=language,
    )
    trajectory_dict, plot_info_dict, keys = check_input_dir(
        input_dir=input_dir, prefix_dict=prefix_dict, separator=separator
    )

    trajectory_dict[keys[0]].to_csv(
        output_dir + "/" + keys[0] + "_traj.csv", index=True
    )

    plot_altitude(
        trajectory_dict=trajectory_dict,
        output_dir=output_dir,
        separator=separator,
        language=language,
    )

    # plot_map(outpath=output_dir)

    print("--- Done.")
