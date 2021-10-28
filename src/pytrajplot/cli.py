"""Command line interface of pytrajplot."""
# Standard library
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
from .utils import count_to_log_level


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
def main(
    *,
    info_prefix: str,
    start_prefix: str,
    traj_prefix: str,
    input_dir: str,
    output_dir: str,
) -> None:
    prefix_dict = {
        "start": start_prefix,
        "trajectory": traj_prefix,
        "plot_info": info_prefix,
    }

    # plot_info_dict --> contains all the information of the plot_info file
    # trajectory_dict --> dict of dataframes, contining the combined information of all start&trajectory files.
    # keys (list) --> contains all keys that occur in the trajectory dict (for each start/traj pair one key)
    # traj_block (dict) --> contains for each start/traj pair, number of trajectories and the number of rows that make up one individual trajectory
    # (it also exactly the same keys as the trajectory_dict, therefore the entries correspond to oneanother)

    trajectory_dict = check_input_dir(input_dir=input_dir, prefix_dict=prefix_dict)

    plot_altitude(trajectory_dict=trajectory_dict, output_dir=output_dir)

    print("--- Done.")
