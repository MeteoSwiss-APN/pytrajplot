"""Command line interface of pytrajplot."""
# Standard library
import json
import logging
import os

# Third-party
import click

# Local
from . import __version__
from .get_data import *
from .utils import count_to_log_level


@click.command()
@click.argument("dir_path", type=click.Path(exists=True))
@click.option(
    "--prefixes",
    type=(str, str, str),
    default=("startf_", "tra_geom_", "plot_info"),
    help="Enter the prefixes for the start, trajectory and plot-info file. (ORDER RELEVANT)",
)
# @click.option(
#     "--start-prefix",
#     #default="startf_",
#     type=str,
#     help="Prefix for the start files. Default: startf_",
# )
# @click.option(
#     "--traj-prefix",
#     #default="tra_geom_",
#     type=str,
#     help="Prefix for the start files. Default: tra_geom_",
# )
# @click.option(
#     "--info-prefix",
#     #default="plot_info",
#     type=str,
#     help="Prefix for the plot info files. Default: plot_info",
# )
def main(*, dir_path: str, prefixes: tuple) -> None:

    # prefix_dict = {
    #     "start": start_prefix,
    #     "trajectory": traj_prefix,
    #     "plot_info": info_prefix,
    # }

    print(type(dir_path))

    prefix_dict = {
        "start": prefixes[0],
        "trajectory": prefixes[1],
        "plot_info": prefixes[2],
    }

    plot_info_dict, trajectory_dict, keys = check_input_dir(
        dir_path=dir_path, prefix_dict=prefix_dict
    )

    print("--- Done.")
