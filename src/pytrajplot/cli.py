"""Command line interface of pytrajplot."""
# Standard library
import logging
import os

# Third-party
import click

# Local
from . import __version__
from .get_data import *
from .utils import count_to_log_level


@click.command()
@click.option(
    "--dry-run",
    "-n",
    flag_value="dry_run",
    default=False,
    help="Perform a trial run with no changes made",
)
@click.option(
    "--verbose",
    "-v",
    count=True,
    help="Increase verbosity (specify multiple times for more)",
)
@click.option(
    "--version",
    "-V",
    is_flag=True,
    help="Print version",
)
def main(*, dry_run: bool, verbose: int, version: bool) -> None:
    # plot_info = read_plot_info(os.getcwd() + "/zmichel/21101100_407/lagranto_c/plot_info")

    # plot_info_dict = check_input_dir()
    # print(f'plot_info_dict: \n {plot_info} \n')
    start_df = read_startf(
        os.getcwd() + "/zmichel/21101100_407/lagranto_c/startf_003-033F"
    )
    traj_df = read_tra_files(
        os.getcwd() + "/zmichel/21101100_407/lagranto_c/tra_geom_003-033F",
        start_df=start_df,
    )

    print("--- Done.")
