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
    # plot_info = read_plot_info()
    # startf = read_startf()
    # plot_info_dict = check_input_dir()
    # print(f'plot_info_dict: \n {plot_info_dict} \n')
    read_tra_files()

    print("--- Done.")
