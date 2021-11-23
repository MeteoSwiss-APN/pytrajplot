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
from .assemble_pdf import assemble_pdf
from .get_data import check_input_dir
from .plot_altitude import plot_altitude
from .plot_map import plot_map
from .plot_map_and_altitude import generate_pdf
from .utils import count_to_log_level

# from pytrajplot import plot_map_and_altitude


def interpret_options(start_prefix, traj_prefix, info_prefix, language):
    """Reformat command line inputs.

    Args:
        start_prefix (str): Prefix of start files
        traj_prefix (str):  Prefix of trajectory files
        info_prefix (str):  Prefix of plot info files
        language (str):     Language for plot annotations

    Returns:
        prefix_dict (dict): Dictionary, collecting the prefix-inputs
        language (str):     Correctly assigned language, based on various inputs

    """
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
    "--altitude",
    type=bool,
    # is_flag=True,
    default=True,
    help="Choose, to create altitude plots. isFlag: True",
)
@click.option(
    "--map",
    type=bool,
    default=True,
    # is_flag=True,
    help="Choose, to create map plots. isFlag: True",
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
@click.option(
    "--domain",
    type=click.Choice(
        ["ch_hd", "ch", "europe", "centraleurope", "alps", "dynamic"],
        case_sensitive=False,
    ),
    multiple=True,
    default=("centraleurope", "europe", "dynamic", "ch_hd"),
    help="Choose domains for map plots. Default: centraleurope, europe, dynamic",
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
    altitude: bool,
    map: bool,
    domain: str,
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

    if altitude and map:
        print("--- Assembling PDF.")
        generate_pdf(
            trajectory_dict=trajectory_dict,
            output_dir=output_dir,
            separator=separator,
            language=language,
            domains=domain,
        )
        print("--- Done.")

    return

    if altitude:
        alt_plot_dict = plot_altitude(
            trajectory_dict=trajectory_dict,
            output_dir=output_dir,
            separator=separator,
            language=language,
        )

    if map:
        map_plot_dict = plot_map(
            trajectory_dict=trajectory_dict,
            separator=separator,
            output_dir=output_dir,
            domains=domain,
            language=language,
        )

    # TODO: write assemble pdf pipeline
    assemble_pdf(altitude_axes=alt_plot_dict, map_axes=map_plot_dict, domains=domain)
