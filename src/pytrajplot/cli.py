"""Command line interface of pytrajplot."""

# Standard library
import time

# Third-party
import click

# Local
from . import __version__
from .generate_pdf import generate_pdf
from .parse_data import check_input_dir
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
    domain: str,
) -> None:

    prefix_dict, language = interpret_options(
        start_prefix=start_prefix,
        traj_prefix=traj_prefix,
        info_prefix=info_prefix,
        language=language,
    )
    end = time.perf_counter()

    print("--- Parsing Input Files")
    start_parse = time.perf_counter()
    trajectory_dict, plot_info_dict, keys = check_input_dir(
        input_dir=input_dir, prefix_dict=prefix_dict, separator=separator
    )
    end_parse = time.perf_counter()
    print(f"Parsing Input Files took {end_parse-start_parse} s")

    print("--- Assembling PDF")
    start_pdf = time.perf_counter()
    generate_pdf(
        trajectory_dict=trajectory_dict,
        plot_info_dict=plot_info_dict,
        output_dir=output_dir,
        separator=separator,
        language=language,
        domains=domain,
    )
    end_pdf = time.perf_counter()
    print(f"Assembling final PDF for: {domain} took {end_pdf-start_pdf} s")

    print("--- Done.")

    return
