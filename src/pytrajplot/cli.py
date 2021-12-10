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


# pylint: disable=W0613  # unused-argument (param)
def print_version(ctx, param, value: bool) -> None:
    """Print the version number and exit."""
    if value:
        click.echo(__version__)
        ctx.exit(0)


def interpret_options(start_prefix, traj_prefix, info_name, language):
    """Reformat command line inputs.

    Args:
        start_prefix        str       Prefix of start files.
        traj_prefix         str       Prefix of trajectory files.
        info_prefix         str       Prefix of info file.
        language            str       language for plot annotations

    Returns:
        prefix_dict         dict      The various prefixes are stored in this dict.
        language            str       language for plot annotations after mapping. "en" or "de"

    """
    prefix_dict = {
        "start": start_prefix,
        "trajectory": traj_prefix,
        "plot_info": info_name,
    }

    if language[0] == "e":
        language = "en"
    else:
        language = "de"

    return prefix_dict, language


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
    "--info-name",
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
    default=("centraleurope", "europe", "dynamic", "ch_hd", "ch", "alps"),
    help="Choose domains for map plots. Default: centraleurope, europe, dynamic",
)
@click.option(
    "--datatype",
    type=click.Choice(
        [
            "eps",
            "jpeg",
            "jpg",
            "pdf",
            "pgf",
            "png",
            "ps",
            "raw",
            "rgba",
            "svg",
            "svgz",
            "tif",
            "tiff",
        ],
        case_sensitive=False,
    ),
    multiple=True,
    default=["pdf"],
    help="Choose data type(s) of final result. Default: pdf",
)
def main(
    *,
    info_name: str,
    start_prefix: str,
    traj_prefix: str,
    input_dir: str,
    output_dir: str,
    separator: str,
    language: str,
    domain: str,
    datatype: str,
) -> None:
    start_complete = time.perf_counter()
    start = time.perf_counter()
    prefix_dict, language = interpret_options(
        start_prefix=start_prefix,
        traj_prefix=traj_prefix,
        info_name=info_name,
        language=language,
    )
    end = time.perf_counter()
    print(f"check 1:\t{end-start} s")

    start = time.perf_counter()
    trajectory_dict, plot_info_dict, _ = check_input_dir(
        input_dir=input_dir, prefix_dict=prefix_dict, separator=separator
    )
    end = time.perf_counter()
    print(f"check 2:\t{end-start} s")

    start = time.perf_counter()
    generate_pdf(
        trajectory_dict=trajectory_dict,
        plot_info_dict=plot_info_dict,
        output_dir=output_dir,
        separator=separator,
        language=language,
        domains=domain,
        output_types=datatype,
    )
    end = time.perf_counter()
    print(f"check 3:\t{end-start} s")

    print("--- Done.")
    end_complete = time.perf_counter()
    print(f"The whole pipeline took:\t\t{end_complete-start_complete} seconds!")
    return
