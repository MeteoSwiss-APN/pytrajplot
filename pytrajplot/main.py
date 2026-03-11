"""Command line interface of pytrajplot."""
from typing import Tuple, Dict
import logging
import os
from pathlib import Path

# Third-party
import click
import boto3

# First-party
from pytrajplot import __version__
from pytrajplot.generate_pdf import generate_pdf
from pytrajplot.parse_data import check_input_dir
from pytrajplot.utils import count_to_log_level

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

def print_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Print the version number and exit."""
    if value:
        click.echo(__version__)
        ctx.exit(0)

def replace_variables(template_content: str) -> str:
    """
    Replace $VAR with actual environment variable values.
    Args:
        template_content: Template string with $VARIABLE placeholders
    Returns:
        String with variables replaced by environment values
    """
    result = template_content
    # Get all environment variables as dict
    env_vars = dict(os.environ)

    # Replace variables found in the template
    for env_key, env_value in env_vars.items():
        placeholder = f'${env_key}'
        if placeholder in result:
            result = result.replace(placeholder, env_value)
            logger.info(f"Replaced {placeholder} with {env_value}")
    return result


def check_plot_info_file(input_dir: str, info_name: str, ssm_parameter_path: str = None) -> bool:
    """
    Check if plot_info file exists in input directory.
    If not found, fetch from SSM parameter and create it replacing variables.
    Args:
        input_dir: Input directory path
        info_name: Name of the plot info file
        ssm_parameter_path: SSM parameter path (optional, uses env var if not provided)
    Returns:
        bool: True if file exists or was created successfully, False otherwise
    """
    input_path = Path(input_dir)
    plot_info_file = input_path / info_name

    # Check if plot_info file already exists
    if plot_info_file.exists():
        logger.info(f"Plot info file already exists: {plot_info_file}")
        return True

    # File doesn't exist, try to create it from SSM parameter
    logger.info(f"Plot info file not found: {plot_info_file}")

    try:
        # Get SSM parameter path from argument or environment
        ssm_param_path = ssm_parameter_path or os.environ.get('SSM_PARAMETER_PATH', '/pytrajplot/icon/plot_info')
        logger.info(f"Fetching SSM parameter: {ssm_param_path}")

        # Fetch template from SSM Parameter
        ssm_client = boto3.client('ssm')
        response = ssm_client.get_parameter(
            Name=ssm_param_path,
            WithDecryption=True
        )

        # Get the template content
        template_content = response['Parameter']['Value']
        logger.info(f"Template content length: {len(template_content)} chars")

        # Replace variables with environment variable values
        substituted_content = replace_variables(template_content)

        # Create the plot_info file
        with open(plot_info_file, 'w') as f:
            f.write(substituted_content)

        logger.info(f"Successfully created plot info file: {plot_info_file}")
        return True

    except Exception as e:
        logger.error(f"Failed to create plot info file from SSM parameter: {str(e)}")
        logger.error(f"SSM parameter path: {ssm_parameter_path or os.environ.get('SSM_PARAMETER_PATH', 'not_set')}")
        return False

def interpret_options(start_prefix: str, traj_prefix: str, info_name: str, language: str) -> Tuple[Dict[str, str], str]:
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

    if language[0] == "e" or language[0] == "E":
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
    help="Name of plot_info file. Default: plot_info",
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
        ["en", "english", "de", "ger", "german", "Deutsch", "deutsch"],
        case_sensitive=False,
    ),
    multiple=False,
    default=("en"),
    help="Choose language. Default: en",
)
@click.option(
    "--domain",
    type=click.Choice(
        [
            "ch",
            "europe",
            "centraleurope",
            "alps",
            "dynamic",
            "dynamic_zoom",
        ],
        case_sensitive=True,
    ),
    multiple=True,
    default=("centraleurope", "europe", "dynamic", "ch", "alps", "dynamic_zoom"),
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
@click.option(
    "--ssm-parameter-path",
    type=str,
    help="SSM parameter path for plot_info template. Uses SSM_PARAMETER_PATH env var if not specified.",
)
@click.option(
    "--skip-ssm-fallback",
    is_flag=True,
    default=False,
    help="Skip SSM parameter fallback if plot_info file is missing.",
)
@click.option(
    "--version",
    "-V",
    help="Print version and exit.",
    is_flag=True,
    expose_value=False,
    callback=print_version,
)
def cli(
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
    ssm_parameter_path: str = None,
    skip_ssm_fallback: bool = False,
) -> None:
    # Check if plot_info file exists (create from SSM if needed)
    if not skip_ssm_fallback:
        plot_info_created = check_plot_info_file(
            input_dir=input_dir,
            info_name=info_name,
            ssm_parameter_path=ssm_parameter_path
        )

        if not plot_info_created:
            logger.error("Failed to check if plot_info file exists. Use --skip-ssm-fallback to continue anyway.")
            raise click.ClickException("Missing plot_info file and failed to create from SSM parameter.")

    prefix_dict, language = interpret_options(
        start_prefix=start_prefix,
        traj_prefix=traj_prefix,
        info_name=info_name,
        language=language,
    )
    trajectory_dict, plot_info_dict = check_input_dir(
        input_dir=input_dir, prefix_dict=prefix_dict, separator=separator
    )
    generate_pdf(
        trajectory_dict=trajectory_dict,
        plot_info_dict=plot_info_dict,
        output_dir=output_dir,
        separator=separator,
        language=language,
        domains=domain,
        output_types=datatype,
    )
    print("--- Done.")
