"""Command line interface of pytrajplot."""
from typing import Tuple, Dict
import logging
import os
import tempfile
from datetime import datetime

# Third-party
import boto3
import click

# First-party
from pytrajplot import __version__
from pytrajplot.generate_pdf import generate_pdf
from pytrajplot.parse_data import check_input_dir
from pytrajplot.utils import count_to_log_level
from pytrajplot.parsing.plot_info import check_plot_info_file
from pytrajplot.s3_utils import download_s3_prefix, upload_dir_to_s3

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

_PRODUCT_TYPE_MAP = {
    "ICON-CH1-EPS": "forecast-iconch1eps-trajectories",
    "IFS": "forecast-ifs-trajectories",
}


def print_version(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    """Print the version number and exit."""
    if value:
        click.echo(__version__)
        ctx.exit(0)

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


def _run_pytrajplot(
    input_dir: str,
    output_dir: str,
    info_name: str,
    start_prefix: str,
    traj_prefix: str,
    separator: str,
    language: str,
    domain: tuple,
    datatype: tuple,
    ssm_parameter_path: str | None,
) -> None:
    plot_info_created = check_plot_info_file(
        input_dir=input_dir,
        info_name=info_name,
        ssm_parameter_path=ssm_parameter_path,
    )

    if not plot_info_created:
        if ssm_parameter_path:
            logger.error("File %s/%s doesn't exist and couldn't be created from SSM parameter.", input_dir, info_name)
            raise click.ClickException("Missing plot_info file and failed to create from SSM parameter.")

        logger.error("File %s/%s does not exist.", input_dir, info_name)
        raise click.ClickException("Missing plot_info file.")

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

@click.command()
@click.argument("input_dir", type=click.Path(), required=False, default=None)
@click.argument("output_dir", type=click.Path(), required=False, default=None)
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
    default=None,
    help="SSM parameter path for plot_info template. Falls back to SSM_PARAMETER_PATH env var if not specified.",
)
@click.option(
    "--s3-input-bucket",
    default=None,
    envvar="S3_INPUT_BUCKET",
    help="S3 bucket containing input files. Triggers S3 mode.",
)
@click.option(
    "--model-name",
    default=None,
    envvar="LM_NL_C_TTAG",
    help="Model name, first path segment of the S3 input prefix (e.g. 'ICON-CH1-EPS'). Required in S3 mode.",
)
@click.option(
    "--model-base-time",
    default=None,
    envvar="MODEL_BASE_TIME",
    help="Model base time in YYYYMMDDHHMM format (e.g. '202504030900'). Required in S3 mode.",
)
@click.option(
    "--s3-output-bucket",
    default=None,
    envvar="S3_OUTPUT_BUCKET",
    help="S3 bucket for output files. Required in S3 mode.",
)
@click.option(
    "--s3-output-prefix",
    default="",
    envvar="S3_OUTPUT_PREFIX",
    help="S3 key prefix for output files. Defaults to the input prefix.",
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
    input_dir: str | None,
    output_dir: str | None,
    info_name: str,
    start_prefix: str,
    traj_prefix: str,
    separator: str,
    language: str,
    domain: tuple,
    datatype: tuple,
    ssm_parameter_path: str | None,
    s3_input_bucket: str | None,
    model_name: str | None,
    model_base_time: str | None,
    s3_output_bucket: str | None,
    s3_output_prefix: str,
) -> None:
    """
    Pytrajplot can be run on CSCS or on AWS backed by S3.
    In CSCS mode provide INPUT_DIR and OUTPUT_DIR.
    In AWS mode provide --s3-input-bucket, --model-name, --model-base-time and
    --s3-output-bucket; INPUT_DIR and OUTPUT_DIR are managed internally.
    S3 options can also be set via environment variables for AWS deployments.
    """
    s3_mode = s3_input_bucket is not None

    if s3_mode:
        missing = [
            flag for flag, val in [
                ("--model-name", model_name),
                ("--model-base-time", model_base_time),
                ("--s3-output-bucket", s3_output_bucket),
            ] if val is None
        ]
        if missing:
            raise click.UsageError(f"S3 mode requires: {', '.join(missing)}")

        try:
            base_time = datetime.strptime(model_base_time, "%Y%m%d%H%M")
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="'--model-base-time'") from e

        s3_input_prefix = f"{model_name}/{base_time:%Y%m%d}_{base_time:%H%M}"
        s3_output_prefix = s3_output_prefix or s3_input_prefix

        product_type = _PRODUCT_TYPE_MAP.get(model_name.upper())
        if product_type is None:
            product_type = f"forecast-{model_name.lower().replace('-', '')}-trajectories"
            logger.warning("Unknown model '%s'; using default product_type '%s'", model_name, product_type)
        s3_metadata = {"product_type": product_type}

        s3_client = boto3.client("s3")
        with tempfile.TemporaryDirectory() as tmp_input, tempfile.TemporaryDirectory() as tmp_output:
            logger.info("Downloading input from s3://%s/%s", s3_input_bucket, s3_input_prefix)
            download_s3_prefix(s3_client, s3_input_bucket, s3_input_prefix, tmp_input)
            _run_pytrajplot(tmp_input, tmp_output, info_name, start_prefix, traj_prefix, separator, language, domain, datatype, ssm_parameter_path)
            logger.info("Uploading output to s3://%s/%s (metadata: %s)", s3_output_bucket, s3_output_prefix, s3_metadata)
            upload_dir_to_s3(s3_client, tmp_output, s3_output_bucket, s3_output_prefix, metadata=s3_metadata)
        logger.info("Output files uploaded successfully!")
    else:
        if input_dir is None or output_dir is None:
            raise click.UsageError("input_dir and output_dir are required when not using S3 mode.")
        _run_pytrajplot(input_dir, output_dir, info_name, start_prefix, traj_prefix, separator, language, domain, datatype, ssm_parameter_path)
