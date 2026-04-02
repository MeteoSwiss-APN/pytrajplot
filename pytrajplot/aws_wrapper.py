"""AWS wrapper for pytrajplot.

Downloads input files from S3, invokes the standard pytrajplot CLI, then uploads output to S3.
Intended for use in AWS Step Functions / ECS Fargate tasks.

S3-specific options are defined here; all other pytrajplot options are passed through unchanged.
"""
import logging
import os
import tempfile

import boto3
import click

from pytrajplot.main import cli as pytrajplot_cli
from pytrajplot.main import print_version
from pytrajplot.s3_utils import download_s3_prefix, upload_dir_to_s3

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


@click.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    epilog="Any additional options are forwarded to pytrajplot unchanged (see: pytrajplot --help).",
)
@click.option(
    "--s3-input-bucket",
    required=True,
    envvar="S3_INPUT_BUCKET",
    help="S3 bucket containing input files.",
)
@click.option(
    "--model-name",
    required=True,
    envvar="LM_NL_C_TTAG",
    help="Model name, first path segment of the S3 input prefix (e.g. 'ICON-CH1-EPS').",
)
@click.option(
    "--model-base-time",
    required=True,
    envvar="MODEL_BASE_TIME",
    help="Model base time in YYYYMMDDHHMM format (e.g. '202504030900').",
)
@click.option(
    "--s3-output-bucket",
    required=True,
    envvar="S3_OUTPUT_BUCKET",
    help="S3 bucket for output files.",
)
@click.option(
    "--s3-output-prefix",
    default="",
    envvar="S3_OUTPUT_PREFIX",
    help="S3 key prefix (folder) for output files. Defaults to empty (bucket root).",
)
@click.option(
    "--version",
    "-V",
    help="Print version and exit.",
    is_flag=True,
    expose_value=False,
    callback=print_version,
)
@click.argument("pytrajplot_args", nargs=-1, type=click.UNPROCESSED)
def cli(
    *,
    s3_input_bucket: str,
    model_name: str,
    model_base_time: str,
    s3_output_bucket: str,
    s3_output_prefix: str,
    pytrajplot_args: tuple[str, ...],
) -> None:
    """Run pytrajplot with input/output backed by S3.

    S3 options can be supplied as environment variables for ECS / Step Functions deployments.
    All standard pytrajplot options (--language, --domain, --datatype, etc.) are passed through.
    """
    s3_client = boto3.client("s3")
    s3_input_prefix = f"{model_name}/{model_base_time[:8]}_{model_base_time[8:]}"
    s3_output_prefix = s3_output_prefix or s3_input_prefix

    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        logger.info("Downloading input files from s3://%s/%s", s3_input_bucket, s3_input_prefix)
        download_s3_prefix(s3_client, s3_input_bucket, s3_input_prefix, input_dir)

        pytrajplot_cli.main(
            args=[input_dir, output_dir, *pytrajplot_args],
            standalone_mode=False,
        )

        logger.info("Uploading output files to s3://%s/%s", s3_output_bucket, s3_output_prefix)
        upload_dir_to_s3(s3_client, output_dir, s3_output_bucket, s3_output_prefix)

    print("--- Done.")
