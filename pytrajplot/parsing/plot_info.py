"""plot info file support."""

# Standard library
from typing import Any
from typing import Dict
import datetime
import os
import logging
from pathlib import Path

# Third-party
import boto3


# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

class PLOT_INFO:
    """Support plot_info files.

    Attributes:
        data: Data part of the atab file.

    """

    def __init__(self, file: str | Path) -> None:
        """Create an instance of ``PLOT_INFO``.

        Args:
            file: Input file.

            sep (optional): Separator for data.

        """
        # Set instance variables
        self.file = file
        self.data: Dict[str, Any] = {}
        self._parse()

    def _parse(self) -> None:
        """Parse the plot info file."""
        # read the plot_info file
        with open(self.file, "r") as file:
            for line in file:
                elements = line.strip().split(":", maxsplit=1)
                # Skip extraction of header information if line contains no ":"
                if len(elements) == 1:
                    continue
                key, data = elements[0], elements[1].lstrip()
                if key == "Model base time":
                    self.data["mbt"] = "".join(data)
                if key == "Model name":
                    self.data["model_name"] = "".join(data)


def _format_model_base_time(raw: str) -> str:
    """Convert MODEL_BASE_TIME from YYYYMMDDHHMM to YYYY-MM-DD HH:MM UTC.

    Args:
        raw: Model base time string in YYYYMMDDHHMM format (e.g. '202504030900')

    Returns:
        Formatted string suitable for plot_info (e.g. '2025-04-03 09:00 UTC')
    """
    return datetime.datetime.strptime(raw, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M UTC")


def replace_variables(template_content: str) -> str:
    """
    Replace $VAR with actual environment variable values.
    MODEL_BASE_TIME is converted from YYYYMMDDHHMM to YYYY-MM-DD HH:MM UTC before substitution.
    Args:
        template_content: Template string with $VARIABLE placeholders
    Returns:
        String with variables replaced by environment values
    """
    result = template_content
    env_vars = dict(os.environ)

    if "MODEL_BASE_TIME" in env_vars:
        env_vars["MODEL_BASE_TIME"] = _format_model_base_time(env_vars["MODEL_BASE_TIME"])

    for env_key, env_value in env_vars.items():
        placeholder = f'${env_key}'
        if placeholder in result:
            result = result.replace(placeholder, env_value)
            logger.info("Replaced %s with %s", placeholder, env_value)
    return result


def check_plot_info_file(input_dir: str, info_name: str, ssm_parameter_path: str | None = None) -> bool:
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

    # If file exists, use it regardless of SSM config
    if plot_info_file.exists():
        logger.info("Plot info file already exists: %s", plot_info_file)
        return True
    # File doesn't exist, try to create it from SSM parameter
    ssm_param_path = ssm_parameter_path or os.environ.get('SSM_PARAMETER_PATH')
    if not ssm_param_path:
        logger.error("Plot info file not found and no ssm parameter set: %s", plot_info_file)
        return False

    try:
        logger.info("Fetching SSM parameter: %s", ssm_param_path)

        # Fetch template from SSM Parameter
        ssm_client = boto3.client('ssm')
        response = ssm_client.get_parameter(
            Name=ssm_param_path,
            WithDecryption=True
        )

        # Get the template content
        template_content = response['Parameter']['Value']
        logger.info("Template content length: %s chars", len(template_content))

        # Replace variables with environment variable values
        substituted_content = replace_variables(template_content)

        # Create the plot_info file
        with open(plot_info_file, 'w') as f:
            f.write(substituted_content)

        logger.info("Successfully created plot info file: %s", plot_info_file)
        return True

    except Exception as e:
        logger.error("Failed to create plot info file from SSM parameter: %s", e)
        logger.error("SSM parameter path: %s", ssm_parameter_path or os.environ.get('SSM_PARAMETER_PATH', 'not_set'))
        return False
