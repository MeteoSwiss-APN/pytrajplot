"""plot info file support."""

# Standard library
from typing import Any
from typing import Dict
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

    def __init__(self, file) -> None:
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
