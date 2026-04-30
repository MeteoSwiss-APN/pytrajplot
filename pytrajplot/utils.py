"""Utils for the command line tool."""

# Standard library
import logging
import os

_PRODUCT_TYPE_MAP: dict[str, str] = {
    "ICON-CH1-EPS": "forecast-iconch1eps-trajectories",
    "IFS": "forecast-ifs-trajectories",
}


def get_product_type(model_name: str) -> str:
    """Derive the product_type identifier from a model name.

    Falls back to a generated name if the model is not in the map.
    Appends '-test' suffix when running on AWS (S3_BUCKET is set).
    """
    product_type = _PRODUCT_TYPE_MAP.get(model_name.upper())
    if product_type is None:
        product_type = f"forecast-{model_name.lower().replace('-', '')}-trajectories"
    if os.environ.get("S3_BUCKET"):
        product_type += "-test"
    return product_type


def count_to_log_level(count: int) -> int:
    """Map occurrence of the command line option verbose to the log level."""
    if count == 0:
        return logging.ERROR
    elif count == 1:
        return logging.WARNING
    elif count == 2:
        return logging.INFO
    else:
        return logging.DEBUG
