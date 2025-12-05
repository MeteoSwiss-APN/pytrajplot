"""Top-level package for PyTrajPlot."""

# Standard library
import importlib.metadata
from pathlib import Path

# Third-party
import cartopy

__author__ = "Michel Zeller"
__email__ = "michel.zeller@meteoswiss.ch"
__version__ = importlib.metadata.version(__package__)

del importlib


def _check_dir_exists(path):
    """Check that a directory exists."""
    if not path.exists():
        raise Exception("data directory is missing", path)
    if not path.is_dir():
        raise Exception("data directory is not a directory", path)


# Set data paths
_data_path = Path(__file__).parent / "resources"
earth_data_path = _data_path / "naturalearthdata"
cities_data_path = _data_path / "cities"
_check_dir_exists(_data_path)
_check_dir_exists(earth_data_path)
_check_dir_exists(cities_data_path)


# Point cartopy to storerd offline data
cartopy.config["pre_existing_data_dir"] = earth_data_path
cartopy.config["data_dir"] = earth_data_path
