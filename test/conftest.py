import os
import shutil
from pathlib import Path
from pytrajplot import main
import pytest

def pytest_sessionstart(session):
    print(f"\n=== pytrajplot version: {main.__version__} ===")
    print(f"\n=== pytrajplot: {shutil.which('pytrajplot')} ===")

@pytest.fixture
def hres_tests()-> Path:
    pwd: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    return pwd / 'integration' / 'hres'

@pytest.fixture
def cosmo_tests() -> Path:
    pwd: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    return pwd / 'integration' / 'cosmo'

@pytest.fixture
def output_dir() -> Path:
    pwd: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    return pwd / 'integration' / 'output'
