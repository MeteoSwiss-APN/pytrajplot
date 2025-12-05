import os
import shutil
from pathlib import Path
from pytrajplot import main
import pytest

def pytest_sessionstart(session):
    print(f"\n=== pytrajplot version: {main.__version__} ===")
    print(f"\n=== pytrajplot: {shutil.which('pytrajplot')} ===")

@pytest.fixture(scope="session")
def hres_input()-> Path:
    pwd: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    return pwd / 'integration' / 'hres'

@pytest.fixture(scope="session")
def cosmo_input() -> Path:
    pwd: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    return pwd / 'integration' / 'cosmo'

@pytest.fixture(scope="session")
def output_dir() -> Path:
    """Return the output directory for integration tests.
    Delete it at the start of the session.
    """
    pwd = Path(os.path.dirname(os.path.realpath(__file__)))
    out = pwd / "integration" / "output"

    if out.exists():
        shutil.rmtree(out)
    return out
