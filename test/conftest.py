import os
import shutil
from pathlib import Path
from pytrajplot import main
import pytest

def pytest_sessionstart(session):
    print(f"\n=== pytrajplot version: {main.__version__} ===")
    print(f"\n=== pytrajplot: {shutil.which('pytrajplot')} ===")


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


@pytest.fixture(scope="function")
def input_dir(request) -> Path:
    """Return the appropriate input directory based on the active test's
    parameterized 'input_args' value (uses its 'model' key). Falls back to
    'hres' if the model cannot be determined.
    """
    pwd: Path = Path(os.path.dirname(os.path.realpath(__file__)))

    # default model
    model = 'hres'

    # Try to extract the 'input_args' parameter from the current test's
    # callspec so we can pick the correct model directory.
    callspec = getattr(getattr(request, 'node', None), 'callspec', None)
    if callspec is not None:
        input_args = callspec.params.get('input_args')
        if isinstance(input_args, dict):
            model = input_args.get('model', model)

    return pwd / 'integration' / model
