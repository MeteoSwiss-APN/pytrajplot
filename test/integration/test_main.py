"""Test module ``pytrajplot``."""

# Third-party
from click.testing import CliRunner
import pytest

# First-party
from pytrajplot import main


hres_test_input = [
    ['1_altitudes', '1_altitudes', {'datatype': 'png', 'domain': 'europe'}],
    ['2_altitudes', '2_altitudes', {'datatype': 'png', 'domain': 'europe'}],
    ['3_altitudes', '3_altitudes', {'datatype': 'png', 'domain': 'europe'}],
    ['4_altitudes', '5_altitudes', {'datatype': 'png', 'domain': 'europe'}],

    # backward
    ['backward', 'backward', {'datatype': 'png', 'domain': ['europe','dynamic']}],

    # all domains combined
    ['4_altitudes', '4_altitudes', {
        'datatype': 'png',
        'domain': ['ch', 'alps', 'centraleurope', 'europe', 'dynamic']
    }],

    # dateline crossing
    ['dateline', 'dateline', {'datatype': 'png', 'domain': 'dynamic'}],

    # german case (dateline + language)
    ['dateline', 'dateline/german', {
        'datatype': 'png',
        'domain': 'dynamic',
        'language': 'de'
    }],

    # zero longitude from east
    ['zero_lon_from_east', 'zero_lon_from_east', {}],

    # zero last longitude
    ['zero_last_lon', 'zero_last_lon', {}],

    # zero lon + dateline crossing
    ['zero_lon_dateline', 'zero_lon_dateline', {
        'datatype': 'png',
        'domain': 'dynamic'
    }],
]


def create_args(input_dir: str, output_dir: str, opts: dict):
    args = []

    # Positional arguments
    args.append(input_dir)
    args.append(output_dir)

    # Keyword arguments
    for key, value in opts.items():
        cli_key = f"--{key}"

        if isinstance(value, list):
            # Multiple flags: --domain ch --domain europe ...
            for v in value:
                args.append(cli_key)
                args.append(str(v))
        elif value is not None:
            args.append(cli_key)
            args.append(str(value))
        else:
            # Flags with no value (not needed here, but safe)
            args.append(cli_key)
    return args

@pytest.mark.parametrize("input", hres_test_input)
def test_hres(input, hres_tests, output_dir):
    input_dir, out_dir, opts = input
    input_dir = str(hres_tests / input_dir)
    out_dir = str(output_dir / out_dir)
    args = create_args(input_dir, out_dir, opts)
    print(args)
    runner = CliRunner()
    runner.invoke(main.cli, args)
    assert False
