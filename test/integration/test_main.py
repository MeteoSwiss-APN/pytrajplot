"""Test module ``pytrajplot``."""

# Third-party
from click.testing import CliRunner
import pytest

# First-party
from pytrajplot import main
from pathlib import Path

# Arguments and expected outputs for pytrajplot tests
test_sets = [

    # HRES tests
    {
        'model': 'hres',
        'arguments': ['1_altitudes', '1_altitudes', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            '20210503T12_Linate_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_048_europe.png'
        ]
    },
    {
        'model': 'hres',
        'arguments': ['2_altitudes', '2_altitudes', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            '20210503T12_Linate_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_048_europe.png'
        ]
    },
    {
        'model': 'hres',
        'arguments': ['3_altitudes', '3_altitudes', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            '20210503T12_Linate_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_048_europe.png'
        ]
    },

    # backward
    {
        'model': 'hres',
        'arguments': ['backward', 'backward', {'datatype': 'png', 'domain': ['europe', 'dynamic']}],
        'expected_output': [
            '20210505T00_Geneve_LAGRANTO-IFS-HRES-Europe_Trajektorien_B_036_europe.png'
        ]
    },

    # all domains combined
    {
        'model': 'hres',
        'arguments': ['4_altitudes', '4_altitudes', {
            'datatype': 'png',
            'domain': ['ch', 'alps', 'centraleurope', 'europe', 'dynamic']
        }],
        'expected_output': [
            '20210503T12_Linate_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_048_centraleurope.png',
            '20210503T12_Linate_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_048_alps.png',
            '20210503T12_Linate_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_048_dynamic.png',
            '20210503T12_Linate_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_048_ch.png',
            '20210503T12_Linate_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_048_europe.png'
        ]
    },

    # dateline crossing
    {
        'model': 'hres',
        'arguments': ['dateline', 'dateline', {'datatype': 'png', 'domain': 'dynamic'}],
        'expected_output': [
            '20210503T18_Punggye-ri_LAGRANTO-IFS-HRES_Trajektorien_F_138_dynamic.png'
        ]
    },

    # german case (dateline + language)
    {
        'model': 'hres',
        'arguments': ['dateline', 'dateline/german', {
            'datatype': 'png',
            'domain': 'dynamic',
            'language': 'de'
        }],
        'expected_output': [
            '20210503T18_Punggye-ri_LAGRANTO-IFS-HRES_Trajektorien_F_138_dynamic.png'
        ]
    },

    # zero longitude from east
    {
        'model': 'hres',
        'arguments': ['zero_lon_from_east', 'zero_lon_from_east', {}],
        'expected_output': [
            '20220401T00_Chernobyl_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_alps.pdf',
            '20220401T00_Chernobyl_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_centraleurope.pdf',
            '20220401T00_Chernobyl_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_ch.pdf',
            '20220401T00_Chernobyl_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic.pdf',
            '20220401T00_Chernobyl_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic_zoom.pdf',
            '20220401T00_Chernobyl_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_europe.pdf',
            '20220401T00_Chmelnyzkyj_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_alps.pdf',
            '20220401T00_Chmelnyzkyj_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_alps.pdf',
            '20220401T00_Chmelnyzkyj_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_centraleurope.pdf',
            '20220401T00_Chmelnyzkyj_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_ch.pdf',
            '20220401T00_Chmelnyzkyj_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic.pdf',
            '20220401T00_Chmelnyzkyj_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic_zoom.pdf',
            '20220401T00_Chmelnyzkyj_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_europe.pdf',
            '20220401T00_Riwne_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_alps.pdf',
            '20220401T00_Riwne_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_centraleurope.pdf',
            '20220401T00_Riwne_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_ch.pdf',
            '20220401T00_Riwne_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic.pdf',
            '20220401T00_Riwne_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic_zoom.pdf',
            '20220401T00_Riwne_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_europe.pdf',
            '20220401T00_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_alps.pdf',
            '20220401T00_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_centraleurope.pdf',
            '20220401T00_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_ch.pdf',
            '20220401T00_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic.pdf',
            '20220401T00_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic_zoom.pdf',
            '20220401T00_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_europe.pdf',
            '20220401T00_Zaporozhye_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_alps.pdf',
            '20220401T00_Zaporozhye_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_centraleurope.pdf',
            '20220401T00_Zaporozhye_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_ch.pdf',
            '20220401T00_Zaporozhye_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic.pdf',
            '20220401T00_Zaporozhye_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic_zoom.pdf',
            '20220401T00_Zaporozhye_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_europe.pdf'
        ]
    },

    # zero last longitude
    {
        'model': 'hres',
        'arguments': ['zero_last_lon', 'zero_last_lon', {}],
        'expected_output': [
            '20230205T12_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic.pdf',
            '20230205T12_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_ch.pdf',
            '20230205T12_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_alps.pdf',
            '20230205T12_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_centraleurope.pdf',
            '20230205T12_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_europe.pdf',
            '20230205T12_Sued-Ukraine_LAGRANTO-IFS-HRES-Europe_Trajektorien_F_084_dynamic_zoom.pdf'
        ]
    },

    # zero lon + dateline crossing
    {
        'model': 'hres',
        'arguments': ['zero_lon_dateline', 'zero_lon_dateline', {
            'datatype': 'png',
            'domain': 'dynamic'
        }],
        'expected_output': [
            '20240913T00_Lugano_LAGRANTO-IFS-HRES_Trajektorien_B_144_dynamic.png'
        ]
    },

    # COSMO tests
    {
        'model': 'cosmo',
        'arguments': ['forward', 'forward', {'datatype': 'png', 'domain': ['ch', 'alps']}],
        'expected_output': [
            '20211011T00_Geneve_LAGRANTO-COSMO-1E_Trajektorien_F_033_ch.png',
            '20211011T00_Geneve_LAGRANTO-COSMO-1E_Trajektorien_F_033_alps.png'
        ]
    },
    {
        'model': 'cosmo',
        'arguments': ['backward', 'backward', {'datatype': 'png', 'domain': ['ch', 'alps']}],
        'expected_output': [
            '20211103T09_Geneve_LAGRANTO-COSMO-1E_Trajektorien_B_033_alps.png',
            '20211103T09_Geneve_LAGRANTO-COSMO-1E_Trajektorien_B_033_ch.png'
        ]
    }
]

def create_args(input_dir: str, output_dir: str, opts: dict) -> list:
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

@pytest.mark.parametrize("input_args", test_sets)
def test_pytrajplot(input_args, input_dir, output_dir):
    in_pathname, out_pathname, opts = input_args['arguments']
    expected = input_args.get('expected_output', [])
    model = input_args.get('model')

    # compute source input and output paths
    input_path = str(input_dir / in_pathname)
    output_path = str(output_dir / model / out_pathname)

    args = create_args(input_path, output_path, opts)
    runner = CliRunner()
    result = runner.invoke(main.cli, args)
    assert result.exit_code == 0, f"CLI exited with non-zero status: {result.exit_code}\nOutput:\n{result.output}"

    # verify expected output files were created
    for rel in expected:
        expected_file = Path(output_path) / Path(rel).name
        assert expected_file.exists(), f"Expected output not found: {expected_file}"
