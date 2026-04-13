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
            'forecast-ifshreseurope-trajectories~20210503T12~forward~europe~Linate~20210503T12.png'
        ]
    },
    {
        'model': 'hres',
        'arguments': ['2_altitudes', '2_altitudes', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            'forecast-ifshreseurope-trajectories~20210503T12~forward~europe~Linate~20210503T12.png'
        ]
    },
    {
        'model': 'hres',
        'arguments': ['3_altitudes', '3_altitudes', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            'forecast-ifshreseurope-trajectories~20210503T12~forward~europe~Linate~20210503T12.png'
        ]
    },

    # backward
    {
        'model': 'hres',
        'arguments': ['backward', 'backward', {'datatype': 'png', 'domain': ['europe', 'dynamic']}],
        'expected_output': [
            'forecast-ifshreseurope-trajectories~20210505T00~backward~europe~Geneve~20210506T12.png'
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
            'forecast-ifshreseurope-trajectories~20210503T12~forward~centraleurope~Linate~20210503T12.png',
            'forecast-ifshreseurope-trajectories~20210503T12~forward~alps~Linate~20210503T12.png',
            'forecast-ifshreseurope-trajectories~20210503T12~forward~dynamic~Linate~20210503T12.png',
            'forecast-ifshreseurope-trajectories~20210503T12~forward~ch~Linate~20210503T12.png',
            'forecast-ifshreseurope-trajectories~20210503T12~forward~europe~Linate~20210503T12.png'
        ]
    },

    # dateline crossing
    {
        'model': 'hres',
        'arguments': ['dateline', 'dateline', {'datatype': 'png', 'domain': 'dynamic'}],
        'expected_output': [
            'forecast-ifshres-trajectories~20210503T18~forward~dynamic~Punggye-ri~20210504T00.png'
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
            'forecast-ifshres-trajectories~20210503T18~forward~dynamic~Punggye-ri~20210504T00.png'
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
            'forecast-ifshres-trajectories~20240913T00~backward~dynamic~Lugano~20240919T00.png'
        ]
    },

    # COSMO tests
    {
        'model': 'cosmo',
        'arguments': ['forward', 'forward', {'datatype': 'png', 'domain': ['ch', 'alps']}],
        'expected_output': [
            'forecast-cosmo1e-trajectories~20211011T00~forward~ch~Geneve~20211011T00.png',
            'forecast-cosmo1e-trajectories~20211011T00~forward~alps~Geneve~20211011T00.png'
        ]
    },
    {
        'model': 'cosmo',
        'arguments': ['backward', 'backward', {'datatype': 'png', 'domain': ['ch', 'alps']}],
        'expected_output': [
            'forecast-cosmo1e-trajectories~20211103T09~backward~alps~Geneve~20211104T18.png',
            'forecast-cosmo1e-trajectories~20211103T09~backward~ch~Geneve~20211104T18.png'
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
