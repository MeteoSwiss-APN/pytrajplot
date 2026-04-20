"""Test module ``pytrajplot``."""

# Third-party
from click.testing import CliRunner
import pytest

# First-party
from pytrajplot import main
from pathlib import Path

# Arguments and expected outputs for pytrajplot tests
test_sets = [

    # ICON-CH1-EPS tests
    {
        'model': 'icon-ch1-eps',
        'arguments': ['backward_033-000', 'backward_033-000', {'datatype': 'png', 'domain': ['ch', 'alps']}],
        'expected_output': [
            'forecast-iconch1eps-trajectories~20260414T12~backward~alps~Geneve~20260415T21.png',
            'forecast-iconch1eps-trajectories~20260414T12~backward~ch~Geneve~20260415T21.png',
        ]
    },
    {
        'model': 'icon-ch1-eps',
        'arguments': ['forward_000-033', 'forward_000-033', {'datatype': 'png', 'domain': ['ch', 'alps']}],
        'expected_output': [
            'forecast-iconch1eps-trajectories~20260414T12~forward~alps~Geneve~20260414T12.png',
            'forecast-iconch1eps-trajectories~20260414T12~forward~ch~Geneve~20260414T12.png',
        ]
    },
    {
        'model': 'icon-ch1-eps',
        'arguments': ['forward_003-045', 'forward_003-045', {'datatype': 'png', 'domain': ['ch', 'alps']}],
        'expected_output': [
            'forecast-iconch1eps-trajectories~20260414T03~forward~alps~Beznau~20260414T06.png',
            'forecast-iconch1eps-trajectories~20260414T03~forward~alps~Goesgen~20260414T06.png',
            'forecast-iconch1eps-trajectories~20260414T03~forward~alps~Leibstadt~20260414T06.png',
            'forecast-iconch1eps-trajectories~20260414T03~forward~ch~Beznau~20260414T06.png',
            'forecast-iconch1eps-trajectories~20260414T03~forward~ch~Goesgen~20260414T06.png',
            'forecast-iconch1eps-trajectories~20260414T03~forward~ch~Leibstadt~20260414T06.png',
        ]
    },

    # IFS-Europe tests
    {
        'model': 'ifs-europe',
        'arguments': ['backward_072-000', 'backward_072-000', {'datatype': 'png', 'domain': ['alps', 'centraleurope', 'europe']}],
        'expected_output': [
            'forecast-ifseurope-trajectories~20260414T12~backward~alps~Geneve~20260417T12.png',
            'forecast-ifseurope-trajectories~20260414T12~backward~alps~Jungfraujoch~20260417T12.png',
            'forecast-ifseurope-trajectories~20260414T12~backward~centraleurope~Geneve~20260417T12.png',
            'forecast-ifseurope-trajectories~20260414T12~backward~centraleurope~Jungfraujoch~20260417T12.png',
            'forecast-ifseurope-trajectories~20260414T12~backward~europe~Geneve~20260417T12.png',
            'forecast-ifseurope-trajectories~20260414T12~backward~europe~Jungfraujoch~20260417T12.png',
        ]
    },
    {
        'model': 'ifs-europe',
        'arguments': ['forward_000-048', 'forward_000-048', {'datatype': 'png', 'domain': ['alps', 'centraleurope', 'europe']}],
        'expected_output': [
            'forecast-ifseurope-trajectories~20260414T12~forward~alps~Attisholz~20260414T12.png',
            'forecast-ifseurope-trajectories~20260414T12~forward~alps~Chateau_dOex~20260414T12.png',
            'forecast-ifseurope-trajectories~20260414T12~forward~centraleurope~Attisholz~20260414T12.png',
            'forecast-ifseurope-trajectories~20260414T12~forward~centraleurope~Chateau_dOex~20260414T12.png',
            'forecast-ifseurope-trajectories~20260414T12~forward~europe~Attisholz~20260414T12.png',
            'forecast-ifseurope-trajectories~20260414T12~forward~europe~Chateau_dOex~20260414T12.png',
        ]
    },
    {
        'model': 'ifs-europe',
        'arguments': ['forward_006-090', 'forward_006-090', {'datatype': 'png', 'domain': ['alps', 'centraleurope', 'europe']}],
        'expected_output': [
            'forecast-ifseurope-trajectories~20260414T12~forward~alps~Chernobyl~20260414T18.png',
            'forecast-ifseurope-trajectories~20260414T12~forward~centraleurope~Chernobyl~20260414T18.png',
            'forecast-ifseurope-trajectories~20260414T12~forward~europe~Chernobyl~20260414T18.png',
        ]
    },

    # IFS-Global tests
    {
        'model': 'ifs-global',
        'arguments': ['backward_144-000', 'backward_144-000', {'datatype': 'png', 'domain': ['dynamic', 'dynamic_zoom']}],
        'expected_output': [
            'forecast-ifsglobal-trajectories~20260414T12~backward~dynamic~Geneve~20260420T12.png',
            'forecast-ifsglobal-trajectories~20260414T12~backward~dynamic~Jungfraujoch~20260420T12.png',
            'forecast-ifsglobal-trajectories~20260414T12~backward~dynamic_zoom~Geneve~20260420T12.png',
            'forecast-ifsglobal-trajectories~20260414T12~backward~dynamic_zoom~Jungfraujoch~20260420T12.png',
        ]
    },
    {
        'model': 'ifs-global',
        'arguments': ['forward_000-144', 'forward_000-144', {'datatype': 'png', 'domain': ['dynamic', 'dynamic_zoom']}],
        'expected_output': [
            'forecast-ifsglobal-trajectories~20260414T12~forward~dynamic~Bushehr~20260414T12.png',
            'forecast-ifsglobal-trajectories~20260414T12~forward~dynamic~Punggye-ri~20260414T12.png',
            'forecast-ifsglobal-trajectories~20260414T12~forward~dynamic_zoom~Bushehr~20260414T12.png',
            'forecast-ifsglobal-trajectories~20260414T12~forward~dynamic_zoom~Punggye-ri~20260414T12.png'
        ]
    },
    {
        'model': 'ifs-global',
        'arguments': ['forward_006-144', 'forward_006-144', {'datatype': 'png', 'domain': ['dynamic', 'dynamic_zoom']}],
        'expected_output': [
            'forecast-ifsglobal-trajectories~20260414T12~forward~dynamic~Bushehr~20260414T18.png',
            'forecast-ifsglobal-trajectories~20260414T12~forward~dynamic~Punggye-ri~20260414T18.png',
            'forecast-ifsglobal-trajectories~20260414T12~forward~dynamic_zoom~Bushehr~20260414T18.png',
            'forecast-ifsglobal-trajectories~20260414T12~forward~dynamic_zoom~Punggye-ri~20260414T18.png',
        ]
    },

    # IFS test for special cases
    # 1, 2, 3, altitudes
    {
        'model': 'ifs-testcases',
        'arguments': ['1_altitudes', '1_altitudes', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            'forecast-ifseurope-trajectories~20210503T12~forward~europe~Linate~20210503T12.png'
        ]
    },
    {
        'model': 'ifs-testcases',
        'arguments': ['2_altitudes', '2_altitudes', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            'forecast-ifseurope-trajectories~20210503T12~forward~europe~Linate~20210503T12.png'
        ]
    },
    {
        'model': 'ifs-testcases',
        'arguments': ['3_altitudes', '3_altitudes', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            'forecast-ifseurope-trajectories~20210503T12~forward~europe~Linate~20210503T12.png'
        ]
    },

    # 4 altitudes, all domains
    {
        'model': 'ifs-testcases',
        'arguments': ['4_altitudes', '4_altitudes', {
            'datatype': 'png',
            'domain': ['ch', 'alps', 'centraleurope', 'europe', 'dynamic']
        }],
        'expected_output': [
            'forecast-ifseurope-trajectories~20210503T12~forward~alps~Linate~20210503T12.png',
            'forecast-ifseurope-trajectories~20210503T12~forward~centraleurope~Linate~20210503T12.png',
            'forecast-ifseurope-trajectories~20210503T12~forward~ch~Linate~20210503T12.png',
            'forecast-ifseurope-trajectories~20210503T12~forward~dynamic~Linate~20210503T12.png',
            'forecast-ifseurope-trajectories~20210503T12~forward~europe~Linate~20210503T12.png'
        ]
    },

    # basetime after reftime
    {
        'model': 'ifs-testcases',
        'arguments': ['basetime_after_reftime', 'basetime_after_reftime', {
            'datatype': 'png',
            'domain': 'dynamic'
        }],
        'expected_output': [
            'forecast-ifseurope-trajectories~20210503T18~forward~dynamic~Linate~20210503T12.png'
        ]
    },

    # dateline crossing, german annotations (dateline + language)
    {
        'model': 'ifs-testcases',
        'arguments': ['dateline', 'dateline', {
            'datatype': 'png',
            'domain': 'dynamic',
            'language': 'de'
        }],
        'expected_output': [
            'forecast-ifsglobal-trajectories~20210503T12~forward~dynamic~Punggye-ri~20210503T18.png'
        ]
    },

    # dateline crossing from west
    {
        'model': 'ifs-testcases',
        'arguments': ['dateline_from_west', 'dateline_from_west', {
            'datatype': 'png',
            'domain': 'dynamic',
            'language': 'de'
        }],
        'expected_output': [
            'forecast-ifsglobal-trajectories~20211223T00~backward~dynamic~Geneve~20211229T00.png'
        ]
    },

    # zero last longitude
    {
        'model': 'ifs-testcases',
        'arguments': ['zero_last_lon', 'zero_last_lon', {'datatype': 'png', 'domain': 'europe'}],
        'expected_output': [
            'forecast-ifseurope-trajectories~20230205T06~forward~europe~Sued-Ukraine~20230205T12.png'
        ]
    },

    # zero lon + dateline crossing
    {
        'model': 'ifs-testcases',
        'arguments': ['zero_lon_dateline', 'zero_lon_dateline', {
            'datatype': 'png',
            'domain': 'dynamic'
        }],
        'expected_output': [
            'forecast-ifsglobal-trajectories~20240907T00~backward~dynamic~Lugano~20240913T00.png'
        ]
    },

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
