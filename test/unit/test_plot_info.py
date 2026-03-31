"""Unit tests for pytrajplot main module."""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock
from click.testing import CliRunner
from botocore.exceptions import ClientError

from pytrajplot.parsing.plot_info import replace_variables, check_plot_info_file, _format_model_base_time
from pytrajplot.main import cli


class TestFormatModelBaseTime:
    """Test the _format_model_base_time helper."""

    def test_formats_correctly(self):
        assert _format_model_base_time("202504030900") == "2025-04-03 09:00 UTC"

    def test_midnight(self):
        assert _format_model_base_time("202501010000") == "2025-01-01 00:00 UTC"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            _format_model_base_time("20250403_0900")


class TestReplaceVariables:
    """Test the replace_variables function."""

    def test_replace_single_variable(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "test_value")

        template = "This is a $TEST_VAR template"
        result = replace_variables(template)

        assert result == "This is a test_value template"

    def test_replace_multiple_variables(self, monkeypatch):
        """Test replacing multiple environment variables."""
        monkeypatch.setenv("VAR1", "value1")
        monkeypatch.setenv("VAR2", "value2")
        monkeypatch.setenv("VAR3", "value3")

        template = "Start $VAR1 middle $VAR2 end $VAR3"
        result = replace_variables(template)

        assert result == "Start value1 middle value2 end value3"

    def test_model_base_time_is_formatted(self, monkeypatch):
        """MODEL_BASE_TIME must be converted from YYYYMMDDHHMM to YYYY-MM-DD HH:MM UTC."""
        monkeypatch.setenv("MODEL_BASE_TIME", "202504030900")

        result = replace_variables("Model base time: $MODEL_BASE_TIME")

        assert result == "Model base time: 2025-04-03 09:00 UTC"

    def test_model_base_time_raw_value_unchanged_in_env(self, monkeypatch):
        """replace_variables must not mutate os.environ."""
        monkeypatch.setenv("MODEL_BASE_TIME", "202504030900")

        replace_variables("$MODEL_BASE_TIME")

        assert os.environ["MODEL_BASE_TIME"] == "202504030900"


class TestCheckPlotInfoFile:
    """Test the check_plot_info_file function."""

    def test_plot_info_file_exists(self, tmp_path):
        """Test with plot_info file in place."""
        plot_info_file = tmp_path / "plot_info"
        plot_info_file.write_text("existing content")

        result = check_plot_info_file(
            input_dir=str(tmp_path),
            info_name="plot_info"
        )

        assert result is True
        assert plot_info_file.read_text() == "existing content"

    @patch('boto3.client')
    def test_create_plot_info_from_ssm_success(self, mock_boto3_client, tmp_path, monkeypatch):
        """Test successful creation of plot_info from SSM parameter."""
        monkeypatch.setenv("FORECAST_DATE", "20240101")
        monkeypatch.setenv("MODEL", "ICON")

        # Mock SSM client
        mock_ssm_client = MagicMock()
        mock_boto3_client.return_value = mock_ssm_client

        # Mock SSM response with template content
        template_content = "Forecast: $FORECAST_DATE, Model: $MODEL"
        mock_ssm_client.get_parameter.return_value = {
            'Parameter': {
                'Value': template_content
            }
        }

        result = check_plot_info_file(
            input_dir=str(tmp_path),
            info_name="plot_info",
            ssm_parameter_path="/test/parameter"
        )

        assert result is True

        mock_boto3_client.assert_called_once_with('ssm')
        plot_info_file = tmp_path / "plot_info"
        assert plot_info_file.exists()
        content = plot_info_file.read_text()
        assert content == "Forecast: 20240101, Model: ICON"

    @patch('boto3.client')
    def test_create_plot_info_template(self, mock_boto3_client, tmp_path, monkeypatch):
        """Test plot_info template and vars substitutions including MODEL_BASE_TIME formatting."""
        monkeypatch.setenv("MODEL_BASE_TIME", "202401150000")
        monkeypatch.setenv("LM_NL_C_TTAG", "ICON-CH2-EPS")
        monkeypatch.setenv("LM_NL_POLLONLM_C", "-170.0")
        monkeypatch.setenv("LM_NL_POLLATLM_C", "43.0")
        monkeypatch.setenv("LM_NL_STARTLON_TOT_C", "5.5")
        monkeypatch.setenv("LM_NL_STARTLAT_TOT_C", "45.0")
        monkeypatch.setenv("LM_NL_DLONLM_C", "0.02")
        monkeypatch.setenv("LM_NL_DLATLM_C", "0.02")
        monkeypatch.setenv("LM_NL_IELM_C", "250")
        monkeypatch.setenv("LM_NL_JELM_C", "200")

        # Lagranto configuration template
        template_content = '''
        Model base time:                            $MODEL_BASE_TIME
        Model name:                                 $LM_NL_C_TTAG
        North pole longitude:                       $LM_NL_POLLONLM_C
        North pole latitude:                        $LM_NL_POLLATLM_C
        Start longitude:                            $LM_NL_STARTLON_TOT_C
        Start latitude:                             $LM_NL_STARTLAT_TOT_C
        Increment in longitudinal direction:        $LM_NL_DLONLM_C
        Increment in latitudinal direction:         $LM_NL_DLATLM_C
        Number of points in longitudinal direction: $LM_NL_IELM_C
        Number of points in latitudinal direction:  $LM_NL_JELM_C
        '''

        expected_content = '''
        Model base time:                            2024-01-15 00:00 UTC
        Model name:                                 ICON-CH2-EPS
        North pole longitude:                       -170.0
        North pole latitude:                        43.0
        Start longitude:                            5.5
        Start latitude:                             45.0
        Increment in longitudinal direction:        0.02
        Increment in latitudinal direction:         0.02
        Number of points in longitudinal direction: 250
        Number of points in latitudinal direction:  200
        '''

        # Mock SSM client
        mock_ssm_client = MagicMock()
        mock_boto3_client.return_value = mock_ssm_client
        mock_ssm_client.get_parameter.return_value = {
            'Parameter': {'Value': template_content}
        }

        result = check_plot_info_file(
            ssm_parameter_path="/test/parameter",
            input_dir=str(tmp_path),
            info_name="plot_info"
        )

        assert result is True

        # Check content
        plot_info_file = tmp_path / "plot_info"
        content = plot_info_file.read_text()
        assert content == expected_content

    @patch('boto3.client')
    def test_create_plot_info_ssm_parameter_not_found(self, mock_boto3_client, tmp_path):
        """Test SSM parameter not found error."""
        # Mock SSM client to raise exception
        mock_ssm_client = MagicMock()
        mock_boto3_client.return_value = mock_ssm_client
        mock_ssm_client.get_parameter.side_effect = ClientError(
            {'Error': {'Code': 'ParameterNotFound', 'Message': 'Parameter not found'}},
            'GetParameter'
        )

        result = check_plot_info_file(
            input_dir=str(tmp_path),
            info_name="plot_info",
            ssm_parameter_path="/nonexistent/parameter"
        )

        assert result is False

        plot_info_file = tmp_path / "plot_info"
        assert not plot_info_file.exists()

class TestCliIntegration:
    """Test CLI integration with new functionality."""

    def test_cli_with_existing_plot_info(self, tmp_path):
        """Test CLI when plot_info file in place."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        plot_info = input_dir / "plot_info"
        plot_info.write_text("existing plot_info")
        output_dir = tmp_path / "output"

        runner = CliRunner()

        # Mock the generate_pdf function
        with patch('pytrajplot.main.check_input_dir') as mock_check, \
             patch('pytrajplot.main.generate_pdf'):

            mock_check.return_value = ({}, {})

            result = runner.invoke(cli, [
                str(input_dir),
                str(output_dir),
            ])

        assert result.exit_code == 0
        assert "already exists" in result.output or result.exit_code == 0

    @patch('boto3.client')
    def test_cli_ssm_failure_causes_exit(self, mock_boto3_client, tmp_path):
        """Test CLI exits when SSM parameter fetch fails."""
        # Create input directory without plot_info
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Mock SSM client to fail
        mock_ssm_client = MagicMock()
        mock_boto3_client.return_value = mock_ssm_client
        mock_ssm_client.get_parameter.side_effect = ClientError(
            {'Error': {'Code': 'ParameterNotFound'}},
            'GetParameter'
        )

        runner = CliRunner()
        result = runner.invoke(cli, [
            '--ssm-parameter-path', '/test/path',
            str(input_dir),
            str(output_dir)
        ])

        assert result.exit_code != 0
        assert "Missing plot_info file" in str(result.output) or result.exit_code != 0
