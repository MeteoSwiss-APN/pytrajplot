"""Unit tests for pytrajplot.aws_wrapper."""
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from pytrajplot.aws_wrapper import cli
from pytrajplot.main import __version__


REQUIRED_ARGS = [
    "--s3-input-bucket", "input-bucket",
    "--model-name", "ICON-CH1-EPS",
    "--model-base-time", "202504030900",
    "--s3-output-bucket", "output-bucket",
]


class TestAwsWrapperCli:
    """Tests for the aws_wrapper CLI command."""

    def call(self, args=None):
        runner = CliRunner()
        return runner.invoke(cli, args)

    def test_help(self):
        result = self.call(["--help"])
        assert result.exit_code == 0
        assert "S3" in result.output

    def test_version(self):
        result = self.call(["-V"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_missing_required_options_fails(self):
        result = self.call([])
        assert result.exit_code != 0

    @patch("pytrajplot.aws_wrapper.upload_dir_to_s3")
    @patch("pytrajplot.aws_wrapper.download_s3_prefix")
    @patch("pytrajplot.aws_wrapper.pytrajplot_cli")
    @patch("pytrajplot.aws_wrapper.boto3.client")
    def test_success_flow(self, mock_boto3, mock_pytrajplot, mock_download, mock_upload):
        mock_boto3.return_value = MagicMock()

        result = self.call(REQUIRED_ARGS)

        assert result.exit_code == 0
        mock_download.assert_called_once()
        mock_pytrajplot.main.assert_called_once()
        mock_upload.assert_called_once()

    @patch("pytrajplot.aws_wrapper.upload_dir_to_s3")
    @patch("pytrajplot.aws_wrapper.download_s3_prefix")
    @patch("pytrajplot.aws_wrapper.pytrajplot_cli")
    @patch("pytrajplot.aws_wrapper.boto3.client")
    def test_s3_input_prefix_format(self, mock_boto3, mock_pytrajplot, mock_download, mock_upload):
        """Input prefix is built as model_name/YYYYMMDD_HHMM."""
        mock_boto3.return_value = MagicMock()

        self.call(REQUIRED_ARGS)

        _, _, prefix, _ = mock_download.call_args.args
        assert prefix == "ICON-CH1-EPS/20250403_0900"

    @patch("pytrajplot.aws_wrapper.upload_dir_to_s3")
    @patch("pytrajplot.aws_wrapper.download_s3_prefix")
    @patch("pytrajplot.aws_wrapper.pytrajplot_cli")
    @patch("pytrajplot.aws_wrapper.boto3.client")
    def test_passthrough_args_forwarded_to_pytrajplot(
        self, mock_boto3, mock_pytrajplot, mock_download, mock_upload
    ):
        """Extra args after required options are forwarded to pytrajplot."""
        mock_boto3.return_value = MagicMock()

        self.call(REQUIRED_ARGS + ["--language", "de", "--domain", "ch"])

        call_args = mock_pytrajplot.main.call_args
        forwarded = call_args.kwargs.get("args") or call_args.args[0]
        assert "--language" in forwarded
        assert "de" in forwarded
        assert "--domain" in forwarded
        assert "ch" in forwarded

    @patch("pytrajplot.aws_wrapper.upload_dir_to_s3")
    @patch("pytrajplot.aws_wrapper.download_s3_prefix")
    @patch("pytrajplot.aws_wrapper.pytrajplot_cli")
    @patch("pytrajplot.aws_wrapper.boto3.client")
    def test_output_prefix_passed_to_upload(
        self, mock_boto3, mock_pytrajplot, mock_download, mock_upload
    ):
        mock_boto3.return_value = MagicMock()

        self.call(REQUIRED_ARGS + ["--s3-output-prefix", "results/2025/"])

        _, _, bucket, prefix = mock_upload.call_args.args
        assert bucket == "output-bucket"
        assert prefix == "results/2025/"

    @patch("pytrajplot.aws_wrapper.upload_dir_to_s3")
    @patch("pytrajplot.aws_wrapper.download_s3_prefix")
    @patch("pytrajplot.aws_wrapper.pytrajplot_cli")
    @patch("pytrajplot.aws_wrapper.boto3.client")
    def test_output_prefix_defaults_to_input_prefix(
        self, mock_boto3, mock_pytrajplot, mock_download, mock_upload
    ):
        """When --s3-output-prefix is omitted, output prefix matches the input prefix."""
        mock_boto3.return_value = MagicMock()

        self.call(REQUIRED_ARGS)

        _, _, _, prefix = mock_upload.call_args.args
        assert prefix == "ICON-CH1-EPS/20250403_0900"
