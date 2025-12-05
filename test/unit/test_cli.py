"""Test module ``pytrajplot``."""

# Third-party
from click.testing import CliRunner

# First-party
from pytrajplot import main


class TestCLI:
    """Test the command line interface."""

    def call(self, args=None):
        runner = CliRunner()
        return runner.invoke(main.cli, args)

    def test_help(self):
        result = self.call(["--help"])
        assert result.exit_code == 0
        assert "Show this message and exit." in result.output

    def test_version(self):
        result = self.call(["-V"])
        assert result.exit_code == 0
        assert main.__version__ in result.output
