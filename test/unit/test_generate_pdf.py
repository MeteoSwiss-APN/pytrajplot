"""Unit tests for generate_filename."""
from datetime import datetime

from pytrajplot.generate_pdf import generate_filename


PLOT_INFO = {"model_name": "ICON-CH1-EPS"}

START_TIME = datetime(2026, 4, 1, 12, 0)

PLOT_DICT = {
    "altitude_1": {
        "start_time": START_TIME,
        "trajectory_direction": "F",
    }
}


class TestGenerateFilenamePdf:
    def test_pdf_format(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT, "Beznau", "alps", "000030", "pdf")
        assert result == "20260401T12_Beznau_LAGRANTO-ICON-CH1-EPS_Trajektorien_F_030_alps"

    def test_pdf_backward_direction(self):
        plot_dict = {"altitude_1": {"start_time": START_TIME, "trajectory_direction": "B"}}
        result = generate_filename(PLOT_INFO, plot_dict, "Beznau", "europe", "000030", "pdf")
        assert result == "20260401T12_Beznau_LAGRANTO-ICON-CH1-EPS_Trajektorien_B_030_europe"


class TestGenerateFilenamePng:
    def test_png_format(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT, "Beznau", "alps", "003030", "png")
        assert result == "forecast-iconch1eps-trajectories~20260401T12~forward~alps~Beznau~20260401T15"

    def test_png_zero_offset(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT, "Beznau", "alps", "000030", "png")
        assert result == "forecast-iconch1eps-trajectories~20260401T12~forward~alps~Beznau~20260401T12"

    def test_png_backward_direction(self):
        plot_dict = {"altitude_1": {"start_time": START_TIME, "trajectory_direction": "B"}}
        result = generate_filename(PLOT_INFO, plot_dict, "Beznau", "europe", "003030", "png")
        assert result == "forecast-iconch1eps-trajectories~20260401T12~backward~europe~Beznau~20260401T15"

    def test_png_unknown_model_fallback(self):
        plot_info = {"model_name": "ICON-CH1-CTRL"}
        result = generate_filename(plot_info, PLOT_DICT, "Beznau", "alps", "003030", "png")
        assert result == "forecast-iconch1ctrl-trajectories~20260401T12~forward~alps~Beznau~20260401T15"

    def test_png_offset_crosses_day(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT, "Beznau", "alps", "030030", "png")
        assert result == "forecast-iconch1eps-trajectories~20260401T12~forward~alps~Beznau~20260402T18"
