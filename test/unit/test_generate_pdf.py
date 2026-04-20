"""Unit tests for generate_filename."""
from datetime import datetime

from pytrajplot.generate_pdf import generate_filename


PLOT_INFO = {
    "mbt": "2026-04-01 12:00 UTC",
    "model_name": "ICON-CH1-EPS"}

START_TIME_F = datetime(2026, 4, 1, 15, 0)
START_TIME_F00 = datetime(2026, 4, 1, 12, 0)
START_TIME_F30 = datetime(2026, 4, 2, 18, 0)
START_TIME_B = datetime(2026, 4, 2, 21, 0)

PLOT_DICT_F = {
    "altitude_1": {
        "start_time": START_TIME_F,
        "trajectory_direction": "F",
    }
}

PLOT_DICT_F00 = {
    "altitude_1": {
        "start_time": START_TIME_F00,
        "trajectory_direction": "F",
    }
}

PLOT_DICT_F30 = {
    "altitude_1": {
        "start_time": START_TIME_F30,
        "trajectory_direction": "F",
    }
}

PLOT_DICT_B = {
    "altitude_1": {
        "start_time": START_TIME_B,
        "trajectory_direction": "B",
    }
}


class TestGenerateFilenamePdf:
    def test_pdf_format(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT_F, "Beznau", "alps", "003033", "pdf")
        assert result == "20260401T15_Beznau_LAGRANTO-ICON-CH1-EPS_Trajektorien_F_030_alps"

    def test_pdf_backward_direction(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT_B, "Beznau", "europe", "033000", "pdf")
        assert result == "20260402T21_Beznau_LAGRANTO-ICON-CH1-EPS_Trajektorien_B_033_europe"


class TestGenerateFilenamePng:
    def test_png_format(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT_F, "Beznau", "alps", "003030", "png")
        assert result == "forecast-iconch1eps-trajectories~20260401T12~forward~alps~Beznau~20260401T15"

    def test_png_zero_offset(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT_F00, "Beznau", "alps", "000030", "png")
        assert result == "forecast-iconch1eps-trajectories~20260401T12~forward~alps~Beznau~20260401T12"

    def test_png_backward_direction(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT_B, "Beznau", "europe", "033000", "png")
        assert result == "forecast-iconch1eps-trajectories~20260401T12~backward~europe~Beznau~20260402T21"

    def test_png_unknown_model_fallback(self):
        plot_info = {"mbt": "2026-04-01 12:00 UTC","model_name": "ICON-CH1-CTRL"}
        result = generate_filename(plot_info, PLOT_DICT_F, "Beznau", "alps", "003033", "png")
        assert result == "forecast-iconch1ctrl-trajectories~20260401T12~forward~alps~Beznau~20260401T15"

    def test_png_offset_crosses_day(self):
        result = generate_filename(PLOT_INFO, PLOT_DICT_F30, "Beznau", "alps", "030033", "png")
        assert result == "forecast-iconch1eps-trajectories~20260401T12~forward~alps~Beznau~20260402T18"
