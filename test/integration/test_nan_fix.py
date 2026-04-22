import numpy as np
import pandas as pd

from pytrajplot.plotting.plot_map import _filter_nan_coordinates


def test_filter_nan_coordinates_filters_series_with_nan():
    lon_series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    lat_series = pd.Series([10.0, np.nan, 30.0, 40.0, 50.0])

    lon_filtered, lat_filtered = _filter_nan_coordinates(lon_series, lat_series)

    assert len(lon_filtered) == 3
    assert list(lon_filtered.values) == [1.0, 4.0, 5.0]
    assert list(lat_filtered.values) == [10.0, 40.0, 50.0]


def test_filter_nan_coordinates_preserves_numpy_array_format():
    lon_array = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    lat_array = np.array([10.0, np.nan, 30.0, 40.0, 50.0])

    lon_filtered, lat_filtered = _filter_nan_coordinates(lon_array, lat_array)

    assert isinstance(lon_filtered, np.ndarray)
    assert isinstance(lat_filtered, np.ndarray)
    assert len(lon_filtered) == 3
    assert list(lon_filtered) == [1.0, 4.0, 5.0]
    assert list(lat_filtered) == [10.0, 40.0, 50.0]


def test_filter_nan_coordinates_preserves_valid_values():
    lon_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    lat_series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

    lon_filtered, lat_filtered = _filter_nan_coordinates(lon_series, lat_series)

    assert len(lon_filtered) == 5
    assert list(lon_filtered.values) == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(lat_filtered.values) == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_filter_nan_coordinates_returns_empty_for_all_nan():
    lon_series = pd.Series([np.nan, np.nan, np.nan])
    lat_series = pd.Series([np.nan, np.nan, np.nan])

    lon_filtered, lat_filtered = _filter_nan_coordinates(lon_series, lat_series)

    assert len(lon_filtered) == 0
    assert len(lat_filtered) == 0
