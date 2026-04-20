#!/usr/bin/env python3
"""Test script to verify the NaN filtering fix works correctly."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/users/kaufmann/src/pytrajplot')

from pytrajplot.plotting.plot_map import _filter_nan_coordinates

# Test Case 1: Series with NaN values
print("Test Case 1: Filtering Series with NaN values")
lon_series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
lat_series = pd.Series([10.0, np.nan, 30.0, 40.0, 50.0])

lon_filtered, lat_filtered = _filter_nan_coordinates(lon_series, lat_series)
print(f"Original lon: {lon_series.values}")
print(f"Original lat: {lat_series.values}")
print(f"Filtered lon: {lon_filtered.values}")
print(f"Filtered lat: {lat_filtered.values}")
assert len(lon_filtered) == 3, "Expected 3 valid coordinates"
assert list(lon_filtered.values) == [1.0, 4.0, 5.0], "Longitude filtering failed"
assert list(lat_filtered.values) == [10.0, 40.0, 50.0], "Latitude filtering failed"
print("✓ Test Case 1 passed\n")

# Test Case 2: NumPy arrays with NaN values
print("Test Case 2: Filtering NumPy arrays with NaN values")
lon_array = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
lat_array = np.array([10.0, np.nan, 30.0, 40.0, 50.0])

lon_filtered, lat_filtered = _filter_nan_coordinates(lon_array, lat_array)
print(f"Original lon type: {type(lon_array)}")
print(f"Filtered lon type: {type(lon_filtered)}")
assert isinstance(lon_filtered, np.ndarray), "NumPy array input should return NumPy array"
assert isinstance(lat_filtered, np.ndarray), "NumPy array input should return NumPy array"
assert len(lon_filtered) == 3, "Expected 3 valid coordinates"
print("✓ Test Case 2 passed\n")

# Test Case 3: All valid values (no NaN)
print("Test Case 3: Filtering arrays with no NaN values")
lon_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
lat_series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])

lon_filtered, lat_filtered = _filter_nan_coordinates(lon_series, lat_series)
assert len(lon_filtered) == 5, "Expected 5 valid coordinates"
assert list(lon_filtered.values) == [1.0, 2.0, 3.0, 4.0, 5.0], "All values should be preserved"
print("✓ Test Case 3 passed\n")

# Test Case 4: All NaN values
print("Test Case 4: Filtering arrays with all NaN values")
lon_series = pd.Series([np.nan, np.nan, np.nan])
lat_series = pd.Series([np.nan, np.nan, np.nan])

lon_filtered, lat_filtered = _filter_nan_coordinates(lon_series, lat_series)
assert len(lon_filtered) == 0, "Expected 0 valid coordinates"
print("✓ Test Case 4 passed\n")

print("="*50)
print("All tests passed! ✓")
print("="*50)
print("\nThe fix successfully filters NaN values from trajectory coordinates")
print("before passing them to Shapely, which eliminates the warning:")
print("  RuntimeWarning: invalid value encountered in linestrings")
