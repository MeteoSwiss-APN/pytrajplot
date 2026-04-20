# Fix for Shapely RuntimeWarning: Invalid Value in Linestrings

## Problem
When running pytrajplot, the following warning was generated:
```
RuntimeWarning: invalid value encountered in linestrings
    return lib.linestrings(coords, np.intc(handle_nan), out=out, **kwargs)
```

This warning originates from Shapely (via Cartopy) when processing trajectory coordinates that contain NaN (Not-a-Number) values.

## Root Cause
The pytrajplot codebase intentionally sets NaN values in trajectory data to mark invalid/missing coordinates (in `parse_data.py` lines 363-375). These NaN values represent:
- Negative altitude values (z < 0)
- Negative surface height values (hsurf < 0)
- Missing coordinates marked with -999.00

When these trajectories are plotted with `ax.plot()` in `plotting/plot_map.py`, the NaN values are passed directly to Matplotlib/Cartopy, which internally uses Shapely to create LineStrings. Shapely produces the warning when encountering these NaN coordinates.

## Solution
Added a helper function `_filter_nan_coordinates()` in `pytrajplot/plotting/plot_map.py` that:
1. Filters out any coordinates containing NaN values before plotting
2. Maintains compatibility with both pandas Series and NumPy arrays
3. Preserves the original data format (Series → Series, array → array)

The function is applied at three key plotting locations in `add_trajectories_within_domain()`:
1. **Main trajectory plotting with domain filtering** (lines ~850-878)
2. **Dateline slice plotting** (lines ~880-895)
3. **Simple trajectory plotting (no side trajectories)** (lines ~914-930)

## Implementation Details

### New Function: `_filter_nan_coordinates()`
```python
def _filter_nan_coordinates(
    lon: pd.Series | np.ndarray,
    lat: pd.Series | np.ndarray
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """Filter out NaN values from longitude and latitude arrays."""
```

This function:
- Creates a mask for valid (non-NaN) coordinates
- Removes rows where either longitude OR latitude is NaN
- Returns filtered data in the same format as input
- Only plots segments that contain valid coordinates

### Changes Made
All three trajectory plotting sections now:
1. Call `_filter_nan_coordinates()` before passing to `ax.plot()`
2. Check if the filtered data has any valid coordinates (`if len(...) > 0`)
3. Only plot if valid data exists

## Benefits
✓ **Eliminates the RuntimeWarning** - No more Shapely warnings in pytrajplot output
✓ **Maintains data integrity** - Invalid coordinates are properly excluded from visualization
✓ **Preserves behavior** - Trajectories still look the same; only NaN gaps are omitted
✓ **Handles edge cases** - Works with both pandas Series and NumPy arrays
✓ **No breaking changes** - Fully backward compatible

## Testing
All test cases pass:
- ✓ Filtering pandas Series with NaN values
- ✓ Filtering NumPy arrays with NaN values
- ✓ Handling arrays with no NaN values (all preserved)
- ✓ Handling arrays with all NaN values (empty result)

## Files Modified
- `pytrajplot/plotting/plot_map.py` - Added `_filter_nan_coordinates()` and applied it in `add_trajectories_within_domain()`

## Verification
Run the test to verify the fix:
```bash
conda activate pytrajplot
python test_nan_fix.py
```
