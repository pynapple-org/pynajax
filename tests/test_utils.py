import pytest
from pynajax.utils import _revert_epochs
import numpy as np


@pytest.mark.parametrize("time_array, starts, ends, expected_out", [
    (np.arange(10), [0, 4], [3, 10], np.array([3, 2, 1, 0, 9, 8, 7, 6, 5, 4])),  # Regular case
    (np.arange(5), [0, 2.1], [2, 5], np.array([2, 1, 0, 4, 3])),                  # Smaller array
    (np.arange(6), [0], [6], np.array([5, 4, 3, 2, 1, 0])),                     # Single epoch
    (np.arange(0), [], [], np.array([])),                                       # Empty array
    (np.arange(10), [0, 3.1, 7.1], [3, 7, 10], np.array([3, 2, 1, 0, 7, 6, 5, 4, 9, 8])),  # Multiple epochs
    (np.arange(10), [0, 3.1], [3, 3.2], np.array([3, 2, 1, 0])),                       # Zero-length epoch
    (np.arange(10), [2.1], [2.3], np.array([])),                                    # Start equals end
    (np.arange(10), [0], [10], np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])),       # Full array
    (np.arange(10), [5, 7.1], [7, 7.5], np.array([7, 6, 5])),                          # Overlapping epochs
])
def test_revert_epoch(time_array, starts, ends, expected_out):
    irev = _revert_epochs(time_array, np.array(starts), np.array(ends))
    assert np.array_equal(expected_out, time_array[irev]), f"Failed for starts={starts}, ends={ends}"

