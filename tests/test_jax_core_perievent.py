import numpy as np
from pynajax.jax_process_perievent import event_trigger_average, pad_and_roll
from pynajax.utils import _get_idxs
import pytest

# Define test cases for each function

@pytest.mark.parametrize("count_array, windows, expected_result", [
    (
        np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]),
        (2, 0),
        np.array([
            [[np.nan, np.nan, np.nan]],
            [[1., 2., 3.]],
            [[np.nan, np.nan, np.nan]]
        ])
    ),
    # Add more test cases with correct shapes
])
def test_pad_and_roll(count_array, windows, expected_result):
    result = pad_and_roll(count_array, windows)
    np.testing.assert_array_equal(np.nan_to_num(result), np.nan_to_num(expected_result), err_msg="Arrays do not match.")
    assert np.all(np.isnan(result) == np.isnan(expected_result)), "NaN positions do not match."


def test_get_idxs():
    # Define test input
    time_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    starts = np.array([2, 3, 5])
    ends = np.array([2.9, 4.9, 7])

    # Call the function
    idx_start, idx_end = _get_idxs(time_array, starts, ends)

    # Define the expected output
    expected_idx_start = np.array([1, 2, 4])
    expected_idx_end = np.array([2, 4, 7])

    # Assert statements to check the result
    assert np.array_equal(idx_start, expected_idx_start)
    assert np.array_equal(idx_end, expected_idx_end)


def test_event_trigger_average():
    # Define test input
    time_target_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    count_array = np.array([[1, 2, 3], [4, 5, 6]])
    time_array = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    data_array = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
    starts = np.array([0.1, 0.2])
    ends = np.array([0.3, 0.4])
    windows = (1, 1)
    binsize = 0.01
    batch_size = 2

    # Call the function
    result = event_trigger_average(time_target_array, count_array, time_array, data_array, starts, ends, windows,
                                   binsize, batch_size)

    # Define the expected output
    expected_result = np.array(
        [[[0.1, 0.2], [0.2, 0.3]], [[0.3, 0.4], [0.4, 0.5]], [[0.5, 0.6], [0.6, 0.7]], [[0.7, 0.8], [0.8, 0.9]],
         [[0.9, 1.0], [1.0, 1.1]]])

    # Assert statements to check the result
    assert np.array_equal(result, expected_result)
