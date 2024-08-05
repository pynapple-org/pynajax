import numpy as np
import pytest
from pynajax.jax_process_perievent import event_trigger_average
from pynajax.utils import _get_idxs, pad_and_roll


@pytest.mark.parametrize(
    "windows, array, expected_out",
    [
        (
                (1, 0),
                np.array([[1., 2.], [3., 4.], [5., 6.]]),
                np.array([[[1., 2.], [3., 4.], [5., 6.]], [[np.nan, np.nan], [1., 2.], [3., 4.]]])
        ),
        (
                (1, 1),
                np.array([[1., 2.], [3., 4.], [5., 6.]]),
                np.array([[[3., 4.], [5., 6.],[np.nan, np.nan]], [[1., 2.], [3., 4.], [5., 6.]], [[np.nan, np.nan], [1., 2.], [3., 4.]]])
        ),
        (
                (0, 1),
                np.array([[1., 2.], [3., 4.], [5., 6.]]),
                np.array([[[3., 4.], [5., 6.], [np.nan, np.nan]], [[1., 2.], [3., 4.], [5., 6.]]])
        ),
        (
                (2, 0),
                np.array([[1., 2.], [3., 4.], [5., 6.]]),
                np.array([[[1., 2.], [3., 4.], [5., 6.]], [[np.nan, np.nan], [1., 2.], [3., 4.]], [[np.nan, np.nan], [np.nan, np.nan], [1., 2.]]])
        ),
        (
                (2, 2),
                np.array([[1., 2.], [3., 4.], [5., 6.]]),
                np.array([
                    [[5., 6.], [np.nan, np.nan],  [np.nan, np.nan]],
                    [[3., 4.], [5., 6.], [np.nan, np.nan]],
                    [[1., 2.], [3., 4.], [5., 6.]],
                    [[np.nan, np.nan], [1., 2.], [3., 4.]],
                    [[np.nan, np.nan], [np.nan, np.nan], [1., 2.]],
                ])
        ),
        (
                (0, 2),
                np.array([[1., 2.], [3., 4.], [5., 6.]]),
                np.array([
                    [[5., 6.], [np.nan, np.nan],  [np.nan, np.nan]],
                    [[3., 4.], [5., 6.], [np.nan, np.nan]],
                    [[1., 2.], [3., 4.], [5., 6.]]
                ])
        ),
    ]
)
def test_pad_and_roll(windows, array, expected_out):
    out = np.array(pad_and_roll(array, windows))
    np.testing.assert_array_equal(out, expected_out)


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


@pytest.mark.parametrize(
    "counts, stimuli, expected_out",
    [
        (
                np.array([[0.], [0.], [1.], [0.], [0.]]),
                np.array([0., 0., 2., 0., 0.]),
                np.array([[0.], [2.], [0.]])
        ),
        (
                np.array([[0.], [1.], [1.], [0.], [0.]]),
                np.array([0., 0., 2., 0., 0.]),
                np.array([[0.], [1.], [1.]])
        ),
        (
                np.array([[0.], [0.], [1.], [1.], [0.]]),
                np.array([0., 0., 2., 0., 0.]),
                np.array([[1.], [1.], [0.]])
        ),
        (
                np.array([[0.], [0.], [0.], [0.], [1.]]),
                np.array([0., 0., 0., 2., 0.]),
                np.array([[2.], [0.], [0.]])
        )
    ]
)
def test_event_trigger_average_single_ep(counts, stimuli, expected_out):
    time = np.arange(counts.shape[0])
    windows = (1, 1)
    binsize = 1.
    starts = [0]
    ends = [6]
    out = event_trigger_average(time, counts, time, stimuli, starts, ends, windows, binsize, batch_size=128)
    np.testing.assert_array_equal(out, expected_out)

@pytest.mark.parametrize(
    "counts, stimuli, expected_out",
    [
        (
                np.array([[0.], [0.], [1.], [1.], [0], [0.]]),
                np.array([0., 0., 2., 2., 0., 0.]),
                np.array([[0.], [2.], [0.]])  # this would be [[1], [2], [1]] in a single epoch setting
        )
    ]
)
def test_event_trigger_average_multi_ep(counts, stimuli, expected_out):
    time = np.arange(counts.shape[0])
    windows = (1, 1)
    binsize = 1.
    starts = [0, 3]
    ends = [2, 7]
    out = event_trigger_average(time, counts, time, stimuli, starts, ends, windows, binsize, batch_size=128)
    np.testing.assert_array_equal(out, expected_out)


@pytest.mark.parametrize(
    "counts, stimuli, expected_out",
    [
        (
                np.array([[0.], [0.], [1.], [0.], [0.]]),
                np.zeros((5, 3, 4)),
                np.zeros((3, 1, 3, 4))
        ),
        (
                np.repeat(
                    np.array([[0.], [0.], [1.], [0.], [0.]]),
                    2,1
                    ),
                np.zeros((5, 12)),
                np.zeros((3, 2, 12))
        )
    ]
)
def test_event_trigger_average_batch_size(counts, stimuli, expected_out):
    stimuli[2] = 2.0
    expected_out[1] = 2.0

    time = np.arange(counts.shape[0])
    windows = (1, 1)
    binsize = 1.
    starts = [0]
    ends = [6]
    out = event_trigger_average(
        time, counts, time, stimuli, starts, ends,
        windows, binsize, batch_size=5
        )
    np.testing.assert_array_equal(out, expected_out)
