import pytest
from pynajax.utils import _revert_epochs, _get_complement_slicing, _odd_ext_multiepoch
import numpy as np
from scipy.signal._arraytools import odd_ext
import pynapple as nap


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


@pytest.mark.parametrize("starts, ends, max_len, expected_out",
                         [
                             ([0, 10], [4, 15], 30, [np.hstack([np.arange(4, 10), np.arange(15, 30)]), [4, 15], [10, 30]]),
                             ([0, 10], [4, 30], 30, [np.hstack([np.arange(4, 10)]), [4], [10]]),
                             ([3, 10], [4, 30], 30, [np.hstack([np.arange(0, 3), np.arange(4, 10)]), [0, 4], [3, 10]]),
                             ([1, 10], [4, 29], 30, [np.hstack([np.arange(0, 1), np.arange(4, 10), [29]]), [0, 4, 29], [1, 10, 30]]),
                         ])
def test_complementary_indices(starts, ends, max_len, expected_out):
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    ci, si, ei = _get_complement_slicing(starts, ends, max_len)

    arr = np.arange(max_len)
    out = np.hstack([arr[s:e] for s, e in zip(si, ei)])
    assert np.all(out == arr[ci])
    assert np.all(ci == expected_out[0])
    assert np.all(si == expected_out[1])
    assert np.all(ei == expected_out[2])


@pytest.mark.parametrize("shape", [(100, ), (100, 2), (100, 2, 3)])
@pytest.mark.parametrize("npts", [10, 11])
def test_padding_strategy_single_ep(shape: tuple, npts):
    dat = np.random.normal(size=shape)
    tt = np.arange(shape[0])
    start = [0]
    end = [shape[0]]
    out_sci = odd_ext(dat, npts, axis=0)
    out_jax = _odd_ext_multiepoch(npts, tt, dat, start, end)[0]
    assert np.all(out_sci == out_jax)


@pytest.mark.parametrize("starts, ends", [([0, 30], [20, 100]), ([1, 30], [20, 90]), ([1, 30, 50], [20, 40, 90])])
@pytest.mark.parametrize("shape", [(100, ), (100, 2), (100, 2, 3)])
@pytest.mark.parametrize("npts", [10, 9])
def test_padding_strategy_multi_ep(shape: tuple, npts, starts, ends):
    dat = np.random.normal(size=shape)
    tt = np.arange(shape[0])

    if len(shape) == 1:
        tsd = nap.Tsd(tt, d=dat, time_support=nap.IntervalSet(starts, ends))
    elif len(shape) == 2:
        tsd = nap.TsdFrame(tt, d=dat, time_support=nap.IntervalSet(starts, ends))
    elif len(shape) == 3:
        tsd = nap.TsdTensor(tt, d=dat, time_support=nap.IntervalSet(starts, ends))

    # recompute data after initializing tsd
    dat = tsd.d
    tt = tsd.t
    starts = tsd.time_support.start
    ends = tsd.time_support.end

    # loop over epochs
    out_sci = []
    for ep in tsd.time_support:
        out_sci.append(odd_ext(tsd.restrict(ep).d, npts, axis=0))
    out_sci = np.concatenate(out_sci, axis=0)

    out_jax = _odd_ext_multiepoch(npts, tt, dat, starts, ends)[0]
    assert np.all(out_sci == out_jax)
