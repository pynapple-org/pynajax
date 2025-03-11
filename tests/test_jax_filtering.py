import pdb

import numpy as np
import pynapple as nap
import pytest
import scipy.signal as signal
from pynajax.jax_process_filtering import jax_sosfiltfilt

from contextlib import contextmanager
import jax
import warnings


@contextmanager
def disable_jax_x64():
    original_setting = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", original_setting)


def _naive_multiepoch_filtering(sos, tsdframe: nap.TsdFrame):
    """filter epochs in a loop."""
    out = []
    for iset in tsdframe.time_support:
        out.append(signal.sosfiltfilt(sos, tsdframe.restrict(iset).d, axis=0))
    return nap.TsdFrame(t=tsdframe.t, d=np.vstack(out), time_support=tsdframe.time_support)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 3])
def test_filter_recursion_single_band(wn, order, btype, n_sig):
    sig = np.random.normal(size=(1000, n_sig))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    out_sci = signal.sosfiltfilt(sos, sig, axis=0)
    out_jax = jax_sosfiltfilt(sos, np.arange(sig.shape[0]), sig, [0], [1000])
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [[0.1, 0.4], [0.3, 0.5]])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("btype", ["bandpass", "bandstop"])
@pytest.mark.parametrize("n_sig", [1, 3])
def test_filter_recursion_range(wn, order, btype, n_sig):
    sig = np.random.normal(size=(1000, n_sig))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    out_sci = signal.sosfiltfilt(sos, sig, axis=0)
    out_jax = jax_sosfiltfilt(sos, np.arange(sig.shape[0]), sig, [0], [1000])
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 3])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1000]),
        nap.IntervalSet(start=[0, 50], end=[40, 1000]),
        nap.IntervalSet(start=[0, 50, 100], end=[40, 90, 1000])
    ]
)
def test_filter_recursion_single_band_multiepoch_tsdframe(wn, order, btype, n_sig, ep):
    sig = np.random.normal(size=(1000, n_sig))
    sig = nap.TsdFrame(t=np.arange(sig.shape[0]), d=sig, time_support=ep)
    sos = signal.butter(order, wn, btype=btype, output="sos")
    out_sci = _naive_multiepoch_filtering(sos, sig)
    out_jax = jax_sosfiltfilt(sos, sig.t, sig.d, sig.time_support.start, sig.time_support.end)
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 3])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1000]),
        nap.IntervalSet(start=[0, 50], end=[40, 1000]),
        nap.IntervalSet(start=[0, 50, 100], end=[40, 90, 1000])
    ]
)
def test_filter_recursion_single_band_multiepoch_tsd(wn, order, btype, n_sig, ep):
    sig = np.random.normal(size=(1000, ))
    sig = nap.Tsd(t=np.arange(sig.shape[0]), d=sig, time_support=ep)
    sos = signal.butter(order, wn, btype=btype, output="sos")
    out_sci = _naive_multiepoch_filtering(sos, sig[:, None])[:, 0]
    out_jax = jax_sosfiltfilt(sos, sig.t, sig.d, sig.time_support.start, sig.time_support.end)
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 3])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1000]),
        nap.IntervalSet(start=[0, 50], end=[40, 1000]),
        nap.IntervalSet(start=[0, 50, 100], end=[40, 90, 1000])
    ]
)
def test_filter_recursion_single_band_multiepoch_tsdtensor(wn, order, btype, n_sig, ep):
    sig = np.random.normal(size=(1000, 2, 3))
    sig = nap.TsdTensor(t=np.arange(sig.shape[0]), d=sig, time_support=ep)
    sig_frame = nap.TsdFrame(t=sig.t, d=sig.d.reshape(sig.shape[0], -1), time_support=ep)
    sos = signal.butter(order, wn, btype=btype, output="sos")
    out_sci = _naive_multiepoch_filtering(sos, sig_frame).reshape(sig.shape)
    out_jax = jax_sosfiltfilt(sos, sig.t, sig.d, sig.time_support.start, sig.time_support.end)
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 3])
def test_warning_filtering_conversion(wn, order, btype, n_sig):
    sig = np.random.randint(size=(1000,), dtype=np.int16, low=0, high=2 ** 15)
    time = np.arange(len(sig))
    sos = signal.butter(order, wn, btype=btype, output="sos")
    with disable_jax_x64():
        with pytest.warns(UserWarning, match="Precision mismatch: sos coefficients "):
            jax_sosfiltfilt(sos, time, sig, np.array([0]), np.array([1000]))

    # check that float64 enabled do not raise warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jax_sosfiltfilt(sos, time, sig, np.array([0]), np.array([1000]))


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 3])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1000]),
        nap.IntervalSet(start=[0, 50], end=[40, 1000]),
        nap.IntervalSet(start=[0, 50, 100], end=[40, 90, 1000])
    ]
)
def test_integer_signal_filtering_matches_float64(wn, order, btype, n_sig, ep):
    sig = np.random.randint(size=(1000,), dtype=np.int16, low=0, high=2 ** 15)
    time = np.arange(len(sig))
    sig_int = nap.Tsd(time, sig, time_support=ep)
    sig_float = nap.Tsd(time.astype(np.float64), sig.astype(np.float64), time_support=ep)
    sos = signal.butter(order, wn, btype=btype, output="sos")
    out1 = jax_sosfiltfilt(sos, sig_int.t, sig_int.d, np.array([0]), np.array([1000]))
    out2 = jax_sosfiltfilt(sos, sig_float.t, sig_float.d, np.array([0]), np.array([1000]))
    assert np.all(out1 == out2)
