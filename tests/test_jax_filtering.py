import numpy as np
import pynapple as nap
import pytest
import scipy.signal as signal
from pynajax.jax_process_filtering import jax_sosfiltfilt


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

