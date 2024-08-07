import numpy as np
import pynapple as nap
import pytest
import scipy.signal as signal
from pynajax.iir_filtering import lfilter#, filtfilt


def _naive_multiepoch_filtering(b, a, tsdframe: nap.TsdFrame, zi: np.ndarray = None):
    """filter epochs in a loop."""
    out = []
    for iset in tsdframe.time_support:
        if zi is None:
            out.append(signal.lfilter(b, a, tsdframe.restrict(iset).d, axis=0, zi=zi))
        else:
            out.append(signal.lfilter(b, a, tsdframe.restrict(iset).d, axis=0, zi=zi)[0])
    return nap.TsdFrame(t=tsdframe.t, d=np.vstack(out), time_support=tsdframe.time_support)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 2, 3])
@pytest.mark.parametrize("use_zi", [True, False])
def test_filter_recursion_single_band(wn, order, btype, n_sig, use_zi):
    sig = np.random.normal(size=(1000, n_sig))
    b, a = signal.butter(order, wn, btype=btype)
    if use_zi:
        zi = signal.lfilter_zi(b, a)
        zi_shape = [1] * sig.ndim
        zi_shape[0] = zi.size
        zi2 = np.reshape(zi, zi_shape)
        out_sci, _ = signal.lfilter(b, a, sig, axis=0, zi=zi2)
    else:
        out_sci = signal.lfilter(b, a, sig, axis=0)
        zi = None
    out_jax = lfilter(b, a, np.arange(sig.shape[0]), sig, [0], [1000], zi=zi)
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [[0.1, 0.4], [0.3, 0.5]])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["bandpass", "bandstop"])
@pytest.mark.parametrize("n_sig", [1, 2, 3])
@pytest.mark.parametrize("use_zi", [True, False])
def test_filter_recursion_range(wn, order, btype, n_sig, use_zi):
    sig = np.random.normal(size=(1000, n_sig))
    b, a = signal.butter(order, wn, btype=btype)
    if use_zi:
        zi = signal.lfilter_zi(b, a)
        zi_shape = [1] * sig.ndim
        zi_shape[0] = zi.size
        zi2 = np.reshape(zi, zi_shape)
        out_sci, _ = signal.lfilter(b, a, sig, axis=0, zi=zi2)
    else:
        out_sci = signal.lfilter(b, a, sig, axis=0)
        zi = None
    out_jax = lfilter(b, a, np.arange(sig.shape[0]), sig, [0], [1000], zi=zi)
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 2, 3])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1000]),
        nap.IntervalSet(start=[0, 50], end=[40, 1000]),
        nap.IntervalSet(start=[0, 50, 100], end=[40, 90, 1000])
    ]
)
@pytest.mark.parametrize("use_zi", [True, False])
def test_filter_recursion_single_band_multiepoch_tsdframe(wn, order, btype, n_sig, ep, use_zi):
    sig = np.random.normal(size=(1000, n_sig))
    sig = nap.TsdFrame(t=np.arange(sig.shape[0]), d=sig, time_support=ep)
    b, a = signal.butter(order, wn, btype=btype)
    if use_zi:
        zi = signal.lfilter_zi(b, a)
        zi_shape = [1] * sig.ndim
        zi_shape[0] = zi.size
        zi2 = np.reshape(zi, zi_shape)
        out_sci = _naive_multiepoch_filtering(b, a, sig, zi=zi2)
    else:
        out_sci = _naive_multiepoch_filtering(b, a, sig)
        zi = None
    out_jax = lfilter(b, a, sig.t, sig.d, sig.time_support.start, sig.time_support.end, zi=zi)
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 2, 3])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1000]),
        nap.IntervalSet(start=[0, 50], end=[40, 1000]),
        nap.IntervalSet(start=[0, 50, 100], end=[40, 90, 1000])
    ]
)
@pytest.mark.parametrize("use_zi", [True, False])
def test_filter_recursion_single_band_multiepoch_tsd(wn, order, btype, n_sig, ep, use_zi):
    sig = np.random.normal(size=(1000, ))
    sig = nap.Tsd(t=np.arange(sig.shape[0]), d=sig, time_support=ep)
    b, a = signal.butter(order, wn, btype=btype)
    if use_zi:
        zi = signal.lfilter_zi(b, a)
        out_sci = _naive_multiepoch_filtering(b, a, sig[:, None], zi=zi[:, None])[:, 0]
    else:
        zi = None
        out_sci = _naive_multiepoch_filtering(b, a, sig[:, None])[:, 0]
    out_jax = lfilter(b, a, sig.t, sig.d, sig.time_support.start, sig.time_support.end, zi=zi)
    assert np.allclose(out_jax, out_sci)


@pytest.mark.parametrize("wn", [0.1, 0.3])
@pytest.mark.parametrize("order", [2, 4, 6])
@pytest.mark.parametrize("btype", ["lowpass", "highpass"])
@pytest.mark.parametrize("n_sig", [1, 2, 3])
@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(start=[0], end=[1000]),
        nap.IntervalSet(start=[0, 50], end=[40, 1000]),
        nap.IntervalSet(start=[0, 50, 100], end=[40, 90, 1000])
    ]
)
@pytest.mark.parametrize("use_zi", [True, False])
def test_filter_recursion_single_band_multiepoch_tsdtensor(wn, order, btype, n_sig, ep, use_zi):
    sig = np.random.normal(size=(1000, 2, 3))
    sig = nap.TsdTensor(t=np.arange(sig.shape[0]), d=sig, time_support=ep)
    sig_frame = nap.TsdFrame(t=sig.t, d=sig.d.reshape(sig.shape[0], -1), time_support=ep)

    b, a = signal.butter(order, wn, btype=btype)
    if use_zi:
        zi = signal.lfilter_zi(b, a)
        zi_shape = [1] * sig_frame.ndim
        zi_shape[0] = zi.size
        zi2 = np.reshape(zi, zi_shape)
        out_sci = _naive_multiepoch_filtering(b, a, sig_frame, zi=zi2).reshape(sig.shape)
    else:
        out_sci = _naive_multiepoch_filtering(b, a, sig_frame).reshape(sig.shape)
        zi = None
    out_jax = lfilter(b, a, sig.t, sig.d, sig.time_support.start, sig.time_support.end, zi=zi)
    assert np.allclose(out_jax, out_sci)


# @pytest.mark.parametrize("wn", [0.1, 0.3])
# @pytest.mark.parametrize("order", [2, 4, 6])
# @pytest.mark.parametrize("btype", ["lowpass", "highpass"])
# @pytest.mark.parametrize("n_sig", [1, 2, 3])
# @pytest.mark.parametrize("epoch", [nap.IntervalSet([0, 100], [98, 1000])])
# def test_lfit_single_band_multi_ep(wn, order, btype, n_sig):
#     sig = np.random.normal(size=(1000, n_sig))
#     b, a = signal.butter(order, wn, btype=btype)
#     out_sci = signal.filtfilt(b, a, sig, axis=0)
#     out_jax = filtfilt(b, a, np.arange(sig.shape[0]), sig, [0], [1000])
#     assert np.allclose(out_jax, out_sci)
#
#
# @pytest.mark.parametrize("wn", [[0.1, 0.4], [0.3, 0.5]])
# @pytest.mark.parametrize("order", [2, 4, 6])
# @pytest.mark.parametrize("btype", ["bandpass", "bandstop"])
# @pytest.mark.parametrize("n_sig", [1, 2, 3])
# def test_filtfilt_range(wn, order, btype, n_sig):
#     sig = np.random.normal(size=(1000, n_sig))
#     b, a = signal.butter(order, wn, btype=btype)
#     out_sci = signal.lfilter(b, a, sig, axis=0)
#     out_jax = lfilter(b, a, np.arange(sig.shape[0]), sig, [0], [1000])
#     assert np.allclose(out_jax, out_sci)
#
#
# @pytest.mark.parametrize("wn", [0.1, 0.3])
# @pytest.mark.parametrize("order", [2, 4, 6])
# @pytest.mark.parametrize("btype", ["lowpass", "highpass"])
# @pytest.mark.parametrize("n_sig", [1, 2, 3])
# @pytest.mark.parametrize(
#     "ep",
#     [
#         nap.IntervalSet(start=[0], end=[1000]),
#         nap.IntervalSet(start=[0, 50], end=[40, 1000]),
#         nap.IntervalSet(start=[0, 50, 80], end=[40, 90, 1000])
#     ]
# )
# def test_filtfilt_single_band_multiepoch_tsdframe(wn, order, btype, n_sig, ep):
#     sig = np.random.normal(size=(1000, n_sig))
#     sig = nap.TsdFrame(t=np.arange(sig.shape[0]), d=sig, time_support=ep)
#     b, a = signal.butter(order, wn, btype=btype)
#     out_sci = _naive_multiepoch_filtering(b, a, sig)
#     out_jax = filtfilt(b, a, sig.t, sig.d, sig.time_support.start, sig.time_support.end)
#     assert np.allclose(out_jax, out_sci)
#
#
# @pytest.mark.parametrize("wn", [0.1, 0.3])
# @pytest.mark.parametrize("order", [2, 4, 6])
# @pytest.mark.parametrize("btype", ["lowpass", "highpass"])
# @pytest.mark.parametrize("n_sig", [1, 2, 3])
# @pytest.mark.parametrize(
#     "ep",
#     [
#         nap.IntervalSet(start=[0], end=[1000]),
#         nap.IntervalSet(start=[0, 50], end=[40, 1000]),
#         nap.IntervalSet(start=[0, 50, 80], end=[40, 90, 1000])
#     ]
# )
# def test_filtfilt_single_band_multiepoch_tsd(wn, order, btype, n_sig, ep):
#     sig = np.random.normal(size=(1000, ))
#     sig = nap.Tsd(t=np.arange(sig.shape[0]), d=sig, time_support=ep)
#     b, a = signal.butter(order, wn, btype=btype)
#     out_sci = _naive_multiepoch_filtering(b, a, sig[:, None])[:, 0]
#     out_jax = lfilter(b, a, sig.t, sig.d, sig.time_support.start, sig.time_support.end)
#     assert np.allclose(out_jax, out_sci)