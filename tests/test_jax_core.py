import jax
import pynajax as jnap
import pytest
import numpy as np
import itertools

import pynajax.jax_core
import pynapple as nap
from mock import MockArray

@pytest.mark.parametrize(
    "shape_array1, shape_array2",
    [
        ((3, ), (10, 2)),
        ((3, ), (10, 2, 3))
    ]
)
def test_2d_conv_vec(shape_array1, shape_array2):
    np.random.seed(111)
    arr1 = np.random.normal(size=shape_array1)
    arr2 = np.random.normal(size=shape_array2)
    if arr2.ndim == 1:
        arr2 = nap.Tsd(t=np.arange(arr2.shape[0]), d=arr2)
    elif arr2.ndim == 2:
        arr2 = nap.TsdFrame(t=np.arange(arr2.shape[0]), d=arr2)
    else:
        arr2 = nap.TsdTensor(t=np.arange(arr2.shape[0]), d=arr2)

    res_numpy = np.zeros((arr2.shape[0], *arr2.shape[1:]))
    for indices in itertools.product(*[range(dim) for dim in arr2.shape[1:]]):
        full_indices = (slice(None),) + indices
        res_numpy[full_indices] = np.convolve(arr1, arr2[full_indices].d, mode="same")

    res_pynajax = jnap.jax_core.convolve(arr2, arr1)
    assert np.allclose(res_pynajax.d, res_numpy)