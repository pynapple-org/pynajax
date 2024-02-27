import jax
import pynajax as jnap
import pytest
import numpy as np
import itertools


@pytest.mark.parametrize(
    "shape_array1, shape_array2",
    [
        ((3, ), (10, 2)),
        ((3, ), (10, 2, 3))
    ]
)
def test_2d_conv(shape_array1, shape_array2):
    np.random.seed(111)
    arr1 = np.random.normal(size=shape_array1)
    arr2 = np.random.normal(size=shape_array2)

    res_numpy = np.zeros((arr2.shape[0] - arr1.shape[0]+1, *arr2.shape[1:]))
    for indices in itertools.product(*[range(dim) for dim in arr2.shape[1:]]):
        full_indices = (slice(None),) + indices
        res_numpy[full_indices] = np.convolve(arr1, arr2[full_indices], mode="valid")
