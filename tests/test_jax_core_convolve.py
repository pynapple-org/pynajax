import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from pynajax import jax_core_convolve

from contextlib import nullcontext as does_not_raise


@pytest.mark.parametrize(
    "shape_array1, shape_array2", [((3,), (10, 2)), ((3,), (10, 2, 3))]
)
def test_2d_convolve_epoch_vec(shape_array1, shape_array2):
    """Compare convolution with numpy for 1D kernel"""
    np.random.seed(111)
    arr1 = np.random.normal(size=shape_array1)
    arr2 = np.random.normal(size=shape_array2)

    res_numpy = np.zeros(arr2.shape)
    for indices in itertools.product(*[range(dim) for dim in arr2.shape[1:]]):
        full_indices = (slice(None),) + indices
        res_numpy[full_indices] = np.convolve(arr1, arr2[full_indices], mode="same")

    res_pynajax = jax_core_convolve.convolve_epoch(jnp.asarray(arr2), arr1)
    assert np.allclose(res_pynajax, res_numpy)


@pytest.mark.parametrize(
    "shape_array1, shape_array2", [((3, 2), (10, 2)), ((3, 2), (10, 2, 3))]
)
def test_2d_convolve_epoch_mat(shape_array1, shape_array2):
    """Compare convolution with numpy for 2D kernel"""
    np.random.seed(111)
    arr1 = np.random.normal(size=shape_array1)
    arr2 = np.random.normal(size=shape_array2)

    res_numpy = np.zeros((*arr2.shape, arr1.shape[1]))
    for j in range(arr1.shape[1]):
        for indices in itertools.product(*[range(dim) for dim in arr2.shape[1:]]):
            full_indices = (slice(None),) + indices
            res_numpy[(*full_indices, j)] = np.convolve(
                arr1[:, j], arr2[full_indices], mode="same"
            )

    res_pynajax = jax_core_convolve.convolve_epoch(arr2, arr1)
    assert np.allclose(res_pynajax, res_numpy)
    assert isinstance(res_pynajax, jnp.ndarray)


@pytest.mark.parametrize(
    "ep",
    [
        np.array([[0, 100]]),
        np.array([[0, 19],[20,100]])
    ],
)
@pytest.mark.parametrize(
    "time_array",
    [
        np.arange(100),
    ],
)
@pytest.mark.parametrize(
    "data_array",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((10,)),
        np.ones((10, 1)),
        np.ones((10, 2)),
        jnp.ones((10,)),
        jnp.ones((10, 1)),
        jnp.ones((10, 2)),
    ],
)
def test_convolve_intervals(time_array, data_array, ep, kernel):
    """Run convolution on single and multi interval."""
    jax_core_convolve.convolve_intervals(
        time_array, 
        data_array, 
        ep[:,0],
        ep[:,1],
        kernel
        )


@pytest.mark.parametrize(
    "iset",
    [
        np.array([[0, 100]]),
        np.array([[0, 19],[20,100]])
    ],
)
@pytest.mark.parametrize(
    "data",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "time",
    [
        np.arange(100),
    ],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((10,)),
    ],
)
def test_convolve_intervals_shape_1d_kernel(time, data, iset, kernel):
    """Check that the shape of input and output matches if kernel is 1D."""    
    expected_shape = data.shape
    res = jax_core_convolve.convolve_intervals(time, data, iset[:,0], iset[:,1], kernel)
    assert all(res.shape[i] == expected_shape[i] for i in range(data.ndim))


@pytest.mark.parametrize(
    "iset",
    [
        np.array([[0, 100]]),
        np.array([[0, 19],[20,100]])
    ],
)
@pytest.mark.parametrize(
    "data",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "time",
    [
        np.arange(100),
    ],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((10, 1)),
        np.ones((10, 2)),
    ],
)
def test_convolve_intervals_shape_2d_kernel(time, data, iset, kernel):
    """
    Check that the shape of output is equal to that of the input
    plus the second dimension of the kernel attached at the end.
    """    
    expected_shape = (*data.shape, kernel.shape[1])
    res = jax_core_convolve.convolve_intervals(time, data, iset[:,0], iset[:,1], kernel)
    assert all(res.shape[i] == expected_shape[i] for i in range(data.ndim))





@pytest.mark.parametrize(
    "ep",
    [
        np.array([[0, 100]]),
        np.array([[0, 19],[20,100]])
    ],
)
@pytest.mark.parametrize(
    "time_array",
    [
        np.arange(100),
    ],
)
@pytest.mark.parametrize(
    "data_array",
    [jnp.ones((100,)), jnp.ones((100, 1)), jnp.ones((100, 2)), jnp.ones((100, 2, 3))],
)
@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((10,)),
        np.ones((10, 1)),
        np.ones((10, 2)),
        jnp.ones((10,)),
        jnp.ones((10, 1)),
        jnp.ones((10, 2)),
    ],
)
def test_convolve(time_array, data_array, ep, kernel):
    """Run convolution on single and multi interval."""
    jax_core_convolve.convolve(
        time_array, 
        data_array, 
        ep[:,0],
        ep[:,1],
        kernel
        )

def test_raise_error():
    """Run convolution on single and multi interval."""
    with pytest.raises(IOError, match=r"Provide a kernel with at least 1 dimension, current kernel has 0 dimensions"):
        jax_core_convolve.convolve(
            np.arange(100),
            jnp.ones((100,)),
            np.array([0]),
            np.array([100]),
            np.array(0)
            )