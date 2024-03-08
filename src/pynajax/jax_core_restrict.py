"""Vectorized restrict """

from functools import partial

import jax
import jax.numpy as jnp
import pynapple as nap
import numpy as np
from numba import jit

@jax.jit
def _get_index(startend, time_array):
	return jnp.searchsorted(time_array, startend)	


def restrict(time_array, data_array, starts, ends):
	"""
	Restrict function of pynapple for `Tsd` objects
	"""
	tree = [jnp.array([s,e]) for s, e in zip(starts, ends)]

	func = partial(_get_index, time_array=time_array)

	index = jax.tree_map(lambda x:func(startend=x), tree)

	new_time_array = jnp.concatenate(jax.tree_map(lambda x:time_array[x[0]:x[1]], index))

	if data_array is not None:
		new_data_array = jnp.concatenate(jax.tree_map(lambda x:data_array[x[0]:x[1]], index), axis=0)
		return (new_time_array, new_data_array)
	else:
		return new_time_array


@jit(nopython=True)
def _get_idxs(time_array, starts, ends):
	idx_start = np.searchsorted(time_array, starts)
	idx_end = np.searchsorted(time_array, ends)
	return idx_start, idx_end


@jit(nopython=True)
def _get_indices(idx_start, idx_end, out):
	cnt = 0
	for k in range(idx_start.shape[0]):
		for j in range(idx_start[k], idx_end[k]):
			out[cnt] = j
			cnt += 1
	return out


def restrict_mixed(time_array, data_array, starts, ends):
	idx_start, idx_end = _get_idxs(time_array, starts, ends)
	size = np.sum(idx_end - idx_start)
	out = np.zeros(size, dtype=int)
	indexes = _get_indices(idx_start, idx_end, out)
	new_time_array = np.take(time_array, indexes)
	if data_array is not None:
		return new_time_array, jnp.take(data_array, indexes, axis=0)
	return new_time_array


if __name__ == "__main__":
	from time import perf_counter
	T = 10000
	time_array = np.arange(T)
	data_array = np.arange(2*T).reshape(T, 2)
	starts = np.arange(1, T-1, 20)
	ends = np.arange(1, T-1, 20) + 2

	ep = nap.IntervalSet(start=starts, end=ends)
	tsd = nap.TsdFrame(t=time_array, d=data_array)

	restrict(time_array, data_array, starts, ends)
	t0 = perf_counter()
	restrict(time_array, data_array, starts, ends)
	print("restrict", perf_counter()-t0)

	restrict_mixed(time_array, data_array, starts, ends)
	t0 = perf_counter()
	restrict_mixed(time_array, data_array, starts, ends)
	print("restrict_mixed", perf_counter()-t0)

	tsd.restrict(ep)
	t0 = perf_counter()
	tsd.restrict(ep)
	print("pynapple restrict", perf_counter() - t0)