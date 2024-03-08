"""Vectorized restrict """

from functools import partial

import jax
import jax.numpy as jnp
import pynapple as nap


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

