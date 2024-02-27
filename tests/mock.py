import jax.numpy as jnp
import numpy as np
import pynapple as nap


def create_nap(t, d, time_support):
    if d.ndim == 1:
        return nap.Tsd(t, d, time_support=time_support)
    elif d.ndim == 2:
        return nap.TsdFrame(t, d, time_support=time_support)
    else:
        return nap.TsdTensor(t, d, time_support=time_support)


class MockArray:
    """
    A mock array class designed for testing purposes. It mimics the behavior of array-like objects
    by providing necessary attributes and supporting indexing and iteration, but it is not a direct
    instance of numpy.ndarray.
    """

    def __init__(self, t, d, time_support=None):
        """
        Initializes the MockArray with data.
        Parameters
        ----------
        data : Union[numpy.ndarray, List]
            A list of data elements that the MockArray will contain.
        """
        self._nap = create_nap(t=np.asarray(t), d=np.asarray(d), time_support=time_support)
        self.d = jnp.asarray(d)
        self.t = np.asarray(t)
        self.time_support = self._nap.time_support
        self.shape = self.d.shape # Simplified shape attribute
        self.dtype = 'float64'  # Simplified dtype; in real scenarios, this should be more dynamic
        self.ndim = self.d.ndim  # Simplified ndim for a 1-dimensional array

    def __getitem__(self, index):
        """
        Supports indexing into the mock array.
        Parameters
        ----------
        index : int or slice
            The index or slice of the data to access.
        Returns
        -------
        The element(s) at the specified index.
        """
        return self.d[index]

    def __iter__(self):
        """
        Supports iteration over the mock array.
        """
        return iter(self.d)

    def __len__(self):
        """
        Returns the length of the mock array.
        """
        return len(self.d)

    def get(self, start, end=None):
        nap_new = self._nap.get(start, end)
        return MockArray(t=nap_new.t, d=jnp.asarray(nap_new.d), time_support=nap_new.time_support)
