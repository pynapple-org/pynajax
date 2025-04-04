import jax
import jax.numpy as jnp
import numpy as np
from numba import jit


@jit(nopython=True)
def _get_mask_and_count(time_array, starts, ends):
    """
    Calculate mask and count of time points within specified epochs.

    Parameters
    ----------
    time_array : numpy.ndarray
        Array of time points.
    starts : numpy.ndarray
        Start times of the epochs.
    ends : numpy.ndarray
        End times of the epochs.

    Returns
    -------
    ix : numpy.ndarray
        Boolean array where True indicates a time point is within an epoch.
    count : numpy.ndarray
        Array of counts of time points within each epoch.
    """
    # Initialize variables: total number of time points, total epochs,
    # a mask for time points within epochs, and a count of time points per epoch.
    n = len(time_array)
    m = len(starts)
    ix = np.zeros(n, dtype=np.bool_)
    count = np.zeros(m, dtype=np.int64)

    # Epoch index
    k = 0
    # Time point index
    t = 0

    # Advance through epochs until finding one that includes the first time point.
    while ends[k] < time_array[t]:
        k += 1

    # Process all epochs.
    while k < m:
        # Advance the time index until reaching a time point within the current epoch.
        while t < n and time_array[t] < starts[k]:
            t += 1

        # Process all time points within the current epoch.
        while t < n and time_array[t] <= ends[k]:
            ix[t] = True  # Mark time point as within an epoch.
            count[k] += 1  # Increment count for the current epoch.
            t += 1

        # Prepare for next epoch or exit if done.
        k += 1
        if k == m or t == n:
            break

    return ix, count


@jit(nopython=True)
def _get_bin_edges(time_array, starts, ends, binsize):
    """
    Compute bin edges for averaging within specified epochs and identify
    within-epoch time points.

    Parameters
    ----------
    time_array : numpy.ndarray
        Array of time points.
    starts : numpy.ndarray
        Start times of the epochs.
    ends : numpy.ndarray
        End times of the epochs.
    binsize : float
        Size of each bin.

    Returns
    -------
    ix : numpy.ndarray
        Boolean array where True indicates a time point is within an epoch.
    edges : numpy.ndarray
        Array of bin edges.
    in_epoch : numpy.ndarray
        Boolean array indicating if the interval between edges lies within an epoch.
    """
    # Preliminary calculations: mask and count of time points in each epoch.
    ix, count_in_epoch = _get_mask_and_count(time_array, starts, ends)

    # Calculate the total number of bins per epoch and overall edges needed.
    m = starts.shape[0]
    nb_bins = np.zeros(m, dtype=np.int32)
    for k in range(m):
        if (ends[k] - starts[k]) > binsize:
            nb_bins[k] = int(np.ceil((ends[k] + binsize - starts[k]) / binsize))
        else:
            nb_bins[k] = 1

    # Initialize arrays for bin edges and their epoch inclusion status.
    nb = np.sum(nb_bins) + starts.shape[0]
    edges = np.zeros(nb, dtype=np.float64)
    in_epoch = np.ones(nb, dtype=np.bool_)

    k = 0  # Epoch index
    edge_idx = 0  # Edge index
    t = 0  # sample index
    b = 0  # bin index

    while k < m:
        maxb = b + nb_bins[k]
        maxt = t + count_in_epoch[k]
        lbound = starts[k]

        # add all left edges
        while b < maxb:
            xpos = lbound
            if xpos + binsize / 2 > ends[k]:
                break
            else:
                edges[edge_idx] = np.around(xpos, 9)
                lbound += binsize
                b += 1
                edge_idx += 1

        # add final edge of epoch
        edges[edge_idx] = edges[edge_idx - 1] + np.around(binsize, 9)
        in_epoch[edge_idx] = False

        edge_idx += 1

        t = maxt
        k += 1

    return ix, edges[:edge_idx], in_epoch[: edge_idx - 1]


@jax.jit
def jit_average(bins, data, edges):
    # Initialize arrays for sums and counts per bin, and perform bin-wise addition.
    n_bins = len(edges) - 1
    sums = jnp.zeros((n_bins, *data.shape[1:])).at[bins].add(data)
    counts = jnp.zeros(n_bins).at[bins].add(1)

    average = (sums.T / counts).T
    return average


def bin_average(time_array, data_array, starts, ends, binsize):
    """
    Perform bin-averaging of data array based on time array within specified epochs.

    Parameters
    ----------
    time_array : numpy.ndarray
        Array of time points.
    data_array : jax.numpy.ndarray
        Multidimensional data array to be averaged, where the first dimension matches time_array.
    starts : numpy.ndarray
        Start times of the epochs.
    ends : numpy.ndarray
        End times of the epochs.
    binsize : float
        Size of each bin.

    Returns
    -------
    time_array_new : jax.numpy.ndarray
        New time array corresponding to the bin centers.
    data_array_new : jax.numpy.ndarray
        New data array containing the averaged values.
    """
    if not isinstance(data_array, jnp.ndarray):
        data_array = jnp.asarray(data_array)

    # Calculate bin edges and identify time points within epochs for averaging.
    ix, edges, in_epoch = _get_bin_edges(time_array, starts, ends, binsize=binsize)

    # Digitize time points to find corresponding bins, adjusting indices to be 0-based.
    bins = np.digitize(time_array[ix], edges) - 1
    if len(bins):
        average = jit_average(bins, data_array[ix], edges)
        # Create a new time array with bin centers, and filter by in-epoch bins.
        time_array_new = edges[:-1] + binsize / 2
        return time_array_new[in_epoch], average[in_epoch]
    else:
        return np.array([]), jnp.asarray([])
