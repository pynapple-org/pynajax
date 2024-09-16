# Benchmarks

Performance comparison between `pynajax` and the normal `pynapple` backend based on `numba`.
The functions that have been optimized with `pynajax` are :

- [`convolve`](https://pynapple-org.github.io/pynapple/reference/core/time_series/#pynapple.core.time_series.BaseTsd.convolve)

- [`bin_average`](https://pynapple-org.github.io/pynapple/reference/core/time_series/#pynapple.core.time_series.BaseTsd.bin_average)

- [`threshold`](https://pynapple-org.github.io/pynapple/reference/core/time_series/#pynapple.core.time_series.Tsd.threshold)

- [`event_trigger_average`](https://pynapple-org.github.io/pynapple/reference/process/perievent/#pynapple.process.perievent.compute_event_trigger_average)

- filtering