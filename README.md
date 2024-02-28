# pynajax

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/flatironinstitute/nemos/blob/main/LICENSE)
![Python version](https://img.shields.io/badge/python-3.10-blue.svg)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)


pynajax is backend for [pynapple](https://github.com/pynapple-org/pynapple) in [jax](https://github.com/google/jax). It offers a fast acceleration for the core pynapple functions using CPU or GPU. 

__The package is under active development and more methods will be added in the future.__


## Installation



## Basic usage

To use pynajax, you need to change the pynapple backend using `nap.nap_config.set_backend`. See the example below : 

```python
import pynapple as nap
import numpy as np
nap.nap_config.set_backend("jax")

tsd = nap.Tsd(t=np.arange(100), d=np.random.randn(100))

tsd.convolve(np.ones(11)) # This will run on GPU or CPU depending on the jax installation
```

## Benchmarks

This benchmark for the `convolve` function was run on a NVIDIA GeForce GTX 1060.

![benchmark_convolve](./benchmarks/convolve_benchmark.png)
